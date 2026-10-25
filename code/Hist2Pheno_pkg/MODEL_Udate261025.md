# Hist2Pheno 修改报告：实现、验证与计算成本

版本基线：GitHub main `b63cfe2f32ba92439050311e7cb4892d13cc727c`。这是独立本地文件夹，未提交/推送 GitHub；原 `/workspace/Hist2Pheno` 未改动。原结构及中英文注释尽量保留，新增代码以 `2026.10.24` 标注。文件清单见 `CHANGED_FILES.md`。

## 首先处理的结论：本轮已落实的更新

优先修正训练/推理一致性及验证，再比较新模型。本轮实现了这些修正，并加入可关闭的层级/空间模块。它们通过了合成数据的正确性检查，但没有真实数据/GPU重训证据，不能声称生物学准确率已经提高。修正偏差后，原先过高的分数也可能下降。

| 审查项 | 本轮处理 |
|---|---|
| F01 scaler 与模型不绑定 | 每折保存 `.pt.artifact.pt`，包含确切训练 scaler、三头类别顺序、fold/行身份、数据指纹、空间/层级配置；新版泛癌及外部预测不再隐式 fit 全局 scaler |
| F02 验证前缀偏差 | 每轮 forward 全部验证 batch，然后进行保证类别覆盖的有上限 AUROC 采样；不再只看前 300k 行 |
| F03 仅凭 checkpoint 文件存在复用 | 只有完成标记、权重 checksum、数据/划分/参数/采样/代码签名匹配才复用。中断 fold 从头重训；没有实现 optimizer-step 级断点继续 |
| F04 best fold 和固定 fold 0 混淆 | 默认训练 section 的重新预测逐 section 选择对应 fold；内部 fold-0 页面使用 fold-0 模型与 OOF 结果；部署单模型必须显式 `--deployment` |
| F05 层级 sampler | 使用父/子类别数量及叶类细胞数计算等概率层级抽样，修复祖先细胞数反复倒数导致的过度补偿 |
| F06 attention 重载 | 根据 state_dict 和 artifact 恢复 attention、gate、尺度及几何结构；保存/加载已测试 |
| F07 过期标签缓存 | h5ad 完整内容、ontology/workbook/映射实现参与 cache stamp；失效/损坏后重建；写入采用原子替换 |
| F08 L1 口径混用 | 三头真实 OOF 概率分别保存；汇总 L1 head 使用自己的概率。原聚合 L1 别名保留兼容，同时增加 `l1_from_l2_agg_*` 明确名称 |
| F09 空间零初始化覆盖 | 通用初始化跳过 spatial_fusion，保留残差投影零初始化 |
| F10 StarDist 缩进 | 修复可选 ST 输入及 segmentation 分支的缩进，代码可解析；未实际运行 TensorFlow/StarDist 图像分割 |
| F11 HIPT | 按用户要求不处理；原 `Image_feature_extraction.py` 保持不变 |
| F12 远距离核匹配 | 共享匹配增加 `max_distance`（像素）参数，默认20px；已有 BRCA/CRC/GBM 的50px wrapper 参数明确传递。阈值应按配准精度校准，不等于已验证的通用20px阈值；None 可恢复旧行为 |
| F13 ROC 来源 | 新泛癌结果写 `gt_matched_roc`，标明 GT 中心匹配细胞；为旧读表工具保留 CSV 的历史文件名前缀。真正 StarDist 中心评价仍须使用对应标签匹配数据 |
| F14 部分预测漏空间上下文 | 单队列 StarDist 调用/共享指标 helper 传递坐标；空间模型缺上下文显式报错；标签预测与概率预测共用实现 |
| 额外一致性 | 优先使用 spatial_HE 坐标；模型加载后 eval；完成模型的复制同时复制 artifact；旧单队列 in-sample helper 使用正确 scaler/上下文 |

## 1. 保留三个 head，增加完整相邻层级损失

`hierarchy_loss.py` 提供相邻映射和 JS 一致性。PanCancer 的完整 ontology 明确生成 L2→L12、L12→L1 索引，三头类别顺序写入模型 artifact。

目标仍包含三个直接 CE，另外有：

```
p12_from_l2 = softmax(logits_l2) @ map_l2_l12
p1_from_l12 = softmax(logits_l12) @ map_l12_l1

L = 原三个直接 CE
  + legacy_weight * NLL(L1_from_L2, y1)
  + a * NLL(p12_from_l2, y12)
  + b * NLL(p1_from_l12, y1)
  + c * JS(p12_head, p12_from_l2)
  + d * JS(p1_head, p1_from_l12)
```

上述是示意公式；实现函数直接使用 torch 张量。没有硬 argmax 路由，梯度可到达所有 head。软一致性会减少概率矛盾，但不保证三个独立 argmax 永远属于同一路径；本轮保留独立 head 输出，不强制后处理改写类别。

| 参数 | 默认 | 用途 |
|---|---:|---|
| `--hierarchy adjacent/legacy` | adjacent | legacy 关闭所有新增项，保留原目标形式 |
| `--l2-l12-nll` | 0.5 | L2 聚合到 L12 的监督 |
| `--l12-l1-nll` | 0.5 | L12 聚合到 L1 的监督 |
| `--l2-l12-js` | 0.1 | L12 head 与聚合概率的一致性 |
| `--l12-l1-js` | 0.1 | L1 head 与聚合概率的一致性 |
| `--legacy-l2-l1-weight` | 1.0 | 原直接 L2→L1 聚合项，原 PanCancer 默认是2 |

每个新增权重可独立设0。权重是起始实验值，不是调参后的最优值。严格复现旧损失形式可用 `--hierarchy legacy --legacy-l2-l1-weight 2`；其他修复仍生效，因此不是旧代码逐位复现。

可能改善：跨谱系混淆、稀有细类的共享统计、层级一致性。不能凭映射解决 H&E 本身无法区分的 CD4/CD8/Treg 等问题。没有加入单 L2 输出路线、PCA 或随机投影。

## 2. 可插拔的空间模块

核心在 `spatial_context.py` 和原 `base.py` 的 SpatialContextFusion。

| 开关 | 默认 | 说明 |
|---|---|---|
| `--use-spatial-context` | 未指定即关闭 | non-spatial 对照；指定后默认 k=8 mean |
| `--spatial-k` | 8 | 每个尺度的最大邻居数，按 section 构图 |
| `--spatial-radii-um 16 32` | 空：普通KNN | 一个或多个物理半径；半径之外不连边 |
| `--spatial-mpp 0.5` | 无 | 所有图像确实同一 µm/px 时才用 |
| `--mpp-map file.csv` | 无 | 每个 section 的真实 µm/px，推荐跨队列使用 |
| `--spatial-coordinates-um` | 关闭 | 仅当输入坐标本来就是µm时声明；不能对像素坐标乱开 |
| `--spatial-weighting uniform/gaussian` | uniform | 固定均值或距离衰减 |
| `--spatial-sigma-um` | 16 | gaussian 的距离尺度 |
| `--spatial-gate` | 关闭 | 学习中心细胞对应的 context 强度 |
| `--spatial-geometry` | 关闭 | count、平均距离、密度代理等单位化特征 |
| `--spatial-mode mean/attention` | mean | attention 使用图像 query/key，可加入距离偏置 |
| `--context-cache-dir` | /tmp/hist2pheno_context | 固定聚合值的磁盘缓存 |
| `--context-chunk-size` | 1024 | 聚合预处理的分块行数 |

mean 支持多尺度、gate和geometry。attention 当前支持单尺度和可选gate，不支持多尺度/geometry，非法组合会明确拒绝。没有邻居时使用中心特征，不会连接其它切片的第一个细胞。稀疏邻居用mask，避免重复首邻居造成额外权重。geometry中的密度是基于最多K个邻居的代理，不是全半径内的精确细胞密度。

缓存保存 raw embedding 的固定均值，batch 内使用当前折 scaler 转换；标准化与固定归一权重均值可交换。cell-mixed CV 必须按每折 partition 重建 context，不能使用未屏蔽全图缓存。attention 权重随模型变化，不能预先缓存最终聚合值，只做按批次邻居读取。

CUDA 传输改为 batch 范围；`predict_all_label_heads` 统一调用概率预测 API，避免每批搬运全体 embedding。大规模 scaled train/val 默认超过256MB时使用分块 StandardScaler 与磁盘 memmap，不再额外建立无用的全体 X_all。

环境覆盖：

```bash
export HIST2PHENO_ARRAY_CACHE=/your/writable/scratch/scaled
export HIST2PHENO_CONTEXT_CACHE=/your/writable/scratch/context
export HIST2PHENO_MEMMAP_THRESHOLD_MB=256
```

旧脚本/小数据仍可使用 NumPy 数组。内部 `_scale_split_float32` 的第四个返回值默认None，需要旧全矩阵时必须显式 `build_all=True`。不能把不同fold、不同图、不同embedding版本的context混用。

## 3. CV 和模型产物

泛癌默认启用 `--inner-val-fraction 0.2`：每个 outer train 再按 patient（有真实映射时）或 section 分 inner train/val，scaler仅拟合inner train；inner val选择epoch，outer val不参加早停。当前没有在选择epoch后再用全部outer train重训，因此每折实际参数训练约用全体数据的0.8×0.8=64%（五折时）。这是可解释的保守评价，不是最终全量部署模型训练。

`--patient-map patients.csv` 格式：

```csv
section_id,patient_id
BRCA::rep1,BRCA_patient_001
BRCA::rep2,BRCA_patient_001
```

真实元数据必须由使用者提供，不从名字猜患者。没有该表时，只能称section-level CV，不能称patient-level。相同患者不跨outer fold，也不跨inner train/val。患者数量不足时明确报错，不静默回退细胞随机分割。多section患者的fold可能不均衡，这是保持患者隔离的代价。

`mpp.csv` 格式：

```csv
section_id,mpp_um
BRCA::rep1,0.5
BRCA::rep2,0.5
```

以上0.5只是格式示例，不能直接当真实图像标定。使用映射时训练统一到µm，并保存映射；预测恢复每个section标定。外部队列通过原geometry接口按每张slide标定，不沿用训练slide的像素尺寸。使用已按mpp-map训练的物理模型预测新的原生像素切片时，需要显式提供标定；通用predict_block也支持 `HIST2PHENO_INFERENCE_MPP`。不同标定的slide应逐张传入，不用一个环境值覆盖全部。

结果目录中新增：

- `fold_plan.csv`：本次结果目录自己的折计划，不再由不同实验共享覆盖。
- `oof_predictions.npz`：三头概率/标签及原行索引。
- `oof_rows.csv`：section、cell_id、fold、可选patient_id。
- `hce_group_fold_N.pt.artifact.pt`：精确scaler、类别顺序、划分/配置/完成状态/checksum。

新版 artifact 缺失或校验不匹配时不允许把任意 scaler 当作正确部署变换；旧 plain state_dict 仍可通过底层模型加载接口读取，但新版部署主流程要求重新训练得到完整 artifact。不要仅复制 .pt 而丢掉 companion，也不要手工为旧权重猜一个scaler。

完成fold复用的是完整训练后的最佳模型，不是恢复optimizer/scheduler的中间step。中断或配置不兼容的fold从头训练，避免把第一个best-so-far误认为已完成。大h5ad内容指纹计算需要一次顺序读，增加准备阶段I/O；这是本轮选择的保守正确性代价。

`--class-weight-mode` 可选 none/balanced/class_balanced；启用sampler时默认none，避免默认双重补偿。

cell-stratified 保留作为开发对照，但必须显式 `--inner-val-fraction 0`；它不是新患者泛化评价，其真正逐细胞OOF保存在npz；按section整体重预测必须传 --deployment，不能标成OOF。GroupKFold的实际section计划从各折artifact回写；重复shuffle不作为唯一OOF入口。patient-map要求默认preassigned section splitter，避免被section GroupKFold绕过患者约束。旧单队列 stratified/in-sample 入口也保留，不自动变成严格患者级外层测试。

## 4. 建议运行方法

在自己的conda环境和仓库根目录执行。数据路径仍按原项目组织；不要把示例cache路径当作真实farm路径。

小规模功能检查：

```bash
python code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
  --datasets BRCA GBM --max-cells-per-section 5000 \
  --cv-k 2 --max-epochs 2 --patience 2 \
  --inner-val-fraction 0.2 --use-spatial-context --spatial-k 8 \
  --save-result result_updated_smoke --ablation-tag updated_smoke \
  --steps prepare train he_validate
```

示例小队列不保证有足够独立患者；真正泛化评价需要足够患者及真实patient-map。CPU诊断可额外传 `--allow-cpu-train`；全量任务应通过farm调度，不在登录节点运行。

物理双尺度实验：

```bash
python code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \
  --patient-map /path/to/patients.csv --mpp-map /path/to/mpp.csv \
  --use-spatial-context --spatial-k 8 --spatial-radii-um 16 32 \
  --spatial-weighting gaussian --spatial-sigma-um 16 \
  --spatial-gate --spatial-geometry \
  --context-cache-dir /your/scratch/hist2pheno_context \
  --save-result result_updated_multiscale --ablation-tag updated_multiscale \
  --steps prepare train he_validate
```

训练section的逐折OOF重预测：

```bash
python code/PanCancer/PanCancer_predict_visualize.py \
  --checkpoint data/PanCancer/result_updated_multiscale/cross_dataset_cv/updated_multiscale/best_mlp_gpu.pt \
  --pred-root data/PanCancer/pred_updated_multiscale
```

默认写到 `pred_updated_multiscale/oof/`，使用该artifact所在结果目录的fold_plan。新外部数据使用External_val入口；若显式对全部section应用单一部署模型，传 `--deployment`，这些结果不能称为全体OOF。

三头队列内指标和患者bootstrap：

```bash
python code/Hist2Pheno_pkg/oof_evaluation.py \
  data/PanCancer/result_updated_multiscale \
  data/PanCancer/result_updated_multiscale/cross_dataset_cv/updated_multiscale/best_mlp_gpu.pt \
  --bootstrap 500
```

输出每队列和每类别的支持数、macro-F1、AUROC、AUPRC。没有正例或负例的类别AUC为NaN。bootstrap仅实现患者cluster的macro-F1置信区间，利用患者混淆矩阵，不重复推理；没有真实patient_id时拒绝生成“患者CI”。没有实现患者bootstrap AUROC/完整自动超参搜索，最终独立test仍应另外保留。

历史图像输出保留在 `code/PanCancer/4_Pred_statistic_visual_pancancer_all.ipynb`，那是已发表的 UNI cross-section 与 cross-celltype 比较，不代表新版结果。UNI2 A0–A3 在 `code/PanCancer/2_Pred_statistic_visual_pancancer_uni2_ablation.ipynb`，读取各架构的 `oof_predictions.npz`。旧路径、全局 scaler 和 in-sample 图应先对照本报告迁移，不能把旧保存输出当新版验证。

## 5. 时间、内存与潜在性能提升

以下按 N=5.5M、D=1024、K=8、float32；GB为十进制。

| 项目 | 估算 |
|---|---:|
| 原始embedding | 22.53GB |
| 每个尺度的固定mean context磁盘缓存 | 22.53GB |
| 2/3个尺度context缓存 | 45.06 / 67.58GB |
| 每尺度int64邻居索引 / float32距离 | 0.352 / 0.176GB |
| 一个完整scaled train+val集合的memmap | 约22.53GB |
| B=4096、单尺度中心+context输入GPU | 约33.6MB，不含MLP/梯度/其他张量 |
| 若显式取(B,K,D)邻居张量 | 约134.2MB，不含attention临时张量 |

默认流式路径去除了把约22.5GB的全体embedding放在GPU上的要求，但CPU准备阶段仍会把原始embedding池化并可能产生堆叠/过滤临时副本；不能保证32GB RAM足够。大数据请先按约45–70GB级准备峰值主存，再实测。memmap改善scaled副本常驻内存，不代表整个数据准备已完全out-of-core。

缓存需要足够scratch：双尺度45GB加scaled副本22.5GB，至少约68GB，再加余量、其它实验缓存和输入数据。不要写满home/NFS配额；在共享farm选择合适scratch。不同实验缓存不会自动删除。

固定mean预计算避免每个epoch重复邻居聚合。本次CPU微测试：50k细胞、128维、k=8，构图+缓存约0.568s，重复在线聚合中位数0.0533s，缓存读取归约约0.00148s。它只测热缓存CPU聚合，不含MLP、GPU、冷网络盘和训练，不能据此声称整体训练快36倍。

特别注意：原GPU全矩阵驻留时，gather主要读GPU显存；流式缓存则增加主机/PCIe或磁盘读取，显存充裕的机器未必更快。本轮首先改善可运行规模和避免重复全矩阵传输，端到端速度仍需在目标farm基准测试。

单尺度中心+context每完整pass约读45GB。若有效I/O只有0.5GB/s，光读取就约90s；5GB/s约9s。缓存命中系统page cache时可能更低，随机访问和共享盘争用可能更高。五折×50epoch、每折约0.8N用于inner训练/验证，输入读取量粗略9TB，对应上述带宽约5小时或0.5小时的I/O下界，尚未计入计算、额外outer评价、特征准备和启动开销。

### GraphSAGE/GAT接口与粗略成本

只提供 `register_graph_encoder/create_graph_encoder` 扩展工厂，以及 `estimate_graph_cost`，没有默认增加GNN或声称已经实现其完整训练链。后续接入应使用1–2层、邻居采样/分块，不构建稠密N×N attention。

两层、hidden=128的一阶均值图编码器量级：44M有向边，单层完整hidden约2.82GB，前向约1.64TFLOP（首层1024→128、后续128→128及邻居归约）。假设有效0.5–5TFLOP/s，纯前向计算下界约3.3–0.33s，考虑反向粗乘3约10–1s；这些数值不含图采样、随机I/O、传输、kernel开销、MLP和验证，不能作为实际epoch耗时。GAT还有边注意力及softmax开销，通常更贵。

B=4096、fanout=8的两跳邻域最多约4096×(1+8+64)=299k节点；若全部保留1024维FP32输入，仅该batch的原始特征就约1.22GB，尚未含梯度/激活。需要更小batch或受控采样。图层hidden128只是GNN成本假设，不是本轮加入PCA/特征压缩。

实际建议：先记录原MLP每epoch的训练/验证时间、峰值主存/显存、I/O，再对比缓存mean、多尺度mean，最后才接GNN。模型精度、校准、层级冲突率和速度应一起衡量。

## 6. 批量图像特征I/O策略，保留旧实现

原 `Image_feature_extraction.py` 和逐patch/.pth接口完全保留，HIPT与UNI裁剪参数不变。新增 `batched_embedding_io.py` 是可用的可选适配器，不是已经接管所有队列的图像CLI：

1. 用原crop/preprocessing产生 `(cell_id,x,y,PIL_patch)` 迭代器。
2. 提供 `embed_batch(patches)` 回调，复用现有编码器、原输入尺度和torch.inference_mode，返回(B,D)。建议先试batch64/128，依据显存调整。
3. `extract_to_h5` 分批写入embedding/cell_id/xy，记录encoder checkpoint及预处理版本，采用原子替换；不覆盖旧路径，除非显式overwrite。
4. `iter_h5_embeddings` 分块读取，后续可接AnnData构建；本轮未把所有legacy .pth reader自动改成读取此HDF5，需显式接入。

它避免B次单样本GPU调用和海量小文件的潜在收益通常比继续优化Python文件扫描更有价值，但尚无真实WSI吞吐测量。默认不启用，不强迫用户重做已有UNI embedding。

## 7. 验证范围与剩余事项

详细日志、合成集结果和微测试见本目录的JSON/log文件；使用的CPU环境记录在runtime.json。

完成：原有共享/ESCC测试、新增回归测试、患者分组+内层早停的两折三头CLI训练、完整OOF覆盖、完成fold复用、三头重新载入预测及概率一致性、attention训练/重载、缓存失效、空间稀疏/重复坐标、memmap和批量HDF5测试，以及源代码语法检查。

未验证：真实550万细胞全量训练、CUDA吞吐/峰值显存、真实WSI embedding、TensorFlow/StarDist分割、全部历史notebook及单队列完整真实数据链。不会据合成数据宣称AUROC/F1提升。实际标定、真实患者映射、数据/权重和farm调度配置仍由使用者提供。

建议实验顺序：新版legacy-loss/non-spatial基线 → adjacent loss → k=8 cached mean → 物理半径/多尺度/gate/geometry分项ablation → 最后接浅层GNN。每个实验独立结果目录，固定患者划分、种子和预算，避免把多个改动的效果混在一起。


### 本轮最终验证记录

- 新增回归测试21项；原共享测试21项；ESCC兼容测试2项，合计44项通过。
- 源码/测试共109个Python文件语法检查通过；新版PanCancer与Internal_val shell入口通过bash语法检查。
- 合成集320细胞、8个section、4个患者，两折×两epoch：三头OOF覆盖、inner/outer患者隔离、已完成fold复用、重载后标签及概率一致性通过。
- 另外验证GroupKFold实际计划与artifact一致，以及cell-stratified拒绝伪装成逐section OOF。
- OOF统计生成9个队列/head组合、99个类别组合和3个患者bootstrap F1区间。它们只证明实现能够运行，不代表真实数据性能。

交付包均不含Git元数据、虚拟环境、原始数据或模型权重。完整包保留原历史notebook及其旧输出，体积较大；精简修改包只含有变化和新增的文件，适合先解压到新目录审阅，再手动合并到你自己的clone。补丁以基线SHA为前提，可先自行运行 `git apply --check`，确认无冲突后再决定应用及提交。
