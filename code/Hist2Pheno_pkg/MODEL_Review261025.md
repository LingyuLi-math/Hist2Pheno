# Hist2Pheno 模型与验证代码审查

审查版本：GitHub main 的 `b63cfe2f32ba92439050311e7cb4892d13cc727c`。本地工作仓库仍是旧版本 `0d50dc1`，未修改。以下结论针对前者，不应直接当作 farm 上某次训练产物的鉴定。

阅读重点：共享 base/model、标签映射、交叉验证、PanCancer prepare/train/predict、内部验证、外部预测、特征提取，以及多个单队列 CLI 的相关调用链。下载并解析了 102 个 Python 文件；源文件快照不是完整 Git checkout，也不包含所有 notebook。未获得真实数据、fold_plan、scaler、模型权重、患者对应表或训练日志，因此不能量化每项问题造成的性能损失，不能保证已找出所有错误。

## 首先处理的结论

优先修正训练/推理一致性及验证，再比较新模型。当前存在能够让结果偏移或解释失真的问题，单纯加大 K 或换 GNN 会掩盖这些问题。

| ID | 优先级 | 问题 | 影响范围 |
|---|---|---|---|
| F01 | P1 | 各折 scaler 没有随权重保存，推理另行拟合 scaler | 泛癌部署及相关共享路径 |
| F02 | P1 | 每轮验证只看按顺序的前约 300k 细胞 | 大数据训练的 early stopping/epoch selection |
| F03 | P1 | checkpoint 存在即跳过训练，无协议核验或完成标记 | resume；更改数据/折/配置后的复用 |
| F04 | P1 | 动态选择 best fold，但内部验证固定 fold 0 | 内部验证的 held-out 声明 |
| F05 | P1 | 层级 sampler 权重不实现其声明的层级均衡 | 默认启用 sampler 的 cell-stratified 流程 |
| F06 | P1 | attention 模型重载时被构建成 mean | attention 选项 |
| F07 | P1 | 标签缓存不核验源数据或 ontology 版本 | 重建 h5ad、改标签、更新 ontology 后 |
| F08 | P2 | 父类汇总、父 head、叶预测映射三种 L1 定义混用 | L1 指标的跨路径比较 |
| F09 | P2 | 空间投影的零初始化被后续初始化覆盖 | 所有带空间的 MLP |
| F10 | P1 | StarDist 分割脚本存在缩进语法错误 | 直接执行/导入该分割入口 |
| F11 | P1 | HIPT 分支传入 URL 却仅检查本地文件，可能返回随机模型 | HIPT 特征提取；不等于 UNI 也有此错误 |
| F12 | P2 | GT↔StarDist 匹配没有最大距离约束 | HCC/PDAC/GIST 等未额外筛距离的调用 |
| F13 | P2 | 泛癌 stardist_roc 实际保存 GT-matched ROC | 指标命名、来源与科学解释 |
| F14 | P1/P2 | 某些单样本 StarDist 路径未传空间邻居 | 例如 GIST step_stardist；泛癌/pooled 多处已正确传递 |

P1 表示建议在下一次正式训练或发布指标前处理；某个可选分支的问题只影响使用该分支的实验。P2 表示结果解释、稳健性或实现与设计不一致的问题，仍应修复。

## 1. 三层 mapping：数据有完整树，损失没有完整使用

`code/Hist2Pheno_pkg/dataset/build_pancancer_celltype_v1.py:35` 已定义 `L12_PARENT`；原生标签映射输出 `(L1,L12,L2)`。实际 Excel `celltype_pancancer` 的 20 个 L2 均只有一个 L12，每个 L12 也只有一个 L1，共 20/8/5 类。本次检查没有发现该泛癌 ontology 的父节点冲突。

`pancancer_labels.py:52` 只提供直接的 L2→L1 索引数组。`model.py:1017` 内的 UnifiedHCELoss 使用：

```
CE(L1 head) + CE(L2 head) + w12*NLL(sum_child P(L2) -> L1) + CE(L12 head)
```

PanCancer 的默认 `hce_w12=2`。这里 w12 的名字表示 L1/L2 的层级损失权重，不是 L12 head 的权重；L12 head 由 `hce_w_l12head` 控制。没有 L2→L12、L12→L1 的聚合监督，也没有三个 head 之间的显式概率一致性项。

建议首先保留共享 backbone 和三个直接 CE，添加以下相邻关系的 ablation：

- A 为 L2×L12 的 one-hot 归属矩阵，B 为 L12×L1 的矩阵。
- `q12 = p2 @ A`，`q1 = p12 @ B`。
- 比较添加 `NLL(q12,y12)`、`NLL(q1,y1)`，以及 `JS(p12,q12)`、`JS(p1,q1)` 的效果。
- 已有 L2→L1 关系等价于 A@B，不能不调权重就把所有项叠加，造成粗层级被反复计权。
- consistency 使用软概率；不要用 L1 argmax 硬截断 L2，避免上层误判无法恢复。

另一条更简洁的对照路线：只输出 20 类 L2，把 L12/L1 概率由求和得到，并对三个层级分别监督。这样概率边缘分布严格一致，但失去独立粗 head 的灵活性。若要求最终三个离散标签必为同一路径，可选最佳叶类别后映射两个祖先，或做受约束的联合路径解码；分别对三个边缘分布 argmax，即便概率合法，也不保证标签路径一致。

增加 mapping 有机会改善一致性和跨谱系误判，但不能保证 L2 的 CD4/CD8/Treg 等细分类性能提高。它不会产生图像中原本没有的形态证据。

本版本还核对了 KEEP/DROP 筛选：虽然 inventory 里有“未细分 T→NK”“Mast→Neutrophil”等旧合并记录，当前 pancancer 的训练 KEEP 规则已把这些不合适的原生标签排除。不能仅看字典上半部分就断言它们仍进入当前训练。可考虑以后对只有粗标签的细胞使用部分标签监督，而非丢弃或强行细分类。

## 2. 空间上下文：K=8 可作为基线，需要物理尺度与高效实现

当前 `base.py:1650` 的 mean 模式为：

```
c_i = mean(x_j for j in KNN(i))
x_fused = x_i + dropout(W c_i + b)
```

坐标用于选邻居，之后模型看不到距离、相对方向、局部密度。attention 模式依靠 query/key 图像特征，也没有显式距离项。邻居按 sample 分组构建，这是正确的，避免把不同切片相同像素坐标连接起来。

K=8 没有逻辑错误，但在细胞密集区和稀疏区代表不同物理范围，且固定 k 会跨越组织空白区域。建议：

1. 保留 non-spatial、k=8 mean 为对照，用相同患者划分比较 k=4/8/16/32。K 由开发验证集选择，不看最终 test 后决定。
2. 统一坐标到 µm，再比较“半径邻居 + 最大 K”的图。可从 16/32/64 µm 做开发实验，不把这些初值视为最优。稀疏邻居应有 mask，避免用重复首邻居或 self 无意加权。
3. 加入距离衰减权重、邻居数、局部密度；需要时加相对坐标。绝对 x/y 直接跨片输入容易学到扫描位置，不能替代生物学上下文。
4. 比较局部细胞环境和更大组织环境的两个尺度，用 gate 选择 context 强度，以减少边界过度平滑。
5. 在前述简洁模型有效后，再考虑 1–2 层 GraphSAGE/GAT、位置编码或局部图 Transformer；使用局部采样/分块，不做全切片稠密 attention。

### 同时改善速度的首选方案

mean 模式的邻居 embedding 是固定输入，与训练权重无关。可以把每个细胞的邻居均值预先缓存，训练时读取 `(x_i,c_i)`，保留原来的可学习投影。这样避免每个 epoch 重复 gather `(B,K,1024)`，也不必把全体 550 万细胞的 embedding 常驻 GPU。

- 使用 memmap/分块文件存储中心特征和缓存 context，按 batch 预取到 GPU。
- 固定、权重和为 1 的均值/距离加权均值与 StandardScaler 的仿射变换可交换：先聚合 raw embedding，再使用各折训练 scaler 转换，便于正确复用缓存。
- cell-stratified 若采用当前 split mask，context 必须针对各折 mask 重算；不能拿全图预聚合结果冒充已屏蔽的上下文。
- attention 权重依赖训练参数，不能同样预先缓存最终聚合值。
- 若仍慢，再比较训练折拟合 PCA 的 128/256 维输入或压缩 context 分支。固定随机投影也可作效率对照，但需验证性能。

`model.py:4150` 的旧 `predict_all_label_heads` 仍在 batch 循环里调用 `X_t.to(device)`，CUDA 上可能重复传整个特征矩阵。较新的 `plot.py:2997` 概率推理路径已把整矩阵搬运移到循环外，两者应统一。`_scale_split_float32` 即使 non-spatial 也构造全体 scaled 矩阵，存在可去除的内存开销。

特征提取也仍按单 patch 推理并写单独 .pth：可批量推理并分块写 HDF5/Zarr，减少 GPU 小任务和海量小文件。真实加速比需要在 farm 的数据和存储上测量。

UNI 当前常用 16×16 输入，目标 0.5 µm/px 时只覆盖约 8 µm，且 ViT patch_size=16，只有一个图像 patch token。动态尺寸允许运行，但上下文和预训练输入分布都值得验证。建议比较更大真实视野的 crop 和标准 encoder preprocessing。把原有 16×16 单纯放大到 224×224 不会恢复缺失的组织信息；新特征需写到不同版本路径以保留基线。

## 3. CV：基础拆分可用，但当前完整评价不能直接认定严格正确

本次确认正确的部分：

- 默认 PanCancer 使用按队列打散、按 section 分配的预定义折；一张 section 整体 train 或 val。
- `_scale_split_float32` 只在当前训练折拟合 scaler。
- kNN 在 section 内构建。
- cell-mixed 的空间索引会屏蔽跨 train/val 邻居。
- 实际 96 细胞、4 section、2 折、每折 2 epoch 的空间三头训练成功，96 个细胞均有 OOF 预测。

仍须修正或限定的部分：

**F01 — 标准化器与权重未绑定。** `model.py:2294` 得到 fold scaler，但 `1687` 只保存 state_dict。PanCancer `step_train:586` 复制 best-fold 权重，预测脚本 `204`、`561` 则另从全 index/最多 400k 原始细胞拟合或读取 scaler。后者不保证等于该折 scaler，也未严格限定该折训练细胞与训练保留标签。示例中同一输入在原 scaler 下为 `[99,99]`，替代 scaler 下为 `[0.98,0.98]`，足以改变模型行为；这是合成示例，不代表真实数据损失的幅度。建议每折保存权重、scaler、类别顺序、ontology hash、训练样本/患者、fold plan、空间模式/K、特征版本的完整 artifact。预测禁止隐式重新 fit scaler。

**F02 — 验证选模型只看前缀。** `model.py:466–528` 达到 300k 后 break，val_loader shuffle=False，数据按 section/队列拼接。后面的队列可能根本未参与 epoch selection。示例 20 行里只访问前 8 行，某类别完全未被评估却能返回宏 AUC=1。后续 AUC 函数的随机子采样不能补回已经没做 forward 的细胞。建议提前构建固定、分队列/患者/类覆盖的开发验证子集，或完整流式验证；报告支持数与不可评价类别。

**F03 — resume 只认文件存在。** `model.py:2349` 及其它 CV 分支无数据/参数 hash、无完成状态校验。第一轮 best 权重保存后若进程中断，下次会跳过整个 fold；改标签、K、fold plan 后也可能复用旧权重。应分别实现 best artifact、last training state 与 fold-complete marker，保存 optimizer/scheduler/AMP/RNG；不匹配时拒绝复用。当前行为可以被定义成“用已有 best-so-far 做评价”，但不应当作继续完成训练。

**F04 — fold 0 不是固定的 best-fold held-out。** `model.py:3144` 动态取最高得分折，`PanCancer_train_validate_cv_UNIlabel.py:586` 复制该折权重；`Internal_val/internal_val.py:36` 与 demo.sh 则固定 0。只有实际选中 fold=0，相关切片才没参加该模型参数训练。即便如此，该验证集参与过早停/模型选择，也不是独立最终 test。应从 artifact 元数据选对应 held-out，或用每个 section 对应的 fold 模型做真正 OOF；最终 test 必须独立保留。

**F05 — 层级 sampler 的样本权重公式过度放大少数父类。** `base.py:6156–6185` 同时乘 L1、L12、L2 的细胞数倒数。例：两个 L1 各仅一个 L12/L2，分别有 100 和 10 细胞，采样概率为 0.99%/99.01%，而不是声明的父类均衡 50%/50%。单父树上的等概率层级采样应使用 `1/(K1*K12(parent1)*K2(parent12)*n_leaf)`，K 是该层子类别数量，不是重复使用祖先细胞数量。也应避免 sampler 与 balanced CE 不经验证地重复补偿稀有类。默认 section 流程关闭 sampler，cell-stratified 默认启用，影响范围不同。

**F07 — 标签缓存可能无声错配。** `PanCancer_train_validate_cv_UNIlabel.py:215` 读缓存后不核验 h5ad 内容、obs_names 顺序、n_raw 或 ontology。复现：同一 h5ad 的 CD4 改为 CD8，仍返回缓存中的 CD4；无缓存才返回 CD8。必须保存并检查数据/细胞 ID/ontology 版本；重排后尤其危险，因为缓存包含位置索引。

**F08 — L1 指标口径不统一。** fold evaluation 同时有 L1 head 和由 L2 概率求和的 L1；OOF 汇总 `model.py:3276` 却直接把 argmax L2 映射到父类。叶概率 `[0.35,0.33,0.32]`，三者的父归属 `[A,B,B]` 时，映射 argmax 得 A、概率求和得 B。这两种规则都能定义成有效任务，但指标必须分别命名，不能当同一个 L1 数值比较。建议保存每折三头 OOF 概率、真实行 ID 和两类父级定义。

另有轻量问题：旧 fold_plan 的 fold ID 没有范围校验；请求 k=2 可保留已有 fold=4。分折参数改变时必须拒绝旧计划或显式重建。summary 中部分文字/计数仍按 GroupShuffleSplit 0.7 计算，即使实际跑预定义 k-fold，也可能输出错误的训练/验证组数；cell-mixed 中 `n_groups-len(held)` 更可能错误显示为 0。

### 科学评价建议

section CV 不是 patient CV，也不是 leave-one-cancer-out。BRCA 邻近重复切片、GBM 初发/复发以及 TMA 同一患者多 core，都要依据真实 patient metadata 检查 train/test 是否共享患者；代码当前没有强制患者级隔离，不能仅凭名字推断实际泄漏已发生。

建议主结果使用 patient-grouped outer CV，开发训练内部再拆患者用于 early stopping/调参。细胞级分层只作为辅助诊断，空间 block 划分也不能自动替代患者隔离。若声称泛癌跨器官迁移，增加 leave-one-cohort/cancer-out 对照。

报告每队列、每患者/section 的 macro-F1、balanced accuracy、per-class recall/precision、AUPRC、AUROC、校准及支持数，给出患者 bootstrap CI。pooled OvR AUROC 对器官特异细胞可因其它器官提供大量易负例而偏高，应同时看队列内分类与混淆矩阵。没有真值的外部队列只能做空间/生物学验证，不能宣称分类准确率已验证。

训练图由有标签且保留的 GT 细胞构建，部署图常为全部 StarDist 核；节点密度、邻居身份及分割中心均不同。应检查该输入分布变化，并比较“用全部可见核构图，只在有标签目标核计算损失”的设计，使用图像/坐标特征、不把验证标签当邻居特征。对于同一切片可见的无标签邻居，应事先定义是 transductive 还是 inductive 任务。

## 4. 其它模型与管线错误

**F06 — attention checkpoint 无法严格重载。** `model.py:827` 只推断是否 spatial，没有恢复 spatial_mode，默认建 mean；attention 权重多出 attn_q/attn_k，strict load 报 Unexpected key。已用实际三头小模型复现。修复后才有意义比较 mean/attention，也要保存 K 与架构配置。

**F09 — 空间零初始化被覆盖。** `SpatialContextFusion` 将 proj.weight/bias 置零，但父 MLP 的 `_initialize_weights` (`base.py:1820`) 遍历全部 Linear 又把 proj.weight 改成 Kaiming。独立 fusion 权重 0 个非零，完整模型同一 4×4 层为 16 个非零。并非不能训练，但违背“从 non-spatial 模型附近启动”的设计。应在总体初始化之后重置 fusion projection，或让总体初始化跳过它。

**F10 — 分割脚本无法解析。** `code/StarDist_nuclei_segmente.py:98–107` 的 if/else 缩进错误，AST 直接报 IndentationError。102 个检查文件中发现这一处语法错误。它不等于已经准备好的 StarDist CSV 不能用，也不影响直接训练已有 embedding，但阻断该分割入口。

**F11 — HIPT 预训练权重未真正加载。** `Image_feature_extraction.py:721` 给 get_vit256 一个 GitHub blob URL，函数 `704` 却只在 os.path.isfile 时加载，没有下载和失败分支。文件检查为 False 时仍返回模型，随后日志称 loaded successfully。使用小 ViT 替身执行原函数，确认 load_state_dict 从未调用。应要求可验证本地 checkpoint 或实现正式下载与校验；缺权重立即失败。UNI 分支另外使用 hf_hub_download，不受此 URL 判断问题影响。

**F12 — 匹配缺少距离上限。** `base.py:4344` 的 greedy 1:1 匹配保留可用最近核，不限制物理距离。HCC/PDAC/GIST 的 wrapper 只 dropna，没有筛 matched_distance；本次合成例距 100000 px 仍被匹配。BRCA/CRC/GBM wrapper 另有默认 50 px 阈值，不能把问题泛化为全部队列。建议按 µm 设置并验证阈值，保存覆盖率与距离分布，检查相邻密集细胞标签转移误差。阈值应由配准与核直径决定，不盲用 50 px。

**F13 — ROC 文件来源误导。** `PanCancer_predict_visualize.py:330–331` 写 stardist_roc 时使用 load_matched 得到的 GT 坐标 embedding 与 got；到后面才读取 StarDist-all。图的 ROC 行自己标为 matched，相比之下 CSV 使用 external_stardist_matched 的通用命名，容易在 pooled 图中误当成分割核评价。应区分 GT-center matched ROC、StarDist-center matched ROC、all-nuclei unlabeled prediction。再者一个 best-fold 模型预测所有训练 section 的 ROC 不是全体 OOF，约 4/5 section 对该模型是训练数据；真正 OOF 要逐折调用对应权重和 scaler。

**F14 — 部分单样本预测丢失 spatial context。** 例如 `CODEX_gist/GIST_train_validate_cv_UNIlabel.py:1010` 的 step_stardist 中，多处 evaluate/softmax/stardist_head_preds 未传 neighbor_index。下游 helper 默认 None，空间模型静默回退到只看中心特征。较新的 pooled GIST 路径以及泛癌 predict_block、external_validation 多处已经正确构图传递。建议统一一个预测 API：空间模型若无图且不是显式 ablation，应报错；同时核对这类历史输出是否要重算。

## 实证与可复现文件

- 现有共享测试 21 项通过，ESCC 兼容性测试 2 项通过。这些测试主要覆盖绘图和导入，不能说明模型/CV 无误。
- `reproduce_findings.py` 12 项小规模核对完成。部分测试预期错误行为，测试“通过”表示问题成功复现，不是问题已修好。证据见 evidence.json/reproductions.log。
- `smoke_cv.py` 真正执行 2 折 × 2 epoch，96 个细胞均生成 OOF，见 smoke_cv.log 与 toy-cv/result.json。
- syntax.json 记录 102 个 Python 文件的语法检查结果；runtime.json 记录 CPU 审查环境。动态测试使用 Python 3.12、NumPy 1.26.4、Scanpy 1.11.5、torch 2.14.1 CPU，不是原训练机器的 Python 3.11/GPU 环境。
- source/ 保存已下载的版本化源码；source-version.json 记录 SHA 和范围。真实 UNI/HIPT 权重及原始数据未下载，未进行完整 GPU 重训。

运行小规模复现：

```bash
export XDG_CACHE_HOME=/workspace/.cache
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export NUMBA_CACHE_DIR=/workspace/.cache/numba
export MPLBACKEND=Agg
/workspace/.venvs/hist2pheno-audit/bin/python /workspace/hist2pheno-review/reproduce_findings.py
```

## 建议的实验顺序

1. 修复 artifact/scaler、验证前缀、resume/cache、最佳折对应；确认真实患者划分。先重算可信 baseline，不预设分数一定上升。
2. 修复 sampler、初始化和空间推理一致性；有用到 attention/HIPT/分割入口时先修对应错误。
3. 固定数据、fold plan、种子与训练预算，比较“当前层级损失”与“相邻 mapping 一致性”，并保留每队列指标。
4. 在相同协议下比较 non-spatial、k=8 mean、距离加权、多尺度；先用可缓存的 mean context 获得效率基线。
5. 最后比较更大物理视野的图像 embedding 和浅层 GNN。以新患者指标、运行时间、峰值内存/显存、层级冲突率共同选择，不只看 pooled AUROC。

要鉴定你已训练模型的实际结果，后续最有价值的材料是 group_cv_folds.csv、fold_plan.csv、训练日志、scaler.joblib、checkpoint 的 SHA/配置、per-section 指标，以及 section→patient 对应表；不需要先搬整套大图像。
