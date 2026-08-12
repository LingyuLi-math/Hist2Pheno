# CODEX HCC development notes

本文件记录 `code/CODEX_hcc` 及其依赖的 Hist2Pheno 代码更新、修改原因、兼容性和验证结果。
后续每次功能性更新应在下方 changelog 顶部增加一条记录。

## Hist2Pheno hierarchy terminology

当前模型内部命名与 HCC Excel 列的对应关系如下：

- L2 / fine head：`celltype_level2` → `final_CT`
- L12 / intermediate head：`celltype_level1` → `final_sublineage`
- L1 / coarse head：`celltype_level0` → `final_lineage`

这里的 `L1`、`L12`、`L2` 是历史模型命名，不等同于 Excel 的 level 编号。

## Current architecture status

Hist2Pheno 目前共享一个 MLP backbone，每一个监督层级有独立的 linear classification head。
现有代码支持以下固定模式：

1. single-head：fine label
2. dual-head：fine + coarse
3. three-head：fine + intermediate + coarse（CODEX HCC）
4. five-head：L2 + L1 + L12 + CNiche + TNiche（Xenium lung）

因此，当前代码最多可以运行五个 label heads，但还不是“输入任意层数即可自动建模”的动态架构。
head 数量、target 名称、DataLoader batch 格式和部分 evaluation 逻辑仍有固定分支。

### Current hierarchy-aware loss

每个 head 都有自己的 cross-entropy loss。当前真正使用 mapping matrix 的层次一致性项只有：

```text
fine L2 probabilities → coarse L1 probabilities
```

其 loss 为：

```text
total =
    w_l1 * CE(coarse_head)
  + w_l2 * CE(fine_head)
  + w_map * NLL(mapped_fine_to_coarse)
  + optional direct CE losses for other heads
```

three-head 中 intermediate head 目前有独立 CE，但尚未计算：

```text
fine → intermediate
intermediate → coarse
```

five-head 中 L12、CNiche、TNiche 目前也主要使用独立 CE；它们之间没有通用 mapping-matrix loss。

## Target unified design

最终统一框架不应继续增加 `ThreeHead...`、`FiveHead...` 等固定类。建议改为：

- 输入一个 hierarchy specification（Excel、CSV 或标准化 DataFrame）
- 自动识别 label levels、类别数和由粗到细的顺序
- 使用 `nn.ModuleDict` 按层级动态建立任意数量的独立 heads
- 指定唯一的 fine L2 → coarse L1 mapping matrix
- loss 同时包含：
  - 每个 head 的直接监督 CE
  - fine L2 probabilities 经 mapping matrix 聚合后的 coarse L1 consistency loss
- checkpoint 保存 hierarchy metadata、class names、mapping matrices 和 level order
- training、prediction、metrics、plotting 全部遍历 level specification，不再按 L1/L12/L3/L4 写固定分支

建议的统一 loss：

```text
L_total =
    Σ_level α_level * CE(direct_head[level], target[level])
  + β_l2_to_l1 * NLL(
        softmax(logits_fine_l2) @ mapping[fine_l2→coarse_l1],
        target_coarse_l1
    )
```

intermediate、CNiche 和 TNiche 等其他 heads 只计算各自的直接 CE；
设计目标不包含 fine → intermediate 或 intermediate → coarse mapping loss。

Excel hierarchy 必须满足：

- 每个 fine class 在同一 parent level 只能映射到一个 parent class
- 不允许空 label
- 每个模型 head 的 class order 必须随 checkpoint 保存
- mapping matrix 的行列顺序必须与 child/parent encoders 完全一致

## HCC and Xenium lung model comparison

### Conclusion

两个 notebook 使用的是同一套 Hist2Pheno modeling framework：

- 相同的 1024-dimensional UNI image embeddings
- 相同的 shared MLP backbone：`1024 → 1024 → 512 → 256`
- 相同的 dropout（0.2）、optimizer/training implementation 和 HCE core
- 相同的 five-fold `StratifiedKFold`，`stratify_target="joint"`
- 相同的 loss core：
  - `1.0 * CE(L1 coarse head)`
  - `2.0 * CE(L2 fine head)`
  - `1.0 * NLL(L2 probabilities mapped to L1)`
- spatial-context fusion 均关闭

两者不是同一个输出层实例。它们共享 backbone 设计和训练框架，但根据数据集 hierarchy
建立不同数量、不同 output dimensions 的独立 heads。

### Dataset and output-head differences

CODEX HCC：

- 33,037 matched cells in the executed notebook
- 12 L2 fine classes、6 L12 intermediate classes、4 L1 coarse classes
- `ThreeHeadImprovedMLPClassifier`
- heads：L2 + L12 + L1
- parameters：1,714,966
- auxiliary direct loss：`1.0 * CE(L12)`
- checkpoint selection：`three_tier_auc_sum`

Xenium lung (`VUILD107MA`)：

- 30,614 filtered matched cells
- 44 L2 fine classes、6 L12 classes、4 L1 classes、12 CNiche classes、12 TNiche classes
- `FiveHeadImprovedMLPClassifier`
- heads：L2 + L12 + L1 + CNiche + TNiche
- parameters：1,729,358
- auxiliary direct losses：
  - `1.0 * CE(L12)`
  - `1.0 * CE(CNiche)`
  - `1.0 * CE(TNiche)`
- persisted notebook result used `four_term_sum` for best-fold selection

参数量只相差 14,392（约 0.84%），因为主要参数位于共享 backbone；增加 label heads
只增加最后一层的少量 weights 和 biases。因此两个数据集上的模型容量基本一致，
主要差异来自 label task complexity，而不是 backbone 大小。

### Executed five-fold CV results

以下数字来自两个 notebook 当前保存的 outputs，而不是 synthetic smoke tests。

CODEX HCC：

- L2 accuracy：0.2913 ± 0.0169
- L2 macro-F1：0.2155 ± 0.0081
- L2 weighted-F1：0.2966 ± 0.0135
- L1 mapped accuracy：0.7363 ± 0.0041
- L1 mapped macro-F1：0.5066 ± 0.0099
- L1 mapped weighted-F1：0.6959 ± 0.0079
- L1 direct-head macro-F1：0.5207 ± 0.0253
- L12 macro-F1：0.3894 ± 0.0092
- L12 weighted-F1：0.5278 ± 0.0073

Xenium lung：

- L2 accuracy：0.2594 ± 0.0150
- L2 macro-F1：0.1144 ± 0.0033
- L2 weighted-F1：0.2660 ± 0.0084
- L1 mapped accuracy：0.6829 ± 0.0047
- L1 mapped macro-F1：0.5564 ± 0.0050
- L1 mapped weighted-F1：0.6740 ± 0.0039
- L1 direct-head macro-F1：0.5548 ± 0.0037
- L12 macro-F1：0.4240 ± 0.0075
- L12 weighted-F1：0.5722 ± 0.0061
- CNiche macro-F1：0.2148 ± 0.0024
- TNiche macro-F1：0.3027 ± 0.0049

### Interpretation

- HCC 的 L2 macro-F1 高于 Xenium（0.2155 vs 0.1144），但不能据此认定 HCC
  image phenotype 更容易学习。HCC 只有 12 个 fine classes，而 Xenium 有 44 个，
  Xenium 的类别不平衡和细粒度区分任务明显更难。
- HCC 的 L2 accuracy/weighted-F1 也略高，和较少的 fine classes 一致。
- Xenium 的 L12 macro-F1 高于 HCC（0.4240 vs 0.3894），说明其六类 intermediate
  hierarchy 在当前 UNI features 下稍容易区分。
- L1 coarse 层面两者接近。HCC 的 mapped L1 accuracy/weighted-F1 较高，但
  Xenium 的 macro-F1 较高；这表示 HCC 对多数 coarse classes 表现较好，而
  Xenium 在各 coarse classes 之间相对更均衡。
- HCC 中 direct L1 head 的 macro-F1（0.5207）略高于 L2→L1 mapped prediction
  （0.5066），但 mapped accuracy 更高（0.7363 vs direct-head 0.6135）。
  两种输出强调的类别平衡不同，均应保留并分别报告。
- Xenium 中 direct L1 和 mapped L1 macro-F1 几乎相同（0.5548 vs 0.5564），
  说明独立 coarse head 与 fine-to-coarse mapping 的结果较一致。

### Generalization caution

两个 notebook 的 “in-sample” 指标包含 best-fold model 训练时见过的 cells，
不能作为独立验证结果。尤其 Xenium L2 in-sample macro-F1 为 0.7245，而 CV
macro-F1 只有 0.1144，差距很大；模型效果判断必须以 CV 指标为主。

此外，两个已保存结果的 checkpoint selection metric 不一致：

- HCC：`three_tier_auc_sum`
- Xenium persisted run：`four_term_sum`

因此当前结果适合描述各数据集内的性能，但不是严格受控的跨数据集 benchmark。
若要正式比较，需固定随机种子、fold policy、class weighting、selection metric，
并明确报告每层类别频率与 per-class metrics。

## Compatibility policy

- 已有 Xenium five-head checkpoint 的参数名和 tensor shape 保持不变。
- `use_five_head=True` 仍构建原 `FiveHeadImprovedMLPClassifier`。
- Xenium 的 CNiche/TNiche 数据加载与 five-head loss 路径保持原行为。
- 新增 three-head 路径通过独立开关和 checkpoint head detection 启用。
- 在动态统一架构完成前，不应删除 legacy single/dual/five-head loader，以便旧 checkpoint 可继续预测。
- 任何 loss 或 checkpoint schema 的修改都必须分别运行 dual、three、five-head smoke tests。

## Changelog

### 2026-08-12 — HCC preprocess CSVs + StarDist CSV source + one-shot h5ad

Updated:

- Added `code/CODEX_hcc/match_codex_cells_with_pixel.py`
  (Xenium analog of `code/Xenium_lung/match_HEanno_with_sample_pix.py`).

  Prebuild functions and I/O:

  1. `{MATCHED_HE}_cells_with_pixel.csv`
     - Function: `build_cells_with_pixel()`
     - Inputs:
       - CODEX `cell_data`:
         `s4769/{acq_id}/{acq_id}.cell_data.csv` (e.g. `CELL_ID`, `X`, `Y`)
       - CODEX `cell_types`:
         `s4769/{acq_id}/{acq_id}.cell_types.csv` (`CELL_ID`, `ANNOTATION_LABEL`)
       - Excel hierarchy (via `s4769_img_cell_mapping.load_codex_celltype_hierarchy()`):
         `s4769/HE/s4769_he_mapping_updated_Visium.xlsx` sheet `Celltype`
         (`celltype_level2` → `celltype_level1` → `celltype_level0`)
     - Output:
       `s4769/HE/{MATCHED_HE}/{MATCHED_HE}_cells_with_pixel.csv`

  2. `{MATCHED_HE}_cells_matched_by_stardist.csv`
     - Function: `build_cells_matched_by_stardist()`
       (calls `Hist2Pheno_pkg.base.match_celltype2stardist`)
     - Inputs:
       - GT table from (1): `*_cells_with_pixel.csv`
       - Prepared StarDist nuclei only from
         `data/CODEX/HCC/StarDist_Segment/{MATCHED_HE}/{MATCHED_HE}_Float_prob0.01_nms_0.3.csv`
         (no rebuild from UNI `.pth` filenames)
     - Output:
       `s4769/HE/{MATCHED_HE}/{MATCHED_HE}_cells_matched_by_stardist.csv`

- `HCC_train_validate_cv_UNIlabel_single.ipynb`
  - Loads the two prebuilt CSVs above instead of regenerating them inline
  - Uses StarDist tables only from `data/CODEX/HCC/StarDist_Segment`
  - Builds matched h5ad in the notebook, once each
    (`FORCE_REBUILD_*` stays False when the file already exists):
    - `{MATCHED_HE}_matched_features.h5ad` (GT UNI embeddings)
    - `{MATCHED_HE}_matched_features_stardist.h5ad` (StarDist UNI embeddings)

Reason:

- Keep shared CSV preprocessing outside the training notebook (same pattern as Xenium lung).
- Fail clearly if the StarDist CSV is missing under `StarDist_Segment`.
- Matched h5ad remains a notebook-side cache and should not be rebuilt every run.

Usage:

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py \
  --he-key awy-98938_aligned_0d535a74
# or all ALIGNED annotated regions:
conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py
```

Then open `HCC_train_validate_cv_UNIlabel_single.ipynb` and run; h5ad files are created on first pass.

### 2026-08-12 — HCC cross-dataset notebook (`HCC_train_validate_cv_UNIlabel_all.ipynb`)

Updated:

- Added `code/CODEX_hcc/HCC_train_validate_cv_UNIlabel_all.ipynb` as the twin of
  `code/Xenium_lung/Lung_train_validate_cv_UNIlabel_all.ipynb`.
- Pools all **36** ALIGNED annotated HCC regions; dataset-level group CV.
- Three-head prediction visibility:
  - Internal: L2 OOF + L1/L12 confusion matrices
  - StarDist per tier: `levels=["l2"]`, `["l1"]`, `["l12"]`
- Spatial context on by default (`k=8`, `mean`); selection metric `three_tier_auc_sum`.
- Outputs under `s4769/result_all_spatial/` (and `.../stardist/{MATCHED_HE}/`).

### 2026-08-12 — HCC multi-sample CLI (`HCC_train_validate_cv_UNIlabel.py`)

Updated:

- Added `code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py` as the CODEX HCC twin of
  `code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py`.
- Default sample set: all **36** ALIGNED annotated regions from
  `list_aligned_annotated_regions()` (not Xenium Complete_Cases).
- Paths: `data/CODEX/HCC/Michael_data_transfer/s4769/HE/{MATCHED_HE}/project_all_UNI/`.
- Three-head hierarchy + `three_tier_auc_sum` checkpoint selection.
- Spatial neighbor fusion enabled by default with lung defaults:
  `build_spatial_neighbor_index` / `--spatial-k 8` / `--spatial-mode mean`
  (disable with `--no-use-spatial-context`).
- GT / StarDist CSV inputs from `match_codex_cells_with_pixel.py`;
  StarDist centroids from `data/CODEX/HCC/StarDist_Segment`.

Usage:

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

# One MATCHED_HE, full pipeline (spatial on by default)
python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py \
  --sample awy-98938_aligned_0d535a74

# All 36 annotated ALIGNED regions (per-sample)
python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py

# Cross-dataset pool + spatial kNN (same lung spatial defaults)
python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_bs4096

# HE h5ad only
python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py \
  --sample awy-98938_aligned_0d535a74 --steps he_h5ad
```

### 2026-08-11 — HCC versus Xenium lung architecture and performance review

Updated:

- 比较 `HCC_train_validate_cv_UNIlabel_single.ipynb` 与
  `Lung_train_validate_cv_UNIlabel_single.ipynb` 的 shared backbone、heads、loss、
  CV configuration、parameter count 和 persisted evaluation results。
- 在本 README 增加跨数据集结果解释与 generalization cautions。
- 根据确认后的设计修正 unified framework 描述：任意数量 heads 均有直接 CE，
  但 mapping consistency 只计算 fine L2 → coarse L1。

Reason:

- 说明两个数据集使用同一 modeling framework，但不是相同 output heads。
- 避免把类别数量不同、checkpoint selection metric 不同的 CV 数字解释成严格 benchmark。
- 保持文档设计目标与实际需要一致。

### 2026-08-11 — CODEX HCC three-head hierarchy

Updated:

- `code/Hist2Pheno_pkg/base.py`
  - 新增 `ThreeHeadImprovedMLPClassifier`
  - 新增 `HCC_H5AD_OBS_COLUMNS`
  - matched h5ad cache 可按数据集指定必须的 obs columns
  - `prepare_data_from_matched_h5ad(..., require_niche_heads=False)`
  - DataLoader 支持只有 L12、没有 CNiche/TNiche 的 batch
- `code/Hist2Pheno_pkg/model.py`
  - model builder、training、validation、checkpoint reload 和 prediction 支持 three-head
  - three-head loss 增加 L12 direct CE
  - 新增 `three_tier_auc_sum`
  - 修复二分类 macro-AUROC 的 `label_binarize` shape
- `code/Hist2Pheno_pkg/plot.py`
  - softmax probability collection 支持 three-head，不再假设有 L12 就一定有 L3/L4
- `HCC_train_validate_cv_UNIlabel_single.ipynb`
  - 从 `s4769_he_mapping_updated_Visium.xlsx` 的 `Celltype` sheet 读取 12 → 6 → 4 hierarchy
  - 删除 HCC 的 CNiche/TNiche placeholder labels
  - 使用 `final_CT`、`final_sublineage`、`final_lineage` 训练三个独立 heads
  - checkpoint selection 使用 `three_tier_auc_sum`

Reason:

- HCC 只有三层真实 cell-type hierarchy；用 lineage 复制生成 CNiche/TNiche 会引入虚假监督。
- Excel hierarchy 是本数据集的唯一 mapping source，应避免 notebook 内手写重复映射。
- three-head 模式用于完成当前 HCC 分析，同时保持 Xenium five-head checkpoint 兼容。

Validation:

- HCC hierarchy 验证为 12 fine / 6 intermediate / 4 coarse classes。
- 示例 acquisition 共 33,731 cells；保留 33,060 个属于 hierarchy 的 cells。
- 排除不在新 hierarchy 中的 `Unknown` 和 `Stroma Uncharacterized`。
- synthetic three-head train → validation → checkpoint reload → three-head prediction smoke test 通过。
- legacy Xenium five-head train → validation → checkpoint reload → five-head prediction regression test 通过。
- Python syntax 与 notebook JSON 验证通过。

Known limitation:

- 本次更新不是最终的任意层数动态 hierarchy framework。
- three-head 只有 fine → coarse mapping consistency；这是确认后的目标设计，
  不计划增加 fine → intermediate 或 intermediate → coarse mapping loss。

## Legacy notes

旧的批量 UNI feature extraction 后台任务记录仍保存在 `README.txt`。
