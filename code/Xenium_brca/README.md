# Xenium BRCA development notes

本文件记录 `code/Xenium_brca` 及其数据路径更新、修改原因和当前预处理 / 训练状态。
后续每次功能性更新应在下方 changelog 顶部增加一条记录。

运行环境：`conda activate SeededNTM`。

数据来源：10x Xenium FFPE Human Breast Cancer preview（rep1 / rep2 邻切片）及
Janesick et al. GEO 注释 [GSE243275](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243275)。

## Current status

BRCA 已接入与 `code/Xenium_lung` / `code/CODEX_hcc` 同款的 **Hist2Pheno 训练流程**。
细胞类型是 **three-head**（与 HCC 相同：L2 / L12 / L1），不是 lung 的 five-head（无 CNiche / TNiche）。

坐标约定：

- `obsm["spatial"]`：Xenium `x_centroid` / `y_centroid`（µm）
- `obsm["spatial_HE"]`：post-Xenium **HE TIFF 像素**，`inv(he_imagealignment) @ (µm / 0.2125)`
- 不要把 HE TIFF 当成 morphology 的 0.2125 µm/px（那会把坐标画到比 TIFF 更大的画布上）
- UNI：HE ≈ 0.364 µm/px → `--scale 0.728`，`patch_size=16` → ~8 µm
- 注释按 `cell_id` ↔ `Barcode` join，不按行号对齐
- `Unlabeled` 在 match 阶段排除（LY `celltype` 表中 L12/L1 为空）

命令索引见 [`demo.sh`](demo.sh)。推荐评估命令（pool rep1+rep2 + spatial k=8 mean）：

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
python code/Xenium_brca/match_xenium_cells_with_pixel.py
SAMPLE=rep1 bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh gt
SAMPLE=rep1 bash code/Xenium_brca/demo_UNI_feature_extraction_batch.sh stardist
# 对 rep2 重复上面两行，或去掉 SAMPLE= 一次跑两个 replicate
python -u code/Xenium_brca/transer_embedding_label_h5ad.py --sample rep1
python -u code/Xenium_brca/BRCA_train_validate_cv_UNIlabel.py \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_bs4096
```

HE TIFF ~1.5G / replicate，StarDist 核约 25 万，UNI 与训练都是长 GPU 任务；不要在没指定 GPU 的情况下误跑。
两个 replicate 的 cross-dataset CV 是 **leave-one-replicate-out**（CLI 会把 `cv_k` clamp 到样本数）。

本 preview 没有 lung/HCC 那种临床分组，因此没有 niche 指数 / 临床 AUROC 对比；看效果用 per-sample notebook、cross-dataset notebook，以及 `Pred_statistic_visual_brca_all.ipynb` 的 pooled ROC 与 rep1 vs rep2 macro AUROC。

## Data layout

```text
data/Xemium/BRCA/
├── Annotation/                          # 论文 / 校正后的 cell-type 表
│   ├── GSE243275_Barcode_Cell_Type_Matrices.xlsx
│   ├── GSE243275_Barcode_Cell_Type_MatricesLY.xlsx
│   ├── cell_groups.csv                  # 10x companion 使用的 R1 supervised
│   └── GSM7780153_Xenium_R1_Fig1-5_supervised_wrong.csv
├── Results/                             # 预处理 QC + cross-dataset 训练输出
│   ├── xenium_brca_rep{1,2}_*
│   └── result_all_spatial/              # --pooled-save-result（训练后）
├── Cases/{rep1,rep2}/                   # Hist2Pheno 样本目录
│   ├── {sample}_cells_with_pixel.csv
│   ├── {sample}_cells_matched_by_stardist.csv
│   ├── {sample}_matched_features.h5ad
│   ├── {sample}_matched_features_stardist.h5ad
│   ├── {sample}_all_features_stardist.h5ad
│   └── project_all_UNI/
│       ├── ImgEmbeddings_all/sc_pth_16_16/            # UNI on GT pixels
│       └── ImgEmbeddings_all_stardist/sc_pth_16_16/   # UNI on StarDist
├── human_breast_Xenium_rep1/
├── human_breast_Xenium_rep2/
└── StarDist_Segment/
    └── Xenium_FFPE_Human_Breast_Cancer_Rep{1,2}_he_image/
```

旧目录名 `xenium-publication/` 已拆成 `Annotation/`（输入）和 `Results/`（输出）。

## Code

| 文件 | 作用 |
|------|------|
| `demo.sh` | 与 lung/HCC 同款的命令索引 |
| `Data_process_HEcelltype_BRCA.ipynb` | rep1 / rep2 预处理：读 10x matrix + `cells.csv.gz` + supervised 注释 |
| `brca_paths.py` | 样本注册、hierarchy 读取、Cases / StarDist 路径 |
| `match_xenium_cells_with_pixel.py` | 写 `{sample}_cells_with_pixel.csv` 与 StarDist 匹配表 |
| `demo_GT_feature_extraction_Single.sh` | 单 replicate UNI（`gt` / `stardist`） |
| `demo_UNI_feature_extraction_batch.sh` | rep1+rep2 UNI 批处理 |
| `transer_embedding_label_h5ad.py` | GT / StarDist matched h5ad，以及 all-nuclei StarDist h5ad |
| `BRCA_train_validate_cv_UNIlabel.py` | three-head 训练 CLI（per-sample 或 `--mode cross-dataset`） |
| `BRCA_train_validate_cv_UNIlabel_single.ipynb` | 单 replicate 训练 / 验证 / StarDist（默认 `rep1`） |
| `BRCA_train_validate_cv_UNIlabel_all.ipynb` | pool rep1+rep2；`SKIP_POOLED_TRAIN=True` 可只加载 CLI 权重 |
| `Pred_statistic_visual_brca_all.ipynb` | pooled StarDist ROC + 两 replicate macro AUROC |
| `BRCA_plot.py` | 论文 cell-type hex palette（`ctype_hex_map`） |
| `companion_notebook_compare_ROI_istar.ipynb` | 历史 10x companion 副本（路径已过时） |

Palette 在 `code/Hist2Pheno_pkg/plotting_palettes.py`（`PAN_ORGAN="xenium_brca"`）。

## Annotation sheets

Notebook 当前读取 `GSE243275_Barcode_Cell_Type_MatricesLY.xlsx`：

| Replicate | Sheet | 细胞数 |
|-----------|-------|--------|
| rep1 | `Xenium R1 Fig1-5 (supervised)` | 167,780 |
| rep2 | `Xenium R2 Fig1-5 (supervised)LY` | 118,752 |

`GSM7780153_Xenium_R1_Fig1-5_supervised.csv` 中 DCIS 1 / DCIS 2 标反，不要当 GT。
rep1 / rep2 是邻切片；原始 R2 supervised sheet 上同一批导管的 DCIS 1 / DCIS 2 与 rep1 对调，故 rep2 改用 `…(supervised)LY`。

### `celltype` hierarchy（Hist2Pheno three-head）

LY sheet `celltype` 使用 Hist2Pheno 列名：`celltype_level2` → `celltype_level12` → `celltype_level1`
（fine / intermediate / coarse）。`brca_paths.load_brca_celltype_hierarchy` 会把它们规范成
训练 CSV 用的 `final_CT` / `final_sublineage` / `final_lineage`。
`celltype_level2` 与 supervised `Cluster` 下划线写法一致。

| L1 coarse (`celltype_level1`) | L12 intermediate (`celltype_level12`) | L2 fine |
|-------------------------------|----------------------------------------|---------|
| Epithelial | DCIS | `DCIS_1`, `DCIS_2` |
| Epithelial | Invasive tumor | `Invasive_Tumor`, `Prolif_Invasive_Tumor` |
| Epithelial | Myoepithelial | `Myoepi_ACTA2+`, `Myoepi_KRT15+` |
| Epithelial | Tumor–T-cell hybrid | `T_Cell_&_Tumor_Hybrid` |
| Immune | T cells | `CD4+_T_Cells`, `CD8+_T_Cells` |
| Immune | B cells | `B_Cells` |
| Immune | Myeloid | `Macrophages_1`, `Macrophages_2`, `IRF7+_DCs`, `LAMP3+_DCs`, `Mast_Cells` |
| Immune | Stromal–T-cell hybrid | `Stromal_&_T_Cell_Hybrid` |
| Stromal | Fibroblasts | `Stromal` |
| Endothelial | Endothelial | `Endothelial` |
| Endothelial | Perivascular | `Perivascular-Like` |

`Unlabeled` 保留在表中，但 intermediate / coarse 为空，match 时排除（同 HCC `Unknown`）。

可训练层级：**19** fine / **11** intermediate / **4** coarse。
两个 hybrid 现在有不同的 L12 名字，因此 L12→L0 是 1:1（不再共用一个 `Hybrid`）。

## Changelog

### 2026-09-03 — HE TIFF pixels via `he_imagealignment.csv` (not 0.2125 µm/px)

Preprocess had used lung's morphology conversion `X_pix_HE = x_um / 0.2125`.
The BRCA HE TIFF is smaller (rep1 27587×20511) than that canvas (~35400×25800),
so StarDist (run on the TIFF) only matched ~54% of GT cells, with max distance
thousands of pixels.

Fix: map microns through 10x `*_he_imagealignment.csv`
(`HE_px = inv(M) @ (um / 0.2125)`). Keep morphology pixels as `X_pix_morph`.
UNI `--scale` is now **0.728** (HE ≈ 0.364 µm/px → 0.5 µm/px). StarDist match
drops pairs farther than 50 HE px.

### 2026-09-03 — Hist2Pheno train path (UNI + three-head, lung/HCC 同款流程)

Added:

- `brca_paths.py`, `match_xenium_cells_with_pixel.py`, `transer_embedding_label_h5ad.py`
- UNI scripts: `demo_GT_feature_extraction_Single.sh`, `demo_UNI_feature_extraction_batch.sh`
- Train CLI + notebooks: `BRCA_train_validate_cv_UNIlabel.py`, `_single.ipynb`, `_all.ipynb`
- `Pred_statistic_visual_brca_all.ipynb`, `demo.sh`
- `plotting_palettes.py` schemes `xenium_brca_fine/intermediate/coarse`

Hierarchy loader accepts the current LY columns (`celltype_level2/12/1`) and still
understands the older HCC-style `level2/1/0` names.

Reason: 用 Hist2Pheno 跑 10x FFPE Human Breast Cancer preview（rep1+rep2 邻切片 H&E），
看 three-head 细胞类型预测效果。HE 很大，默认不自动跑 UNI/训练。

### 2026-09-03 — BRCA `Celltype` hierarchy in LY workbook

- Replaced the copied HCC `Sheet1` (Fibroblasts / INOS epithelium / …) with
  sheet `celltype` covering all 20 Janesick supervised clusters.
- Mapping is breast-specific: DCIS / invasive / myoepithelial under
  Epithelial; T / B / myeloid under Immune; `Stromal` under Stromal;
  endothelial + perivascular under Vascular. Two paper hybrid classes stay
  as L12 `Hybrid`. `Unlabeled` has empty parents.

### 2026-09-03 — Pooled cell-type bar chart (`plot_pooled_celltype_distribution`)

- `BRCA_plot.plot_pooled_celltype_distribution` wraps the shared
  `plotting_utils` bar chart and colors bars with `ctype_hex_map`.
- `Data_process_HEcelltype_BRCA.ipynb` pools `processed` rep1/rep2
  `cell_info` tables (`cell_type` column; drops `Unlabeled`).
- Output: `Results/xenium_brca_rep1_rep2_pooled_celltype_distribution.png`.

### 2026-09-03 — LY annotation workbook for rep2 DCIS 1 / DCIS 2

Updated:

- `Data_process_HEcelltype_BRCA.ipynb` 注释表改为
  `data/Xemium/BRCA/Annotation/GSE243275_Barcode_Cell_Type_MatricesLY.xlsx`。
- rep2 sheet 改为 `Xenium R2 Fig1-5 (supervised)LY`；rep1 仍用
  `Xenium R1 Fig1-5 (supervised)`。
- QC 空间图使用 `BRCA_plot.ctype_hex_map`（同时接受 `DCIS_1` 与 `DCIS 1` 两种写法）。

Reason:

- 邻切片 QC 图上，rep2 大导管结构的 DCIS 1（橙）与 DCIS 2（粉）相对 rep1 对调。
- `Annotation/READMEanno.txt` 记录原始 R2 sheet 反了，LY sheet 为校正版。

### 2026-09-03 — Split `xenium-publication/` and rename code files

Updated:

- 数据：`xenium-publication/` → `Annotation/`（xlsx / paper CSV）+ `Results/`（h5ad / per-rep CSV / QC）。
- 代码：`companion_functions.py` → `BRCA_plot.py`。
- Notebook：`companion_notebook_compare_ROI_istar.ipynb` 的现行预处理副本为
  `Data_process_HEcelltype_BRCA.ipynb`（路径指向新目录）。

Reason:

- 把论文原始注释与本仓库生成结果分开，避免覆盖 GEO / 10x 输入。

### 2026-09-02 — Dual-replicate Xenium BRCA preprocessing

Updated:

- 对 `human_breast_Xenium_rep1` 与 `human_breast_Xenium_rep2` 分别：
  读 `cell_feature_matrix.h5`、`cells.csv.gz`、supervised `Barcode`/`Cluster`；
  写 `spatial` / `spatial_HE`、`cell_type`，保存 replicate-specific h5ad 与 CSV。
- 坐标转换复用 `code/Xenium_lung/xenium_coords.py`。

Validation:

- rep1：167,780 cells × 313 genes；rep2：118,752 cells × 313 genes。
- 两套 replicate 的 matrix / cells / annotation barcode 均为 1:1，未匹配数为 0。

## Legacy notes

更早的 10x companion 拷贝命令、DCIS 标反备忘见
`data/Xemium/BRCA/Annotation/READMEanno.txt` 与
`data/Xemium/BRCA/README.txt`。
