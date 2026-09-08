# CODEX GBM development notes

本文件记录 `code/CODEX_gbm` 及其数据路径、层次标注和 Hist2Pheno 训练状态。
后续功能性更新应在下方 changelog 顶部增加一条记录。

运行环境：`conda activate SeededNTM`。

数据来源：WangLab **Visium HD** IDH-mutant glioma（Initial `P174511` / Recurrent `P179161`），
放在 `data/CODEX/GBM/WangLab/`。尽管目录名是 CODEX，实际是显微镜 HE + 单核注释，
Hist2Pheno 流程与 [`code/CODEX_hcc`](../CODEX_hcc) / [`code/Xenium_brca`](../Xenium_brca) 同款 **three-head**。

## Current status

- 层次 Excel：[`WangLab/3_Annotation_Table/GBM_sc_seg_celltypes_hierarchy.xlsx`](../../data/CODEX/GBM/WangLab/3_Annotation_Table/GBM_sc_seg_celltypes_hierarchy.xlsx) sheet **`Celltype`**
  （观测三元组，按细胞类型语义挂 head，**不是**把 SN 当 coarsest）：
  - **L2** fine = `subcluster` → `final_CT`
  - **L12** intermediate = `spatial_niche` SN1–SN9 → `final_sublineage`
  - **L1** coarse = `cell_type` → `final_lineage`
  - 丢掉 SN **LowQ / missing**，以及 `Unknown` / `LowQ` cell labels
- 可训练：L2 **16** `subcluster` / L12 **9** SN / L1 **5** `cell_type`
- `loc.csv` 的 `x,y` **已是显微镜 HE 像素**（与 StarDist 同画布；无需 BRCA affine）
- StarDist 匹配（≤50 px）：Ini **46,689**（median **0.66** px）；Rec **155,794**（median **0.78** px）
- Per-sample / 小样本：`resolve_stratified_cv_params`（默认 `auto_cv_k=True`）会在稀有 strata 上
  自动 coarsen `joint→level2→level1` 并降低 `cv_k`，避免 `StratifiedKFold` 崩掉
- Cross-dataset（Ini+Rec）：leave-one-section-out（`cv_k` clamp 到样本数）
- UNI / 训练脚本已就绪，**默认不跑**（Ini HE ~2G；Rec HE ~4G；Rec StarDist ~1.16M 核）
- `_all.ipynb` §5：`step_pooled_stardist_all` → `*_all_features_stardist_label.h5ad`（全 nuclei；无 GT AUROC）

命令索引见 [`demo.sh`](demo.sh)。

推荐评估（pool Ini+Rec + spatial k=8 mean）：

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
# preprocess notebook already mirrored by Results CSVs; rematch if needed:
python -u code/CODEX_gbm/match_codex_cells_with_pixel.py
SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh gt
SAMPLE=P174511_Initial bash code/CODEX_gbm/demo_UNI_feature_extraction_batch.sh stardist
# repeat for P179161_Recurrent, or omit SAMPLE=
python -u code/CODEX_gbm/transer_embedding_label_h5ad.py --sample P174511_Initial
python -u code/CODEX_gbm/GBM_train_validate_cv_UNIlabel.py \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_gbm
```

## Data layout

```text
data/CODEX/GBM/
├── WangLab/                              # 原始分享
│   ├── 1_Bin_Matrix/{P174511_Initial,P179161_Recurrent}/
│   ├── 2_Single_Nuclei_Matrix/*_loc.csv  # HE 像素坐标
│   ├── 3_Annotation_Table/
│   │   ├── GBM_sc_seg_celltypes_hierarchy.xlsx
│   │   ├── 0917_Ini_nuclei_bin_spatialNiches_joined.xlsx
│   │   ├── 1_Bin/0917_ST_HD_200G_{Ini,Rec}_SpatialNiches.xlsx
│   │   ├── 2_Single_Nuclei/0917_{Ini,Rec}_sc_seg_celltypes.mapping.xlsx
│   │   └── 3_Mapping_bin_nuclei/0917_200G_celltype_spatialNiches_mapping_{Ini,Rec}.xlsx
│   ├── 4_Images/Microscope_Image/{174511-2-3,179161-4-3}.tif
│   └── StarDist_Segment/{174511-2-3,179161-4-3}/
├── Cases/{P174511_Initial,P179161_Recurrent}/
│   ├── {sample}_cells_with_pixel.csv
│   ├── {sample}_cells_matched_by_stardist.csv
│   ├── {sample}_matched_features.h5ad            # after transfer
│   └── project_all_UNI/...
└── Results/                              # 预处理 CSV + cross-dataset 训练输出
```

## Code

| 文件 | 作用 |
|------|------|
| `demo.sh` | 命令索引 |
| `gbm_paths.py` | 样本注册、hierarchy、路径、Rec `cell_type` 去前缀 |
| `Data_process_HEcelltype_GBM.ipynb` | join `cellID` ↔ loc + primary SN；写 Results CSV；HE QC |
| `match_codex_cells_with_pixel.py` | GT + StarDist Cases CSV |
| `demo_GT_feature_extraction_Single.sh` / `demo_UNI_feature_extraction_batch.sh` | UNI |
| `transer_embedding_label_h5ad.py` | matched / StarDist h5ad（`stardist_all_h5ad` 在 Rec 上可选） |
| `GBM_train_validate_cv_UNIlabel.py` | three-head 训练 CLI |
| `GBM_train_validate_cv_UNIlabel_single.ipynb` | 单样本（默认 Initial） |
| `GBM_train_validate_cv_UNIlabel_all.ipynb` | pool Ini+Rec；§5 all-nuclei；小样本 stratify/`cv_k` 自动调整 |
| `Pred_statistic_visual_gbm_all.ipynb` | Initial vs Recurrent macro AUROC |

Palette：`PAN_ORGAN="codex_gbm"`（`plotting_palettes.py`）。

## Hierarchy (`Celltype` sheet)

训练用 **观测三元组**（`gbm_paths.GBM_HIERARCHY_SOURCE_COLS` / `match_codex_cells_with_pixel`）：

| Head | 源列 | Cases 列 | 说明 |
|------|------|----------|------|
| L2 fine | `subcluster` | `final_CT` | AC-like / Mac_* / …（约 16，去 Unknown/LowQ） |
| L12 intermediate | `spatial_niche` | `final_sublineage` | SN1–SN9（去 LowQ / missing） |
| L1 coarse | `cell_type` | `final_lineage` | Tumor / Myeloid / Lymph / Oligo / Vascular |

不要套 HCC 的 Malignant / Immune / Glial / Stromal。SN 不是 coarsest lineage。

`cell_type` × `subcluster` 仍是生物学父子关系（示意）：

| L1 `cell_type` | L2 `subcluster` | Ini | Rec |
|----------------|-----------------|-----|-----|
| Tumor | AC-like, MES-like, OC-like, NPC-like, G1S, G2M | 3 states | 6 states |
| Myeloid | Mac_Tmr, Mac_SPP1, Mac_other, Mac_SEPP1 | yes | yes |
| Lymph | Lymphocyte | yes | yes |
| Oligo | Oligodendrocyte | yes | yes |
| Vascular | Vascular, CAF, Collagen_fibrils, lowQ_vas | yes | yes |
| Unknown | Unknown | yes | yes |
| LowQ | LowQ（Ini `subcluster` 为空） | 649 | 0 |

- Rec 原始 `cell_type` 带数字前缀（`0Tumor`…）；`normalize_gbm_cell_type` 去掉。
- Ini 把 `Mac_SEPP1` 标在 Vascular（443），Rec 标在 Myeloid（2309）；层次表允许多 parent（duplicate L2）。
- Bin-level SN 在 `1_Bin`；核↔bin 在 `3_Mapping_bin_nuclei`。Ini 全表见 `0917_Ini_nuclei_bin_spatialNiches_joined.xlsx`。
- 改 hierarchy / rematch 后需重建 matched h5ad（`--force-rebuild`），再训练。

## Changelog

### 2026-09-09 — Three-head mapping + small-data CV auto-adjust

- **Head 语义**（`gbm_paths`）：`subcluster`→L2，`spatial_niche`→L12，`cell_type`→L1
  （修正早期 README 把 L2/L12/L1 写反 / 把 SN 当 L1 的表述）。
- Match CSV：`final_CT` / `final_sublineage` / `final_lineage` 与上表一致。
- Per-sample 训练：`resolve_stratified_cv_params` + `--no-auto-cv-k` 开关；稀有
  strata 时自动降 `cv_k` / coarsen `stratify_target`（Ini 等小样本需要）。
- Ablation tag 与 `demo.sh` 对齐：`D_emph_L2_spatial_gbm`。
- Cross-dataset notebook §5 保留 all-nuclei 推理；与 BRCA 一样，§4 matched
  评估细胞数 ≈ GT，全 nuclei 看 `*_all_features_stardist_label.h5ad`。

### 2026-09-08 — GT tables drop SN LowQ / missing

- Hierarchy, Results CSVs, match, and GT `h5ad` all drop `spatial_niche` LowQ or missing,
  so `bin_barcode` is complete on labeled h5ad. Source annotation xlsx is unchanged.
  StarDist-all (unlabeled) is not filtered.

### 2026-09-04 — Preprocess notebook follows real 2-level labels

- `Data_process_HEcelltype_GBM.ipynb` loads the moved hierarchy (7 L1 / 18 L2),
  prints Ini×Rec crosstabs, and attaches primary `spatial_niche` from
  `3_Mapping_bin_nuclei` (max `intersect_ratio`).

### 2026-09-03 — Ini join table + hierarchy from real counts

- SpatialNiches has `Barcode`, not `cell_id`. Joined Ini nuclei + bin niches via WangLab
  `3_Mapping_bin_nuclei`; 7 mapping columns identical (0 diffs), plus nuclei/BANKSY columns.
- Moved hierarchy to `3_Annotation_Table/GBM_sc_seg_celltypes_hierarchy.xlsx`.
  Union is 7 L1 / 18 L2 (Ini adds `LowQ`). Training still drops Unknown/LowQ.

### 2026-09-03 — GBM hierarchy is two-level (not HCC)

- Rebuilt `GBM_sc_seg_celltypes_hierarchy.xlsx` from Ini/Rec `cell_type` × `subcluster`
  (6 L1 / 17 L2). Dropped the invented Malignant/Immune/Glial/Stromal coarse layer.

### 2026-09-03 — Initial Hist2Pheno port

- Added hierarchy xlsx + `code/CODEX_gbm` twin of BRCA/HCC three-head pipeline.
- Preprocess + StarDist match validated: coordinates already on microscope HE;
  match median &lt; 1 px after ≤50 px filter.
- UNI uses `scale_image=False` (do not apply CytAssist 0.883 µm/px to microscope TIFF).
