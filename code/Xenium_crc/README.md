# Xenium CRC (Visium HD companion)

Folder renamed from `code/VisiumHD_crc` → `code/Xenium_crc` (2026.09.14).

Notebook: `Data_process_HEcelltype_CRC.ipynb`  
Paper metadata clone: `HumanColonCancer_VisiumHD/`  
Raw Space Ranger-style data: `data/VisiumHD/CRC/` (see `data/VisiumHD/README.md`).  
Xenium alignment tables: `data/Xemium/CRC/Xenium_Visium_Alignment/`.

Helpers (same pattern as `code/Xenium_brca`):

- `crc_paths.py` — paths, Flex L2→L1 map, Visium HD loaders, P1/P2/P5 Cases / StarDist / HE
- `CRC_plot.py` — spatial scatter helpers
- `match_xenium_cells_with_visiumhd.py` — transfer RCTD Label1 onto Xenium cells
- `match_xenium_cells_with_pixel.py` — HE pixels + StarDist match → `Cases/{P1,P2,P5}CRC/`
- `demo.sh` — all-sample command index (P1/P2/P5; BRCA-style `_all` path)
- `demo_P2.sh` — P2-only shortcut
- `CRC_train_validate_cv_UNIlabel.py` — per-sample or `--mode cross-dataset` three-head train
- `CRC_train_validate_cv_UNIlabel_single.ipynb` — one-sample three-head train / validate
- `CRC_train_validate_cv_UNIlabel_all.ipynb` — pooled P1+P2+P5 (dataset-level CV; `SKIP_POOLED_TRAIN=True`)
- `Pred_statistic_visual_crc_all.ipynb` — pooled StarDist ROC + per-sample macro AUROC (P1 vs P2 vs P5)

Cell type used here: **`DeconvolutionLabel1`** (RCTD first type, Level2).
L2 colors (`L2_COLORS` in `crc_paths.py`) follow the Oliveira et al. Flex legend.

## Current status

**P1CRC / P2CRC / P5CRC** are all on the BRCA-style Hist2Pheno path: RCTD transfer → pixel + StarDist match → UNI → h5ad → three-head pooled CV → Pred.

| | P1CRC | P2CRC | P5CRC |
|--|-------|-------|-------|
| Excel sheet | `P1_CRC` | `P2_CRC` | `P5_CRC` |
| Transferred singlets on HE | 154,760 | 209,779 | 129,465 |
| StarDist-matched (≤50 px) | 96,858 | 166,503 | 94,194 |
| UNI PNG dumps | skip (`--no_save_patch_images`) | keep | skip |
| Cases h5ads | matched + StarDist matched + all-nuclei | same | same |

Pooled three-head train (`--mode cross-dataset`, spatial k=8 mean, `--ablation-tag D_emph_L2_spatial_crc`):

- OOF is leave-one-dataset-out (3 samples). `best_mlp_gpu.pt` is refit on all cells; LORO fold kept as `best_mlp_gpu_loro.pt`.
- Outputs: `data/Xemium/CRC/Results/result_all_spatial/`
- `_all.ipynb` with `SKIP_POOLED_TRAIN=True` only **loads** that checkpoint (no retrain).
- §5 all-nuclei inference writes `{sample}_all_features_stardist_label.h5ad` and pred maps for **all three** samples.
- Pred notebook: pooled StarDist ROC + P1 vs P2 vs P5 macro AUROC.

Labels remain **transferred RCTD singlet**, not native Xenium GT. No clinical niche-index notebook.

## 2026.09.22 — P5 on the pooled train / Pred path

P5 RCTD transfer (`P5_CRC` sheet), pixel match, UNI (no PNG), StarDist match (~2.05M nuclei CSV), matched + all-nuclei h5ads, then 3-sample cross-dataset with `--no-resume-from-checkpoints` (replaces the earlier P1+P2 `result_all_spatial`). `_all.ipynb` §5 and `Pred_statistic_visual_crc_all.ipynb` include P5.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P5CRC
python code/Xenium_crc/match_xenium_cells_with_pixel.py --sample P5CRC
SAMPLE=P5CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
SAMPLE=P5CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
python -u code/Xenium_crc/transer_embedding_label_h5ad.py --sample P5CRC --steps he_h5ad stardist_h5ad stardist_all_h5ad
python -u code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_crc \
  --no-resume-from-checkpoints
```

## 2026.09.15 — Hist2Pheno train path (P1 / P2 / P5)

Command index: [`demo.sh`](demo.sh). Sample keys **`P1CRC` / `P2CRC` / `P5CRC`**. Excel sheets **`P1_CRC` / `P2_CRC` / `P5_CRC`**.

This is the BRCA-style all-sample path (`BRCA_train_validate_cv_UNIlabel_all.ipynb` analogue):

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py          # all patients
python code/Xenium_crc/match_xenium_cells_with_pixel.py             # all samples
bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
python -u code/Xenium_crc/transer_embedding_label_h5ad.py --steps he_h5ad
python -u code/Xenium_crc/transer_embedding_label_h5ad.py --steps stardist_h5ad stardist_all_h5ad
python -u code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py \
  --mode cross-dataset \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_crc \
  --no-resume-from-checkpoints
# then CRC_train_validate_cv_UNIlabel_all.ipynb (SKIP_POOLED_TRAIN=True)
# then Pred_statistic_visual_crc_all.ipynb (pooled StarDist ROC / P1 vs P2 vs P5 AUROC)
```

P2-only debug remains [`demo_P2.sh`](demo_P2.sh) + `_single.ipynb`.

- HE: `data/Xemium/CRC/HE_images/Xenium_V1_Human_Colon_Cancer_P{1,2,5}_CRC_Add_on_FFPE_he_image.ome.tif` (~0.274 µm/px; UNI `--scale` ≈ 0.548)
- `X_pix_HE` / `Y_pix_HE` = paper `x_centroid_visium_scale` / `y_centroid_visium_scale`
- StarDist CSVs exist for P1, P2, and P5 under `StarDist_Segment/`
- P1/P5 UNI does **not** write `sc_pth_16_16_image` (`--no_save_patch_images`). P2 keeps PNG dumps.
- Labels remain **transferred RCTD singlet**, not Xenium GT
- UNI / train are long GPU jobs; pin a GPU and do not launch them accidentally

## Data layout

```text
data/Xemium/CRC/
├── Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx   # celltype + P{1,2,5}_CRC
├── HE_images/Xenium_V1_Human_Colon_Cancer_P{1,2,5}_CRC_Add_on_FFPE_he_image.ome.tif
├── Xenium_Visium_Alignment/Xenium_P{1,2,5}_cell_info.csv
├── Cases/{P1CRC,P2CRC,P5CRC}/
│   ├── {sample}_cells_with_pixel.csv
│   ├── {sample}_cells_matched_by_stardist.csv
│   ├── {sample}_matched_features.h5ad
│   ├── {sample}_matched_features_stardist.h5ad
│   ├── {sample}_all_features_stardist.h5ad
│   └── project_all_UNI/
│       ├── ImgEmbeddings_all/sc_pth_16_16/
│       └── ImgEmbeddings_all_stardist/sc_pth_16_16/
├── StarDist_Segment/Xenium_V1_Human_Colon_Cancer_P{1,2,5}_CRC_Add_on_FFPE_he_image.ome/
└── Results/result_all_spatial/                      # --pooled-save-result
    ├── cross_dataset_cv/D_emph_L2_spatial_crc/best_mlp_gpu.pt
    └── stardist/{P1CRC,P2CRC,P5CRC}/
```

Visium HD bins live under `data/VisiumHD/CRC/P{1,2,5}_CRC/` (not under `data/Xemium/CRC/`).

## 2026.09.14 — Hist2Pheno train path (P2)

P2 is wired like BRCA (`rep1`): transferred labels → Cases CSVs → UNI → h5ad → three-head notebook.

Command index: [`demo_P2.sh`](demo_P2.sh). Sample key **`P2CRC`**.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC
python code/Xenium_crc/match_xenium_cells_with_pixel.py --sample P2CRC
SAMPLE=P2CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
SAMPLE=P2CRC bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist
# then open CRC_train_validate_cv_UNIlabel_single.ipynb
```

- HE: `data/Xemium/CRC/HE_images/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_image.ome.tif` (~0.274 µm/px; UNI `--scale` ≈ 0.548)
- `X_pix_HE` / `Y_pix_HE` = paper `x_centroid_visium_scale` / `y_centroid_visium_scale` (same canvas as add-on HE / StarDist)
- StarDist: `StarDist_Segment/Xenium_V1_Human_Colon_Cancer_P2_CRC_Add_on_FFPE_he_image.ome/` (~1.5M nuclei)
- Labels remain **transferred RCTD singlet**, not Xenium GT
- UNI / train are long GPU jobs; pin a GPU and do not launch them accidentally

## 2026.09.14 — parquet only for bin annotations

`P{1,2,5}CRC_Metadata.parquet` already contain every column of
`DeconvolutionResults_P{1,2,5}CRC.csv` (`barcode`, `DeconvolutionClass`,
`DeconvolutionLabel1`, `DeconvolutionLabel2`), plus spatial coordinates and
paper annotations (`tissue`, `X`, `Y`, `Periphery`, `UnsupervisedL1/L2`,
`MacrophageSubtype`, `GobletSubcluster`).

**Deleted** (redundant):

- `HumanColonCancer_VisiumHD/MetaData/DeconvolutionResults_P1CRC.csv`
- `HumanColonCancer_VisiumHD/MetaData/DeconvolutionResults_P2CRC.csv`
- `HumanColonCancer_VisiumHD/MetaData/DeconvolutionResults_P5CRC.csv`

**Notebook now reads only:**

- `HumanColonCancer_VisiumHD/MetaData/P1CRC_Metadata.parquet`
- `HumanColonCancer_VisiumHD/MetaData/P2CRC_Metadata.parquet`
- `HumanColonCancer_VisiumHD/MetaData/P5CRC_Metadata.parquet`

Flex single-cell hierarchy is unchanged: `SingleCell_MetaData_2025.csv`
(Hist2Pheno) / `SingleCell_MetaData_2024.csv` (FineST comparison).

## 2026.09.14 — Xenium vs Visium HD overlap

Notebook section 4 reads paper alignment tables
`data/Xemium/CRC/Xenium_Visium_Alignment/Xenium_P{1,2,5}_cell_info.csv`.
These have centroids / counts / areas and Visium-scaled pixels, **not** cell types.

Each Xenium cell is mapped to the nearest on-tissue 8 µm bin
(`tissue_positions.parquet` `pxl_col/row_in_fullres`) if distance &lt; one bin
diameter. P2: ~29% of Xenium cells fall in the Visium capture bbox; mapped
cells sit ~3 µm from the bin center. RCTD `DeconvolutionLabel1` of that bin is
used for L1 / periphery composition vs all Visium HD bins.

## 2026.09.14 — transferred Xenium annotations (P1 / P2 / P5)

Notebook section 5 / `match_xenium_cells_with_visiumhd.py` writes BRCA-style

`data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx`

| Excel column | Meaning |
|--------------|---------|
| `celltype_level2` | Flex Level2 (RCTD `DeconvolutionLabel1`) |
| `celltype_level12` | Flex 2025 Level1, 9 GT classes |
| `celltype_level1` | 4 coarse parents: Epithelial / Stromal / Immune / Endothelial (Neuronal → Stromal) |

`Unlabeled` has empty L1/L12. Each `P{1,2,5}_CRC` sheet is Xenium `cell_id` (`Barcode`) + transferred
L2 (`Cluster`) + L12/L1, RCTD **singlet** only.

These are **transferred** labels (serial sections, RCTD on 8 µm bins), not
Xenium ground truth.

```
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py              # all patients
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --refresh-hierarchy
```
