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
- `CRC_train_validate_cv_UNIlabel_single.ipynb` — one-sample three-head train / validate
- `CRC_train_validate_cv_UNIlabel_all.ipynb` — pooled P1+P2+P5 (dataset-level CV)

Cell type used here: **`DeconvolutionLabel1`** (RCTD first type, Level2).
L2 colors (`L2_COLORS` in `crc_paths.py`) follow the Oliveira et al. Flex legend.

## 2026.09.15 — Hist2Pheno train path (P1 / P2 / P5)

Command index: [`demo.sh`](demo.sh). Sample keys **`P1CRC` / `P2CRC` / `P5CRC`**. Excel sheets **`P1_CRC` / `P2_CRC` / `P5_CRC`**.

This is the BRCA-style all-sample path (`BRCA_train_validate_cv_UNIlabel_all.ipynb` analogue):

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py          # all patients
python code/Xenium_crc/match_xenium_cells_with_pixel.py             # all samples
bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh gt
bash code/Xenium_crc/demo_UNI_feature_extraction_batch.sh stardist  # skips missing StarDist (P5)
# then CRC_train_validate_cv_UNIlabel_all.ipynb
# or: python -u code/Xenium_crc/CRC_train_validate_cv_UNIlabel.py --mode cross-dataset
```

P2-only debug remains [`demo_P2.sh`](demo_P2.sh) + `_single.ipynb`.

- HE: `data/Xemium/CRC/HE_images/Xenium_V1_Human_Colon_Cancer_P{1,2,5}_CRC_Add_on_FFPE_he_image.ome.tif` (~0.274 µm/px; UNI `--scale` ≈ 0.548)
- `X_pix_HE` / `Y_pix_HE` = paper `x_centroid_visium_scale` / `y_centroid_visium_scale`
- StarDist present for P1 and P2; P5 CSV may still be missing
- P1/P5 UNI does **not** write `sc_pth_16_16_image` (`--no_save_patch_images`). P2 keeps PNG dumps.
- Labels remain **transferred RCTD singlet**, not Xenium GT
- UNI / train are long GPU jobs; pin a GPU and do not launch them accidentally

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

## 2026.09.14 — transferred Xenium annotations (P2 first)

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
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC
python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --refresh-hierarchy
```

