# BRCA ST (HER2ST) external validation

2026.10.08. Andersson et al., *Nature Communications* 2021. Thirty-six HE JPEGs and StarDist nuclei, no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. The shared steps live in `code/Hist2Pheno_pkg/external_validation.py`. `brca_st_external.py` points that module at `data/ST/BRCA` and sets a per-slide µm/px.

| Patient | HER2ST | HE stems | Slides | µm/px |
|---------|--------|----------|--------|-------|
| A (9769) | A1–A6 | `9769_*_HE_small` | 6 | 0.914–0.922 |
| B (10737) | B1–B6 | `BT10737_*_HE` | 6 | 0.689–0.691 |
| C (5714) | C1–C6 | `5714_HE_BT_*` | 6 | 0.683–0.690 |
| D (10426) | D1–D6 | `10426_HE_BT_*` | 6 | 0.686–0.688 |
| E (23567) | E1–E3 | `HE_BT23567_*` | 3 | 0.687–0.688 |
| F (23810) | F1–F3 | `HE_BT23810_*` | 3 | 0.688–0.689 |
| G (23903) | G1–G3 | `HE_BT23903_*` | 3 | 0.687–0.689 |
| H (24044) | H1–H3 | `HE_BT24044_*` | 3 | 0.687 |

StarDist tables: `data/ST/BRCA/StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv` (`centroid_x`, `centroid_y`, `probability`). **2,417,153** nuclei in total (about 27k–183k per slide).

## Pixel size: 200 µm spot pitch, not 72 dpi

These JPEGs have no OME `PhysicalSize`. EXIF is 72 dpi and is not used.

HER2ST is the original Spatial Transcriptomics array: spot diameter 100 µm, centre-to-centre **200 µm**. `her2st/data/ST-spotfiles/{A1..H3}_selection.tsv` stores `pixel_x` / `pixel_y` on the same JPEG that StarDist used (the coordinates sit inside the image). µm/px is `200 / median pixel distance` between spots that differ by one array step. The per-slide value is stored in `brca_st_external.py`. Patient A images are the `*_small` JPEGs and come out coarser (~0.92 µm/px). The other patients are ~0.69 µm/px.

A native 16 px crop is about **14.7 µm** on patient A and about **11.0 µm** on B–H. Training crops were **8 µm** (16 px at 0.5 µm/px).

## Crop used for UNI

Rescale each JPEG and its StarDist centroids by `mpp / 0.5` (about 1.83 on patient A, about 1.38 on B–H), then take `patch_size=16`. The physical field is **8 µm**. JPEG resize stays in uint8.

Coordinates stay the original CSV pixels in the `.pth` filename. `obsm['spatial_HE']` is that native grid. Niche radius 55 µm uses that slide's own µm/px.

## Checkpoint

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root. `features` writes one `.pth` per nucleus. Run **one patient or one slide** in `screen`. All 36 slides are about 2.4 million nuclei.

```bash
bash code/External_val/BRCA_ST/demo.sh dry-run
CUDA_DEVICE=2 SAMPLE=A bash code/External_val/BRCA_ST/demo.sh all
CUDA_DEVICE=2 SAMPLE=A1 bash code/External_val/BRCA_ST/demo.sh all
```

`SAMPLE` is a HER2ST id (`A1`), a patient letter (`A`–`H`), or the HE stem (`9769_C1_HE_small`). Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0` by default.

Screen output is appended to `code/External_val/BRCA_ST/BRCA_ST.log`. Each step and the whole run record elapsed time there. Figures are drawn in the patient notebooks.

## Outputs

| Path | Role |
|------|------|
| `data/ST/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/ST/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/ST/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
`Pred_visual_brca_st_A.ipynb` through `Pred_visual_brca_st_H.ipynb` draw the spatial maps and the TLS / SRI / TNI indices, one HER2ST patient per notebook. `SAVE_FIGURES = False` only displays them. There is no ROC: these slides have no ground-truth cell type.

## UNI2 paired_v2 best weights

A2 is fold 1. A3 is fold 0. `demo_uni2.sh` forwards to A3. Outputs are `data/ST/BRCA/pred_uni2_A2_bestmlp_paired_v2_seed42/` and `pred_uni2_A3_bestmlp_paired_v2_seed42/`. `Pred_visual_brca_st_{A..H}_uni2.ipynb` and `_uni2_A3.ipynb` read A3. The `_uni2_A2.ipynb` copies read A2.
