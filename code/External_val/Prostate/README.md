# Prostate external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07. Erickson et al., *Nature* 2022 (Mendeley `svw96g68dv`). JPEG HE and StarDist nuclei, no 10x `tissue_positions` JSON in this folder and no cell-type ground truth. Same checkpoint and steps as BRCA / NPC, via `Hist2Pheno_pkg.external_validation.ExternalCohort`. `prostate_external.py` only adapts the on-disk layout and the pixel size.

Two patients. Section names repeat (`H1_2` is in both), so the sample id is `Patient_1__H1_2`.

| Patient | Sections on disk | JPEG size | µm/px |
|---------|------------------|-----------|-------|
| `Patient_1` | 7 (`H1_2`, `H1_4`, `H1_5`, `H2_1`, `H2_2`, `H2_5`, `V1_2`) | 13184 or ~15821 | 0.573–0.688 |
| `Patient_2` | 28 | 24000 × 21221 / 23242 / 24258 | 0.341–0.349 |

StarDist tables: `data/Visium/Prostate/{Patient}/StarDist_Segment/{section}/{section}_Float_prob0.01_nms_0.3.csv` with `centroid_x`, `centroid_y`, `probability`. Those centroids are pixels on the matching JPEG.

## Pixel size: HEST spot spacing, not 72 dpi

Every JPEG has EXIF `XResolution = 72` dpi. That is a Photoshop placeholder and is not used.

These slides have no OME `PhysicalSize`. µm/px is `pixel_size_um_estimated` from [HEST-1k](https://github.com/mahmoodlab/HEST) (`HEST_v1_1_0.csv`, ids MEND139–MEND162), estimated from Visium spot coordinates at 100 µm centre-to-centre on these same pixel dimensions. `prostate_external.py` stores the per-slide value. Sections that HEST did not list (Patient 2 `H1_*`, `H2_3`–`H2_6`, `H3_3`, `V2_3`–`V2_6`) use the median of the HEST slides with the same width and height. `dry-run` prints which source each slide used.

A native 16 px crop is about 11 µm on Patient 1 and about 5.5 µm on Patient 2. Training crops were **8 µm** (16 px at 0.5 µm/px).

## Crop used for UNI

Rescale each JPEG and its StarDist centroids by `mpp / 0.5`, then take `patch_size=16`. Patient 1 is enlarged (`scale ≈ 1.15–1.38`). Patient 2 is shrunk (`scale ≈ 0.68–0.70`). The physical field is **8 µm**. JPEG resize stays in uint8 (`Image_feature_extraction.py`); a float32 rescale of a 24k RGB slide is several GB.

## Checkpoint

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root. There are 35 sections and on the order of several million nuclei, so run one section or one patient inside `screen`. Do not launch all 35 in one `all` unless that is intentional.

```bash
bash code/External_val/Prostate/demo.sh dry-run

# one section
SAMPLE=Patient_1__H1_2 bash code/External_val/Prostate/demo.sh features
SAMPLE=Patient_1__H1_2 bash code/External_val/Prostate/demo.sh h5ad
SAMPLE=Patient_1__H1_2 bash code/External_val/Prostate/demo.sh predict
SAMPLE=Patient_1__H1_2 bash code/External_val/Prostate/demo.sh plot

# one patient (7 slides)
SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh all

# the other patient
SAMPLE=Patient_2 bash code/External_val/Prostate/demo.sh all
```

Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0`.

Screen output is appended to `code/External_val/Prostate/Prostate.log`. Each step and the whole run record elapsed time there.

```bash
screen -S prostate_p1
conda activate SeededNTM
cd /home/lingyu/ssd2/Python/Hist2Pheno
SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh all
# detach: Ctrl-a d
```

## Outputs

| Path | Role |
|------|------|
| `data/Visium/Prostate/project_all_UNI/{Patient_1__H1_2}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/Prostate/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/Prostate/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
| `data/Visium/Prostate/pred_pancancer_section/composition.csv` | class fractions for the samples included in that `plot` |
| `data/Visium/Prostate/pred_pancancer_section/figures/` | spatial panels and composition bars |

`Pred_visual_prostate.ipynb` loads those figures and the TLS / SRI / TNI niche indices (each index is a 1×3 row). There is no ROC: these slides have no ground-truth cell type. Niche radius 55 µm is converted with that slide's own µm/px, not 0.5.
