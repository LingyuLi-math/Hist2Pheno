# Prostate external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07. Erickson et al., *Nature* 2022 (Mendeley `svw96g68dv`). JPEG HE and StarDist nuclei, no cell-type ground truth. Patient_1 now has Space Ranger `tissue_positions_list.csv` under `Visium_with_annotation/`. Same checkpoint and steps as BRCA / NPC, via `Hist2Pheno_pkg.external_validation.ExternalCohort`. `prostate_external.py` only adapts the on-disk layout and the pixel size.

Two patients. Section names repeat (`H1_2` is in both), so the sample id is `Patient_1__H1_2`.

| Patient | Sections on disk | JPEG size | µm/px |
|---------|------------------|-----------|-------|
| `Patient_1` | 7 (`H1_2`, `H1_4`, `H1_5`, `H2_1`, `H2_2`, `H2_5`, `V1_2`) | 13184 or ~15821 | 0.573–0.688 |
| `Patient_2` | 28 | 24000 × 21221 / 23242 / 24258 | 0.341–0.349 |

StarDist tables: `data/Visium/Prostate/{Patient}/StarDist_Segment/{section}/{section}_Float_prob0.01_nms_0.3.csv` with `centroid_x`, `centroid_y`, `probability`. Those centroids are pixels on the matching JPEG.

## Pixel size: per-section Visium scalefactors, not BRCA_ST pitch

Every JPEG has EXIF `XResolution = 72` dpi. That is a Photoshop placeholder and is not used. The HE has no OME `PhysicalSize` (NPC TIFFs do, so NPC reads `PhysicalSizeX` per slide).

This cohort is **Visium**. Each section that has `Visium_with_annotation/{section}/scalefactors_json.json` gets its own µm/px, the same way NPC gets a factor per TIFF:

```text
µm/px = 65 / spot_diameter_fullres
```

Space Ranger documents `spot_diameter_fullres` as pixels across a theoretical **65 µm** spot. NPC patient 1 is `65 / 139.45 ≈ 0.466`, which matches that slide's OME `0.465`. Do **not** use HER2ST / BRCA_ST `200 µm / array pitch`, and do not use `55 / diameter` (that would disagree with NPC OME).

Patient_2 still uses the HEST table until its `scalefactors_json.json` files are on disk. `dry-run` prints which source each slide used.

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

# one patient (7 slides)
SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh all

# the other patient (or one series: SAMPLE=Patient_2__H1)
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
`Pred_visual_prostate_patient1_{H1,H2,V1}.ipynb` and `Pred_visual_prostate_patient2_{H1,H2,H3,V1,V2}.ipynb` draw the spatial maps and the TLS / SRI / TNI indices. Each patient is one notebook per series so the file stays small. `SAVE_FIGURES = False` only displays them. There is no ROC: these slides have no ground-truth cell type. Niche radius 55 µm is converted with that slide's own µm/px, not 0.5.

## UNI2 paired_v2 best weights

A2 is fold 1 (`result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/.../best_mlp_gpu.pt`). A3 is fold 0 (`result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/.../best_mlp_gpu.pt`). `demo_uni2.sh` forwards to A2. Predictions go to `data/Visium/Prostate/pred_uni2_A2_bestmlp_paired_v2_seed42/` and `pred_uni2_A3_bestmlp_paired_v2_seed42/`. The existing `*_uni2.ipynb` files and `*_uni2_A3.ipynb` read A3; `*_uni2_A2.ipynb` reads A2.

2026-11-15. A6 is fold 1 of the A2 trunk with L12 links = 2 (`result_uni2_A6_l12_links_2_paired_v2_seed42/cross_dataset_cv/A6_l12_links_2/best_mlp_gpu.pt`, pooled OOF L2 macro AUROC 0.8882). `Pred_visual_prostate_patient1_H2_uni2_A6.ipynb` and `demo_uni2_A6.sh` write `pred_uni2_A6_bestmlp_paired_v2_seed42/`.
