# HCC external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07. Seven HE JPEGs, StarDist nuclei, no 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. The shared steps live in `code/Hist2Pheno_pkg/external_validation.py` (`_SIMPLE_COHORTS["HCC"]`). `MPP_UM` is still required.

| Sample | HE | pixels (W × H) | StarDist nuclei |
|--------|----|----------------|-----------------|
| `P1_N` | `data/Visium/HCC/HE_images/P1_N.jpg` | 16512 × 16736 | 134,265 |
| `P1_T` | `data/Visium/HCC/HE_images/P1_T.jpg` | 16768 × 17056 | 176,941 |
| `P2_N` | `data/Visium/HCC/HE_images/P2_N.jpg` | 16928 × 17056 | 252,640 |
| `P3_T` | `data/Visium/HCC/HE_images/P3_T.jpg` | 16512 × 17184 | 177,882 |
| `P5_N` | `data/Visium/HCC/HE_images/P5_N.jpg` | 16736 × 17120 | 178,777 |
| `P5_T` | `data/Visium/HCC/HE_images/P5_T.jpg` | 16736 × 16736 | 164,263 |
| `P7_T` | `data/Visium/HCC/HE_images/P7_T.jpg` | 15904 × 16320 | 257,883 |

StarDist tables: `data/Visium/HCC/StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv` with columns `centroid_x`, `centroid_y`, `probability`. Those centroids are full-resolution pixels on the same JPEG.

## Pixel size is not in the file

These are JPEG, not OME-TIFF, so there is no `PhysicalSizeX` / `PhysicalSizeY`. Several files carry EXIF/JFIF `96 dpi` written by Microsoft Windows Photo Viewer. That is the same kind of placeholder as the NPC TIFF `XResolution=96` tag, and it is not used.

Set the scanner calibration yourself:

```bash
export MPP_UM=0.5   # replace 0.5 with the real µm/px
```

`0.5` is only an example of the training grid. It is not a measurement of these slides. `dry-run` prints `scale = MPP_UM / 0.5`. A native 16 px crop is `16 × MPP_UM` µm. Pan-cancer UNI crops were **8 µm** (16 px at 0.5 µm/px), so the image and the StarDist centroids are rescaled by that factor before `patch_size=16`.

JPEG resize stays uint8 (PIL). A float32 rescale of a ~16k RGB slide would need several GB.

## Checkpoint

Cross-section weights, not the cell-stratified run:

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial context k=8, same as training. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

From the repo root, in `screen`. `features` is the long step (about 1.34 million nuclei, UNI one crop at a time). `predict` also writes `{sample}_l2_probs.npz`.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
export MPP_UM=0.5          # scanner µm/px; required
bash code/External_val/HCC/demo.sh dry-run
bash code/External_val/HCC/demo.sh all
```

One slide: `SAMPLE=P1_T`. Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=2` (the default in `demo.sh`).

## Outputs

| Path | Role |
|------|------|
| `data/Visium/HCC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/HCC/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/HCC/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
`Pred_visual_hcc.ipynb` draws the spatial maps, composition bars, and TLS / SRI / TNI indices. `SAVE_FIGURES = False` only displays them. It uses `MPP_UM = 0.5`, the value in `HCC.log`. There is no ROC: these slides have no ground-truth cell type.
