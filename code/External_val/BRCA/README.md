# BRCA external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07. One HE slide, StarDist nuclei, no 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. Same pipeline as `code/External_val/NPC`.

| Sample | HE | StarDist nuclei |
|--------|----|-----------------|
| `GSM7782699_tissue_image` | `data/Visium/BRCA/HE_images/GSM7782699_tissue_image.tif` (21571 × 19505) | 281,051 |

StarDist table: `data/Visium/BRCA/StarDist_Segment/GSM7782699_tissue_image/GSM7782699_tissue_image_Float_prob0.01_nms_0.3.csv` with columns `centroid_x`, `centroid_y`, `probability`. Those centroids are full-resolution pixels on the same TIFF.

## Pixel size: 0.548 µm/px, not 96 dpi

Read with `tifffile` from the page `ImageDescription` (OME-XML):

```text
PhysicalSizeX="0.54758599602876212" PhysicalSizeXUnit="µm"
PhysicalSizeY="0.54760935218579332"  PhysicalSizeYUnit="µm"
```

X and Y differ only in the fifth decimal, so the slide is **0.548 µm/px**. `brca_external.py` parses `PhysicalSizeX` / `PhysicalSizeY` per TIFF (`ome_um_per_px`); it does not hard-code the number.

The same IFD also has TIFF tags `XResolution = 96`, `YResolution = 96`, `ResolutionUnit = inch`. That is a 96 dpi placeholder and is not used.

A native 16 px crop on this grid is `16 × 0.548 ≈ 8.76 µm`. Pan-cancer UNI crops were **8 µm** (16 px at 0.5 µm/px). Using 16 native pixels here would feed a slightly larger field than training.

## Crop used for UNI

Rescale the HE and the StarDist centroids by `0.548 / 0.5 ≈ 1.095` so the working grid is **0.5 µm/px**, then take `patch_size=16`. The physical field is **8 µm**, the same as training. The factor is greater than 1: this slide is coarser than 0.5 µm/px, so the image is enlarged rather than shrunk. No Visium spot diameter is involved.

`Image_feature_extraction.py` treats this `*.tif` as OME (the XML sits in `ImageDescription` even though the suffix is not `.ome.tif`) and resizes in uint8.

Coordinates stay the original CSV pixels in the `.pth` filename (`{x_scaled}/scale`), so the later h5ad match uses `centroid_x` / `centroid_y` at 1 px tolerance. `obsm['spatial_HE']` is that same native pixel grid. Neighbor identity for k=8 is unchanged by a uniform scale.

## Checkpoint

Cross-section weights, not the cell-stratified run:

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial context k=8, same as training. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root.

```bash
bash code/External_val/BRCA/demo.sh dry-run    # mpp, scale, nucleus counts
bash code/External_val/BRCA/demo.sh features   # UNI via code/Image_feature_extraction.py
bash code/External_val/BRCA/demo.sh h5ad
bash code/External_val/BRCA/demo.sh predict
bash code/External_val/BRCA/demo.sh plot
# or
bash code/External_val/BRCA/demo.sh all
```

One slide: `SAMPLE=GSM7782699_tissue_image`. Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0`.

## Outputs

| Path | Role |
|------|------|
| `data/Visium/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
| `data/Visium/BRCA/pred_pancancer_section/composition.csv` | class fractions |
| `data/Visium/BRCA/pred_pancancer_section/figures/` | spatial panels and composition bars |

`Pred_visual_brca.ipynb` loads those figures and the TLS / SRI / TNI niche indices. There is no ROC: this slide has no ground-truth cell type.
