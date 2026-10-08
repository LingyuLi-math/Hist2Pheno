# CRC external validation (pan-cancer cross-section Hist2Pheno)

2026.10.08. One HE slide, StarDist nuclei, no 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. The shared steps live in `code/Hist2Pheno_pkg/external_validation.py` (`_SIMPLE_COHORTS["CRC"]`, HE suffix `.btf`).

| Sample | HE | StarDist nuclei |
|--------|----|-----------------|
| `Visium_V2_Human_Colon_Cancer_P2_tissue_image` | `data/Visium/CRC/HE_images/Visium_V2_Human_Colon_Cancer_P2_tissue_image.btf` (76190 × 49099, 20×) | 579,587 |

StarDist table: `data/Visium/CRC/StarDist_Segment/Visium_V2_Human_Colon_Cancer_P2_tissue_image/Visium_V2_Human_Colon_Cancer_P2_tissue_image_Float_prob0.01_nms_0.3.csv` with columns `centroid_x`, `centroid_y`, `probability`. Those centroids are full-resolution pixels on the same BigTIFF.

## Pixel size: 0.274 µm/px, not 96 dpi

Read with `tifffile` from the page `ImageDescription` (OME-XML, Olympus VS200, 20×):

```text
PhysicalSizeX="0.27377489065723981" PhysicalSizeXUnit="µm"
PhysicalSizeY="0.27377656059267208"  PhysicalSizeYUnit="µm"
```

X and Y differ only in the sixth decimal, so the slide is **0.274 µm/px**. `external_validation.py` parses `PhysicalSizeX` / `PhysicalSizeY` (`ome_um_per_px`); it does not hard-code the number. No `MPP_UM` is required.

The same IFD also has TIFF tags `XResolution = 96`, `YResolution = 96`, `ResolutionUnit = inch`. That is a 96 dpi placeholder and is not used.

A native 16 px crop on this grid is `16 × 0.274 ≈ 4.38 µm`. Pan-cancer UNI crops were **8 µm** (16 px at 0.5 µm/px). Using 16 native pixels here would feed a smaller field than training.

## Crop used for UNI

Rescale the HE and the StarDist centroids by `0.274 / 0.5 ≈ 0.548` so the working grid is **0.5 µm/px**, then take `patch_size=16`. The physical field is **8 µm**, the same as training. The factor is less than 1: this slide is finer than 0.5 µm/px, so the image is shrunk. `Image_feature_extraction.py` treats `*.btf` as OME and resizes in uint8.

Coordinates stay the original CSV pixels in the `.pth` filename (`{x_scaled}/scale`), so the later h5ad match uses `centroid_x` / `centroid_y` at 1 px tolerance. `obsm['spatial_HE']` is that same native pixel grid. Neighbor identity for k=8 is unchanged by a uniform scale. Niche radius 55 µm is converted with this 0.274 µm/px, not with 0.5.

## Checkpoint

Cross-section weights, not the cell-stratified run:

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial context k=8, same as training. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root. `features` writes one `.pth` per nucleus (~580k files) and is the long step. Run it in `screen`.

```bash
bash code/External_val/CRC/demo.sh dry-run
CUDA_DEVICE=2 bash code/External_val/CRC/demo.sh all
```

One slide: `SAMPLE=Visium_V2_Human_Colon_Cancer_P2_tissue_image`. Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0` by default.

Screen output is appended to `code/External_val/CRC/CRC.log`. Each step and the whole run record elapsed time there.

## Outputs

| Path | Role |
|------|------|
| `data/Visium/CRC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/CRC/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/CRC/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
`Pred_visual_crc.ipynb` draws the spatial maps, composition bars, and TLS / SRI / TNI indices. `SAVE_FIGURES = False` only displays them. There is no ROC: this slide has no ground-truth cell type.
