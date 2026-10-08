# NPC external validation (pan-cancer cross-section Hist2Pheno)

2026.10.06. Two HE slides, StarDist nuclei, no 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. The shared steps live in `code/Hist2Pheno_pkg/external_validation.py` (`_SIMPLE_COHORTS["NPC"]`).

| Sample | HE | StarDist nuclei |
|--------|----|-----------------|
| `20210809-C-AH4199551` | `data/Visium/NPC/HE_images/20210809-C-AH4199551.tif` (17319 × 17351) | 142,184 |
| `20210809-D-AJ3037946` | `data/Visium/NPC/HE_images/20210809-D-AJ3037946.tif` (17290 × 17349) | 120,888 |

StarDist tables: `data/Visium/NPC/StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv` with columns `centroid_x`, `centroid_y`, `probability`. Those centroids are full-resolution pixels on the same TIFF.

## Pixel size: 0.465 µm/px, not 96 dpi

Read with `tifffile` from each page's `ImageDescription` (OME-XML). Both files contain:

```text
PhysicalSizeX="0.46507936507936531" PhysicalSizeXUnit="µm"
PhysicalSizeY="0.4650793650793652"  PhysicalSizeYUnit="µm"
```

X and Y differ only in the last digit, so both slides are **0.465 µm/px**. `external_validation.py` parses `PhysicalSizeX` / `PhysicalSizeY` per TIFF (`ome_um_per_px`); it does not hard-code the number.

The same IFD also has TIFF tags `XResolution = 96`, `YResolution = 96`, `ResolutionUnit = inch`. That is a 96 dpi placeholder (~264 µm/px) and does not match a ~17k slide. It is not used.

A native 16 px crop on this grid is `16 × 0.465 = 7.44 µm`. Pan-cancer UNI crops were **8 µm** (16 px at 0.5 µm/px: CODEX-style `scale_image=False`, or Xenium rescaled with `0.2125/0.5`). Using 16 native pixels here would feed a systematically smaller field.

## Crop used for UNI

Rescale the HE and the StarDist centroids by `0.465 / 0.5 ≈ 0.930` so the working grid is **0.5 µm/px**, then take `patch_size=16`. The physical field is **8 µm**, the same as training. No Visium spot diameter is involved.

`Image_feature_extraction.py` treats these `*.tif` files as OME (the XML sits in `ImageDescription` even though the suffix is not `.ome.tif`) and resizes in uint8. A float32 rescale of a ~17k RGB slide would need several GB.

Coordinates stay the original CSV pixels in the `.pth` filename (`{x_scaled}/scale`), so the later h5ad match uses `centroid_x` / `centroid_y` at 1 px tolerance. `obsm['spatial_HE']` is that same native pixel grid. Neighbor identity for k=8 is unchanged by a uniform scale.

## Checkpoint

Cross-section weights, not the cell-stratified run:

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial context k=8, same as training. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root.

```bash
bash code/External_val/NPC/demo.sh dry-run    # mpp, scale, nucleus counts
bash code/External_val/NPC/demo.sh features   # UNI via code/Image_feature_extraction.py
bash code/External_val/NPC/demo.sh h5ad
bash code/External_val/NPC/demo.sh predict    # h5ad + l2_probs.npz
# or
bash code/External_val/NPC/demo.sh all        # features, h5ad, predict
```

One slide: `SAMPLE=20210809-C-AH4199551`. Redo a finished step: `FORCE=1`. GPU: `CUDA_DEVICE=0`.

Screen output is appended to `code/External_val/NPC/NPC.log`. Each step and the whole run record elapsed time there (`00:26:14 (1574s)`).

## Outputs

| Path | Role |
|------|------|
| `data/Visium/NPC/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/NPC/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/NPC/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
`Pred_visual_npc.ipynb` draws the spatial maps, composition bars, and TLS / SRI / TNI indices. `SAVE_FIGURES = False` only displays them. There is no ROC: these slides have no ground-truth cell type.
