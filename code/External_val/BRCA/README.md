# BRCA external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07 — GSM OME TIFF.  
2026.10.15 — two SpatialVDJ pathology HE JPEGs (same cohort as Zenodo
`Breast_Cancer_Visium_samples.xlsx` pathology annotations).

Shared steps live in `code/Hist2Pheno_pkg/external_validation.py`. BRCA is **not**
a `_SIMPLE_COHORTS` row anymore: mixed HE suffixes and per-sample µm/px are handled
in [`brca_external.py`](brca_external.py).

| Sample | HE | Visium ID | µm/px source | StarDist nuclei |
|--------|----|-----------|--------------|-----------------|
| `GSM7782699_tissue_image` | `HE_images/….tif` (21571 × 19505) | — | OME PhysicalSize ~**0.548** | 281,051 |
| `200715_BC_SA2_C_D_V19F24-113.B1-Spot000001` | `HE_images/….jpg` (48640 × 47104) | **`V10F24-113_B1`** | Visium × HE÷SR ~**0.172** | ~372k |
| `200219_BC_SA3_V19T26-031_CE_V4-Spot000001` | `HE_images/….jpg` (52736 × 52736) | **`V19T26-031_D1`** | Visium × HE÷SR ~**0.172** | ~391k |

StarDist tables:
`data/Visium/BRCA/StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv`
(`centroid_x`, `centroid_y`, `probability` on the HE used for segmentation).

Raw pathology images (source):
`data/Visium/BRCA/npj_data/SpatialVDJ_forZenodo/data/breast_cancer/11_images/raw/`.

Spaceranger (JSON / positions):
`…/5_spatial_GEX/spaceranger_output/{VisiumID}/outs/spatial/`
(`scalefactors_json.json`, `tissue_positions_list.csv`, …).  
Sample map: `Breast_Cancer_Visium_samples.xlsx` (`Raw image` stem → `Visium ID`).
Note the image string says `V19F24` while the Visium ID is `V10F24-113_B1`.

## Pixel size

### GSM OME TIFF (~0.548 µm/px)

`tifffile` / OME-XML `PhysicalSizeX/Y`. TIFF 96 dpi is ignored. Native 16 px ≈ 8.76 µm;
rescale by `0.548/0.5 ≈ 1.095` so UNI crops are **8 µm**.

### Zenodo pathology JPEGs (~0.172 µm/px)

These HE files are **~3.333×** Space Ranger fullres (from
`tissue_hires` size / `tissue_hires_scalef`). On SR fullres,
`65 / spot_diameter_fullres ≈ 0.573 µm/px`. On the StarDist HE:

```text
µm/px_HE = (65 / spot_diameter_fullres) / (HE_width / SR_fullres_width) ≈ 0.172
```

Do **not** apply Prostate’s raw `65 / spot_diameter` to these JPEGs. JPEG EXIF dpi
is not used. UNI then rescales by `0.172/0.5 ≈ 0.344` (shrink) to the 0.5 µm/px
training grid; `patch_size=16` → **8 µm**.

For spot mapping later, scale `tissue_positions` by the same HE÷SR factor (or
downscale nuclei); StarDist centroids are on the large HE, not SR fullres.

## Checkpoint

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads:
L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

`conda activate SeededNTM`, from the repo root.

```bash
bash code/External_val/BRCA/demo.sh dry-run
bash code/External_val/BRCA/demo.sh features
bash code/External_val/BRCA/demo.sh h5ad
bash code/External_val/BRCA/demo.sh predict
# or
bash code/External_val/BRCA/demo.sh all
```

One slide:

```bash
SAMPLE=GSM7782699_tissue_image bash code/External_val/BRCA/demo.sh all
SAMPLE=200715_BC_SA2_C_D_V19F24-113.B1-Spot000001 bash code/External_val/BRCA/demo.sh all
SAMPLE=200219_BC_SA3_V19T26-031_CE_V4-Spot000001 bash code/External_val/BRCA/demo.sh all
```

`FORCE=1` redo. `CUDA_DEVICE=0`. Log: `code/External_val/BRCA/BRCA.log`.

## Outputs

| Path | Role |
|------|------|
| `data/Visium/BRCA/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | UNI `.pth` per in-bounds nucleus |
| `data/Visium/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/BRCA/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels |

`Pred_visual_brca.ipynb` draws spatial maps, composition, and TLS / SRI / TNI.
Niche radius uses **per-sample** `geometry(sample)["mpp_um"]`. No ROC (no cell-type GT).

## QuPath (large JPEG)

Pathology HE JPEGs are ~48k–52k on a side; QuPath/Java often cannot open them.
Convert to a tiled pyramid BigTIFF (keeps the original `.jpg` for Hist2Pheno):

```bash
conda run --no-capture-output -n SeededNTM python -u \
  code/External_val/BRCA/jpg_to_qupath_pyramid_tif.py
```

Outputs next to the JPEGs: `HE_images/{sample}_qupath.tif`. Open those in QuPath.

## UNI2 paired_v2 best weights

A2 is fold 1. A3 is fold 0. `demo_uni2.sh` forwards to A3.

```bash
CUDA_DEVICE=0 bash code/External_val/BRCA/demo_uni2_A2.sh predict
CUDA_DEVICE=1 bash code/External_val/BRCA/demo_uni2_A3.sh predict
```

Outputs: `data/Visium/BRCA/pred_uni2_A2_bestmlp_paired_v2_seed42/` and `pred_uni2_A3_bestmlp_paired_v2_seed42/`. `Pred_visual_brca_uni2.ipynb` and `Pred_visual_brca_uni2_A3.ipynb` read A3. `Pred_visual_brca_uni2_A2.ipynb` reads A2.
