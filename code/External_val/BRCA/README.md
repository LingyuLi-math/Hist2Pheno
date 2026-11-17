# BRCA external validation (pan-cancer cross-section Hist2Pheno)

2026.10.07 — GSM OME TIFF.  
2026.10.15 — two SpatialVDJ pathology HE JPEGs (same cohort as Zenodo
`Breast_Cancer_Visium_samples.xlsx` pathology annotations).  
2026.11.15 — 10x Human Breast Cancer Block A Section 1 (Space Ranger fullres TIFF).

Shared steps live in `code/Hist2Pheno_pkg/external_validation.py`. BRCA is **not**
a `_SIMPLE_COHORTS` row anymore: mixed HE suffixes and per-sample µm/px are handled
in [`brca_external.py`](brca_external.py).

| Sample | HE | Visium ID | µm/px source | StarDist nuclei |
|--------|----|-----------|--------------|-----------------|
| `GSM7782699_tissue_image` | `HE_images/….tif` (21571 × 19505) | — | OME PhysicalSize ~**0.548** | 281,051 |
| `200715_BC_SA2_C_D_V19F24-113.B1-Spot000001` | `HE_images/….jpg` (48640 × 47104) | **`V10F24-113_B1`** | Visium × HE÷SR ~**0.172** | ~372k |
| `200219_BC_SA3_V19T26-031_CE_V4-Spot000001` | `HE_images/….jpg` (52736 × 52736) | **`V19T26-031_D1`** | Visium × HE÷SR ~**0.172** | ~391k |
| `V1_Breast_Cancer_Block_A_Section_1_image` | `10x_data` fullres TIFF (this file **is** SR fullres) | `V1_Breast_Cancer_Block_A_Section_1` | `65/spot_diameter` ~**0.366** | 423,162 |

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

### 10x fullres TIFF (~0.366 µm/px)

`V1_Breast_Cancer_Block_A_Section_1_image.tif` is the Space Ranger fullres image
under `data/Visium/BRCA/10x_data`. HE÷SR = 1, so µm/px =
`65 / spot_diameter_fullres` ≈ **0.366**. There is no OME `PhysicalSize`.
Spatial files are `10x_data/spatial/`. UNI rescales by `0.366/0.5 ≈ 0.732`.

Downloaded from the
[10x Genomics Human Breast Cancer (Block A Section 1)](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0)
dataset (Space Ranger 1.1.0). `10x_data/metadata.tsv` is the pathologist spot
annotation used by Long et al., *Nat Commun* 2023
([GraphST](https://www.nature.com/articles/s41467-023-36796-3)): `annot_type`
(Healthy, Invasive, Surrounding tumor) and `fine_annot_type` (20 regions).
Those labels are Visium spots, not StarDist nucleus GT.
[`Pred_visual_brca_uni2_A2.ipynb`](Pred_visual_brca_uni2_A2.ipynb) plots this
slide with the other three.

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

A2 is fold 1. A3 is fold 0. `demo_uni2.sh` forwards to A2.

```bash
CUDA_DEVICE=0 bash code/External_val/BRCA/demo_uni2_A2.sh predict
CUDA_DEVICE=1 bash code/External_val/BRCA/demo_uni2_A3.sh predict
```

Outputs: `data/Visium/BRCA/pred_uni2_A2_bestmlp_paired_v2_seed42/` and `pred_uni2_A3_bestmlp_paired_v2_seed42/`. `Pred_visual_brca_uni2.ipynb` and `Pred_visual_brca_uni2_A3.ipynb` read A3. `Pred_visual_brca_uni2_A2.ipynb` reads A2.

2026-11-16. A8 is fold 0 of the 1,066-section run (`result_uni2_A8_l12_balanced_paired_v2_seed42/cross_dataset_cv/A8_l12_balanced/best_mlp_gpu.pt`, k=8 mean with gate, balanced L12, pooled OOF L2 macro AUROC 0.9002). `Pred_visual_brca_uni2_A8.ipynb` and `demo_uni2_A8.sh` write `pred_uni2_A8_bestmlp_paired_v2_seed42/`.

```bash
CUDA_DEVICE=0 bash code/External_val/BRCA/demo_uni2_A8.sh predict
```

2026-11-17. A7 is fold 4 of the 1,066-section bf16 run (`result_uni2_A2_l12_balanced_new1066_paired_v2_seed42/cross_dataset_cv/A2_l12_balanced/best_mlp_gpu.pt`, k=8 mean, no gate, balanced L12, pooled OOF L2 macro AUROC 0.9035). `Pred_visual_brca_uni2_A7.ipynb` and `demo_uni2_A7.sh` write `pred_uni2_A7_new1066_paired_v2_seed42/`.

```bash
bash code/External_val/BRCA/demo_uni2_A7.sh predict
```

2026-11-17. The finished ablation weights, named by arm rather than A2–A8, are applied to the 10x slide only (`V1_Breast_Cancer_Block_A_Section_1_image`). [`Pred_visual_brca_10x_axisfix.ipynb`](Pred_visual_brca_10x_axisfix.ipynb) draws one L2 map per arm, including `l12_links_2`, `l12_balanced`, and `focal_gamma1`. `gate_l12_balanced` is not included; that run is still training. Predictions go to `pred_uni2_axisfix_<arm>/`. The notebook does not start a GPU job.

```bash
CUDA_DEVICE=2 bash code/External_val/BRCA/demo_uni2_axisfix.sh predict
```

2026.11.17. Score the arms against the pathologist spots (55 µm disc, four label modes: argmax, decision rule, EM prior shift, kNN smoothing). It also writes `{SAMPLE}_l2_label_modes.csv` next to each prediction:

```bash
python code/External_val/BRCA/brca_10x_annotation_eval.py
python code/External_val/BRCA/brca_10x_annotation_eval.py --arms spatial_k8 l12_links_1 --radius-um 65
```

`STAR_PROB_MIN` other than 0 now writes `pred_uni2_axisfix_<arm>_probX.XX/`; set `PRED_TAG` in the notebook and `--pred-tag` for the script to match.
