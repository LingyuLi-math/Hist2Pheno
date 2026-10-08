# External validation (new HE + StarDist, no cell-type GT)

Apply the pan-cancer **cross-section** checkpoint to a cancer that was not in training. Each cohort has HE and StarDist nuclei only: no 10x `tissue_positions` / `scalefactors` JSON and no ground-truth cell type. There is no ROC. The readout is L1 / L12 / L2 plus TLS / SRI / TNI.

Shared implementation: [`code/Hist2Pheno_pkg/external_validation.py`](../Hist2Pheno_pkg/external_validation.py) (`ExternalCohort`). A cohort folder only sets the data root and calls `main`. Notebooks import `ExternalCohort` the same way.

Checkpoint (section CV, not cell-stratified):

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`. Hidden dims `(768, 384, 192)`.

## Pixel size is read in code

Training UNI crops are **8 µm**: 16 px on a **0.5 µm/px** grid. `ExternalCohort.geometry` computes the scale that puts the new slide on that grid, then `patch_size=16`.

For OME TIFF / BigTIFF (NPC, BRCA, CRC), `ome_um_per_px` opens the file with `tifffile` and parses `PhysicalSizeX` / `PhysicalSizeY` from the first page `ImageDescription`. The numbers in the NPC README (0.465 µm/px), the BRCA README (0.548 µm/px), and the CRC README (0.274 µm/px) are what that function returns for those slides. They are not constants in the cohort scripts. Those scripts do not pass `mpp_um`, so every `dry-run` / `features` call reads the file again. CRC sets `he_suffix=".btf"`. TIFF `XResolution = 96` dpi is not used.

For JPEG (HCC) there is no OME block. Pass the scanner calibration as `MPP_UM`. `geometry` records `mpp_source` as `OME PhysicalSize` or `cohort mpp_um`.

`Image_feature_extraction.py` resizes OME TIFF and JPEG/PNG in uint8. StarDist `centroid_x` / `centroid_y` are native full-resolution pixels. The `.pth` filename stores the scaled coordinate divided by `scale`, so the later h5ad match uses the original CSV coordinates (1 px tolerance). `obsm['spatial_HE']` stays on that native grid, so niche radius 55 µm is converted with the same µm/px.

## Steps

From the repo root, `conda activate SeededNTM` (or `Hist2Pheno`).

```bash
bash code/External_val/<COHORT>/demo.sh dry-run    # mpp, scale, nucleus counts
bash code/External_val/<COHORT>/demo.sh features   # UNI
bash code/External_val/<COHORT>/demo.sh h5ad
bash code/External_val/<COHORT>/demo.sh predict    # also writes {sample}_l2_probs.npz
bash code/External_val/<COHORT>/demo.sh plot
# or
bash code/External_val/<COHORT>/demo.sh all        # features, h5ad, predict, plot
```

`SAMPLE=<id>` limits one slide. `FORCE=1` redoes a finished step. `features` is the long step (UNI, one 16×16 crop per nucleus).

## Cohorts

| Cohort | Data | HE | Slides | Nuclei | µm/px | GPU default |
|--------|------|----|--------|--------|-------|-------------|
| NPC | `data/Visium/NPC` | OME TIFF | 2 | 142,184 + 120,888 | read by `ome_um_per_px` (~0.465) | `CUDA_DEVICE` default 0 |
| BRCA | `data/Visium/BRCA` | OME TIFF | 1 (`GSM7782699_tissue_image`) | 281,051 | read by `ome_um_per_px` (~0.548) | `CUDA_DEVICE` default 0 |
| HCC | `data/Visium/HCC` | JPEG | 7 (`P1_N` `P1_T` `P2_N` `P3_T` `P5_N` `P5_T` `P7_T`) | ~1.34M | `MPP_UM` (required; 96 dpi is not a pixel size) | `CUDA_DEVICE` default **2** |
| CRC | `data/Visium/CRC` | OME BigTIFF | 1 (`Visium_V2_Human_Colon_Cancer_P2_tissue_image`) | 579,587 | read by `ome_um_per_px` (~0.274) | `CUDA_DEVICE` default 0 |
| BRCA_ST | `data/ST/BRCA` | JPEG | 36 HER2ST (A1–H3) | 2,417,153 | 200 µm spot pitch (~0.92 on A, ~0.69 on B–H) | `CUDA_DEVICE` default 0 |

Predicted NPC, BRCA, and HCC outputs are under `data/Visium/{NPC,BRCA,HCC}/pred_pancancer_section/`. CRC and BRCA_ST are wired the same way and have not been predicted yet.

Per-cohort notes: [`NPC/README.md`](NPC/README.md), [`BRCA/README.md`](BRCA/README.md), [`HCC/README.md`](HCC/README.md), [`CRC/README.md`](CRC/README.md), [`BRCA_ST/README.md`](BRCA_ST/README.md).

## After prediction

`Pred_visual_*.ipynb` loads the spatial L1/L12/L2 figures and builds TLS / SRI / TNI from L2 softmax (`ensure_l2_probs`), not from the argmax label. Radius is 55 µm on native `spatial_HE` pixels. Each index is a 1×3 row: abundance, spatial score, spatial score with the within-slide 95th-percentile candidates. Lung ARI / FRI are not used.

A new cancer with the same folder layout (`HE_images/`, `StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv`) is:

```python
from Hist2Pheno_pkg.external_validation import ExternalCohort

cohort = ExternalCohort(name="NAME", data_root=repo / "data" / "Visium" / "NAME", repo=repo)
cohort.main(["dry-run"])
```

JPEG with one scanner calibration: also pass `he_suffix=".jpg"` and `mpp_um=...`. BigTIFF: pass `he_suffix=".btf"`. HER2ST (`BRCA_ST`) and the prostate cohort set a per-slide µm/px in the cohort script, from spot spacing, because 72 dpi is not a pixel size. Each `demo.sh` appends screen output and per-step plus whole-run elapsed time to `code/External_val/<COHORT>/<COHORT>.log`.
