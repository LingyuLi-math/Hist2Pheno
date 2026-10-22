# HCC Nature external validation (pan-cancer cross-section Hist2Pheno)

2026.10.21. Seventeen H&E TIFFs + StarDist nuclei from `data/Visium/HCC/nature_data`. No 10x `tissue_positions` / `scalefactors` JSON and no cell-type ground truth. Apply the pan-cancer **cross-section** checkpoint and read out L1 / L12 / L2. Shared steps live in `code/Hist2Pheno_pkg/external_validation.py` (`_SIMPLE_COHORTS["HCC_NATURE"]`).

Slides are the H&E set from Jia *et al.*, *Nature* (2025) [Spatial immune scoring system predicts hepatocellular carcinoma recurrence](https://www.nature.com/articles/s41586-025-08668-x) (TIMES). Nature `/figures/6` is **Extended Data Fig. 1**. Panel **a** marks adjacent stroma (AS), invasive front (IF), and tumour centre (TC) with H&E / GPC3 / Reticulin (example: `L-8012_HE.ome`). High-resolution extras: [OSF](https://osf.io/db3uf/). This folder has **H&E + StarDist only** — no IHC, no Visium GT, no TIMES genes, no REC / non-REC labels.

This is **not** [`External_val/HCC`](../HCC/README.md) (seven Visium JPEGs under `data/Visium/HCC/`). Outputs stay under `nature_data/` so the two HCC tracks do not overwrite each other.

StarDist tables: `StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv` with `centroid_x`, `centroid_y`, `probability`. Sample ids are the StarDist folder names: plain TIFFs are `L-8001_HE`, OME TIFFs are `L-8006_HE.ome` (HE file `{sample}.tif`).

| Sample | HE | pixels (H × W) | StarDist nuclei | µm/px source |
|--------|----|----------------|-----------------|--------------|
| `L-8001_HE` | `L-8001_HE.tif` | 8112 × 8175 | 78,152 | Aperio `MPP=1.0058` (10× page; `OriginalMPP=0.5029`) |
| `L-8002_HE` | `L-8002_HE.tif` | 8291 × 8287 (RGBA) | 158,054 | no calibration → fallback **0.5** |
| `L-8003_HE` | `L-8003_HE.tif` | 14824 × 17420 | 372,075 | fallback **0.5** |
| `L-8004_HE` | `L-8004_HE.tif` | 8291 × 8287 (RGBA) | 92,723 | fallback **0.5** |
| `L-8005_HE` | `L-8005_HE.tif` | 8291 × 8282 (RGBA) | 111,552 | fallback **0.5** |
| `L-8006_HE.ome` | `L-8006_HE.ome.tif` | 15546 × 15830 | 177,405 | OME **0.5005** |
| `L-8007_HE` | `L-8007_HE.tif` | 5017 × 5014 (RGBA) | 10,349 | fallback **0.5** |
| `L-8008_HE.ome` | `L-8008_HE.ome.tif` | 15628 × 15272 | 324,976 | OME **0.5005** |
| `L-8009_HE.ome` | `L-8009_HE.ome.tif` | 15576 × 15412 | 313,262 | OME **0.5005** |
| `L-8010_HE.ome` | `L-8010_HE.ome.tif` | 15595 × 15660 | 371,645 | OME **0.5005** |
| `L-8011_HE.ome` | `L-8011_HE.ome.tif` | 15518 × 16908 | 267,587 | OME **0.5005** |
| `L-8012_HE.ome` | `L-8012_HE.ome.tif` | 15610 × 15384 | 263,433 | OME **0.5005** |
| `L-8013_HE.ome` | `L-8013_HE.ome.tif` | 15628 × 15340 | 268,287 | OME **0.5005** |
| `L-8014_HE.ome` | `L-8014_HE.ome.tif` | 15714 × 15908 | 267,970 | OME **0.5005** |
| `L-8015_HE.ome` | `L-8015_HE.ome.tif` | 7762 × 7788 | 162,228 | OME 84.67 is 300 dpi; fallback **0.5** |
| `L-8016_HE.ome` | `L-8016_HE.ome.tif` | 15743 × 16491 | 167,225 | OME **0.5005** |
| `L-8017_1_HE.ome` | `L-8017_1_HE.ome.tif` | 15628 × 15627 | 274,469 | OME **0.5005** |

Total: **3,681,392** StarDist nuclei. 96 dpi / 300 dpi tags are not used. `MPP_UM` is optional and only fills slides with no plausible OME / Aperio value.

A native 16 px crop on the OME 0.5005 grid is already ~8.01 µm (`scale ≈ 1.001`). L-8001 is rescaled by `1.0058 / 0.5 ≈ 2.01` so the UNI crop is still 8 µm.

## Checkpoint

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`.

## Commands

From the repo root, in `screen`. `features` is the long step (~3.68 million nuclei).

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
bash code/External_val/HCC_nature/demo.sh dry-run
CUDA_DEVICE=0 bash code/External_val/HCC_nature/demo.sh all
```

If `demo.sh all` is still extracting UNI, predict the slides that already have embeddings (do **not** reuse GPU 0):

```bash
CUDA_DEVICE=1 bash code/External_val/HCC_nature/predict_ready.sh
```

That runs `h5ad` then `predict` with the pan-cancer `best_mlp_gpu.pt` on every sample whose UNI `.pth` count is ≥90% of the StarDist CSV. All 17 slides now have `*_stardist_pred.h5ad`. Log: `HCC_nature_predict_ready.log`. Pin slides with `SAMPLE=L-8001_HE,L-8002_HE`. Remaining slides only: `SAMPLE=L-8006_HE.ome,...,L-8017_1_HE.ome`. Finished steps skip unless `FORCE=1`.

One slide: `SAMPLE=L-8001_HE` or `SAMPLE=L-8006_HE.ome`. GPU: `CUDA_DEVICE=0` (override if that card is busy).

## What Hist2Pheno can do vs the TIMES paper

The paper’s recurrence score uses **spatial protein/RNA** of SPON2, ZFP36L2, ZFP36, VIM, HLA-DRB1, plus pathologist IF/TC and NK IHC. Hist2Pheno sees only H&E. Implement the H&E analogues in `Pred_visual_hcc_nature.ipynb` (last cells); do **not** claim a TIMES score.

| Paper readout | In this folder | Notes |
|--|--|--|
| AS / IF / TC from GPC3 + Reticulin | **Yes (surrogate)** | Smoothed tumour-fraction grid; **dashed contour** = tumour-domain edge. IF = 160 µm inside + 80 µm into the cap (`BAND_UM`, `AS_SIDE_UM`). TC = core, AS = rest of the cap. A 300 µm inside-only rim painted most of the L-8012 nodules orange |
| NK density at IF vs TC | **Limited** | L2 `NK` is near zero on several slides (e.g. L-8012). A flat NK bar means the class is not predicted. No REC / non-REC labels |
| Immune context at IF | **Yes** | Plot **AS vs IF vs TC** for T / B-plasma / DC / TLS-like (lymphocytes sit in the cap). L-8012 cell also scatters those nuclei on the HE |
| TLS / SRI / TNI | **Yes** | Same niche indices as other External_val cohorts (L2 softmax, R = 55 µm) |
| TIMES 5-gene spatial score | **No** | Cannot read those markers from H&E |
| REC / non-REC, DFS | **No** | Not in `nature_data` |

L-8012 zoom box (native HE pixels, left edge of the first crop, right edge extended): `[1301:9167] × [5463:10666]` (~3937 × 2604 µm at 0.5005 µm/px).

## Outputs

| Path | Role |
|------|------|
| `data/Visium/HCC/nature_data/project_all_UNI/{sample}/ImgEmbeddings_all_stardist/sc_pth_16_16/` | one UNI `.pth` per in-bounds nucleus |
| `data/Visium/HCC/nature_data/pred_pancancer_section/{sample}/{sample}_stardist_features.h5ad` | embeddings + native `spatial_HE` |
| `data/Visium/HCC/nature_data/pred_pancancer_section/{sample}/{sample}_stardist_pred.h5ad` | L1/L12/L2 labels and max-softmax |
| `pred_pancancer_section/figures/L-8012_HE.ome_EDFig1a_*.png` | Extended Data Fig. 1a HE zoom + L1/L12/L2 |
| `pred_pancancer_section/figures/L-8012_HE.ome_IF_TC_hist2pheno.png` | prediction-derived AS/IF/TC on that zoom |
| `pred_pancancer_section/figures/HCC_nature_IF_TC_L2_summary.csv` | 17-slide IF vs TC composition (NK, T, B, TLS-like) |
| `pred_pancancer_section/figures/HCC_nature_NK_IF_vs_TC.png` | NK fraction at IF vs TC, all slides |

`Pred_visual_hcc_nature.ipynb` is the full 17-slide notebook (spatial maps, L-8012 zoom, IF/TC analogue, TLS / SRI / TNI). `Pred_visual_hcc_nature_ready.ipynb` is the same plots restricted to the first five slides. `SAVE_FIGURES = False` only displays them; saved PNGs/CSV go under `pred_pancancer_section/figures/` (ready notebook: `figures_ready/`). There is no ROC: these slides have no ground-truth cell type.
