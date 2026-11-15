# External validation (new HE + StarDist, no cell-type GT)

Apply the pan-cancer **cross-section** checkpoint to a cancer that was not in training. Each cohort has HE and StarDist nuclei and no nucleus-level cell-type GT, so there is no ROC. The readout is L1 / L12 / L2 plus TLS / SRI / TNI. The 10x breast slide is the exception that ships Space Ranger `spatial/` plus pathologist spot labels (`data/Visium/BRCA/10x_data/metadata.tsv`); those spots are not the StarDist GT.

Shared implementation: [`code/Hist2Pheno_pkg/external_validation.py`](../Hist2Pheno_pkg/external_validation.py). NPC, CRC, HCC, and HCC_NATURE are rows in `_SIMPLE_COHORTS`; `demo.sh` calls `external_validation.py <COHORT> <step>`. Prostate, BRCA, and BRCA_ST keep their own scripts (`*_external.py`) because HE suffix and/or µm/px are not one flat folder + one global factor.

Checkpoint (section CV, not cell-stratified):

`data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt`

Scaler: `data/PanCancer/result_all_spatial/scaler.joblib`. Spatial k=8. Heads: L1 `pred_lineage`, L12 `pred_sublineage`, L2 `pred_CT`. Hidden dims `(768, 384, 192)`.

## Pixel size is read in code

Training UNI crops are **8 µm**: 16 px on a **0.5 µm/px** grid. `ExternalCohort.geometry` computes the scale that puts the new slide on that grid, then `patch_size=16`.

For OME TIFF / BigTIFF (NPC, CRC; BRCA GSM slide), `ome_um_per_px` opens the file with `tifffile` and parses `PhysicalSizeX` / `PhysicalSizeY` from the first page `ImageDescription`. The numbers in the NPC README (0.465 µm/px), BRCA GSM (~0.548 µm/px), and CRC (~0.274 µm/px) are what that function returns. CRC sets `he_suffix=".btf"`. TIFF `XResolution = 96` dpi is not used.

For JPEG (HCC) there is no OME block. Pass the scanner calibration as `MPP_UM`. Mixed Nature TIFFs (`HCC_nature`) resolve µm/px **per slide**: plausible OME `PhysicalSize` (0.05–5 µm/px), else Aperio `MPP`, else fallback 0.5. A 300 dpi OME tag (84.67 µm/px on `L-8015`) is skipped. `geometry` records `mpp_source` as `OME PhysicalSize`, `Aperio MPP`, `cohort mpp_um`, or `cohort mpp_um fallback`.

BRCA also has two SpatialVDJ pathology JPEGs: µm/px comes from Visium `scalefactors_json.json` **divided by HE÷SR fullres** (~3.333), ≈0.172 µm/px — see [`BRCA/brca_external.py`](BRCA/brca_external.py). Do not use raw `65/spot_diameter` on those HE files. The fourth BRCA slide, `V1_Breast_Cancer_Block_A_Section_1_image`, is the 10x Space Ranger fullres TIFF in `data/Visium/BRCA/10x_data` (HE÷SR = 1), so µm/px is raw `65/spot_diameter` ≈ 0.366. Spot labels in `10x_data/metadata.tsv` are the GraphST pathologist annotation, not nucleus GT.

`Image_feature_extraction.py` resizes OME TIFF and JPEG/PNG in uint8. StarDist `centroid_x` / `centroid_y` are native full-resolution pixels. The `.pth` filename stores the scaled coordinate divided by `scale`, so the later h5ad match uses the original CSV coordinates (1 px tolerance). `obsm['spatial_HE']` stays on that native grid, so niche radius 55 µm is converted with the same µm/px.

## Steps

From the repo root, `conda activate SeededNTM` (or `Hist2Pheno`).

```bash
bash code/External_val/<COHORT>/demo.sh dry-run    # mpp, scale, nucleus counts
bash code/External_val/<COHORT>/demo.sh features   # UNI
bash code/External_val/<COHORT>/demo.sh h5ad
bash code/External_val/<COHORT>/demo.sh predict    # h5ad + {sample}_l2_probs.npz
# or
bash code/External_val/<COHORT>/demo.sh all        # features, h5ad, predict
```

`SAMPLE=<id>` limits one slide. `FORCE=1` redoes a finished step. `features` is the long step (UNI, one 16×16 crop per nucleus).

## After a new pan-cancer train

The classifier is `result_all_spatial/.../best_mlp_gpu.pt` (section CV). External val writes under each cohort’s `pred_pancancer_section/`, not `data/PanCancer/pred_all_spatial/`. You can run these `predict` jobs while `bash code/PanCancer/demo.sh section repredict` is rewriting the pan-cancer pred tree.

Do **not** re-run `features` / `h5ad` / `all`. UNI embeddings do not change with the MLP. Existing pred h5ads are skipped unless `FORCE=1`:

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno

# small cohorts — one GPU each; GPU 2 is the PanCancer demo.sh default
CUDA_DEVICE=0 FORCE=1 bash code/External_val/BRCA/demo.sh predict
CUDA_DEVICE=1 FORCE=1 bash code/External_val/NPC/demo.sh predict
CUDA_DEVICE=3 FORCE=1 bash code/External_val/CRC/demo.sh predict
```

### HCC (7 JPEGs, ~1.34M nuclei)

`demo.sh` **requires** `MPP_UM` (no OME tag; 96 dpi is not a pixel size). The finished feature run used **`MPP_UM=0.5`** (`HCC.log` / `Pred_visual_hcc.ipynb`). Default GPU is **2** — override it if PanCancer `section repredict` is on GPU 2.

```bash
CUDA_DEVICE=0 FORCE=1 MPP_UM=0.5 bash code/External_val/HCC/demo.sh predict
# one slide: SAMPLE=P1_T
```

Then `code/External_val/HCC/Pred_visual_hcc.ipynb`. Keep `MPP_UM=0.5` unless you truly change the scanner calibration (that would also require re-extracting UNI).

### HCC Nature (17 TIFFs, ~3.68M nuclei)

Distinct from the seven JPEG Visium slides. Data: `data/Visium/HCC/nature_data/`. µm/px is **per slide** (OME ~0.501, Aperio `MPP` on `L-8001_HE`, fallback 0.5). `MPP_UM` is optional. Default GPU is **0**. OME sample ids keep the `.ome` stem (`SAMPLE=L-8006_HE.ome`).

```bash
CUDA_DEVICE=0 bash code/External_val/HCC_nature/demo.sh dry-run
CUDA_DEVICE=0 bash code/External_val/HCC_nature/demo.sh all
```

Then `code/External_val/HCC_nature/Pred_visual_hcc_nature.ipynb`.

### Prostate (35 JPEGs, ~7.43M nuclei)

No `MPP_UM`: µm/px comes from HEST spot spacing in `prostate_external.py` (~0.57–0.69 on Patient 1, ~0.34–0.35 on Patient 2). Do not start all 35 slides on a GPU that is already busy. Default GPU is 0. `SAMPLE` is a patient or one section (`Patient_1__H1_2`).

```bash
CUDA_DEVICE=1 FORCE=1 SAMPLE=Patient_1 bash code/External_val/Prostate/demo.sh predict
CUDA_DEVICE=3 FORCE=1 SAMPLE=Patient_2 bash code/External_val/Prostate/demo.sh predict
```

Notebooks: `Pred_visual_prostate_patient1_{H1,H2,V1}.ipynb` (Patient_1, 7 sections) and `Pred_visual_prostate_patient2_{H1,H2,H3,V1,V2}.ipynb` (Patient_2, 28 sections), each split by series.

BRCA_ST (36 HER2ST slides) is the same `FORCE=1 predict` pattern; pin its own `CUDA_DEVICE`.

## Cohorts

| Cohort | Data | HE | Slides | Nuclei | µm/px | GPU default |
|--------|------|----|--------|--------|-------|-------------|
| NPC | `data/Visium/NPC` | OME TIFF | 2 | 142,184 + 120,888 | read by `ome_um_per_px` (~0.465) | `CUDA_DEVICE` default 0 |
| BRCA | `data/Visium/BRCA` | OME TIFF + 2 pathology JPEG + 10x fullres TIFF | 4 (GSM + `V10F24-113_B1` / `V19T26-031_D1` + `V1_Breast_Cancer_Block_A_Section_1`) | ~281k + ~372k + ~391k + 423,162 | GSM: OME ~0.548; JPEGs: Visium×HE÷SR ~0.172; 10x fullres: `65/spot_diameter` ~0.366 (`brca_external.py`) | `CUDA_DEVICE` default 0 |
| HCC | `data/Visium/HCC` | JPEG | 7 (`P1_N` `P1_T` `P2_N` `P3_T` `P5_N` `P5_T` `P7_T`) | ~1.34M | `MPP_UM` (required; 96 dpi is not a pixel size) | `CUDA_DEVICE` default **2** |
| HCC_nature | `data/Visium/HCC/nature_data` | TIFF + OME TIFF | 17 (`L-8001_HE` … `L-8017_1_HE.ome`) | ~3.68M | per slide (OME ~0.501 / Aperio / fallback 0.5) | `CUDA_DEVICE` default 0 |
| CRC | `data/Visium/CRC` | OME BigTIFF | 1 (`Visium_V2_Human_Colon_Cancer_P2_tissue_image`) | 579,587 | read by `ome_um_per_px` (~0.274) | `CUDA_DEVICE` default 0 |
| BRCA_ST | `data/ST/BRCA` | JPEG | 36 HER2ST (A1–H3) | 2,417,153 | 200 µm spot pitch (~0.92 on A, ~0.69 on B–H) | `CUDA_DEVICE` default 0 |
| Prostate | `data/Visium/Prostate` | JPEG | 35 (patient 1: 7; patient 2: 28) | 7.43M | HEST spot spacing (~0.57–0.69 on patient 1, ~0.34–0.35 on patient 2) | `CUDA_DEVICE` default 0 |

Unlabeled leftovers from the **seven training cohorts** (StarDist-all UNI already extracted; no cell-type GT; notebooks draw 3–5 slides; **no niches**):

| Folder | Data | Slides | What is plotted |
|--------|------|--------|-----------------|
| [`Lung_Incomplete`](Lung_Incomplete/README.md) | Xenium `Incomplete_Cases` | 20 (UNI preds in `pred_all_spatial/incomplete_lung/`) | all 20; top 5 L2 AUROC first. UNI2 A2: `Pred_visual_lung_incomplete_uni2_A2.ipynb` |
| [`HCC_unlabel`](HCC_unlabel/README.md) | CODEX HCC rest (not Visium `HCC/`) | 8 | 5 largest |
| [`PDAC_unlabel`](PDAC_unlabel/README.md) | CODEX PDAC Incomplete_Cases | 195 | 5 largest |

Shared CLI: [`unlabeled_external.py`](unlabeled_external.py). HCC/PDAC preds go to `data/PanCancer/pred_unlabel/`.

Predicted outputs are under `data/Visium/{NPC,BRCA,CRC,HCC,Prostate}/pred_pancancer_section/`, `data/Visium/HCC/nature_data/pred_pancancer_section/`, and `data/ST/BRCA/pred_pancancer_section/`.

Per-cohort notes: [`NPC/README.md`](NPC/README.md), [`BRCA/README.md`](BRCA/README.md), [`HCC/README.md`](HCC/README.md), [`HCC_nature/README.md`](HCC_nature/README.md), [`CRC/README.md`](CRC/README.md), [`BRCA_ST/README.md`](BRCA_ST/README.md), [`Prostate/README.md`](Prostate/README.md).

## After prediction

`Pred_visual_*.ipynb` draws the maps from the pred h5ad. Spatial is a **2×3** figure: row 1 L1/L12/L2 on the HE, row 2 the same points with no HE. Composition is the **next cell**; bars use the pan-cancer L1 / L12 / L2 palette (`plotting_palettes`), not a sample color cycle. `SAVE_FIGURES`, `FIGSIZE`, `DPI`, `SAVE_DPI`, `HE_MAX_SIDE`, and `MARKER_PX` are at the top of the notebook. `DPI` is the notebook preview (160 or 200); saved files are **300 dpi PNG**. The default only displays the figure. `demo.sh plot` is an optional batch export of the same panels; `all` does not call it. TLS / SRI / TNI use L2 softmax (`ensure_l2_probs`), not the argmax label. Radius is 55 µm on native `spatial_HE` pixels. Each index is a 1×3 row: abundance, spatial score, spatial score with the within-slide 95th-percentile candidates. Lung ARI / FRI are not used.

### Run every Pred_visual notebook (no manual Run All)

Do not click Run All in the editor: that embeds rasters into the `.ipynb` and Prostate / BRCA_ST files become tens of MB. Use the batch runner. It executes the 20 plotting notebooks **one after another** with a headless Agg backend, leaves the source notebooks untouched, and writes PNGs / niche h5ads as side effects.

```bash
cd /home/lingyu/ssd2/Python/Hist2Pheno
conda activate SeededNTM

# all cohorts (BRCA → NPC → CRC → HCC → HCC_nature → BRCA_ST A–H → Prostate series)
# default SAVE_FIGURES=1 → 300 dpi PNG under pred_pancancer_section/figures/
screen -S extval_nb
bash code/External_val/demo.sh notebooks
# detach: Ctrl-a d
```

Subset, smoke (no PNG), or stop at the first error:

```bash
ONLY=BRCA,CRC bash code/External_val/demo.sh notebooks
ONLY=Prostate_patient1 bash code/External_val/demo.sh notebooks
ONLY=brca_st_A bash code/External_val/demo.sh notebooks
SAVE_FIGURES=0 bash code/External_val/demo.sh notebooks
STOP_ON_ERROR=1 bash code/External_val/demo.sh notebooks
```

`ONLY` is a comma-separated substring of the path (`BRCA` also matches `BRCA_ST`). Index pages `Pred_visual_prostate_patient1.ipynb` / `_patient2.ipynb` are skipped (markdown only). Log: `code/External_val/Pred_visual_batch.log`. `MARKER_PX` / `DPI` stay as set in each notebook. Predict must already be done; this step does not call UNI or the MLP. Saved panels are 300 dpi PNG (`SAVE_DPI`).

A new cancer with the same folder layout (`HE_images/`, `StarDist_Segment/{sample}/{sample}_Float_prob0.01_nms_0.3.csv`) is:

```python
from Hist2Pheno_pkg.external_validation import ExternalCohort

cohort = ExternalCohort(name="NAME", data_root=repo / "data" / "Visium" / "NAME", repo=repo)
cohort.main(["dry-run"])
```

JPEG with one scanner calibration: also pass `he_suffix=".jpg"` and `mpp_um=...`. BigTIFF: pass `he_suffix=".btf"`. HER2ST (`BRCA_ST`) uses the 200 µm ST array pitch. Prostate Visium uses that section's `scalefactors_json.json` (`65 / spot_diameter_fullres`), like NPC reads OME per TIFF. Each `demo.sh` appends screen output and per-step plus whole-run elapsed time to `code/External_val/<COHORT>/<COHORT>.log`.
