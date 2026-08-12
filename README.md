# Hist2Pheno

**Histology image embeddings → cell phenotype prediction → spatial niche indices.**

Hist2Pheno maps H&E patch embeddings (UNI / HIPT / Virchow2) to multi-tier cell-type labels via a five-head MLP, validates on independent StarDist nuclei, and supports histology-derived niche biomarkers (TLS, FRI, ARI, APNB).

This repository contains:

| Path | Description |
|------|-------------|
| [`code/Hist2Pheno_pkg/`](code/Hist2Pheno_pkg/) | Core library — `base.py`, `model.py`, `plot.py` |
| [`code/Image_feature_extraction.py`](code/Image_feature_extraction.py) | Per-cell UNI/HIPT embedding extraction from H&E (Xenium + CODEX HCC) |
| [`code/Xenium_lung/`](code/Xenium_lung/) | **GSE250346 lung fibrosis Xenium** pipeline — Complete + Incomplete cohorts ([README](code/Xenium_lung/README.md)) |
| [`code/CODEX_hcc/`](code/CODEX_hcc/) | **CODEX HCC s4769** pipeline — Visium-aligned HE, GT + StarDist UNI features ([README](code/CODEX_hcc/README.md)) |

## Quick start

```bash
git clone https://github.com/LingyuLi-math/Hist2Pheno.git
cd Hist2Pheno

# Conda env with PyTorch, scanpy, spatialdata (e.g. SeededNTM)
conda activate SeededNTM
```

Add the package to `PYTHONPATH` (or let project scripts add it automatically):

```bash
export PYTHONPATH="${PWD}/code/Hist2Pheno_pkg:${PYTHONPATH}"
```

---

## Xenium lung pipeline

Full step-by-step instructions: **[`code/Xenium_lung/README.md`](code/Xenium_lung/README.md)**

### Cohorts

| Cohort | Samples | Role |
|--------|---------|------|
| `Complete_Cases` | 25 | Train pooled cross-dataset model; pathologist GT; StarDist matched validation |
| `Incomplete_Cases` | 20 | Discovery cohort — no `data.zarr`; all-StarDist prediction + clinical niche burden (APNB) |

Replace `<DATA>` with your local GSE250346 processed data root  
(e.g. `.../Spatial-PF-Processed/Data`). Raw data are **not** included in this repo.

### Phase A — Complete_Cases (training cohort)

```bash
# 1. Spatial coords + HE annotation match + StarDist copy
python code/Xenium_lung/extract_cell_spatial_coords.py --data-dir <DATA>/Complete_Cases
python code/Xenium_lung/match_HEanno_with_sample_pix.py --cases-dir <DATA>/Complete_Cases
python code/Xenium_lung/copy_stardist_to_cases.py

# 2. UNI features (single script: GT or StarDist × Complete or Incomplete)
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt complete
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist complete

# 3. Build matched / StarDist h5ad
python -u code/Xenium_lung/transer_embedding_label_h5ad.py
python -u code/Xenium_lung/transer_embedding_label_h5ad.py --steps stardist_all_h5ad

# 4. Cross-dataset train + StarDist matched validation (spatial context)
python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
  --mode cross-dataset --cases-set complete \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_bs4096

# 5. All StarDist nuclei → five-head label h5ad (Complete)
python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
  --mode cross-dataset --cases-set complete \
  --pooled-save-result result_all_spatial \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --ablation-tag D_emph_L2_spatial_bs4096 \
  --steps stardist_all
```

**Main outputs** under `<DATA>/result_all_spatial/`:

- `cross_dataset_cv/D_emph_L2_spatial_bs4096/best_mlp_gpu.pt` — pooled checkpoint
- `stardist/{sample}/validation_external_stardist_matched_AUROC.csv` — matched nuclei + GT
- `stardist/{sample}/{sample}_all_features_stardist_label.h5ad` — all nuclei, five-head softmax

### Phase B — Incomplete_Cases (discovery cohort)

Incomplete samples lack Xenium zarr; skip `extract_cell_spatial_coords.py`. Run HE match, StarDist copy, UNI extraction, and h5ad build on the incomplete set, then apply the **Complete-trained** checkpoint:

```bash
python code/Xenium_lung/match_HEanno_with_sample_pix.py --cases-dir <DATA>/Incomplete_Cases
python code/Xenium_lung/copy_stardist_to_cases.py --cases-dir <DATA>/Incomplete_Cases

bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh stardist incomplete
python -u code/Xenium_lung/transer_embedding_label_h5ad.py \
  --cases-set incomplete --steps stardist_all_h5ad

# Predict all StarDist nuclei with pooled Complete model
python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
  --mode cross-dataset --cases-set incomplete \
  --pooled-save-result result_all_spatial \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --ablation-tag D_emph_L2_spatial_bs4096 \
  --steps stardist_all
```

**Outputs:** `<DATA>/result_all_spatial/stardist_Incomplete_Cases/{sample}/{sample}_all_features_stardist_label.h5ad`

### Phase C — Niche indices and clinical analysis

Open **[`code/Xenium_lung/histology_derived_niche_index.ipynb`](code/Xenium_lung/histology_derived_niche_index.ipynb)**:

| Section | Content |
|---------|---------|
| §6 | Complete_Cases — TLS / FRI / ARI on StarDist matched nuclei; pathologist validation |
| §7 | Incomplete_Cases — Active Proliferative Niche Burden (APNB) on **all** StarDist nuclei; clinical comparison vs `41588_2025_2080_MOESM5_ESM.xlsx`; five-tier spatial maps (`plot_stardist_all_label_spatial_tiers`) |

Key Python API in [`histology_derived_niche_index.py`](code/Xenium_lung/histology_derived_niche_index.py):

```python
import histology_derived_niche_index as hdni

# Complete — matched StarDist + spatial TLS/FRI/ARI
df, class_names, probs, coords, paths = hdni.load_cross_dataset_sample_with_spatial("VUILD107MA")

# Incomplete — all nuclei from label h5ad
hdni.plot_stardist_all_label_spatial_tiers("TILD299MA", tier="l2")
```

### Notebooks (Xenium)

| Notebook | Purpose |
|----------|---------|
| [`Lung_train_validate_cv_UNIlabel_all.ipynb`](code/Xenium_lung/Lung_train_validate_cv_UNIlabel_all.ipynb) | Cross-dataset training; §4 StarDist matched; §5–§6 Complete / Incomplete all-nuclei prediction |
| [`histology_derived_niche_index.ipynb`](code/Xenium_lung/histology_derived_niche_index.ipynb) | Niche indices, APNB, clinical stratification, spatial tier plots |
| [`Data_process_visual_xenium_all.ipynb`](code/Xenium_lung/Data_process_visual_xenium_all.ipynb) | Preprocessing QC and cohort overview |

---

## CODEX HCC pipeline (s4769)

Full step-by-step instructions: **[`code/CODEX_hcc/README.md`](code/CODEX_hcc/README.md)**

**CODEX hepatocellular carcinoma** — 38 Visium-aligned HE regions from the Michael s4769 transfer dataset. Uses the same Hist2Pheno five-head MLP stack after per-cell UNI extraction from aligned H&E TIFFs.

### Data layout

```
data/HCC/Michael_data_transfer/s4769/
├── HE/s4769_he_mapping_updated_Visium.xlsx   # 38 ALIGNED=='Y' regions
├── HE/{HE_KEY}/figures/{HE_KEY}.tif          # registered H&E (~0.5 µm/px)
├── {ACQ_ID}/{ACQ_ID}.cell_data.csv           # GT CODEX cell coords (X, Y in HE px)
└── HE/{HE_KEY}/project_all_UNI/
    ├── ImgEmbeddings_all/                    # GT UNI .pth
    └── ImgEmbeddings_all_stardist/           # StarDist UNI .pth
```

StarDist CSVs (external): `{STARDIST_ROOT}/{HE_KEY}/{HE_KEY}_Float_prob0.01_nms_0.3.csv`

### Workflow

```bash
# 1. Map CODEX cell types onto aligned HE (QC figures)
python -u code/CODEX_hcc/s4769_img_cell_mapping.py

# 2. UNI feature extraction — single region
bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh gt
bash code/CODEX_hcc/demo_GT_feature_extraction_Single.sh stardist

# 3. Batch UNI — all 38 regions (GT then StarDist)
bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh gt
bash code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh stardist

# 4. Train / validate (see HCC_train_validate_cv_UNIlabel.py)
python -u code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py
```

Both shell wrappers call [`code/Image_feature_extraction.py`](code/Image_feature_extraction.py) with region-specific `--position` CSV and HE TIFF. CODEX HCC uses **native HE resolution** (`scale_image=False`, patch size 16 → ~8 µm); Xenium lung uses `scale=0.425` for 0.2125 µm/px HE.

### CODEX scripts

| Script | Role |
|--------|------|
| [`s4769_img_cell_mapping.py`](code/CODEX_hcc/s4769_img_cell_mapping.py) | Load CODEX annotations; render cell types on aligned HE |
| [`demo_GT_feature_extraction_Single.sh`](code/CODEX_hcc/demo_GT_feature_extraction_Single.sh) | One region — GT (`cell_data.csv`) or StarDist coords |
| [`demo_UNI_feature_extraction_batch.sh`](code/CODEX_hcc/demo_UNI_feature_extraction_batch.sh) | Batch over 38 ALIGNED regions |
| [`HCC_train_validate_cv_UNIlabel.py`](code/CODEX_hcc/HCC_train_validate_cv_UNIlabel.py) | HCE training + validation (adapted from Xenium lung) |
| [`HCC_train_validate_cv_UNIlabel_single.ipynb`](code/CODEX_hcc/HCC_train_validate_cv_UNIlabel_single.ipynb) | Interactive training / visualization |
| [`Data_process_visual_codex.ipynb`](code/CODEX_hcc/Data_process_visual_codex.ipynb) | CODEX preprocessing and QC |

Env overrides for batch extraction: `ACQ_ID`, `HE_KEY`, `STARDIST_ROOT`, `SKIP_IF_DONE`, `CLEAN_UNI_OUTPUT`.

---

## Package overview

### `Hist2Pheno_pkg`

- **`base.py`** — coordinate matching, AnnData / NPZ builders, spatial kNN index, five-head MLP architectures, embedding I/O
- **`model.py`** — stratified / LOGO cross-validation training, spatial-context fusion, checkpoint selection
- **`plot.py`** — confusion matrices, ROC, spatial cell-type maps, five-head softmax collection

### Five prediction heads

| Head | Tier | Example labels |
|------|------|----------------|
| L2 | Fine cell type | B cells, AT2, Myofibroblasts, … |
| L1 | Lineage | Epithelial, Immune, Mesenchymal, … |
| L12 | Level 1-1-2 | Intermediate grouping |
| L3 | CNiche | C1–C12 |
| L4 | TNiche | T1–T12 |

### `Image_feature_extraction.py`

Shared UNI/HIPT/Virchow2 patch embedder used by both pipelines. Accepts a coordinate CSV (`X`/`Y` or Xenium-specific columns) and an H&E image path; writes one `.pth` per cell under `ImgEmbeddings_all/` or `ImgEmbeddings_all_stardist/`.

---

## Data

Processed datasets live outside this repository:

| Project | Location (local example) |
|---------|--------------------------|
| Xenium lung | `.../Spatial-PF-Processed/` — see [`code/Xenium_lung/README.md`](code/Xenium_lung/README.md) |
| CODEX HCC | `.../data/HCC/Michael_data_transfer/s4769/` |

Clinical metadata for Xenium lung: `Annotation/HE_Annotations/41588_2025_2080_MOESM5_ESM.xlsx` (`Supplementary Table 1` / `Clinical_info`).

---

## Citation

If you use this code, please cite the associated publication (TBD) and the original datasets:

> Kedlian et al. — spatial multi-omics lung fibrosis atlas ([GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346))

## License

Apache-2.0 — see [LICENSE](LICENSE).
