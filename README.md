# Hist2Pheno

**Histology image embeddings → cell phenotype prediction → spatial niche indices.**

Hist2Pheno maps H&E patch embeddings (UNI / HIPT / Virchow2) to multi-tier cell-type labels via a five-head MLP, validates on independent StarDist nuclei, and supports histology-derived niche biomarkers (TLS, FRI, ARI).

This repository contains:

| Path | Description |
|------|-------------|
| [`code/Hist2Pheno_pkg/`](code/Hist2Pheno_pkg/) | Core library — `base.py`, `model.py`, `plot.py` |
| [`code/Image_feature_extraction.py`](code/Image_feature_extraction.py) | Per-cell UNI/HIPT embedding extraction from H&E |
| [`code/Xenium_lung/`](code/Xenium_lung/) | End-to-end **GSE250346 lung fibrosis Xenium** pipeline (see [README](code/Xenium_lung/README.md)) |

## Quick start

```bash
git clone https://github.com/LingyuLi-math/Hist2Pheno.git
cd Hist2Pheno

# Conda env with PyTorch, scanpy, spatialdata (e.g. SeededNTM)
conda activate SeededNTM
```

Add the package to `PYTHONPATH` (or let Xenium scripts add it automatically):

```bash
export PYTHONPATH="${PWD}/code/Hist2Pheno_pkg:${PYTHONPATH}"
```

### Xenium lung pipeline (Complete_Cases)

Full step-by-step instructions: **[`code/Xenium_lung/README.md`](code/Xenium_lung/README.md)**

```bash
# 1. Spatial coords + HE annotation match + StarDist copy
python code/Xenium_lung/extract_cell_spatial_coords.py --data-dir <DATA>/Complete_Cases
python code/Xenium_lung/match_HEanno_with_sample_pix.py --cases-dir <DATA>/Complete_Cases
bash code/Xenium_lung/demo_UNI_feature_extraction_batch.sh gt complete
python -u code/Xenium_lung/transer_embedding_label_h5ad.py

# 2. Cross-dataset train + StarDist validation
python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
  --mode cross-dataset --cases-set complete \
  --use-spatial-context --spatial-k 8 --spatial-mode mean \
  --pooled-save-result result_all_spatial \
  --ablation-tag D_emph_L2_spatial_bs4096

# 3. Niche indices (TLS / FRI / ARI) — see histology_derived_niche_index.ipynb
```

Replace `<DATA>` with your local GSE250346 processed data root  
(e.g. `.../Spatial-PF-Processed/Data`). Raw data are **not** included in this repo.

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

## Data

Processed Xenium lung data (GSE250346, Weiqin et al.) live outside this repository under a local `data/` tree. See [`code/Xenium_lung/README.md`](code/Xenium_lung/README.md) for directory layout, sample sets (`Complete_Cases` ×25, `Incomplete_Cases` ×20), and expected file names.

## Citation

If you use this code, please cite the associated publication (TBD) and the original Xenium dataset:

> Kedlian et al. — spatial multi-omics lung fibrosis atlas ([GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346))

## License

Apache-2.0 — see [LICENSE](LICENSE).
