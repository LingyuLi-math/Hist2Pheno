# CODEX ESCC

This directory contains the CODEX ESCC pipeline. `NCRT` remains the clinical
cohort and path identifier; notebook filenames, data paths, environment
variables, tumor slugs, and values such as `therapy_data="NCRT"` intentionally
keep that name.

## Plotting scheme and tiers

New plotting code uses the canonical scheme ID `codex_escc`. The lowercase
`ncrt` plotting ID remains a backward-compatible alias only.

Most plotting APIs also accept `pan_organ="codex_escc"` as an optional unified
selector; explicit `codex_escc` / `ncrt` scheme strings continue to work.

CODEX ESCC uses one plotting scheme with a tier hint:

| Model tier | Annotation column / tier hint |
|------------|-------------------------------|
| L2 fine cell type | `celltype` |
| L1 cell type | `celltype_level1` |
| L12 intermediate cell type | `celltype_level12` |
| L3 coarse compartment | `celltype_level0` |
| L4 lineage bucket | `celltype_level01` |

Spatial plots select the tier through `celltype_col`. ROC APIs retain the
parameter name `ncrt_color_tier` for compatibility; pass the same column name
as its value. The parameter name is not a request to use the legacy plotting
scheme ID.

## Notebook helper

`codex_escc_uni_nb_helpers.py` is the canonical five-tier notebook helper. Its
ESCC semantics differ from the shared Xenium/HCC helper and should remain
dataset-local. `ncrt_uni_nb_helpers.py` is a deprecated import shim that
re-exports the canonical module, including private attributes used by older
notebook module namespaces.

## Active entry points

- `NCRT_train_validate_tumor1_cv_UNIlabel012clean.ipynb`
- `NCRT_train_validate_cv_label012.ipynb`
- `NCRT_train_validate_tumor1_cv_UNIlabel012.ipynb`
- `NCRT_train_validate_tumor1_cv_UNIlabel012GAT.ipynb`
- `demo.sh` and `demo_level012.sh`
- `model_train_validate.py`, `model_train_validate_cv_method.py`, and
  `model_train_validate_cv_labmeeting.py`

Dated and archive notebooks are retained for reproducibility and are not
migrated as active plotting entry points.
