# Hist2Pheno package

Shared, dataset-neutral modeling, transformation, and plotting code for Hist2Pheno.
Dataset directories such as `CODEX_hcc/` and `Xenium_lung/` remain responsible
for data paths, file I/O, cohort-specific orchestration, and report assembly.

## 2026-20-25 Which embedder to run next

`data/` currently keeps **UNI** results only. The next embedder to run is **UNI2-h** (`--method UNI2`), with `--patch_size 28` on slides already scaled to 0.5 µm/px. Write the `.pth` files to a new directory and retrain the MLP. The published pan-cancer checkpoint stays the UNI one.

Hist2Pheno labels one nucleus at a time. The H&E is scaled to 0.5 µm/px, a square is cropped on the centroid, and one vector goes into the multi-head MLP (L2 / L12 / L1). Neighbor context is a later residual on those vectors (`SpatialContextFusion` in `base.py`). TLS / SRI / TNI are computed from the predicted labels and coordinates. The embedder’s job is the morphology of that cell. A crop that already mixes the surrounding cells makes two adjacent nuclei with different ground-truth labels look alike, and the L2 head (CD4 / CD8 / Treg, macrophage / DC, fibroblast / myofibroblast) is the head that suffers.

What each `--method` actually saves, given `Image_feature_extraction.py`:

| `--method` | Vector the `.pth` stores | Crop that matches this code | Field at 0.5 µm/px | Fit to a per-nucleus label |
|------------|--------------------------|-----------------------------|--------------------|----------------------------|
| `UNI` | class token, 1024-d | 16 px = one ViT-L/16 token | 8 µm | current results; native token equals the nucleus crop |
| **`UNI2`** | class token, 1536-d | **28 px = 2×2 ViT-H/14 tokens** | **14 µm** | same one-vector contract; crop still covers one cell |
| `H-optimus-1` | class token, 1536-d | 224 px required | 112 µm | one vector per centroid, but the tile contains many cells |
| `HIPT` | patch tokens after dropping token 0, concatenated | 16 px → a single 384-d token | 8 µm | bottom ViT-256 stage only; the 4K hierarchy is unused |
| `Virchow2` | every token after the first five, concatenated | 16 px is not a multiple of 14 | — | a bag of patch tokens, flattened into the MLP |

**UNI2-h** is the one that keeps the UNI contract and upgrades the encoder.

- The loader already stores one class token and flattens it. `input_dim` is read from that vector, so 1024 → 1536 needs a new training run and no change to matching, h5ad I/O, or the spatial residual. UNI2-h is the later MahmoodLab encoder (ViT-H/14, 1536-d) trained as the successor to UNI; the model card reports stronger H&E tile features than UNI. That is the signal this MLP reads.
- Its token is 14 px, so the crop has to be a multiple of 14. `patch_size=28` is 14 µm: nucleus plus a rim of cytoplasm, pooled by the class token over four tokens. That is the smallest crop on which the class token does the pooling UNI2 was trained to do, and it stays on one cell. `patch_size=14` (7 µm, one token) also runs, and is the closer twin of the UNI 8 µm crop if the comparison must hold the field fixed. `patch_size=16` only triggers a warning: 16 is not an integer number of 14 px tokens, so that crop is not a UNI2 input.
- Weights are gated (~2.6 GiB). Accept the license, then follow [`code/UNI2/assets/ckpts/uni2-h/README.md`](../UNI2/assets/ckpts/uni2-h/README.md).

The other three answer a different question than “replace UNI on the cells already embedded.”

- **H-optimus-1** is a ViT-G/14 trained on 224×224 tiles at 0.5 µm/px, and this script refuses any other `patch_size`. At that size the window is 112 µm, so centroids tens of microns apart share almost the same pixels. The embedding is a tissue tile. It is the right scale for a later context-window ablation, and the wrong scale for the first non-UNI cell run. The weights are **CC BY-NC-ND 4.0** (~4.3 GiB), which also constrains redistribution of anything derived from them.
- **Virchow2** is a strong tile encoder, but this script drops the first five tokens and saves the rest. The h5ad loader concatenates them, so a 224 px tile becomes hundreds of 1280-d tokens in one MLP input. A 16 px nucleus crop is not a multiple of its 14 px token. Using it as a UNI replacement means changing the representation, not only the weights.
- **HIPT** at `patch_size=16` keeps one 384-d token from `vit256_small_dino`. That model was trained on 256×256 tiles, and the second stage that makes HIPT hierarchical is not called. The vector is smaller and comes from an earlier pretraining recipe than UNI.

Practical split from the UNI tree: new folder for UNI2 `.pth`, new h5ad, new MLP checkpoints. Leave the UNI embeddings in place so the published track and this run stay comparable.

## Changelog

### 2026-10-25 — Next embedder is UNI2-h

- Recorded why `--method UNI2` with `--patch_size 28` is the run to add while `data/` still holds only UNI. H-optimus-1’s required 224 px tile, Virchow2’s flattened token bag, and HIPT’s single 384-d token do not match the per-nucleus MLP.

### 2026-10-21 — HCC Nature mixed TIFFs in `external_validation`

- `_SIMPLE_COHORTS["HCC_NATURE"]` points at `data/Visium/HCC/nature_data`.
- Per-slide µm/px: plausible OME `PhysicalSize`, else Aperio `MPP`, else fallback 0.5.
- OME sizes outside 0.05–5 µm/px (300 dpi placeholders) are ignored.

### 2026-10-08 — Ontology and fold assignment live in the package

- `pancancer_labels.py` and `pancancer_cv.py` moved here from `code/PanCancer`. The PanCancer copies re-export them so existing scripts and notebooks keep `from pancancer_labels import ...`.
- `palette_for` is in `plotting_palettes.py`. `pancancer_viz` imports it. `write_palette_swatch.py` writes the paper swatch from that same map.
- `external_validation.py` no longer puts `code/PanCancer` on `sys.path`.
- Cohort paths, the section index, and the train / predict CLIs stay in `code/PanCancer`.

### 2026-10-08 — Pan-cancer L12 / L2 colors

- `Cell_Type_COLORS_PANCANCER_level0` (L1) is unchanged.
- L1 stays the five lineage colors. L12 and L2 use the parent hue (epithelial warm, T blue, B teal, myeloid ochre, stromal green, endothelial orange, neural purple), with lightness changes inside each family so neighboring classes stay apart. `get_palette("pancancer", ...)` is the only map.

### 2026-10-07 — StarDist external validation

- Added `external_validation.py`. `ome_um_per_px` reads OME `PhysicalSizeX` / `PhysicalSizeY` from the TIFF `ImageDescription` at run time. `ExternalCohort.geometry` turns that into the scale that maps the slide onto 0.5 µm/px (16 px = 8 µm). TIFF 96 dpi tags are not read.
- JPEG slides have no OME block. Pass `mpp_um` (HCC: env `MPP_UM`). `mpp_source` in the geometry dict records which path was used.
- Cohort folders under `code/External_val/` only set the data root and call `ExternalCohort.main`.

### 2026-08-27 — Shared histology niche-index engine

- Added `histology_niche_index.py`: dataset-neutral TLS / SRI / TNI (AUROC load,
  neighborhood scores, cohort Q4 / burden, source-metric grid).
- Register a new organ with `OrganNicheConfig` + `register_organ_config` (GBM /
  BRCA can call this without copying HCC/PDAC modules).
- CODEX HCC / PDAC / GIST keep thin wrappers for paths and clinical I/O.
- Xenium lung ARI / FRI / pathologist code stays in
  `Xenium_lung/histology_derived_niche_index.py`.

### 2026-08-13 — Dataset-aware plotting refactor

- Moved canonical CODEX ESCC, Xenium lung, and CODEX HCC palettes and palette
  resolution into `plotting_palettes.py`.
- Moved reusable DataFrame count/composition transforms and generic bar,
  stacked-composition, and lineage-panel plots into `plotting_utils.py`.
- Kept spatial plotting in `plot.py`; dataset modules now provide only
  paths, I/O, CLI/report orchestration, and compatibility wrappers/re-exports.
- Removed the package-side dynamic import of the Xenium directory. Package
  modules no longer depend on dataset folders.
- Validated dataset/scheme aliases, semantic-tier resolution, deterministic
  fine-label fallbacks, legacy wrapper imports, and representative CODEX ESCC,
  Xenium, and HCC spatial/bar/composition plotting paths. The historical
  `ncrt` plotting ID remains a backward-compatible alias of `codex_escc`.
- Registered explicit five-tier Xenium lung schemes and migrated shared
  UNI-label spatial/ROC defaults to tier-aware canonical schemes.

### 2026-08-12 — Shared UNI-label CV helpers

Canonical cross-dataset UNI-label CV, internal-validation, and StarDist-tier
helpers moved from `Xenium_lung/xenium_uni_nb_helpers.py` to
`uni_label_cv_helpers.py`. The old Xenium module remains a compatibility shim.

## Module index

| Module | Responsibility |
|--------|----------------|
| [`base.py`](base.py) | Coordinate matching, embedding/AnnData/NPZ data preparation, spatial-neighbor indices, model classes, and low-level evaluation utilities. |
| [`model.py`](model.py) | CV training, checkpoint selection/reload, spatial-context fusion, prediction, and model-level validation. |
| [`plot.py`](plot.py) | Dataset-neutral ROC, confusion-matrix, and spatial plotting APIs, including multi-tier prediction plots. |
| [`plotting_palettes.py`](plotting_palettes.py) | Canonical palette registry, dataset/tier normalization, scheme aliases, RGBA conversion, and deterministic palette resolution. |
| [`plotting_utils.py`](plotting_utils.py) | Generic DataFrame count/composition transforms and reusable bar, stacked-composition, and lineage-panel plots. |
| [`histology_niche_index.py`](histology_niche_index.py) | Shared TLS / SRI / TNI engine (`OrganNicheConfig`, neighborhood scores, cohort burden). |
| [`uni_label_cv_helpers.py`](uni_label_cv_helpers.py) | Shared UNI-label CV, internal-validation, and StarDist prediction/report helpers used by dataset pipelines and notebooks. |

## Architecture boundary

The package owns:

- dataset-neutral transforms, model logic, metrics, and plots;
- registered palette data and palette/tier resolution;
- stable public APIs shared by multiple datasets.

Dataset folders own:

- local and cohort-specific paths, file discovery, and file I/O;
- sample/acquisition mappings and dataset metadata;
- CLI entry points, batch orchestration, and cohort reports;
- thin compatibility wrappers that preserve historical imports/defaults.

The dependency direction is one-way: dataset folders may import
`Hist2Pheno_pkg`, but package modules must not import or dynamically load
modules from dataset folders.

## Semantic tiers and legacy mappings

New APIs use semantic tier names. Historical head names remain accepted where
the registry can map them unambiguously.

| Normalized tier | CODEX HCC | Xenium lung | CODEX ESCC (historical NCRT naming) |
|-----------------|-----------|-------------|------------------------|
| `fine` | Excel `celltype_level2`, `final_CT`, L2 | `celltype_level2`, `final_CT`, L2 | `celltype`, L2 |
| `intermediate` | Excel `celltype_level1`, `final_sublineage`, L12 | `final_sublineage`, L12 | `celltype_level12`, L12 |
| `lineage` | Not a separate HCC model target; use `coarse` | `final_lineage`, `celltype_level1`, L1 | `celltype_level1`, L1 |
| `coarse` | Excel `celltype_level0`, `final_lineage`, L1 | Palette alias of Xenium lineage | `celltype_level0`, L3 |
| `lineage_bucket` | Not registered | Not registered | `celltype_level01`, L4 |
| `cniche` | Not registered | `CNiche`, L3 | Not used |
| `tniche` | Not registered | `TNiche`, L4 | Not used |

For HCC, historical model head `L1` means the four-class coarse target.
For Xenium, `L1` means lineage. `L2`, `L12`, `L3`, and `L4` map to `fine`,
`intermediate`, `cniche`, and `tniche`, respectively.
For ESCC, all five tiers use the single `codex_escc` scheme. Spatial callers
select L2/L1/L12/L3/L4 with `celltype_col` values `celltype`,
`celltype_level1`, `celltype_level12`, `celltype_level0`, and
`celltype_level01`. ROC callers use the same values through the legacy-named
`ncrt_color_tier` compatibility parameter.

## Dataset IDs and scheme aliases

Canonical dataset IDs are:

- `codex_escc` (aliases: `escc`, `ncrt`; `ncrt` is retained for backward compatibility)
- `xenium_lung` (aliases: `xenium`, `lung`)
- `codex_hcc` (aliases: `codex`, `hcc`)

Supported schemes are:

- `codex_escc` (`ncrt` is a backward-compatible alias)
- `xenium_lung_fine`, `xenium_lung_intermediate`, `xenium_lung_coarse`,
  `xenium_lung_CNiche`, `xenium_lung_TNiche`
- `codex_hcc`, `codex_hcc_fine`, `codex_hcc_intermediate`,
  `codex_hcc_coarse`

| Xenium source/head | Canonical scheme |
|--------------------|------------------|
| `final_CT` / L2 | `xenium_lung_fine` |
| `final_sublineage` / L12 | `xenium_lung_intermediate` |
| `final_lineage` / L1 | `xenium_lung_coarse` |
| `CNiche` / L3 | `xenium_lung_CNiche` |
| `TNiche` / L4 | `xenium_lung_TNiche` |

The resolver is case-insensitive. `xenium`, `xenium_auto`, `xenium_ct`, and
`xenium_lineage` remain backward-compatible aliases; new code should use the
explicit canonical schemes above.

### `pan_organ` (optional, unified caller API)

Most plotting APIs accept either explicit `*_color_scheme` values **or** an
optional `pan_organ` selector (`codex_hcc`, `xenium_lung`, `codex_escc`). When
`pan_organ` is set, callers can omit per-tier scheme strings and let the
package choose the canonical scheme for L2/L1/L12/L3/L4 based on the plotting
context (e.g. `celltype_col` or `ncrt_color_tier`).

Clinical boxplots use the same selector for x-axis group order:

```python
from plotting_palettes import clinical_group_order, clinical_columns_for_pan_organ

clinical_group_order("xenium_lung")
# Status: Control, Disease; Sample_Affect_Pairing: Unaffected → More_Affected

clinical_group_order("codex_hcc")
# Response: Responder, Non_Responder; diagnosis: Pre, Post

clinical_columns_for_pan_organ("codex_hcc")
# ("Response", "diagnosis", "treatment")
```

`histology_derived_niche_index.test_q4_by_clinical_groups` /
`plot_q4_clinical_comparison` and `uni_label_cv_helpers.analyze_stardist_macro_auroc_by_clinical`
accept `pan_organ=` so HCC and Xenium do not share one mixed order dict.

Precedence is:

- explicit `color_overrides` / `class_color_overrides`
- explicit scheme argument (`spatial_color_scheme` / `roc_color_scheme`)
- `pan_organ` + tier/head context
- legacy default

## Plotting examples

Add `code/Hist2Pheno_pkg` to `PYTHONPATH` before importing these modules.

### Resolve a palette

```python
from plotting_palettes import resolve_palette

colors = resolve_palette(
    ["CD4 T cells", "CD8 T cells", "B cells"],
    scheme="codex_hcc_fine",
)
# Values are matplotlib-compatible RGBA tuples.
```

You can also route through `pan_organ` when you do not want to specify a
scheme explicitly:

```python
colors = resolve_palette(
    ["AT1", "AT2"],
    pan_organ="xenium_lung",
    tier="fine",
)
```

For an incomplete fine-label subset, pass the full training order through
`canonical_labels` to keep fallback colors stable across samples:

```python
colors = resolve_palette(
    labels_present,
    dataset="xenium_lung",
    tier="fine",
    canonical_labels=global_l2_class_names,
)
```

### Spatial plotting

```python
from plot import plot_celltype_spatial_distribution

plot_celltype_spatial_distribution(
    cells,
    x_col="X_pix_HE",
    y_col="Y_pix_HE",
    celltype_col="final_CT",
    spatial_color_scheme="xenium_lung_fine",
    save_path="figures/final_ct.jpg",
    show=False,
)
```

Use `codex_hcc_fine`, `codex_hcc_intermediate`, or `codex_hcc_coarse` for HCC
tiers. `plot_tier_spatial_distribution` is available for encoded prediction
arrays and optional ground-truth panels.

### Generic bars and composition plots

```python
from plotting_palettes import resolve_palette
from plotting_utils import (
    celltype_counts_from_df,
    plot_celltype_count_bars,
    plot_celltype_proportions_stacked,
)

counts = celltype_counts_from_df(cells, celltype_col="final_CT")
colors = resolve_palette(counts.index, dataset="xenium_lung", tier="fine")
plot_celltype_count_bars(
    counts,
    celltype_colors=colors,
    save_path="figures/cell_counts.pdf",
    show=False,
)

# Rows are cell types; columns are samples; values are counts.
plot_celltype_proportions_stacked(
    counts_by_sample,
    celltype_colors=resolve_palette(
        counts_by_sample.index, dataset="xenium_lung", tier="fine"
    ),
    save_path="figures/composition.pdf",
    show=False,
)
```

## Compatibility wrappers

Legacy dataset modules may re-export package constants/functions or wrap them
to preserve established names, signatures, and dataset-specific defaults.
These wrappers should be thin: new dataset-neutral behavior belongs in the
package, while path lookup, I/O, CLI behavior, and report composition stay in
the dataset module. New code should import package APIs directly.

Compatibility re-exports are not a second source of truth. Palette values,
semantic-tier rules, and generic plotting implementations must be changed only
in `plotting_palettes.py` or `plotting_utils.py`.

## Registering a future dataset

1. Add its canonical ID and aliases to `_DATASET_ALIASES` in
   `plotting_palettes.py`.
2. Register optional user-facing schemes and default tiers in
   `_SCHEME_DEFAULTS`.
3. Add canonical palette mappings to `get_palette()` and tier inference order
   to `_infer_tier()`.
4. Extend `normalize_tier()` only for genuine legacy column/head aliases.
5. Exercise `resolve_palette()` with every supported tier, unknown labels,
   deterministic fine-label fallback, and explicit overrides.
6. Keep all palette data in the package. The dataset folder may import and
   re-export it, but the package must never import the dataset folder.

This keeps dependency direction stable and allows a new dataset to use shared
plots without coupling package imports to its local directory layout.
