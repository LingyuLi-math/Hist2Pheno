## 2026.08.14 LLY: CODEX HCC counterpart of Xenium_lung/histology_derived_niche_index.py
"""Histology-derived niche indices for CODEX HCC (s4769).

This module mirrors the Xenium lung TLS / ARI / FRI workflow, but uses HCC
Level-2 ``final_CT`` names and HCC clinical keys (``matched_he``, Response,
diagnosis, treatment).

Index mapping (lung → HCC)
--------------------------
TLS / TLS_spatial
    Same biology: B + T + DC adaptive-immune aggregate.
SRI (stromal remodeling index)  ← lung ARI
    Fibroblasts / epithelium. Broad stromal replacement of epithelial mass.
TNI (TME niche index)  ← lung FRI
    Geometric co-localization of fibroblasts × M2-like macrophages × endothelium,
    relative to local epithelium. Active immunosuppressive / angiogenic niche.

Spatial scores use ``spatial_HE`` pixel coordinates from
``{matched_he}_matched_features_stardist.h5ad``. Neighborhood radius is specified
in μm and converted with ``DEFAULT_HCC_UM_PER_HE_PIXEL`` (override if the HE
scale is known more precisely).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_CODEX_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CODEX_DIR.parent / "Hist2Pheno_pkg"
_XENIUM_DIR = _CODEX_DIR.parent / "Xenium_lung"
for _p in (_CODEX_DIR, _PKG_DIR, _XENIUM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import histology_derived_niche_index as hdni  # noqa: E402
from s4769_plot import load_hcc_clinical_info  # noqa: E402

PAN_ORGAN = "codex_hcc"

DEFAULT_CODEX_HCC_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer"
)
S4769_SAMPLE = "s4769"
DEFAULT_DATA_ROOT = DEFAULT_CODEX_HCC_DIR / S4769_SAMPLE
DEFAULT_HE_ROOT = DEFAULT_DATA_ROOT / "HE"
DEFAULT_STARDIST_ROOT = DEFAULT_DATA_ROOT / "result_all_spatial" / "stardist"
DEFAULT_CLINICAL_XLSX = DEFAULT_HE_ROOT / "s4769_he_mapping_updated_Visium.xlsx"
DEFAULT_CLINICAL_SHEET = "Clinical_info"

AUROC_CSV_NAME = "validation_external_stardist_matched_AUROC.csv"
AUROC_CLASS_NAMES_CSV_NAME = "validation_external_stardist_matched_AUROC_class_names.csv"

# HE pixel scale is not stored in the matched h5ad; 0.5 μm/px is a 20×-like default.
# Override per call when a calibrated scale is available.
DEFAULT_HCC_UM_PER_HE_PIXEL = 0.5
DEFAULT_HCC_SPATIAL_RADIUS_UM = 55.0
DEFAULT_HOTSPOT_PERCENTILE = 95.0
DEFAULT_DEMO_SAMPLE = "awy-98938_aligned_0d535a74"

################################################################################
# HCC Level-2 compartments (must match AUROC class_names / final_CT)
################################################################################
TLS_B_CELLS = ("B cells",)
TLS_T_CELLS = ("CD4 T cells", "CD8 T cells")
TLS_DC_CELLS = ("Dendritic cells",)
TLS_CELL_TYPES = TLS_B_CELLS + TLS_T_CELLS + TLS_DC_CELLS

EPITHELIUM_CELLS = ("Epithelium (INOS+)", "Epithelium (INOS-)")
FIBROBLAST_CELLS = ("Fibroblasts",)
ENDOTHELIAL_CELLS = ("Endothelial cells",)
M2_MACROPHAGE_CELLS = ("Macrophages M2-like",)
MACROPHAGE_CELLS = ("Macrophages", "Macrophages M2-like")

HCC_CLINICAL_COLUMNS = ("Response", "diagnosis", "treatment")

HCC_QUADRANT_LABELS = {
    "Q1": "Normal / epithelial-dominant",
    "Q2": "Stromal remodeling (epithelial replacement by fibroblasts)",
    "Q3": "Immune / angiogenic activation without extensive remodeling",
    "Q4": "Active TME niche (fibroblasts + M2-like + endothelium)",
}
HCC_QUADRANT_PLOT_LABELS = {
    "Q1": "Q1: Normal / epi-dominant",
    "Q2": "Q2: Stromal remodeling (SRI\u2191)",
    "Q3": "Q3: Activation w/o remodeling",
    "Q4": "Q4: Active TME niche (SRI\u2191+TNI\u2191)",
}


@dataclass(frozen=True)
class HccSpatialPaths:
    sample: str
    auroc_csv: Path
    class_names_csv: Path
    stardist_h5ad: Path


def load_stardist_auroc_csv(auroc_csv, class_names_csv=None):
    """Load matched StarDist AUROC table (wrapper around the shared helper)."""
    return hdni.load_stardist_auroc_csv(auroc_csv, class_names_csv)


def summarize_tls_hotspots(df: pd.DataFrame, **kwargs) -> dict:
    """Count / centroid of ``tls_candidate`` cells (shared helper)."""
    return hdni.summarize_tls_hotspots(df, **kwargs)


def missing_index_cell_types(class_names: Sequence[str]) -> list[str]:
    """Level-2 names required by TLS / SRI / TNI that are absent from ``class_names``."""
    needed = set(
        TLS_CELL_TYPES
        + EPITHELIUM_CELLS
        + FIBROBLAST_CELLS
        + ENDOTHELIAL_CELLS
        + M2_MACROPHAGE_CELLS
    )
    present = {str(name) for name in class_names}
    return sorted(needed - present)


def hcc_spatial_paths(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
) -> HccSpatialPaths:
    """Resolve matched AUROC + StarDist h5ad paths for one ``matched_he`` region."""
    sample = str(sample)
    star = Path(stardist_root) / sample
    return HccSpatialPaths(
        sample=sample,
        auroc_csv=star / AUROC_CSV_NAME,
        class_names_csv=star / AUROC_CLASS_NAMES_CSV_NAME,
        stardist_h5ad=Path(he_root) / sample / f"{sample}_matched_features_stardist.h5ad",
    )


def discover_hcc_spatial_samples(
    clinical: pd.DataFrame,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    sample_col: str = "matched_he",
    require_h5ad: bool = True,
) -> list[str]:
    """Return ``matched_he`` keys that have AUROC (+ optional matched h5ad)."""
    samples = []
    for sample in clinical[sample_col].dropna().astype(str).unique():
        paths = hcc_spatial_paths(sample, stardist_root=stardist_root, he_root=he_root)
        if not paths.auroc_csv.is_file() or not paths.class_names_csv.is_file():
            continue
        if require_h5ad and not paths.stardist_h5ad.is_file():
            continue
        samples.append(sample)
    return sorted(samples)


def load_hcc_annotation() -> pd.DataFrame:
    """Load HCC Level2→Level1→Level0 hierarchy (Celltype sheet)."""
    from s4769_img_cell_mapping import load_codex_celltype_hierarchy

    return load_codex_celltype_hierarchy()


def index_formula_table() -> pd.DataFrame:
    """Notebook-facing summary of HCC index definitions."""
    return pd.DataFrame(
        [
            {
                "index": "TLS",
                "lung_analog": "TLS",
                "formula": "sum(B cells + CD4 T + CD8 T + DC)",
            },
            {
                "index": "TLS_spatial",
                "lung_analog": "TLS_spatial",
                "formula": "(B_local × T_local × DC_local)^(1/3) within radius R μm",
            },
            {
                "index": "SRI_ratio",
                "lung_analog": "ARI_ratio",
                "formula": "Fibroblasts / (Epithelium INOS+ + INOS-)",
            },
            {
                "index": "SRI_spatial",
                "lung_analog": "ARI_spatial",
                "formula": "Fib_local / Epi_local within radius R μm",
            },
            {
                "index": "TNI_ratio",
                "lung_analog": "FRI_ratio",
                "formula": "(Fibroblasts + M2-like macrophages + Endothelial) / epithelium",
            },
            {
                "index": "TNI_spatial",
                "lung_analog": "FRI_spatial",
                "formula": "(Fib_local × M2_local × Endo_local)^(1/3) / Epi_local",
            },
        ]
    )


################################################################################
# Abundance + spatial scores
################################################################################
def _compartment(probs, class_names, cell_types) -> np.ndarray:
    return hdni.compartment_prob_per_cell(probs, class_names, cell_types)


def compute_tls_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Per-cell TLS abundance = sum of B + T + DC softmax mass."""
    return hdni.sum_prob_columns(
        probs, hdni.prob_columns_for_cell_types(class_names, TLS_CELL_TYPES)
    )


def compute_sri_ratio_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Per-cell stromal remodeling ratio: fibroblasts / epithelium."""
    numer = _compartment(probs, class_names, FIBROBLAST_CELLS)
    denom = _compartment(probs, class_names, EPITHELIUM_CELLS)
    return hdni.safe_ratio(numer, denom)


def compute_tni_ratio_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Per-cell TME-niche ratio: (fibroblast + M2 + endothelial) / epithelium."""
    numer = (
        _compartment(probs, class_names, FIBROBLAST_CELLS)
        + _compartment(probs, class_names, M2_MACROPHAGE_CELLS)
        + _compartment(probs, class_names, ENDOTHELIAL_CELLS)
    )
    denom = _compartment(probs, class_names, EPITHELIUM_CELLS)
    return hdni.safe_ratio(numer, denom)


def compute_spatial_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HCC TLS_spatial = geometric mean of local B / T / DC mass."""
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    b_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, TLS_B_CELLS), coords, radius_px
    )
    t_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, TLS_T_CELLS), coords, radius_px
    )
    dc_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, TLS_DC_CELLS), coords, radius_px
    )
    spatial = np.power(
        np.maximum(b_loc, eps) * np.maximum(t_loc, eps) * np.maximum(dc_loc, eps),
        1.0 / 3.0,
    )
    return spatial, b_loc, t_loc, dc_loc


def compute_spatial_sri_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SRI_spatial = Fib_local / Epi_local."""
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    fib_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, FIBROBLAST_CELLS), coords, radius_px
    )
    epi_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, EPITHELIUM_CELLS), coords, radius_px
    )
    return hdni.safe_ratio(fib_loc, epi_loc), fib_loc, epi_loc


def compute_spatial_tni_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TNI_spatial = (Fib × M2 × Endo)^(1/3) / Epi_local."""
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    fib_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, FIBROBLAST_CELLS), coords, radius_px
    )
    m2_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, M2_MACROPHAGE_CELLS), coords, radius_px
    )
    endo_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, ENDOTHELIAL_CELLS), coords, radius_px
    )
    epi_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, EPITHELIUM_CELLS), coords, radius_px
    )
    geomean = np.power(
        np.maximum(fib_loc, eps) * np.maximum(m2_loc, eps) * np.maximum(endo_loc, eps),
        1.0 / 3.0,
    )
    return hdni.safe_ratio(geomean, epi_loc), fib_loc, m2_loc, endo_loc, epi_loc


def add_hcc_abundance_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """Add per-cell TLS / SRI / TNI abundance columns (no spatial neighborhood)."""
    out = df.copy()
    out[f"{prefix}TLS"] = compute_tls_per_cell(probs, class_names)
    out[f"{prefix}SRI_ratio"] = compute_sri_ratio_per_cell(probs, class_names)
    out[f"{prefix}TNI_ratio"] = compute_tni_ratio_per_cell(probs, class_names)
    return out


def add_hcc_spatial_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_HOTSPOT_PERCENTILE,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """Add spatial TLS / SRI / TNI columns and 95th-percentile hotspot flags."""
    out = df.copy()
    tls_s, b_loc, t_loc, dc_loc = compute_spatial_tls_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    sri_s, fib_sri, epi_sri = compute_spatial_sri_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    tni_s, fib_tni, m2_loc, endo_loc, epi_tni = compute_spatial_tni_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    out[f"{prefix}TLS_spatial"] = tls_s
    out[f"{prefix}TLS_B_local"] = b_loc
    out[f"{prefix}TLS_T_local"] = t_loc
    out[f"{prefix}TLS_DC_local"] = dc_loc
    tni_num = np.power(
        np.maximum(fib_tni, 1e-8) * np.maximum(m2_loc, 1e-8) * np.maximum(endo_loc, 1e-8),
        1.0 / 3.0,
    )
    out[f"{prefix}SRI_spatial"] = sri_s
    out[f"{prefix}TNI_spatial"] = tni_s
    out[f"{prefix}Epi_local"] = epi_sri
    out[f"{prefix}Fib_local"] = fib_sri
    out[f"{prefix}M2_local"] = m2_loc
    out[f"{prefix}Endo_local"] = endo_loc
    out[f"{prefix}TNI_numerator_local"] = tni_num
    out[f"{prefix}TNI_Fib_local"] = fib_tni
    out[f"{prefix}TNI_M2_local"] = m2_loc
    out[f"{prefix}TNI_Endo_local"] = endo_loc
    out["tls_spatial_radius_um"] = radius_um
    out["tls_um_per_he_pixel"] = um_per_pixel

    tls_flag, tls_thr = hdni.detect_tls_hotspot_cells(tls_s, coords, percentile=hotspot_percentile)
    sri_flag, sri_thr = hdni.detect_tls_hotspot_cells(sri_s, coords, percentile=hotspot_percentile)
    tni_flag, tni_thr = hdni.detect_tls_hotspot_cells(tni_s, coords, percentile=hotspot_percentile)
    out["tls_candidate"] = tls_flag
    out["sri_candidate"] = sri_flag
    out["tni_candidate"] = tni_flag
    out["tls_spatial_threshold"] = tls_thr
    out["sri_spatial_threshold"] = sri_thr
    out["tni_spatial_threshold"] = tni_thr
    return out


def summarize_indices_by_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Mean abundance indices for one region."""
    cols = [c for c in ("idx_TLS", "idx_SRI_ratio", "idx_TNI_ratio") if c in df.columns]
    row = {"Sample": sample, "n_cells": int(len(df))}
    for col in cols:
        row[col] = float(df[col].mean())
    return pd.DataFrame([row])


################################################################################
# Load one region (AUROC + spatial_HE)
################################################################################
def load_sample_with_spatial_indices(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, HccSpatialPaths, float]:
    """Load matched AUROC, attach HE coords, and compute HCC niche indices."""
    paths = hcc_spatial_paths(sample, stardist_root=stardist_root, he_root=he_root)
    if not paths.auroc_csv.is_file():
        raise FileNotFoundError(paths.auroc_csv)
    if not paths.stardist_h5ad.is_file():
        raise FileNotFoundError(paths.stardist_h5ad)

    df, class_names, probs = hdni.load_stardist_auroc_csv(
        paths.auroc_csv, paths.class_names_csv
    )
    missing = missing_index_cell_types(class_names)
    if missing:
        raise KeyError(
            f"{sample}: AUROC class_names missing index members {missing}. "
            "Update TLS/SRI/TNI tuples in hcc_histology_derived_niche_index.py."
        )
    coords_raw, cell_ids = hdni.load_spatial_coords_from_h5ad(paths.stardist_h5ad)
    df, coords = hdni.align_auroc_df_with_h5ad(df, coords_raw, cell_ids)
    prob_cols = [f"prob_{j}" for j in range(len(class_names))]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    df = add_hcc_abundance_indices(df, probs, class_names)
    df = add_hcc_spatial_indices(
        df,
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
    )
    return df, class_names, probs, coords, paths, um_per_pixel


def build_hcc_sample_dict(
    samples: Sequence[str],
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
) -> dict[str, pd.DataFrame]:
    """Load spatial SRI/TNI tables for a list of ``matched_he`` regions."""
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        df, *_ = load_sample_with_spatial_indices(
            sample,
            stardist_root=stardist_root,
            he_root=he_root,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
        )
        out[str(sample)] = df
        print(f"  loaded {sample}: {len(df):,} cells", flush=True)
    return out


################################################################################
# Plots + cohort (reuse lung helpers with HCC column names)
################################################################################
def plot_tls_abundance_vs_spatial(df: pd.DataFrame, sample: str, **kwargs):
    """TLS abundance vs spatial co-localization (same helper as Xenium)."""
    return hdni.plot_tls_abundance_vs_spatial(df, sample, **kwargs)


def plot_sri_tni_abundance_vs_spatial(df: pd.DataFrame, sample: str, **kwargs):
    """2×2 maps: SRI (stromal remodeling) then TNI (active TME niche)."""
    radius_um = kwargs.get("radius_um")
    if radius_um is None and "tls_spatial_radius_um" in df.columns:
        radius_um = float(df["tls_spatial_radius_um"].iloc[0])
    kwargs.setdefault("ari_abundance_col", "idx_SRI_ratio")
    kwargs.setdefault("ari_spatial_col", "idx_SRI_spatial")
    kwargs.setdefault("fri_abundance_col", "idx_TNI_ratio")
    kwargs.setdefault("fri_spatial_col", "idx_TNI_spatial")
    kwargs.setdefault("ari_candidate_col", "sri_candidate")
    kwargs.setdefault("fri_candidate_col", "tni_candidate")
    fig, axes = hdni.plot_ari_fri_abundance_vs_spatial(df, sample, **kwargs)
    rtxt = f"{float(radius_um):g}" if radius_um is not None else "?"
    axes[0, 0].set_title(f"{sample}: SRI abundance — stromal remodeling (idx_SRI_ratio)")
    axes[0, 1].set_title(f"{sample}: SRI spatial (R={rtxt} μm)")
    axes[1, 0].set_title(f"{sample}: TNI abundance — TME niche (idx_TNI_ratio)")
    axes[1, 1].set_title(f"{sample}: TNI spatial (R={rtxt} μm)")
    return fig, axes


@contextmanager
def _hcc_quadrant_label_context():
    old_labels = hdni.ARI_FRI_QUADRANT_LABELS
    old_plot = hdni.ARI_FRI_QUADRANT_PLOT_LABELS
    hdni.ARI_FRI_QUADRANT_LABELS = HCC_QUADRANT_LABELS
    hdni.ARI_FRI_QUADRANT_PLOT_LABELS = HCC_QUADRANT_PLOT_LABELS
    try:
        yield
    finally:
        hdni.ARI_FRI_QUADRANT_LABELS = old_labels
        hdni.ARI_FRI_QUADRANT_PLOT_LABELS = old_plot


def _relabel_sri_tni_hierarchical_figure(fig, sample_name: str) -> None:
    """Replace leftover lung ARI/FRI axis text on the hierarchical figure."""
    axes = getattr(fig, "axes", None)
    if axes:
        axes[0].set_xlabel("Spatial SRI (stromal remodeling)")
        axes[0].set_ylabel("Spatial TNI (active TME niche)")
    fig.suptitle(
        f"{sample_name} — Hierarchical remodeling: stromal SRI \u2192 active TME niche\n"
        f"Normal (Q1) \u2192 remodeling (Q2, SRI\u2191) \u2192 active niche (Q4, SRI\u2191+TNI\u2191)",
        y=1.02,
        fontsize=11,
    )


def analyze_sri_tni_relationship(df: pd.DataFrame, sample_name: str, **kwargs) -> dict:
    """Hierarchical SRI → TNI (Normal → stromal remodeling → active TME niche)."""
    kwargs.setdefault("ari_col", "idx_SRI_spatial")
    kwargs.setdefault("fri_col", "idx_TNI_spatial")
    with _hcc_quadrant_label_context():
        result = hdni.analyze_ari_fri_relationship(df, sample_name, **kwargs)
    result["SRI_median"] = result.get("ARI_median")
    result["TNI_median"] = result.get("FRI_median")
    quad = result.get("quadrant_table")
    if isinstance(quad, pd.DataFrame) and "quadrant" in quad.columns:
        quad = quad.copy()
        quad["label"] = quad["quadrant"].map(HCC_QUADRANT_LABELS)
        result["quadrant_table"] = quad
    figures = result.get("figures") or {}
    for fig in figures.values():
        if fig is not None:
            _relabel_sri_tni_hierarchical_figure(fig, sample_name)
    return result


def batch_analyze_sri_tni_cohort(
    sample_dict: dict[str, pd.DataFrame],
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Cohort Global Q4 % and active niche burden on SRI/TNI."""
    kwargs.setdefault("ari_col", "idx_SRI_spatial")
    kwargs.setdefault("fri_col", "idx_TNI_spatial")
    batch, meta = hdni.batch_analyze_ari_fri_cohort(sample_dict, **kwargs)
    batch = batch.rename(columns={"ARI_median": "SRI_median", "FRI_median": "TNI_median"})
    meta["sri_threshold_global"] = meta.get("ari_threshold_global")
    meta["tni_threshold_global"] = meta.get("fri_threshold_global")
    return batch, meta


def merge_batch_with_hcc_clinical(
    batch_df: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str = "matched_he",
) -> pd.DataFrame:
    """Join per-region SRI/TNI summary to HCC clinical fields."""
    out = batch_df.copy()
    out[sample_col] = out[sample_col].astype(str)
    clin = clinical.copy()
    clin[clinical_key] = clin[clinical_key].astype(str)
    clin = clin.drop_duplicates(clinical_key)
    merged = out.merge(clin, left_on=sample_col, right_on=clinical_key, how="left")
    for col in HCC_CLINICAL_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged


def test_hcc_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mann–Whitney / Kruskal–Wallis (or Welch / ANOVA) with HCC group order."""
    from uni_label_cv_helpers import test_metric_by_clinical_groups

    return test_metric_by_clinical_groups(
        df,
        metric_col=metric_col,
        clinical_columns=tuple(clinical_columns or HCC_CLINICAL_COLUMNS),
        pan_organ=PAN_ORGAN,
        method=method,
        alternative=alternative,
    )


def plot_hcc_clinical_comparison(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    q4_col: str,
    **kwargs,
):
    """Boxplots of a sample-level HCC niche metric by clinical group."""
    kwargs.setdefault("clinical_columns", HCC_CLINICAL_COLUMNS)
    kwargs.setdefault("pan_organ", PAN_ORGAN)
    kwargs.setdefault("sample_col", "Sample")
    return hdni.plot_q4_clinical_comparison(df, stats_summary, q4_col=q4_col, **kwargs)


################################################################################
# 2026.08.14, add source decomposition: SRI / TNI / Epi vs patient-level Response   
################################################################################
HCC_SOURCE_METRIC_LABELS: dict[str, str] = {
    "SRI_mean": "SRI  (Fib / Epi)",
    "TNI_mean": "TNI  (geomean / Epi)",
    "Epi_local_mean": "Epi_local",
    "TNI_numerator_mean": "TNI numerator  (Fib × M2 × Endo)",
    "Fib_local_mean": "Fib_local",
    "M2_local_mean": "M2_local",
    "Endo_local_mean": "Endo_local",
    "active_niche_burden": "Active TME niche burden",
}

HCC_RESPONSE_SOURCE_METRICS: tuple[str, ...] = (
    "SRI_mean",
    "TNI_mean",
    "Epi_local_mean",
    "TNI_numerator_mean",
)


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _column_or_none(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def epi_local_values(df: pd.DataFrame, *, prefix: str = "idx_") -> np.ndarray:
    """Neighborhood epithelium mass; reconstruct from Fib/SRI if the column is absent."""
    epi_col = _column_or_none(df, f"{prefix}Epi_local")
    if epi_col is not None:
        return pd.to_numeric(df[epi_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    sri_col = _column_or_none(df, f"{prefix}SRI_spatial")
    if fib_col is None or sri_col is None:
        raise KeyError(
            "Need idx_Epi_local, or idx_SRI_spatial plus Fib_local, to recover epithelium."
        )
    fib = pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64)
    sri = pd.to_numeric(df[sri_col], errors="coerce").to_numpy(dtype=np.float64)
    return fib / np.maximum(sri, 1e-8)


def tni_numerator_values(df: pd.DataFrame, *, prefix: str = "idx_") -> np.ndarray:
    """(Fib × M2 × Endo)^(1/3) without dividing by epithelium."""
    num_col = _column_or_none(df, f"{prefix}TNI_numerator_local")
    if num_col is not None:
        return pd.to_numeric(df[num_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    m2_col = _column_or_none(df, f"{prefix}M2_local", f"{prefix}TNI_M2_local")
    endo_col = _column_or_none(df, f"{prefix}Endo_local", f"{prefix}TNI_Endo_local")
    if fib_col is None or m2_col is None or endo_col is None:
        raise KeyError("Need Fib/M2/Endo local columns to reconstruct the TNI numerator.")
    fib = np.maximum(pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    m2 = np.maximum(pd.to_numeric(df[m2_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    endo = np.maximum(pd.to_numeric(df[endo_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    return np.power(fib * m2 * endo, 1.0 / 3.0)


def summarize_hcc_source_metrics(
    sample_dict: dict[str, pd.DataFrame],
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """Per-region means that separate ratio inflation (low Epi) from niche colocalization."""
    rows = []
    for sample, df in sample_dict.items():
        sri_col = _column_or_none(df, f"{prefix}SRI_spatial")
        tni_col = _column_or_none(df, f"{prefix}TNI_spatial")
        if sri_col is None or tni_col is None:
            raise KeyError(f"{sample}: missing SRI/TNI spatial columns.")
        fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
        m2_col = _column_or_none(df, f"{prefix}M2_local", f"{prefix}TNI_M2_local")
        endo_col = _column_or_none(df, f"{prefix}Endo_local", f"{prefix}TNI_Endo_local")
        row = {
            "Sample": str(sample),
            "n_cells": int(len(df)),
            "SRI_mean": _finite_mean(df[sri_col]),
            "TNI_mean": _finite_mean(df[tni_col]),
            "Epi_local_mean": _finite_mean(epi_local_values(df, prefix=prefix)),
            "TNI_numerator_mean": _finite_mean(tni_numerator_values(df, prefix=prefix)),
        }
        if fib_col is not None:
            row["Fib_local_mean"] = _finite_mean(df[fib_col])
        if m2_col is not None:
            row["M2_local_mean"] = _finite_mean(df[m2_col])
        if endo_col is not None:
            row["Endo_local_mean"] = _finite_mean(df[endo_col])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_hcc_metrics_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    metric_cols: Sequence[str] | None = None,
    clinical_columns: Sequence[str] = ("Response", "diagnosis", "treatment"),
) -> pd.DataFrame:
    """Unweighted mean of region-level metrics within each patient (equal slide weight)."""
    if patient_col not in df.columns:
        raise KeyError(f"Missing {patient_col!r}; join clinical metadata first.")
    work = df.copy()
    work[patient_col] = work[patient_col].astype(str)
    if metric_cols is None:
        metric_cols = [
            c
            for c in (
                "active_niche_burden",
                "Q4_global_percent",
                "SRI_mean",
                "TNI_mean",
                "SRI_median",
                "TNI_median",
                "Epi_local_mean",
                "TNI_numerator_mean",
                "Fib_local_mean",
                "M2_local_mean",
                "Endo_local_mean",
            )
            if c in work.columns
        ]
    rows = []
    for pid, sub in work.groupby(patient_col, sort=True):
        row: dict = {
            patient_col: str(pid),
            "n_regions": int(len(sub)),
            "n_cells": int(pd.to_numeric(sub["n_cells"], errors="coerce").sum())
            if "n_cells" in sub.columns
            else int(len(sub)),
        }
        for col in metric_cols:
            row[col] = _finite_mean(sub[col])
        for col in clinical_columns:
            if col not in sub.columns:
                continue
            vals = sub[col].astype(str).unique().tolist()
            row[col] = vals[0]
            if len(vals) > 1:
                print(
                    f"  warn: {patient_col}={pid} has mixed {col}={vals}; using {vals[0]}",
                    flush=True,
                )
        rows.append(row)
    return pd.DataFrame(rows)


def test_hcc_response_source_metrics(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = HCC_RESPONSE_SOURCE_METRICS,
    alternative: str = "two-sided",
    method: str = "rank",
) -> pd.DataFrame:
    """One Mann–Whitney / Kruskal row per metric, Response only."""
    frames = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        stats = test_hcc_clinical_groups(
            df,
            metric_col=col,
            alternative=alternative,
            method=method,
            clinical_columns=("Response",),
        )
        frames.append(stats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_hcc_response_source_grid(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = HCC_RESPONSE_SOURCE_METRICS,
    sample_col: str | None = "patient_id",
    group_col: str = "Response",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    point_size: float = 36,
):
    """Side-by-side Response boxplots for SRI, TNI, Epi_local, and TNI numerator."""
    import matplotlib.pyplot as plt
    from plotting_palettes import clinical_group_order

    cols = [c for c in metric_cols if c in df.columns]
    if not cols:
        raise KeyError(f"None of {list(metric_cols)} found in dataframe.")
    n = len(cols)
    legend_w = 1.8 if sample_col and sample_col in df.columns else 0.0
    if figsize is None:
        figsize = (4.2 * n + legend_w, 5.2)

    order_map = clinical_group_order(PAN_ORGAN)
    color_by_sample = bool(sample_col) and sample_col in df.columns
    sample_colors: dict[str, tuple] = {}
    sample_order: list[str] = []
    if color_by_sample:
        sample_order = sorted(df[sample_col].astype(str).unique().tolist())
        sample_colors = hdni._build_sample_color_map(sample_order)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, layout="constrained")
    rng = np.random.default_rng(0)
    for j, col in enumerate(cols):
        ax = axes[0, j]
        sub = df.dropna(subset=[group_col, col]).copy()
        present = sub[group_col].astype(str).unique().tolist()
        preset = [g for g in order_map.get(group_col, ()) if g in present]
        rest = sorted(g for g in present if g not in preset)
        order = preset + rest
        data = [sub.loc[sub[group_col].astype(str) == g, col].to_numpy(float) for g in order]
        positions = np.arange(1, len(order) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set(
                facecolor=hdni.COMPLETE_COHORT_BOX_FACE,
                alpha=0.85,
                edgecolor=hdni.COMPLETE_COHORT_BOX_EDGE,
            )
        for pos, g in zip(positions, order):
            gdf = sub.loc[sub[group_col].astype(str) == g]
            n_pts = len(gdf)
            if n_pts == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=n_pts)
            y_vals = gdf[col].astype(float).to_numpy()
            x_vals = np.full(n_pts, pos) + jitter
            if color_by_sample:
                pt_colors = [sample_colors[str(sid)] for sid in gdf[sample_col].astype(str)]
                ax.scatter(
                    x_vals, y_vals, s=point_size, c=pt_colors,
                    edgecolors="0.35", linewidths=0.4, alpha=0.95, zorder=3,
                )
            else:
                ax.scatter(
                    x_vals, y_vals, s=point_size, c="#b2182b",
                    edgecolors="0.35", linewidths=0.4, alpha=0.95, zorder=3,
                )

        flat = np.concatenate([d for d in data if len(d)]) if data else np.array([0.0])
        y_min, y_max = float(np.min(flat)), float(np.max(flat))
        y_span = max(y_max - y_min, 1e-9)
        pad_top = y_span * 0.28
        ax.set_ylim(y_min - y_span * 0.06, y_max + pad_top)

        hits = stats_summary
        if "metric_col" in stats_summary.columns:
            hits = stats_summary.loc[stats_summary["metric_col"] == col]
        if "clinical_variable" in hits.columns:
            hits = hits.loc[hits["clinical_variable"] == group_col]
        if len(hits) and np.isfinite(float(hits.iloc[0]["p_value"])):
            p = float(hits.iloc[0]["p_value"])
            test = str(hits.iloc[0].get("test", ""))
            stars = hdni.pvalue_to_stars(p)
            ax.text(
                float(np.mean(positions)),
                y_max + pad_top * 0.52,
                f"{test}\np={p:.3g} ({stars})",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
                clip_on=False,
                zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
        ax.set_title(HCC_SOURCE_METRIC_LABELS.get(col, col), fontsize=11, pad=6)
        ax.set_ylabel("mean per region or patient" if j == 0 else "")
        ax.grid(axis="y", alpha=0.25)

    if color_by_sample and sample_order:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=sample_colors[sid], markeredgecolor="0.35",
                markersize=5, label=sid,
            )
            for sid in sample_order
        ]
        fig.legend(
            handles, sample_order, loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=7, title=sample_col, frameon=False, borderaxespad=0.0,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig, axes


__all__ = [
    "PAN_ORGAN",
    "DEFAULT_DEMO_SAMPLE",
    "HccSpatialPaths",
    "load_hcc_clinical_info",
    "load_hcc_annotation",
    "index_formula_table",
    "missing_index_cell_types",
    "hcc_spatial_paths",
    "discover_hcc_spatial_samples",
    "load_stardist_auroc_csv",
    "summarize_tls_hotspots",
    "load_sample_with_spatial_indices",
    "build_hcc_sample_dict",
    "add_hcc_abundance_indices",
    "add_hcc_spatial_indices",
    "summarize_indices_by_sample",
    "plot_tls_abundance_vs_spatial",
    "plot_sri_tni_abundance_vs_spatial",
    "analyze_sri_tni_relationship",
    "batch_analyze_sri_tni_cohort",
    "merge_batch_with_hcc_clinical",
    "test_hcc_clinical_groups",
    "plot_hcc_clinical_comparison",
    "summarize_hcc_source_metrics",
    "aggregate_hcc_metrics_by_patient",
    "test_hcc_response_source_metrics",
    "plot_hcc_response_source_grid",
]
