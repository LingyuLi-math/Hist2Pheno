## 2026.08.20 LLY: s1167 PDAC / GIST counterpart of CODEX_hcc/hcc_histology_derived_niche_index.py
"""Histology-derived niche indices for CODEX PDAC and GIST (s1167 TMA).

TLS / SRI / TNI follow the HCC workflow, with Level-2 names from each TMA
hierarchy. There is no immunotherapy Response field: clinical grouping uses
``coverslip`` and ``SAMPLE_LABEL``.

Index mapping (HCC → s1167)
--------------------------
TLS / TLS_spatial
    B + T + DC adaptive-immune aggregate.
SRI (stromal remodeling)
    Fibroblasts (and GIST stromal cells) / epithelial cells.
TNI (TME niche)
    Geometric co-localization of fibroblasts × macrophages × endothelium,
    relative to local epithelium. PDAC/GIST have no M2-like class; the myeloid
    compartment is ``Macrophages`` (GIST also ``Monocytes``).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_PDAC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PDAC_DIR.parent / "Hist2Pheno_pkg"
_XENIUM_DIR = _PDAC_DIR.parent / "Xenium_lung"
for _p in (_PDAC_DIR, _PKG_DIR, _XENIUM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import histology_derived_niche_index as hdni  # noqa: E402
from plotting_palettes import clinical_columns_for_pan_organ  # noqa: E402
from s1167_img_cell_mapping import (  # noqa: E402
    coverslip_from_acq,
    load_s1167_celltype_hierarchy,
    load_s1167_metadata,
)

DEFAULT_S1167_ROOT = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer/s1167"
)
DEFAULT_DATA_ROOT = DEFAULT_S1167_ROOT
DEFAULT_CASES_ROOT = DEFAULT_S1167_ROOT
DEFAULT_STARDIST_ROOT = DEFAULT_S1167_ROOT / "result_all_spatial" / "stardist"
DEFAULT_CLINICAL_XLSX = DEFAULT_S1167_ROOT / "raw_metadata_updated.xlsx"
DEFAULT_CLINICAL_SHEET = "Clinical_info"

AUROC_CSV_NAME = "validation_external_stardist_matched_AUROC.csv"
AUROC_CLASS_NAMES_CSV_NAME = "validation_external_stardist_matched_AUROC_class_names.csv"

DEFAULT_UM_PER_HE_PIXEL = 0.5
DEFAULT_SPATIAL_RADIUS_UM = 55.0
DEFAULT_HOTSPOT_PERCENTILE = 95.0

S1167_QUADRANT_LABELS = {
    "Q1": "Normal / epithelial-dominant",
    "Q2": "Stromal remodeling (epithelial replacement by fibroblasts)",
    "Q3": "Immune / angiogenic activation without extensive remodeling",
    "Q4": "Active TME niche (fibroblasts + macrophages + endothelium)",
}
S1167_QUADRANT_PLOT_LABELS = {
    "Q1": "Q1: Normal / epi-dominant",
    "Q2": "Q2: Stromal remodeling (SRI\u2191)",
    "Q3": "Q3: Activation w/o remodeling",
    "Q4": "Q4: Active TME niche (SRI\u2191+TNI\u2191)",
}

SOURCE_METRIC_LABELS: dict[str, str] = {
    "SRI_mean": "SRI  (Fib / Epi)",
    "TNI_mean": "TNI  (geomean / Epi)",
    "Epi_local_mean": "Epi_local",
    "TNI_numerator_mean": "TNI numerator  (Fib × Mac × Endo)",
    "Fib_local_mean": "Fib_local",
    "M2_local_mean": "Mac_local",
    "Endo_local_mean": "Endo_local",
    "active_niche_burden": "Active TME niche burden",
}
SOURCE_METRICS: tuple[str, ...] = (
    "SRI_mean",
    "TNI_mean",
    "Epi_local_mean",
    "TNI_numerator_mean",
)


@dataclass(frozen=True)
class OrganNicheConfig:
    pan_organ: str
    cohort: str
    demo_sample: str
    tls_b: tuple[str, ...]
    tls_t: tuple[str, ...]
    tls_dc: tuple[str, ...]
    epithelium: tuple[str, ...]
    fibroblast: tuple[str, ...]
    endothelial: tuple[str, ...]
    macrophage: tuple[str, ...]


ORGAN_CONFIGS: dict[str, OrganNicheConfig] = {
    "codex_pdac": OrganNicheConfig(
        pan_organ="codex_pdac",
        cohort="PDAC",
        demo_sample="Charvill-94_c001_v001_r001_reg001",
        tls_b=("B cells",),
        tls_t=("Cytotoxic T cells", "Helper T cells", "Tregs"),
        tls_dc=("Dendritic cells",),
        epithelium=("Epithelial cells",),
        fibroblast=("Fibroblasts",),
        endothelial=("Lymphatic Endothelial cells",),
        macrophage=("Macrophages",),
    ),
    "codex_gist": OrganNicheConfig(
        pan_organ="codex_gist",
        cohort="GIST TMA",
        demo_sample="Charvill-94_c013_v001_r001_reg002",
        tls_b=("B cells",),
        tls_t=("T cells", "Cytotoxic T cells", "Helper T cells", "Tregs"),
        tls_dc=("Dendritic cells", "DCs"),
        epithelium=("Epithelial cells",),
        fibroblast=("Fibroblasts", "Stromal cells"),
        endothelial=("Endothelial cells", "Lymphatic Endothelial cells"),
        macrophage=("Macrophages", "Monocytes"),
    ),
}

_ACTIVE = "codex_pdac"


def configure(pan_organ: str) -> OrganNicheConfig:
    """Bind module-level defaults to PDAC or GIST (call before using wrappers)."""
    global _ACTIVE
    key = str(pan_organ).strip().lower().replace("-", "_")
    if key in ("pdac", "pancreas", "codex_pdac"):
        _ACTIVE = "codex_pdac"
    elif key in ("gist", "gist_tma", "codex_gist"):
        _ACTIVE = "codex_gist"
    else:
        raise KeyError(f"Unknown s1167 pan_organ={pan_organ!r}")
    return ORGAN_CONFIGS[_ACTIVE]


def cfg() -> OrganNicheConfig:
    return ORGAN_CONFIGS[_ACTIVE]


def pan_organ() -> str:
    return cfg().pan_organ


def clinical_columns() -> tuple[str, ...]:
    return clinical_columns_for_pan_organ(cfg().pan_organ) or ("coverslip", "SAMPLE_LABEL")


@dataclass(frozen=True)
class S1167SpatialPaths:
    sample: str
    auroc_csv: Path
    class_names_csv: Path
    stardist_h5ad: Path


def load_stardist_auroc_csv(auroc_csv, class_names_csv=None):
    return hdni.load_stardist_auroc_csv(auroc_csv, class_names_csv)


def summarize_tls_hotspots(df: pd.DataFrame, **kwargs) -> dict:
    return hdni.summarize_tls_hotspots(df, **kwargs)


def _tls_types(c: OrganNicheConfig | None = None) -> tuple[str, ...]:
    c = c or cfg()
    return c.tls_b + c.tls_t + c.tls_dc


def missing_index_cell_types(
    class_names: Sequence[str],
    *,
    organ: OrganNicheConfig | None = None,
) -> list[str]:
    """Compartment names with **no** matching Level-2 class in ``class_names``."""
    c = organ or cfg()
    present = {str(name) for name in class_names}
    groups = {
        "TLS_B": c.tls_b,
        "TLS_T": c.tls_t,
        "TLS_DC": c.tls_dc,
        "epithelium": c.epithelium,
        "fibroblast": c.fibroblast,
        "endothelial": c.endothelial,
        "macrophage": c.macrophage,
    }
    return [name for name, types in groups.items() if not any(t in present for t in types)]


def s1167_spatial_paths(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
) -> S1167SpatialPaths:
    sample = str(sample)
    star = Path(stardist_root) / sample
    return S1167SpatialPaths(
        sample=sample,
        auroc_csv=star / AUROC_CSV_NAME,
        class_names_csv=star / AUROC_CLASS_NAMES_CSV_NAME,
        stardist_h5ad=Path(cases_root) / sample / f"{sample}_matched_features_stardist.h5ad",
    )


def discover_s1167_spatial_samples(
    clinical: pd.DataFrame,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    sample_col: str = "ACQUISITION_ID",
    require_h5ad: bool = True,
) -> list[str]:
    samples = []
    for sample in clinical[sample_col].dropna().astype(str).unique():
        paths = s1167_spatial_paths(sample, stardist_root=stardist_root, cases_root=cases_root)
        if not paths.auroc_csv.is_file() or not paths.class_names_csv.is_file():
            continue
        if require_h5ad and not paths.stardist_h5ad.is_file():
            continue
        samples.append(sample)
    return sorted(samples)


def load_s1167_annotation(cohort: str | None = None) -> pd.DataFrame:
    return load_s1167_celltype_hierarchy(cohort=cohort or cfg().cohort)


def load_s1167_clinical_info(
    *,
    cohort: str | None = None,
    annotated_only: bool = True,
    with_he_only: bool = True,
    show_overview: bool = True,
) -> pd.DataFrame:
    """Clinical_info rows for the active organ, with ``coverslip`` attached."""
    clinical = load_s1167_metadata(
        cohort=cohort or cfg().cohort,
        annotated_only=annotated_only,
        with_he_only=with_he_only,
    ).copy()
    clinical["coverslip"] = clinical["ACQUISITION_ID"].map(coverslip_from_acq)
    print("Clinical rows:", len(clinical), f"(cohort={cohort or cfg().cohort})")
    if show_overview:
        try:
            from IPython.display import display as display_fn
        except ImportError:
            display_fn = None
        show_cols = [
            c
            for c in (
                "ACQUISITION_ID",
                "coverslip",
                "SAMPLE_LABEL",
                "SAMPLE_ID",
                "tissue_type",
            )
            if c in clinical.columns
        ]
        preview = clinical[show_cols].head(12) if show_cols else clinical.head(12)
        if display_fn is not None:
            display_fn(preview)
        else:
            print(preview.to_string(index=False))
        for col in ("coverslip", "SAMPLE_LABEL"):
            if col not in clinical.columns:
                continue
            print(f"\n{col}")
            counts = clinical[col].value_counts(dropna=False)
            if display_fn is not None:
                display_fn(counts)
            else:
                print(counts.to_string())
    return clinical


def index_formula_table() -> pd.DataFrame:
    c = cfg()
    return pd.DataFrame(
        [
            {
                "index": "TLS",
                "hcc_analog": "TLS",
                "formula": f"sum({'+ '.join(c.tls_b + c.tls_t + c.tls_dc)})",
            },
            {
                "index": "TLS_spatial",
                "hcc_analog": "TLS_spatial",
                "formula": "(B_local × T_local × DC_local)^(1/3) within radius R μm",
            },
            {
                "index": "SRI_ratio",
                "hcc_analog": "SRI_ratio",
                "formula": f"{'+'.join(c.fibroblast)} / {'+'.join(c.epithelium)}",
            },
            {
                "index": "SRI_spatial",
                "hcc_analog": "SRI_spatial",
                "formula": "Fib_local / Epi_local within radius R μm",
            },
            {
                "index": "TNI_ratio",
                "hcc_analog": "TNI_ratio",
                "formula": (
                    f"({'+'.join(c.fibroblast)} + {'+'.join(c.macrophage)} + "
                    f"{'+'.join(c.endothelial)}) / epithelium"
                ),
            },
            {
                "index": "TNI_spatial",
                "hcc_analog": "TNI_spatial",
                "formula": "(Fib_local × Mac_local × Endo_local)^(1/3) / Epi_local",
            },
        ]
    )


def _compartment(probs, class_names, cell_types) -> np.ndarray:
    return hdni.compartment_prob_per_cell(probs, class_names, cell_types)


def compute_tls_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    return hdni.sum_prob_columns(
        probs, hdni.prob_columns_for_cell_types(class_names, _tls_types())
    )


def compute_sri_ratio_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    c = cfg()
    return hdni.safe_ratio(
        _compartment(probs, class_names, c.fibroblast),
        _compartment(probs, class_names, c.epithelium),
    )


def compute_tni_ratio_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    c = cfg()
    numer = (
        _compartment(probs, class_names, c.fibroblast)
        + _compartment(probs, class_names, c.macrophage)
        + _compartment(probs, class_names, c.endothelial)
    )
    return hdni.safe_ratio(numer, _compartment(probs, class_names, c.epithelium))


def compute_spatial_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = cfg()
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    b_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.tls_b), coords, radius_px
    )
    t_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.tls_t), coords, radius_px
    )
    dc_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.tls_dc), coords, radius_px
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
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cfg()
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    fib_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.fibroblast), coords, radius_px
    )
    epi_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.epithelium), coords, radius_px
    )
    return hdni.safe_ratio(fib_loc, epi_loc), fib_loc, epi_loc


def compute_spatial_tni_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = cfg()
    radius_px = hdni.um_to_he_pixel(radius_um, um_per_pixel)
    fib_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.fibroblast), coords, radius_px
    )
    mac_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.macrophage), coords, radius_px
    )
    endo_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.endothelial), coords, radius_px
    )
    epi_loc = hdni.local_neighborhood_mean(
        _compartment(probs, class_names, c.epithelium), coords, radius_px
    )
    geomean = np.power(
        np.maximum(fib_loc, eps) * np.maximum(mac_loc, eps) * np.maximum(endo_loc, eps),
        1.0 / 3.0,
    )
    return hdni.safe_ratio(geomean, epi_loc), fib_loc, mac_loc, endo_loc, epi_loc


def add_abundance_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}TLS"] = compute_tls_per_cell(probs, class_names)
    out[f"{prefix}SRI_ratio"] = compute_sri_ratio_per_cell(probs, class_names)
    out[f"{prefix}TNI_ratio"] = compute_tni_ratio_per_cell(probs, class_names)
    return out


def add_spatial_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_HOTSPOT_PERCENTILE,
    prefix: str = "idx_",
) -> pd.DataFrame:
    out = df.copy()
    tls_s, b_loc, t_loc, dc_loc = compute_spatial_tls_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    sri_s, fib_sri, epi_sri = compute_spatial_sri_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    tni_s, fib_tni, mac_loc, endo_loc, epi_tni = compute_spatial_tni_per_cell(
        probs, class_names, coords, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    tni_num = np.power(
        np.maximum(fib_tni, 1e-8) * np.maximum(mac_loc, 1e-8) * np.maximum(endo_loc, 1e-8),
        1.0 / 3.0,
    )
    out[f"{prefix}TLS_spatial"] = tls_s
    out[f"{prefix}TLS_B_local"] = b_loc
    out[f"{prefix}TLS_T_local"] = t_loc
    out[f"{prefix}TLS_DC_local"] = dc_loc
    out[f"{prefix}SRI_spatial"] = sri_s
    out[f"{prefix}TNI_spatial"] = tni_s
    out[f"{prefix}Epi_local"] = epi_sri
    out[f"{prefix}Fib_local"] = fib_sri
    out[f"{prefix}M2_local"] = mac_loc
    out[f"{prefix}Endo_local"] = endo_loc
    out[f"{prefix}TNI_numerator_local"] = tni_num
    out[f"{prefix}TNI_Fib_local"] = fib_tni
    out[f"{prefix}TNI_M2_local"] = mac_loc
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
    cols = [c for c in ("idx_TLS", "idx_SRI_ratio", "idx_TNI_ratio") if c in df.columns]
    row = {"Sample": sample, "n_cells": int(len(df))}
    for col in cols:
        row[col] = float(df[col].mean())
    return pd.DataFrame([row])


def load_sample_with_spatial_indices(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
):
    paths = s1167_spatial_paths(sample, stardist_root=stardist_root, cases_root=cases_root)
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
            f"{sample}: AUROC class_names missing index compartments {missing}."
        )
    coords_raw, cell_ids = hdni.load_spatial_coords_from_h5ad(paths.stardist_h5ad)
    df, coords = hdni.align_auroc_df_with_h5ad(df, coords_raw, cell_ids)
    prob_cols = [f"prob_{j}" for j in range(len(class_names))]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    df = add_abundance_indices(df, probs, class_names)
    df = add_spatial_indices(
        df, probs, class_names, coords,
        radius_um=radius_um, um_per_pixel=um_per_pixel,
    )
    return df, class_names, probs, coords, paths, um_per_pixel


def build_sample_dict(
    samples: Sequence[str],
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        df, *_ = load_sample_with_spatial_indices(
            sample,
            stardist_root=stardist_root,
            cases_root=cases_root,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
        )
        out[str(sample)] = df
        print(f"  loaded {sample}: {len(df):,} cells", flush=True)
    return out


def plot_tls_abundance_vs_spatial(df: pd.DataFrame, sample: str, **kwargs):
    return hdni.plot_tls_abundance_vs_spatial(df, sample, **kwargs)


def plot_sri_tni_abundance_vs_spatial(df: pd.DataFrame, sample: str, **kwargs):
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
def _quadrant_label_context():
    old_labels = hdni.ARI_FRI_QUADRANT_LABELS
    old_plot = hdni.ARI_FRI_QUADRANT_PLOT_LABELS
    hdni.ARI_FRI_QUADRANT_LABELS = S1167_QUADRANT_LABELS
    hdni.ARI_FRI_QUADRANT_PLOT_LABELS = S1167_QUADRANT_PLOT_LABELS
    try:
        yield
    finally:
        hdni.ARI_FRI_QUADRANT_LABELS = old_labels
        hdni.ARI_FRI_QUADRANT_PLOT_LABELS = old_plot


def _relabel_sri_tni_hierarchical_figure(fig, sample_name: str) -> None:
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
    kwargs.setdefault("ari_col", "idx_SRI_spatial")
    kwargs.setdefault("fri_col", "idx_TNI_spatial")
    with _quadrant_label_context():
        result = hdni.analyze_ari_fri_relationship(df, sample_name, **kwargs)
    result["SRI_median"] = result.get("ARI_median")
    result["TNI_median"] = result.get("FRI_median")
    quad = result.get("quadrant_table")
    if isinstance(quad, pd.DataFrame) and "quadrant" in quad.columns:
        quad = quad.copy()
        quad["label"] = quad["quadrant"].map(S1167_QUADRANT_LABELS)
        result["quadrant_table"] = quad
    for fig in (result.get("figures") or {}).values():
        if fig is not None:
            _relabel_sri_tni_hierarchical_figure(fig, sample_name)
    return result


def batch_analyze_sri_tni_cohort(
    sample_dict: dict[str, pd.DataFrame],
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    kwargs.setdefault("ari_col", "idx_SRI_spatial")
    kwargs.setdefault("fri_col", "idx_TNI_spatial")
    batch, meta = hdni.batch_analyze_ari_fri_cohort(sample_dict, **kwargs)
    batch = batch.rename(columns={"ARI_median": "SRI_median", "FRI_median": "TNI_median"})
    meta["sri_threshold_global"] = meta.get("ari_threshold_global")
    meta["tni_threshold_global"] = meta.get("fri_threshold_global")
    return batch, meta


def merge_batch_with_clinical(
    batch_df: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str = "ACQUISITION_ID",
) -> pd.DataFrame:
    out = batch_df.copy()
    out[sample_col] = out[sample_col].astype(str)
    clin = clinical.copy()
    clin[clinical_key] = clin[clinical_key].astype(str)
    clin = clin.drop_duplicates(clinical_key)
    merged = out.merge(clin, left_on=sample_col, right_on=clinical_key, how="left")
    if "coverslip" not in merged.columns:
        merged["coverslip"] = merged[sample_col].map(coverslip_from_acq)
    for col in clinical_columns():
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged


def test_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    from uni_label_cv_helpers import test_metric_by_clinical_groups

    return test_metric_by_clinical_groups(
        df,
        metric_col=metric_col,
        clinical_columns=tuple(clinical_cols or clinical_columns()),
        pan_organ=cfg().pan_organ,
        method=method,
        alternative=alternative,
    )


def plot_clinical_comparison(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    q4_col: str,
    **kwargs,
):
    kwargs.setdefault("clinical_columns", clinical_columns())
    kwargs.setdefault("pan_organ", cfg().pan_organ)
    kwargs.setdefault("sample_col", "Sample")
    return hdni.plot_q4_clinical_comparison(df, stats_summary, q4_col=q4_col, **kwargs)


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
    epi_col = _column_or_none(df, f"{prefix}Epi_local")
    if epi_col is not None:
        return pd.to_numeric(df[epi_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    sri_col = _column_or_none(df, f"{prefix}SRI_spatial")
    if fib_col is None or sri_col is None:
        raise KeyError("Need idx_Epi_local, or idx_SRI_spatial plus Fib_local.")
    fib = pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64)
    sri = pd.to_numeric(df[sri_col], errors="coerce").to_numpy(dtype=np.float64)
    return fib / np.maximum(sri, 1e-8)


def tni_numerator_values(df: pd.DataFrame, *, prefix: str = "idx_") -> np.ndarray:
    num_col = _column_or_none(df, f"{prefix}TNI_numerator_local")
    if num_col is not None:
        return pd.to_numeric(df[num_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    m2_col = _column_or_none(df, f"{prefix}M2_local", f"{prefix}TNI_M2_local")
    endo_col = _column_or_none(df, f"{prefix}Endo_local", f"{prefix}TNI_Endo_local")
    if fib_col is None or m2_col is None or endo_col is None:
        raise KeyError("Need Fib/Mac/Endo local columns.")
    fib = np.maximum(pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    m2 = np.maximum(pd.to_numeric(df[m2_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    endo = np.maximum(pd.to_numeric(df[endo_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    return np.power(fib * m2 * endo, 1.0 / 3.0)


def summarize_source_metrics(
    sample_dict: dict[str, pd.DataFrame],
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
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


def aggregate_metrics_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = "SAMPLE_LABEL",
    metric_cols: Sequence[str] | None = None,
    extra_clinical: Sequence[str] = ("coverslip",),
) -> pd.DataFrame:
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
    clin_cols = tuple(dict.fromkeys((*clinical_columns(), *extra_clinical)))
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
        for col in clin_cols:
            if col not in sub.columns or col == patient_col:
                continue
            vals = sub[col].astype(str).unique().tolist()
            row[col] = vals[0]
            if len(vals) > 1:
                print(f"  warn: {patient_col}={pid} has mixed {col}={vals}; using {vals[0]}")
        rows.append(row)
    return pd.DataFrame(rows)


def test_source_metrics(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = SOURCE_METRICS,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        frames.append(
            test_clinical_groups(
                df,
                metric_col=col,
                alternative=alternative,
                method=method,
                clinical_cols=clinical_cols,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_source_metric_grid(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = SOURCE_METRICS,
    sample_col: str | None = "ACQUISITION_ID",
    group_col: str = "SAMPLE_LABEL",
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    point_size: float = 36,
):
    """Side-by-side boxplots for SRI / TNI / Epi / TNI numerator."""
    import matplotlib.pyplot as plt
    from plotting_palettes import clinical_group_order

    cols = [c for c in metric_cols if c in df.columns]
    if not cols:
        raise KeyError(f"None of {list(metric_cols)} found in dataframe.")
    n = len(cols)
    legend_w = 1.8 if sample_col and sample_col in df.columns else 0.0
    if figsize is None:
        figsize = (4.2 * n + legend_w, 5.2)

    order_map = clinical_group_order(cfg().pan_organ)
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
            data, positions=positions, widths=0.55, patch_artist=True,
            showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set(
                facecolor=hdni.COMPLETE_COHORT_BOX_FACE, alpha=0.85,
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
                float(np.mean(positions)), y_max + pad_top * 0.52,
                f"{test}\np={p:.3g} ({stars})",
                ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
                clip_on=False, zorder=4,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
        ax.set_title(SOURCE_METRIC_LABELS.get(col, col), fontsize=11, pad=6)
        ax.set_ylabel("mean per core or patient" if j == 0 else "")
        ax.grid(axis="y", alpha=0.25)

    if color_by_sample and sample_order:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=sample_colors[sid], markeredgecolor="0.35",
                markersize=5, label=sid,
            )
            for sid in sample_order[:24]
        ]
        fig.legend(
            handles, sample_order[:24], loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=6, title=sample_col, frameon=False, borderaxespad=0.0,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig, axes


# HCC-style aliases used by the notebooks
add_hcc_abundance_indices = add_abundance_indices
add_hcc_spatial_indices = add_spatial_indices
build_hcc_sample_dict = build_sample_dict
merge_batch_with_hcc_clinical = merge_batch_with_clinical
test_hcc_clinical_groups = test_clinical_groups
plot_hcc_clinical_comparison = plot_clinical_comparison
summarize_hcc_source_metrics = summarize_source_metrics
aggregate_hcc_metrics_by_patient = aggregate_metrics_by_patient
hcc_spatial_paths = s1167_spatial_paths
discover_hcc_spatial_samples = discover_s1167_spatial_samples
load_hcc_clinical_info = load_s1167_clinical_info
load_hcc_annotation = load_s1167_annotation
