## 2026.08.14 LLY: CODEX HCC counterpart of Xenium_lung/histology_derived_niche_index.py
## 2026.08.27 Shared TLS / SRI / TNI engine lives in Hist2Pheno_pkg/histology_niche_index.py
"""Histology-derived niche indices for CODEX HCC (s4769).

Index mapping (lung → HCC)
--------------------------
TLS / TLS_spatial
    B + T + DC adaptive-immune aggregate.
SRI (stromal remodeling index)  ← lung ARI
    Fibroblasts / epithelium.
TNI (TME niche index)  ← lung FRI
    Geometric co-localization of fibroblasts × M2-like macrophages × endothelium,
    relative to local epithelium.

Shared math: ``Hist2Pheno_pkg/histology_niche_index.py``.
This module keeps HCC paths, clinical loaders, and spatial-plot wrappers.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import pandas as pd

_CODEX_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CODEX_DIR.parent / "Hist2Pheno_pkg"
_XENIUM_DIR = _CODEX_DIR.parent / "Xenium_lung"
for _p in (_CODEX_DIR, _PKG_DIR, _XENIUM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import histology_niche_index as hni  # noqa: E402
import histology_derived_niche_index as hdni  # noqa: E402  (Xenium plot helpers)
from s4769_plot import load_hcc_clinical_info  # noqa: E402

PAN_ORGAN = "codex_hcc"
ORGAN = hni.get_organ_config(PAN_ORGAN)

DEFAULT_CODEX_HCC_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer"
)
S4769_SAMPLE = "s4769"
DEFAULT_DATA_ROOT = DEFAULT_CODEX_HCC_DIR / S4769_SAMPLE
DEFAULT_HE_ROOT = DEFAULT_DATA_ROOT / "HE"
DEFAULT_STARDIST_ROOT = DEFAULT_DATA_ROOT / "result_all_spatial" / "stardist"
DEFAULT_CLINICAL_XLSX = DEFAULT_HE_ROOT / "s4769_he_mapping_updated_Visium.xlsx"
DEFAULT_CLINICAL_SHEET = "Clinical_info"

AUROC_CSV_NAME = hni.AUROC_CSV_NAME
AUROC_CLASS_NAMES_CSV_NAME = hni.AUROC_CLASS_NAMES_CSV_NAME

DEFAULT_HCC_UM_PER_HE_PIXEL = ORGAN.um_per_he_pixel
DEFAULT_HCC_SPATIAL_RADIUS_UM = ORGAN.spatial_radius_um
DEFAULT_HOTSPOT_PERCENTILE = ORGAN.hotspot_percentile
DEFAULT_DEMO_SAMPLE = ORGAN.demo_sample

TLS_B_CELLS = ORGAN.tls_b
TLS_T_CELLS = ORGAN.tls_t
TLS_DC_CELLS = ORGAN.tls_dc
TLS_CELL_TYPES = ORGAN.tls_types
EPITHELIUM_CELLS = ORGAN.epithelium
FIBROBLAST_CELLS = ORGAN.fibroblast
ENDOTHELIAL_CELLS = ORGAN.endothelial
M2_MACROPHAGE_CELLS = ORGAN.macrophage
MACROPHAGE_CELLS = ("Macrophages", "Macrophages M2-like")

HCC_CLINICAL_COLUMNS = ("Response", "diagnosis", "treatment")
HCC_QUADRANT_LABELS = ORGAN.quadrant_labels
HCC_QUADRANT_PLOT_LABELS = ORGAN.quadrant_plot_labels

HCC_SOURCE_METRIC_LABELS = dict(hni.SOURCE_METRIC_LABELS)
HCC_SOURCE_METRIC_LABELS["TNI_numerator_mean"] = "TNI numerator  (Fib × M2 × Endo)"
HCC_SOURCE_METRIC_LABELS["M2_local_mean"] = "M2_local"
HCC_RESPONSE_SOURCE_METRICS = hni.SOURCE_METRICS

HccSpatialPaths = hni.SpatialPaths
load_stardist_auroc_csv = hni.load_stardist_auroc_csv
summarize_tls_hotspots = hni.summarize_tls_hotspots
summarize_indices_by_sample = hni.summarize_indices_by_sample


def missing_index_cell_types(class_names: Sequence[str]) -> list[str]:
    return hni.missing_index_cell_types(class_names, ORGAN, require_all=True)


def hcc_spatial_paths(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
) -> hni.SpatialPaths:
    return hni.spatial_paths(
        sample, organ=ORGAN, stardist_root=stardist_root, cases_root=he_root
    )


def discover_hcc_spatial_samples(
    clinical: pd.DataFrame,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    sample_col: str = "matched_he",
    require_h5ad: bool = True,
) -> list[str]:
    return hni.discover_spatial_samples(
        clinical,
        organ=ORGAN,
        stardist_root=stardist_root,
        cases_root=he_root,
        sample_col=sample_col,
        require_h5ad=require_h5ad,
    )


def load_hcc_annotation() -> pd.DataFrame:
    from s4769_img_cell_mapping import load_codex_celltype_hierarchy

    return load_codex_celltype_hierarchy()


def index_formula_table() -> pd.DataFrame:
    table = hni.index_formula_table(ORGAN)
    analogs = {
        "TLS": "TLS",
        "TLS_spatial": "TLS_spatial",
        "SRI_ratio": "ARI_ratio",
        "SRI_spatial": "ARI_spatial",
        "TNI_ratio": "FRI_ratio",
        "TNI_spatial": "FRI_spatial",
    }
    table.insert(1, "lung_analog", table["index"].map(analogs))
    return table


def compute_tls_per_cell(probs, class_names):
    return hni.compute_tls_per_cell(probs, class_names, ORGAN)


def compute_sri_ratio_per_cell(probs, class_names):
    return hni.compute_sri_ratio_per_cell(probs, class_names, ORGAN)


def compute_tni_ratio_per_cell(probs, class_names):
    return hni.compute_tni_ratio_per_cell(probs, class_names, ORGAN)


def compute_spatial_tls_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_tls_per_cell(probs, class_names, coords, ORGAN, **kwargs)


def compute_spatial_sri_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_sri_per_cell(probs, class_names, coords, ORGAN, **kwargs)


def compute_spatial_tni_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_tni_per_cell(probs, class_names, coords, ORGAN, **kwargs)


def add_hcc_abundance_indices(df, probs, class_names, *, prefix: str = "idx_"):
    return hni.add_abundance_indices(df, probs, class_names, ORGAN, prefix=prefix)


def add_hcc_spatial_indices(df, probs, class_names, coords, **kwargs):
    return hni.add_spatial_indices(df, probs, class_names, coords, ORGAN, **kwargs)


def load_sample_with_spatial_indices(
    sample: str,
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
):
    return hni.load_sample_with_spatial_indices(
        sample,
        organ=ORGAN,
        stardist_root=stardist_root,
        cases_root=he_root,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
        require_all_cell_types=True,
    )


def build_hcc_sample_dict(
    samples: Sequence[str],
    *,
    stardist_root: str | Path = DEFAULT_STARDIST_ROOT,
    he_root: str | Path = DEFAULT_HE_ROOT,
    radius_um: float = DEFAULT_HCC_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_HCC_UM_PER_HE_PIXEL,
) -> dict[str, pd.DataFrame]:
    return hni.build_sample_dict(
        samples,
        organ=ORGAN,
        stardist_root=stardist_root,
        cases_root=he_root,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
        require_all_cell_types=True,
    )


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
    with _hcc_quadrant_label_context():
        result = hdni.analyze_ari_fri_relationship(df, sample_name, **kwargs)
    result["SRI_median"] = result.get("ARI_median")
    result["TNI_median"] = result.get("FRI_median")
    quad = result.get("quadrant_table")
    if isinstance(quad, pd.DataFrame) and "quadrant" in quad.columns:
        quad = quad.copy()
        quad["label"] = quad["quadrant"].map(HCC_QUADRANT_LABELS)
        result["quadrant_table"] = quad
    for fig in (result.get("figures") or {}).values():
        if fig is not None:
            _relabel_sri_tni_hierarchical_figure(fig, sample_name)
    return result


def batch_analyze_sri_tni_cohort(sample_dict: dict[str, pd.DataFrame], **kwargs):
    kwargs.setdefault("sri_col", kwargs.pop("ari_col", "idx_SRI_spatial"))
    kwargs.setdefault("tni_col", kwargs.pop("fri_col", "idx_TNI_spatial"))
    kwargs.pop("plot", None)
    kwargs.pop("include_per_sample_q4", None)
    kwargs.pop("analyze_kwargs", None)
    return hni.batch_analyze_sri_tni_cohort(sample_dict, **kwargs)


def merge_batch_with_hcc_clinical(
    batch_df: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str = "matched_he",
) -> pd.DataFrame:
    return hni.merge_batch_with_clinical(
        batch_df,
        clinical,
        sample_col=sample_col,
        clinical_key=clinical_key,
        clinical_columns=HCC_CLINICAL_COLUMNS,
    )


def test_hcc_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    return hni.test_clinical_groups(
        df,
        metric_col=metric_col,
        pan_organ=PAN_ORGAN,
        clinical_columns=tuple(clinical_columns or HCC_CLINICAL_COLUMNS),
        alternative=alternative,
        method=method,
    )


def plot_hcc_clinical_comparison(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    q4_col: str,
    **kwargs,
):
    kwargs.setdefault("clinical_columns", HCC_CLINICAL_COLUMNS)
    kwargs.setdefault("pan_organ", PAN_ORGAN)
    kwargs.setdefault("sample_col", "Sample")
    return hdni.plot_q4_clinical_comparison(df, stats_summary, q4_col=q4_col, **kwargs)


def summarize_hcc_source_metrics(sample_dict, **kwargs):
    return hni.summarize_source_metrics(sample_dict, **kwargs)


def aggregate_hcc_metrics_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    metric_cols: Sequence[str] | None = None,
    clinical_columns: Sequence[str] = HCC_CLINICAL_COLUMNS,
) -> pd.DataFrame:
    return hni.aggregate_metrics_by_patient(
        df,
        patient_col=patient_col,
        metric_cols=metric_cols,
        clinical_columns=clinical_columns,
    )


def test_hcc_response_source_metrics(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = HCC_RESPONSE_SOURCE_METRICS,
    alternative: str = "two-sided",
    method: str = "rank",
) -> pd.DataFrame:
    return hni.test_source_metrics(
        df,
        pan_organ=PAN_ORGAN,
        clinical_columns=("Response",),
        metric_cols=metric_cols,
        alternative=alternative,
        method=method,
    )


def plot_hcc_response_source_grid(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = HCC_RESPONSE_SOURCE_METRICS,
    sample_col: str | None = "patient_id",
    group_col: str = "Response",
    **kwargs,
):
    kwargs.setdefault("metric_labels", HCC_SOURCE_METRIC_LABELS)
    return hni.plot_source_metric_grid(
        df,
        stats_summary,
        pan_organ=PAN_ORGAN,
        group_col=group_col,
        metric_cols=metric_cols,
        sample_col=sample_col,
        **kwargs,
    )
