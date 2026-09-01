## 2026.08.20 LLY: s1167 PDAC / GIST counterpart of CODEX_hcc/hcc_histology_derived_niche_index.py
## 2026.08.27 Shared TLS / SRI / TNI engine lives in Hist2Pheno_pkg/histology_niche_index.py
"""Histology-derived niche indices for CODEX PDAC and GIST (s1167 TMA).

Call :func:`configure` (``codex_pdac`` or ``codex_gist``) before using helpers.
Shared math is in ``Hist2Pheno_pkg/histology_niche_index.py``. This module keeps
s1167 paths, Clinical_info loaders, and spatial-plot wrappers.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import pandas as pd

_PDAC_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PDAC_DIR.parent / "Hist2Pheno_pkg"
_XENIUM_DIR = _PDAC_DIR.parent / "Xenium_lung"
for _p in (_PDAC_DIR, _PKG_DIR, _XENIUM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import histology_niche_index as hni  # noqa: E402
import histology_derived_niche_index as hdni  # noqa: E402
from plotting_palettes import clinical_columns_for_pan_organ  # noqa: E402
from s1167_img_cell_mapping import (  # noqa: E402
    coverslip_from_acq,
    load_s1167_celltype_hierarchy,
    load_s1167_metadata,
)

#####################################################
# 2026.09.01: add GIST clinical information
#####################################################
from s1167_plot import (  # noqa: E402
    GIST_TMA_CLINICAL_GROUP_COLUMNS,
    GIST_TMA_CLINICAL_RAW_COLUMNS,
    recode_gist_tma_clinical,
    summarize_gist_tma_clinical,
)

DEFAULT_S1167_ROOT = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer/s1167"
)
DEFAULT_DATA_ROOT = DEFAULT_S1167_ROOT
DEFAULT_CASES_ROOT = DEFAULT_S1167_ROOT
DEFAULT_CLINICAL_XLSX = DEFAULT_S1167_ROOT / "raw_metadata_updated.xlsx"
DEFAULT_CLINICAL_SHEET = "Clinical_info"

AUROC_CSV_NAME = hni.AUROC_CSV_NAME
AUROC_CLASS_NAMES_CSV_NAME = hni.AUROC_CLASS_NAMES_CSV_NAME
DEFAULT_UM_PER_HE_PIXEL = hni.DEFAULT_UM_PER_HE_PIXEL
DEFAULT_SPATIAL_RADIUS_UM = hni.DEFAULT_SPATIAL_RADIUS_UM
DEFAULT_HOTSPOT_PERCENTILE = hni.DEFAULT_HOTSPOT_PERCENTILE

ORGAN_CONFIGS = hni.ORGAN_CONFIGS
OrganNicheConfig = hni.OrganNicheConfig
S1167SpatialPaths = hni.SpatialPaths
SOURCE_METRIC_LABELS = hni.SOURCE_METRIC_LABELS
SOURCE_METRICS = hni.SOURCE_METRICS
S1167_QUADRANT_LABELS = hni.DEFAULT_SRI_TNI_QUADRANT_LABELS
S1167_QUADRANT_PLOT_LABELS = hni.DEFAULT_SRI_TNI_QUADRANT_PLOT_LABELS

_ACTIVE = "codex_pdac"


def pooled_result_dirname(pan_organ: str | None = None) -> str:
    """Pooled CV / StarDist pred folder under s1167 (PDAC and GIST must not share)."""
    key = hni.normalize_organ_key(pan_organ or _ACTIVE)
    if key == "codex_gist":
        return "result_all_spatial_gist"
    if key == "codex_pdac":
        return "result_all_spatial_pdac"
    raise KeyError(f"No pooled result dir for {key!r}")


def default_stardist_pred_root(pan_organ: str | None = None) -> Path:
    return DEFAULT_S1167_ROOT / pooled_result_dirname(pan_organ) / "stardist"


DEFAULT_STARDIST_ROOT = default_stardist_pred_root()

load_stardist_auroc_csv = hni.load_stardist_auroc_csv
summarize_tls_hotspots = hni.summarize_tls_hotspots
summarize_indices_by_sample = hni.summarize_indices_by_sample
epi_local_values = hni.epi_local_values
tni_numerator_values = hni.tni_numerator_values
summarize_source_metrics = hni.summarize_source_metrics


def configure(pan_organ: str) -> hni.OrganNicheConfig:
    """Bind module-level defaults to PDAC or GIST (call before using wrappers)."""
    global _ACTIVE, DEFAULT_STARDIST_ROOT
    _ACTIVE = hni.normalize_organ_key(pan_organ)
    if _ACTIVE not in ("codex_pdac", "codex_gist"):
        raise KeyError(f"s1167 configure() expects PDAC or GIST, got {pan_organ!r}")
    DEFAULT_STARDIST_ROOT = default_stardist_pred_root()
    return cfg()


def cfg() -> hni.OrganNicheConfig:
    return hni.get_organ_config(_ACTIVE)


def pan_organ() -> str:
    return cfg().pan_organ


def clinical_columns() -> tuple[str, ...]:
    return clinical_columns_for_pan_organ(cfg().pan_organ) or ("coverslip", "SAMPLE_LABEL")


def missing_index_cell_types(
    class_names: Sequence[str],
    *,
    organ: hni.OrganNicheConfig | None = None,
) -> list[str]:
    return hni.missing_index_cell_types(class_names, organ or cfg())


def s1167_spatial_paths(
    sample: str,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
) -> hni.SpatialPaths:
    return hni.spatial_paths(
        sample,
        organ=cfg(),
        stardist_root=stardist_root or default_stardist_pred_root(),
        cases_root=cases_root,
    )


def discover_s1167_spatial_samples(
    clinical: pd.DataFrame,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    sample_col: str = "ACQUISITION_ID",
    require_h5ad: bool = True,
) -> list[str]:
    return hni.discover_spatial_samples(
        clinical,
        organ=cfg(),
        stardist_root=stardist_root or default_stardist_pred_root(),
        cases_root=cases_root,
        sample_col=sample_col,
        require_h5ad=require_h5ad,
    )


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
    if cohort is None:
        cohort = cfg().cohort
    clinical = load_s1167_metadata(
        cohort=cohort,
        annotated_only=annotated_only,
        with_he_only=with_he_only,
    ).copy()
    clinical["coverslip"] = clinical["ACQUISITION_ID"].map(coverslip_from_acq)
    print("Clinical rows:", len(clinical), f"(cohort={cohort})")
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
    table = hni.index_formula_table(cfg())
    table.insert(1, "hcc_analog", table["index"])
    return table


def compute_tls_per_cell(probs, class_names):
    return hni.compute_tls_per_cell(probs, class_names, cfg())


def compute_sri_ratio_per_cell(probs, class_names):
    return hni.compute_sri_ratio_per_cell(probs, class_names, cfg())


def compute_tni_ratio_per_cell(probs, class_names):
    return hni.compute_tni_ratio_per_cell(probs, class_names, cfg())


def compute_spatial_tls_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_tls_per_cell(probs, class_names, coords, cfg(), **kwargs)


def compute_spatial_sri_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_sri_per_cell(probs, class_names, coords, cfg(), **kwargs)


def compute_spatial_tni_per_cell(probs, class_names, coords, **kwargs):
    return hni.compute_spatial_tni_per_cell(probs, class_names, coords, cfg(), **kwargs)


def add_abundance_indices(df, probs, class_names, *, prefix: str = "idx_"):
    return hni.add_abundance_indices(df, probs, class_names, cfg(), prefix=prefix)


def add_spatial_indices(df, probs, class_names, coords, **kwargs):
    return hni.add_spatial_indices(df, probs, class_names, coords, cfg(), **kwargs)


def load_sample_with_spatial_indices(
    sample: str,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
):
    return hni.load_sample_with_spatial_indices(
        sample,
        organ=cfg(),
        stardist_root=stardist_root or default_stardist_pred_root(),
        cases_root=cases_root,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
    )


def build_sample_dict(
    samples: Sequence[str],
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
) -> dict[str, pd.DataFrame]:
    return hni.build_sample_dict(
        samples,
        organ=cfg(),
        stardist_root=stardist_root or default_stardist_pred_root(),
        cases_root=cases_root,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
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


def batch_analyze_sri_tni_cohort(sample_dict: dict[str, pd.DataFrame], **kwargs):
    kwargs.setdefault("sri_col", kwargs.pop("ari_col", "idx_SRI_spatial"))
    kwargs.setdefault("tni_col", kwargs.pop("fri_col", "idx_TNI_spatial"))
    kwargs.pop("plot", None)
    kwargs.pop("include_per_sample_q4", None)
    kwargs.pop("analyze_kwargs", None)
    return hni.batch_analyze_sri_tni_cohort(sample_dict, **kwargs)


def merge_batch_with_clinical(
    batch_df: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str = "ACQUISITION_ID",
    extra_clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    cols = list(clinical_columns())
    if extra_clinical_columns:
        cols.extend(extra_clinical_columns)
    cols = tuple(dict.fromkeys(str(c) for c in cols))
    merged = hni.merge_batch_with_clinical(
        batch_df,
        clinical,
        sample_col=sample_col,
        clinical_key=clinical_key,
        clinical_columns=cols,
    )
    if "coverslip" not in merged.columns:
        merged["coverslip"] = merged[sample_col].map(coverslip_from_acq)
    return merged


def test_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_cols: Sequence[str] | None = None,
    pairwise: bool = True,
    pairwise_correction: str = "holm",
) -> pd.DataFrame:
    return hni.test_clinical_groups(
        df,
        metric_col=metric_col,
        pan_organ=cfg().pan_organ,
        clinical_columns=tuple(clinical_cols or clinical_columns()),
        alternative=alternative,
        method=method,
        pairwise=pairwise,
        pairwise_correction=pairwise_correction,
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


def aggregate_metrics_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = "SAMPLE_LABEL",
    metric_cols: Sequence[str] | None = None,
    extra_clinical: Sequence[str] = ("coverslip",),
) -> pd.DataFrame:
    clin_cols = tuple(dict.fromkeys((*clinical_columns(), *extra_clinical)))
    return hni.aggregate_metrics_by_patient(
        df,
        patient_col=patient_col,
        metric_cols=metric_cols,
        clinical_columns=clin_cols,
    )


def test_source_metrics(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = SOURCE_METRICS,
    alternative: str = "two-sided",
    method: str = "rank",
    clinical_cols: Sequence[str] | None = None,
    pairwise: bool = True,
    pairwise_correction: str = "holm",
) -> pd.DataFrame:
    return hni.test_source_metrics(
        df,
        pan_organ=cfg().pan_organ,
        clinical_columns=tuple(clinical_cols or clinical_columns()),
        metric_cols=metric_cols,
        alternative=alternative,
        method=method,
        pairwise=pairwise,
        pairwise_correction=pairwise_correction,
    )


def plot_source_metric_grid(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = SOURCE_METRICS,
    sample_col: str | None = "ACQUISITION_ID",
    group_col: str = "SAMPLE_LABEL",
    **kwargs,
):
    return hni.plot_source_metric_grid(
        df,
        stats_summary,
        pan_organ=cfg().pan_organ,
        group_col=group_col,
        metric_cols=metric_cols,
        sample_col=sample_col,
        **kwargs,
    )
