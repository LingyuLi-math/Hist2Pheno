## 2026.08.18 plotting wrappers for s1167 PDAC / GIST TMA
## Analog of code/CODEX_hcc/s4769_plot.py. Group cores by coverslip (no Response).

"""Plotting wrappers for the s1167 CODEX PDAC / GIST TMA dataset."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
_PDAC_DIR = Path(__file__).resolve().parent
if str(_PDAC_DIR) not in sys.path:
    sys.path.insert(0, str(_PDAC_DIR))

from plotting_utils import (  # noqa: E402
    celltype_counts_from_df,
    plot_celltype_count_bars as _plot_celltype_count_bars,
    plot_celltype_proportions_stacked as _plot_celltype_proportions_stacked,
    pooled_celltype_counts,
)
from s1167_img_cell_mapping import (  # noqa: E402
    DEFAULT_S1167_CELLTYPE_FILTER,
    S1167_CELLTYPE_COLORS,
    coverslip_from_acq,
    load_s1167_metadata,
)

PDAC_COVERSLIP_ORDER = ("c001", "c003", "c005", "c007")
GIST_COVERSLIP_ORDER = ("c009", "c011", "c013")


def s1167_celltype_color_map(celltypes: Sequence[str]) -> dict[str, str]:
    """Subset ``S1167_CELLTYPE_COLORS`` to the retained Level2 labels."""
    names = [str(name).strip() for name in celltypes if str(name).strip()]
    missing = [name for name in names if name not in S1167_CELLTYPE_COLORS]
    if missing:
        raise ValueError(f"S1167_CELLTYPE_COLORS missing labels: {missing}")
    return {name: S1167_CELLTYPE_COLORS[name] for name in names}


def _drop_filtered(
    counts: pd.Series,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
) -> pd.Series:
    excluded = {str(value).strip() for value in (celltype_filter or ())}
    if not excluded:
        return counts
    keep = [idx for idx in counts.index if str(idx).strip() not in excluded]
    return counts.loc[keep]


def _panel_title(acq_id: str, mapping_df: pd.DataFrame | None) -> str:
    if mapping_df is not None and not mapping_df.empty and "ACQUISITION_ID" in mapping_df.columns:
        rows = mapping_df[mapping_df["ACQUISITION_ID"].astype(str) == str(acq_id)]
        if not rows.empty:
            label = rows.iloc[0].get("SAMPLE_LABEL")
            if pd.notna(label) and str(label).strip():
                return f"{coverslip_from_acq(acq_id)} {str(label).strip()}"
    if re.fullmatch(r"c\d+", str(acq_id)):
        return str(acq_id)
    match = re.search(r"reg(\d+)", str(acq_id), flags=re.IGNORECASE)
    return f"{coverslip_from_acq(acq_id)} reg{match.group(1)}" if match else str(acq_id)


def plot_celltype_proportions_stacked(
    celltype_counts: pd.DataFrame,
    **kwargs,
) -> plt.Figure | None:
    """Stacked composition; default sample id is ``ACQUISITION_ID``."""
    kwargs.setdefault("sample_id_col", "ACQUISITION_ID")
    kwargs.setdefault("celltype_colors", S1167_CELLTYPE_COLORS)
    kwargs.setdefault("xlabel", "")
    hide_xticklabels = kwargs.pop("hide_xticklabels", None)
    n_samples = celltype_counts.drop(index="_TOTAL_", errors="ignore").shape[1]
    if hide_xticklabels is None:
        hide_xticklabels = n_samples > 40
    show = kwargs.pop("show", True)
    fig = _plot_celltype_proportions_stacked(celltype_counts, show=False, **kwargs)
    if fig is not None and hide_xticklabels:
        ax = fig.axes[0]
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")
    if show:
        plt.show()
        if fig is not None:
            plt.close(fig)
        return None
    return fig


def plot_all_acq_celltype_distributions(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame | None = None,
    *,
    n_cols: int = 6,
    celltype_col: str = "celltype",
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
    celltype_colors: dict[str, str] | None = None,
    cmap_name: str = "tab20",
    figsize_per_panel: tuple[float, float] = (4.2, 3.2),
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    suptitle: str | None = None,
) -> plt.Figure | None:
    """One cell-type count panel per acquisition (or pooled coverslip key)."""
    if not cells_by_acq:
        raise ValueError("cells_by_acq is empty")
    colors = celltype_colors or S1167_CELLTYPE_COLORS
    acq_ids = list(cells_by_acq)
    n_panels = len(acq_ids)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, acq_id in zip(axes_flat, acq_ids):
        counts = _drop_filtered(
            celltype_counts_from_df(
                cells_by_acq[acq_id],
                celltype_col=celltype_col,
                exclude_unknown=False,
            ),
            celltype_filter,
        )
        _plot_celltype_count_bars(
            counts,
            title=_panel_title(acq_id, mapping_df),
            ax=ax,
            cmap_name=cmap_name,
            celltype_colors=colors,
            show=False,
        )
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_pooled_celltype_distribution(
    cells_by_acq: dict[str, pd.DataFrame],
    *,
    celltype_col: str = "celltype",
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
    celltype_colors: dict[str, str] | None = None,
    title: str = "s1167 pooled cell-type distribution",
    **kwargs,
) -> plt.Figure | None:
    """One bar chart pooling all acquisitions in ``cells_by_acq``."""
    counts = _drop_filtered(
        pooled_celltype_counts(
            cells_by_acq,
            celltype_col=celltype_col,
            exclude_unknown=False,
        ),
        celltype_filter,
    )
    kwargs.setdefault("cmap_name", "tab20")
    kwargs.setdefault("celltype_colors", celltype_colors or S1167_CELLTYPE_COLORS)
    return _plot_celltype_count_bars(counts, title=title, **kwargs)


#####################################################
# 2026.08.20 Incomplete_Cases StarDist-all spatial maps
#####################################################
PDAC_STARDIST_MACRO_AUROC_TIERS: tuple[str, ...] = ("l2", "l12", "l1")
S1167_STARDIST_CLINICAL_COLUMNS: tuple[str, ...] = ("coverslip", "SAMPLE_LABEL")


def short_pdac_acq_label(sample: str) -> str:
    """Short y-axis / title label: ``c005 reg012``."""
    return _panel_title(sample, None)


def _subset_loaded_by_coverslip(
    loaded: Mapping[str, Mapping],
    *,
    max_per_coverslip: int | None,
) -> dict[str, Mapping]:
    if max_per_coverslip is None:
        return dict(loaded)
    counts: dict[str, int] = {}
    subset: dict[str, Mapping] = {}
    for sample, rec in loaded.items():
        cs = coverslip_from_acq(sample)
        n = counts.get(cs, 0)
        if n >= max_per_coverslip:
            continue
        subset[sample] = rec
        counts[cs] = n + 1
    if len(subset) < len(loaded):
        print(
            f"Overview subset: {len(subset)}/{len(loaded)} cores "
            f"({max_per_coverslip} per coverslip)"
        )
    return subset


def plot_pdac_incomplete_stardist_spatial_maps(
    data_root,
    samples: Sequence[str],
    save_result: str = "result_all_spatial",
    *,
    heads: Sequence[str] = PDAC_STARDIST_MACRO_AUROC_TIERS,
    pan_organ: str = "codex_pdac",
    spatial_point_size: float | None = None,
    fig_size: tuple[float, float] = (10, 8),
    show: bool = False,
) -> dict[str, dict]:
    """Full-size pred-only L2/L12/L1 maps for PDAC Incomplete_Cases label h5ads."""
    from uni_label_cv_helpers import (
        plot_stardist_label_spatial_maps,
        stardist_incomplete_all_label_h5ad_path,
    )

    return plot_stardist_label_spatial_maps(
        data_root,
        samples,
        save_result,
        label_h5ad_path_fn=stardist_incomplete_all_label_h5ad_path,
        heads=heads,
        pan_organ=pan_organ,
        spatial_point_size=spatial_point_size,
        fig_size=fig_size,
        show=show,
        title_prefix_fn=short_pdac_acq_label,
        missing_error="No Incomplete_Cases label h5ads. Run the §6 inference cell first.",
    )


def plot_pdac_incomplete_stardist_spatial_overview(
    loaded: Mapping[str, Mapping],
    data_root,
    save_result: str = "result_all_spatial",
    *,
    heads: Sequence[str] = PDAC_STARDIST_MACRO_AUROC_TIERS,
    pan_organ: str = "codex_pdac",
    point_size: float | None = None,
    show: bool = True,
    save_path=None,
    max_per_coverslip: int | None = 2,
):
    """n×3 overview of Incomplete_Cases predicted spatial maps.

    Defaults to ``max_per_coverslip=2`` so 195 cores do not make a huge grid.
    Pass ``max_per_coverslip=None`` to plot every loaded core.
    """
    from uni_label_cv_helpers import (
        STARDIST_INCOMPLETE_RESULT_SUBDIR,
        plot_stardist_label_spatial_overview,
    )

    plot_loaded = _subset_loaded_by_coverslip(
        loaded, max_per_coverslip=max_per_coverslip,
    )
    if save_path is None:
        save_path = (
            Path(data_root)
            / save_result
            / STARDIST_INCOMPLETE_RESULT_SUBDIR
            / "incomplete_spatial_pred_overview_l2_l12_l1.jpg"
        )
    return plot_stardist_label_spatial_overview(
        plot_loaded,
        heads=heads,
        pan_organ=pan_organ,
        save_path=save_path,
        point_size=point_size,
        sample_labels={s: short_pdac_acq_label(s) for s in plot_loaded},
        suptitle="PDAC Incomplete_Cases StarDist-all predicted spatial maps (pred only)",
        show=show,
    )


def load_s1167_clinical_info(
    *,
    cohort: str = "PDAC",
    annotated_only: bool = True,
    with_he_only: bool = True,
    show_overview: bool = True,
) -> pd.DataFrame:
    """Clinical_info for PDAC or GIST, with ``coverslip`` attached."""
    clinical = load_s1167_metadata(
        cohort=cohort, annotated_only=annotated_only, with_he_only=with_he_only,
    ).copy()
    clinical["coverslip"] = clinical["ACQUISITION_ID"].map(coverslip_from_acq)
    print("Clinical rows:", len(clinical), f"(cohort={cohort})")
    if show_overview:
        try:
            from IPython.display import display as display_fn
        except ImportError:
            display_fn = None
        show_cols = [
            c for c in (
                "ACQUISITION_ID", "coverslip", "SAMPLE_LABEL", "SAMPLE_ID", "tissue_type",
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


def analyze_s1167_stardist_macro_auroc_by_clinical(
    stardist_result_root,
    clinical_info: pd.DataFrame,
    *,
    pan_organ: str = "codex_pdac",
    **kwargs,
) -> dict:
    """Pooled per-core macro AUROC vs coverslip / SAMPLE_LABEL."""
    from uni_label_cv_helpers import analyze_stardist_macro_auroc_by_clinical as _analyze

    kwargs.setdefault("tiers", PDAC_STARDIST_MACRO_AUROC_TIERS)
    kwargs.setdefault("clinical_columns", S1167_STARDIST_CLINICAL_COLUMNS)
    kwargs.setdefault("clinical_key", "ACQUISITION_ID")
    kwargs.setdefault("pan_organ", pan_organ)
    kwargs.setdefault("layout", "pooled_stardist")
    kwargs.setdefault("figsize", (12.0, 5.0))
    kwargs.setdefault("legend_ncol", 2)
    return _analyze(stardist_result_root, clinical_info, **kwargs)


def analyze_pdac_stardist_macro_auroc_by_clinical(
    stardist_result_root, clinical_info: pd.DataFrame, **kwargs,
) -> dict:
    kwargs.setdefault("pan_organ", "codex_pdac")
    return analyze_s1167_stardist_macro_auroc_by_clinical(
        stardist_result_root, clinical_info, **kwargs,
    )


def analyze_gist_stardist_macro_auroc_by_clinical(
    stardist_result_root, clinical_info: pd.DataFrame, **kwargs,
) -> dict:
    kwargs.setdefault("pan_organ", "codex_gist")
    return analyze_s1167_stardist_macro_auroc_by_clinical(
        stardist_result_root, clinical_info, **kwargs,
    )
