## 2026.08.13, add plotting and report wrappers for the s4769 CODEX HCC dataset
## 2026.08.14, add box-plot for clinical for the StarDist macro AUROC by clinical groups



"""Plotting and report wrappers for the s4769 CODEX HCC dataset."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from plotting_palettes import (  # noqa: E402
    DEFAULT_CELLTYPE_FILTER,
    Cell_Type_COLORS_CODEX_hcc_level0,
    Cell_Type_COLORS_CODEX_hcc_level1,
    Cell_Type_COLORS_CODEX_hcc_level2,
)
from plotting_utils import (  # noqa: E402
    _format_cell_count_label,
    _yaxis_k_formatter,
    celltype_counts_from_df,
    format_cell_count_label,
    plot_celltype_count_bars as _plot_celltype_count_bars,
    plot_celltype_proportions_stacked as _plot_celltype_proportions_stacked,
    pooled_celltype_counts,
)

DEFAULT_CELLTYPE_XLSX = (
    Path(__file__).resolve().parents[2]
    / "data/CODEX/HCC/Michael_data_transfer/s4769/HE/s4769_he_mapping_updated_Visium.xlsx"
)


def load_hcc_celltypes(
    xlsx_path: str | Path = DEFAULT_CELLTYPE_XLSX,
    *,
    sheet_name: str = "Celltype",
    celltype_col: str = "celltype_level2",
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_CELLTYPE_FILTER,
) -> list[str]:
    """Load, normalize, deduplicate, and filter HCC Level2 cell types."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if celltype_col not in df.columns:
        raise KeyError(f"{sheet_name!r} sheet missing column {celltype_col!r}")
    excluded = {str(value).strip() for value in (celltype_filter or ())}
    celltypes = (
        df[celltype_col].dropna().astype(str).str.strip().drop_duplicates()
    )
    celltypes = [name for name in celltypes if name and name not in excluded]
    missing_colors = sorted(set(celltypes) - set(Cell_Type_COLORS_CODEX_hcc_level2))
    extra_colors = sorted(set(Cell_Type_COLORS_CODEX_hcc_level2) - set(celltypes))
    if missing_colors or extra_colors:
        raise ValueError(
            "Excel cell types and Cell_Type_COLORS_CODEX_hcc_level2 disagree: "
            f"missing_colors={missing_colors}, extra_colors={extra_colors}"
        )
    return celltypes


def hcc_celltype_color_map(
    xlsx_path: str | Path = DEFAULT_CELLTYPE_XLSX,
) -> dict[str, str]:
    """Return the validated one-to-one color mapping for the retained types."""
    celltypes = load_hcc_celltypes(xlsx_path)
    return {name: Cell_Type_COLORS_CODEX_hcc_level2[name] for name in celltypes}


def _panel_title(acq_id: str, mapping_df: pd.DataFrame | None) -> str:
    """Return the HCC region display label for one acquisition."""
    if mapping_df is not None and not mapping_df.empty:
        rows = mapping_df[
            mapping_df["CODEX_ACQUISITION_ID"].astype(str) == str(acq_id)
        ]
        if not rows.empty:
            label = rows.iloc[0].get("CODEX_REGION_DISPLAY_LABEL")
            if pd.notna(label) and str(label).strip():
                return str(label).strip()
    match = re.search(r"reg(\d+)", str(acq_id), flags=re.IGNORECASE)
    return f"reg{match.group(1)}" if match else str(acq_id)


def plot_celltype_count_bars(counts: pd.Series, **kwargs) -> plt.Figure | None:
    """Compatibility wrapper using the legacy HCC palette and green fallback."""
    kwargs.setdefault("cmap_name", "Greens")
    kwargs.setdefault("celltype_colors", Cell_Type_COLORS_CODEX_hcc_level2)
    return _plot_celltype_count_bars(counts, **kwargs)


def plot_all_acq_celltype_distributions(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame | None = None,
    *,
    n_cols: int = 6,
    celltype_col: str = "celltype",
    exclude_unknown: bool = True,
    cmap_name: str = "Greens",
    figsize_per_panel: tuple[float, float] = (4.2, 3.2),
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    suptitle: str | None = "s4769 CODEX cell-type distribution (36 regions)",
) -> plt.Figure | None:
    """Plot one HCC cell-type count panel per acquisition."""
    if not cells_by_acq:
        raise ValueError("cells_by_acq is empty")
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
        counts = celltype_counts_from_df(
            cells_by_acq[acq_id],
            celltype_col=celltype_col,
            exclude_unknown=exclude_unknown,
        )
        _plot_celltype_count_bars(
            counts,
            title=_panel_title(acq_id, mapping_df),
            ax=ax,
            cmap_name=cmap_name,
            celltype_colors=Cell_Type_COLORS_CODEX_hcc_level2,
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
    exclude_unknown: bool = True,
    title: str = "s4769 pooled cell-type distribution (36 regions, excl. Unknown)",
    **kwargs,
) -> plt.Figure | None:
    """Plot one HCC bar chart pooling all acquisitions."""
    counts = pooled_celltype_counts(
        cells_by_acq,
        celltype_col=celltype_col,
        exclude_unknown=exclude_unknown,
    )
    kwargs.setdefault("cmap_name", "Greens")
    kwargs.setdefault("celltype_colors", Cell_Type_COLORS_CODEX_hcc_level2)
    return _plot_celltype_count_bars(counts, title=title, **kwargs)


def plot_celltype_proportions_stacked(
    celltype_counts: pd.DataFrame,
    **kwargs,
) -> plt.Figure | None:
    """Compatibility wrapper preserving the legacy HCC stacked-plot defaults."""
    kwargs.setdefault("sample_id_col", "CODEX_ACQUISITION_ID")
    kwargs.setdefault("celltype_colors", Cell_Type_COLORS_CODEX_hcc_level2)
    kwargs.setdefault("title", "s4769 CODEX cell-type composition by acquisition")
    kwargs.setdefault("xlabel", "CODEX acquisition")
    return _plot_celltype_proportions_stacked(celltype_counts, **kwargs)



#####################################################
# 2026.08.14 StarDist macro AUROC by clinical groups
# 2026.08.14 HCC clinical + StarDist AUROC helpers
#####################################################
HCC_STARDIST_CLINICAL_COLUMNS: tuple[str, ...] = ("Response", "diagnosis", "treatment")
HCC_STARDIST_MACRO_AUROC_TIERS: tuple[str, ...] = ("l2", "l12", "l1")
HCC_CLINICAL_RENAME: dict[str, str] = {
    "MATCHED_HE": "matched_he",
    "CODEX_ACQUISITION_ID": "region_id",
    "CODEX_REGION_DISPLAY_LABEL": "region_label",
}
HCC_CLINICAL_OVERVIEW_COLS: tuple[str, ...] = (
    "matched_he",
    "region_id",
    "region_label",
    "patient_id",
    "diagnosis",
    "Response",
    "treatment",
    "tissue_type",
    "tissue_subtype",
    "Visium",
    "Annotation",
)
HCC_CLINICAL_COUNT_COLS: tuple[str, ...] = (
    "diagnosis",
    "Response",
    "treatment",
    "tissue_type",
    "tissue_subtype",
    "Visium",
    "Annotation",
)


def load_hcc_clinical_info(
    xlsx_path: str | Path,
    *,
    sheet_name: str = "Clinical_info",
    aligned_only: bool = True,
    annotated_only: bool = True,
    show_overview: bool = True,
) -> pd.DataFrame:
    """Load s4769 clinical rows from the Visium mapping workbook.

    Filters to ``ALIGNED=='Y'`` and (by default) ``Annotation=='Y'``.
    """
    clinical_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    clinical = clinical_raw.copy()
    if aligned_only and "ALIGNED" in clinical.columns:
        clinical = clinical[clinical["ALIGNED"].astype(str).str.upper().eq("Y")]
    if annotated_only and "Annotation" in clinical.columns:
        clinical = clinical[clinical["Annotation"].astype(str).str.upper().eq("Y")]
    clinical = clinical.rename(columns=HCC_CLINICAL_RENAME).copy()

    print("Clinical sheet rows:", len(clinical_raw))
    print("Retained rows      :", len(clinical))
    if show_overview:
        display_fn = None
        try:
            from IPython.display import display as display_fn  # type: ignore
        except ImportError:
            pass
        show_cols = [c for c in HCC_CLINICAL_OVERVIEW_COLS if c in clinical.columns]
        preview = clinical[show_cols].head(12) if show_cols else clinical.head(12)
        if display_fn is not None:
            display_fn(preview)
        else:
            print(preview.to_string(index=False))
        for col in HCC_CLINICAL_COUNT_COLS:
            if col not in clinical.columns:
                continue
            print(f"\n{col}")
            counts = clinical[col].value_counts(dropna=False)
            if display_fn is not None:
                display_fn(counts)
            else:
                print(counts.to_string())
    return clinical


def plot_stardist_macro_auroc_by_clinical(
    merged_auc: pd.DataFrame,
    *,
    tiers: Sequence[str] = HCC_STARDIST_MACRO_AUROC_TIERS,
    clinical_columns: Sequence[str] = HCC_STARDIST_CLINICAL_COLUMNS,
    sample_col: str = "matched_he",
    pan_organ: str = "codex_hcc",
    method: Literal["rank", "parametric"] = "rank",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    figsize: tuple[float, float] = (15.0, 5.0),
    legend_ncol: int = 2,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict:
    """HCC wrapper around ``uni_label_cv_helpers.plot_stardist_macro_auroc_by_clinical``."""
    from uni_label_cv_helpers import plot_stardist_macro_auroc_by_clinical as _plot

    return _plot(
        merged_auc,
        tiers=tiers,
        clinical_columns=clinical_columns,
        sample_col=sample_col,
        pan_organ=pan_organ,
        method=method,
        alternative=alternative,
        figsize=figsize,
        legend_ncol=legend_ncol,
        show=show,
        save_dir=save_dir,
    )


def analyze_hcc_stardist_macro_auroc_by_clinical(
    stardist_result_root,
    clinical_info: pd.DataFrame,
    **kwargs,
) -> dict:
    """One-shot HCC clinical AUROC comparison (table + merge + plot)."""
    from uni_label_cv_helpers import analyze_stardist_macro_auroc_by_clinical as _analyze

    kwargs.setdefault("tiers", HCC_STARDIST_MACRO_AUROC_TIERS)
    kwargs.setdefault("clinical_columns", HCC_STARDIST_CLINICAL_COLUMNS)
    kwargs.setdefault("clinical_key", "matched_he")
    kwargs.setdefault("pan_organ", "codex_hcc")
    kwargs.setdefault("layout", "pooled_stardist")
    kwargs.setdefault("figsize", (15.0, 5.0))
    kwargs.setdefault("legend_ncol", 2)
    return _analyze(stardist_result_root, clinical_info, **kwargs)


# End of s4769 compatibility wrappers.