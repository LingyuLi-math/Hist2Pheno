"""Bar charts for s4769 CODEX cell-type distributions."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns




######################################################
## 2026.08.10 define the color for CODEX_hcc dataset
######################################################
DEFAULT_CELLTYPE_XLSX = (
    Path(__file__).resolve().parents[2]
    / "data/CODEX/HCC/Michael_data_transfer/s4769/HE/s4769_he_mapping_updated_Visium.xlsx"
)
DEFAULT_CELLTYPE_FILTER = ("Unknown", "Stroma Uncharacterized")

# Fixed colors following the supplied HCC paper legend. Keys use the exact
# normalized ``celltype_level2`` labels in the Excel sheet / annotation CSVs.
Cell_Type_COLORS_CODEX_hcc_level2 = {
    "Epithelium (INOS+)": "#17becf",
    "Epithelium (INOS-)": "#1f77b4",
    "Fibroblasts": "#98df8a",
    "Endothelial cells": "#2ca02c",
    "CD4 T cells": "#ff7f0e",
    "CD8 T cells": "#ffbb78",
    "Macrophages": "#d62728",
    "Macrophages M2-like": "#ff9896",
    "Neutrophils": "#e377c2",
    "B cells": "#f7b6d2",
    "Dendritic cells": "#8c564b",
    "INFg+": "#c49c94",
}

Cell_Type_COLORS_CODEX_hcc_level1 = {
    "Stromal": "#98df8a",
    "T cells": "#ff7f0e",
    "Myeloid": "#d62728",
    "Endothelial": "#2ca02c",
    "Epithelial": "#1f77b4",
    "B cells": "#f7b6d2",
}

Cell_Type_COLORS_CODEX_hcc_level0 = {
    "Stromal": "#98df8a",
    "Immune": "#d62728",
    "Endothelial": "#2ca02c",
    "Epithelial": "#1f77b4",
}

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

    excluded = {str(x).strip() for x in (celltype_filter or ())}
    celltypes = (
        df[celltype_col]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
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
    """Return the validated one-to-one color mapping for the 12 retained types."""
    celltypes = load_hcc_celltypes(xlsx_path)
    return {name: Cell_Type_COLORS_CODEX_hcc_level2[name] for name in celltypes}

######################################################


def _format_cell_count_label(n: int | float) -> str:
    n = int(n)
    if n >= 1000:
        text = f"{n / 1000:.1f}k"
        if text.endswith(".0k"):
            return f"{int(n / 1000)}k"
        return text
    return str(n)


def _yaxis_k_formatter() -> mticker.FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        if x == 0:
            return "0K"
        if x >= 1000:
            return f"{int(round(x / 1000))}K"
        return str(int(x))

    return mticker.FuncFormatter(_fmt)


def celltype_counts_from_df(
    df: pd.DataFrame,
    *,
    celltype_col: str = "celltype",
    exclude_unknown: bool = True,
) -> pd.Series:
    """Return cell-type counts for one acquisition DataFrame."""
    if celltype_col not in df.columns:
        raise KeyError(f"DataFrame needs {celltype_col!r}")
    counts = df[celltype_col].value_counts()
    if exclude_unknown and "Unknown" in counts.index:
        counts = counts.drop("Unknown")
    return counts.sort_values(ascending=False)


def pooled_celltype_counts(
    cells_by_acq: dict[str, pd.DataFrame],
    *,
    celltype_col: str = "celltype",
    exclude_unknown: bool = True,
) -> pd.Series:
    """Sum cell-type counts across all acquisitions."""
    parts = []
    for df in cells_by_acq.values():
        parts.append(celltype_counts_from_df(
            df, celltype_col=celltype_col, exclude_unknown=exclude_unknown
        ))
    if not parts:
        return pd.Series(dtype=int)
    table = pd.concat(parts, axis=1).fillna(0)
    return table.sum(axis=1).astype(int).sort_values(ascending=False)


def _panel_title(acq_id: str, mapping_df: pd.DataFrame | None) -> str:
    if mapping_df is not None and not mapping_df.empty:
        rows = mapping_df[
            mapping_df["CODEX_ACQUISITION_ID"].astype(str) == str(acq_id)
        ]
        if not rows.empty:
            label = rows.iloc[0].get("CODEX_REGION_DISPLAY_LABEL")
            if pd.notna(label) and str(label).strip():
                return str(label).strip()
    m = re.search(r"reg(\d+)", str(acq_id), flags=re.IGNORECASE)
    if m:
        return f"reg{m.group(1)}"
    return str(acq_id)


def plot_celltype_count_bars(
    counts: pd.Series,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    cmap_name: str = "Greens",
    ylabel: str = "Number of cells",
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    celltype_colors: dict[str, str] | None = Cell_Type_COLORS_CODEX_hcc_level2,
) -> plt.Figure | None:
    """Bar chart using fixed HCC cell-type colors, with cmap fallback."""
    counts = counts.sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No cell-type counts to plot")

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fallback_colors = sns.color_palette(cmap_name, n_colors=len(counts))
    colors = [
        celltype_colors.get(str(name).strip(), fallback)
        if celltype_colors is not None
        else fallback
        for name, fallback in zip(counts.index, fallback_colors)
    ]
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color=colors, width=0.92, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ymax = float(counts.max())
    ax.set_ylim(0, ymax * 1.12 if ymax else 1)
    ax.yaxis.set_major_formatter(_yaxis_k_formatter())

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            _format_cell_count_label(val),
            ha="center",
            va="bottom",
            fontsize=9,
            # fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_fig:
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
    """One bar chart per acquisition, arranged in a grid."""
    if not cells_by_acq:
        raise ValueError("cells_by_acq is empty")

    acq_ids = list(cells_by_acq.keys())
    n_panels = len(acq_ids)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, acq_id in zip(axes_flat, acq_ids):
        counts = celltype_counts_from_df(
            cells_by_acq[acq_id],
            celltype_col=celltype_col,
            exclude_unknown=exclude_unknown,
        )
        plot_celltype_count_bars(
            counts,
            title=_panel_title(acq_id, mapping_df),
            ax=ax,
            cmap_name=cmap_name,
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
    """Single bar chart pooling all acquisitions."""
    counts = pooled_celltype_counts(
        cells_by_acq,
        celltype_col=celltype_col,
        exclude_unknown=exclude_unknown,
    )
    return plot_celltype_count_bars(counts, title=title, **kwargs)

######################################################
# 2026.08.10 LLY, add the celltype_colors parameter
# 2026.08.11 LLY, add condition parameter
######################################################
def plot_celltype_proportions_stacked(
    celltype_counts: pd.DataFrame,
    *,
    celltype_order: list[str] | tuple[str, ...] | None = None,
    sample_order: list[str] | tuple[str, ...] | None = None,

    # 2026.08.11 condition parameter
    condition_df: pd.DataFrame | None = None,
    condition: str | None = None,
    sample_id_col: str = "CODEX_ACQUISITION_ID",
    condition_order: list[str] | tuple[str, ...] | None = None,
    condition_gap: float = 1.0,
    condition_line_y: float = -0.62,
    condition_label_y: float = -0.68,
    bottom_margin: float = 0.42,

    celltype_colors: dict[str, str] = Cell_Type_COLORS_CODEX_hcc_level2,
    title: str = "s4769 CODEX cell-type composition by acquisition",
    ylabel: str = "Cell proportion (%)",
    xlabel: str = "CODEX acquisition",
    figsize: tuple[float, float] | None = None,
    legend_title: str = "Cell type",
    legend_ncol: int = 1,
    rotation: int = 90,

    # 2026.08.10 LLY, adjust the fontsize of the plot
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 11,

    # 2026.08.10 LLY, add the show_shading, shading_alpha, bar_width, bar_alpha parameter   
    show_shading: bool = True,
    shading_alpha: float = 0.28,
    bar_width: float = 0.72,
    bar_alpha: float = 0.90,

    dpi: int = 300,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot 100%-stacked bars with optional ribbons and clinical grouping.

    The translucent ribbons connect the lower/upper boundaries of each cell type
    between neighboring bars. When ``condition`` is supplied, acquisitions are
    grouped by ``condition_df[condition]`` and ribbons do not cross group boundaries.
    """
    if celltype_counts.empty:
        raise ValueError("celltype_counts is empty")
    if not 0 < bar_width <= 1:
        raise ValueError("bar_width must be in (0, 1]")
    if not 0 <= shading_alpha <= 1 or not 0 <= bar_alpha <= 1:
        raise ValueError("shading_alpha and bar_alpha must be in [0, 1]")
    if condition_gap < 0:
        raise ValueError("condition_gap must be >= 0")
    if not 0 < bottom_margin < 1:
        raise ValueError("bottom_margin must be in (0, 1)")

    counts = celltype_counts.drop(index="_TOTAL_", errors="ignore").copy()
    counts.index = counts.index.astype(str).str.strip()
    if counts.index.has_duplicates:
        counts = counts.groupby(level=0, sort=False).sum()

    order = list(celltype_order or celltype_colors.keys())
    unexpected = sorted(set(counts.index) - set(order))
    if unexpected:
        raise ValueError(
            "celltype_counts contains types outside the 12-color mapping: "
            f"{unexpected}. Apply celltype_filter first."
        )
    counts = counts.reindex(order, fill_value=0)

    samples = list(sample_order or counts.columns)
    missing_samples = sorted(set(samples) - set(counts.columns))
    if missing_samples:
        raise KeyError(f"sample_order contains unknown columns: {missing_samples}")
 
    ######################################################
    # 2026.08.11 condition parameter
    sample_conditions: dict[str, str] | None = None
    condition_values: list[str] | None = None
    if condition is not None:
        if condition_df is None:
            raise ValueError("condition_df is required when condition is set")
        required = {sample_id_col, condition}
        missing_cols = required - set(condition_df.columns)
        if missing_cols:
            raise KeyError(f"condition_df missing columns: {sorted(missing_cols)}")

        clinical = condition_df[[sample_id_col, condition]].dropna().copy()
        clinical[sample_id_col] = clinical[sample_id_col].astype(str).str.strip()
        clinical[condition] = clinical[condition].astype(str).str.strip()
        conflicts = clinical.groupby(sample_id_col)[condition].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ValueError(
                f"Samples map to multiple {condition!r} values: {conflicts.index.tolist()}"
            )
        sample_conditions = (
            clinical.drop_duplicates(sample_id_col)
            .set_index(sample_id_col)[condition]
            .to_dict()
        )
        missing_condition = [sample for sample in samples if sample not in sample_conditions]
        if missing_condition:
            raise KeyError(
                f"No {condition!r} metadata for acquisitions: {missing_condition}"
            )

        groups = list(condition_order or pd.unique([sample_conditions[s] for s in samples]))
        unknown_groups = sorted(set(sample_conditions[s] for s in samples) - set(groups))
        if unknown_groups:
            raise ValueError(
                f"condition_order omits {condition!r} values: {unknown_groups}"
            )
        group_rank = {group: rank for rank, group in enumerate(groups)}
        original_rank = {sample: rank for rank, sample in enumerate(samples)}
        samples = sorted(
            samples,
            key=lambda sample: (
                group_rank[sample_conditions[sample]],
                original_rank[sample],
            ),
        )
        condition_values = [sample_conditions[sample] for sample in samples]
    ######################################################
    counts = counts.loc[:, samples].apply(pd.to_numeric, errors="coerce").fillna(0)

    totals = counts.sum(axis=0)
    if (totals <= 0).any():
        bad = totals.index[totals <= 0].tolist()
        raise ValueError(f"Acquisitions with zero retained cells: {bad}")
    proportions = counts.divide(totals, axis=1) * 100.0

    if figsize is None:
        figsize = (max(12.0, 0.38 * len(samples)), 6.0)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(samples), dtype=float)
    if condition_values is not None:
        offset = 0.0
        for i in range(1, len(samples)):
            if condition_values[i] != condition_values[i - 1]:
                offset += condition_gap
            x[i] += offset
    bottom = np.zeros(len(samples), dtype=float)

    for celltype in order:
        values = proportions.loc[celltype].to_numpy(dtype=float)
        top = bottom + values
        color = celltype_colors[celltype]

        if show_shading and len(samples) > 1:
            for i in range(len(samples) - 1):
                if (
                    condition_values is not None
                    and condition_values[i] != condition_values[i + 1]
                ):
                    continue
                connector_x = [
                    x[i] + bar_width / 2,
                    x[i + 1] - bar_width / 2,
                ]
                ax.fill_between(
                    connector_x,
                    [bottom[i], bottom[i + 1]],
                    [top[i], top[i + 1]],
                    color=color,
                    alpha=shading_alpha,
                    linewidth=0,
                    zorder=1,
                )

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=bar_width,
            color=color,
            alpha=bar_alpha,
            edgecolor="#666666",
            linewidth=0.45,
            label=celltype,
            zorder=2,
        )
        bottom = top

    # 2026.08.10 LLY, adjust the fontsize of the plot
    ax.set_xticks(x)
    ax.set_xticklabels(
        samples,
        rotation=rotation,
        ha="center",
        fontsize=xtick_fontsize,
    )
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)


    ######################################################
    # 2026.08.11 condition parameter
    if condition_values is not None:
        group_start = 0
        for i in range(1, len(samples) + 1):
            at_end = i == len(samples)
            group_changed = not at_end and condition_values[i] != condition_values[i - 1]
            if at_end or group_changed:
                group_end = i - 1
                left = x[group_start] - bar_width / 2
                right = x[group_end] + bar_width / 2
                ax.plot(
                    [left, right],
                    [condition_line_y, condition_line_y],
                    transform=ax.get_xaxis_transform(),
                    color="#333333",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.text(
                    (left + right) / 2,
                    condition_label_y,
                    condition_values[group_start],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=label_fontsize,
                    clip_on=False,
                )
                if not at_end:
                    ax.axvline(
                        (x[group_end] + x[i]) / 2,
                        color="#bdbdbd",
                        linewidth=0.8,
                        zorder=0,
                    )
                group_start = i

    ######################################################
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2026.08.10 LLY, adjust the fontsize of the legend
    ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )


    fig.tight_layout()
    if condition_values is not None:
        fig.subplots_adjust(bottom=bottom_margin)

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
######################################################