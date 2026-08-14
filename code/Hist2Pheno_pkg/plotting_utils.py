"""Reusable, dataset-neutral plotting helpers.

Dataset modules should keep path discovery, file I/O, and report orchestration
local while delegating generic DataFrame transformations and plotting here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from plotting_palettes import (
    load_global_l2_class_names,
    xenium_final_ct_rgba_overrides,
)

__all__ = [
    "format_cell_count_label",
    "celltype_counts_from_df",
    "pooled_celltype_counts",
    "plot_celltype_count_bars",
    "plot_celltype_proportions_stacked",
    "plot_final_ct_by_lineage",
]


def format_cell_count_label(n: int | float) -> str:
    """Format a cell count compactly, using ``k`` for thousands."""
    n = int(n)
    if n >= 1000:
        text = f"{n / 1000:.1f}k"
        return f"{int(n / 1000)}k" if text.endswith(".0k") else text
    return str(n)


# Private legacy name retained for dataset-module re-exports.
_format_cell_count_label = format_cell_count_label


def _yaxis_k_formatter() -> mticker.FuncFormatter:
    """Return the legacy cell-count axis formatter."""

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
    """Return descending cell-type counts for one DataFrame."""
    if celltype_col not in df.columns:
        raise KeyError(f"DataFrame needs {celltype_col!r}")
    counts = df[celltype_col].value_counts()
    if exclude_unknown and "Unknown" in counts.index:
        counts = counts.drop("Unknown")
    return counts.sort_values(ascending=False)


def pooled_celltype_counts(
    cells_by_sample: Mapping[str, pd.DataFrame],
    *,
    celltype_col: str = "celltype",
    exclude_unknown: bool = True,
) -> pd.Series:
    """Sum cell-type counts across multiple samples or acquisitions."""
    parts = [
        celltype_counts_from_df(
            df,
            celltype_col=celltype_col,
            exclude_unknown=exclude_unknown,
        )
        for df in cells_by_sample.values()
    ]
    if not parts:
        return pd.Series(dtype=int)
    table = pd.concat(parts, axis=1).fillna(0)
    return table.sum(axis=1).astype(int).sort_values(ascending=False)


def plot_celltype_count_bars(
    counts: pd.Series,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    cmap_name: str = "viridis",
    ylabel: str = "Number of cells",
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 150,
    celltype_colors: Mapping[str, object] | None = None,
) -> plt.Figure | None:
    """Plot descending cell-type counts with optional label-specific colors."""
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

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            format_cell_count_label(value),
            ha="center",
            va="bottom",
            fontsize=9,
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


def plot_celltype_proportions_stacked(
    celltype_counts: pd.DataFrame,
    *,
    celltype_order: Sequence[str] | None = None,
    sample_order: Sequence[str] | None = None,
    condition_df: pd.DataFrame | None = None,
    condition: str | None = None,
    sample_id_col: str = "sample",
    condition_order: Sequence[str] | None = None,
    condition_gap: float = 1.0,
    condition_line_y: float = -0.62,
    condition_label_y: float = -0.68,
    bottom_margin: float = 0.42,
    celltype_colors: Mapping[str, object] | None = None,
    title: str = "Cell-type composition by sample",
    ylabel: str = "Cell proportion (%)",
    xlabel: str = "Sample",
    figsize: tuple[float, float] | None = None,
    legend_title: str = "Cell type",
    legend_ncol: int = 1,
    rotation: int = 90,
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 10,
    legend_title_fontsize: int = 11,
    show_shading: bool = True,
    shading_alpha: float = 0.28,
    bar_width: float = 0.72,
    bar_alpha: float = 0.90,
    dpi: int = 300,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot 100%-stacked bars with optional ribbons and condition grouping."""
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

    if celltype_order is not None:
        order = [str(value) for value in celltype_order]
    elif celltype_colors is not None:
        order = [str(value) for value in celltype_colors]
    else:
        order = list(counts.index)
    unexpected = sorted(set(counts.index) - set(order))
    if unexpected:
        raise ValueError(f"celltype_order omits cell types: {unexpected}")
    counts = counts.reindex(order, fill_value=0)

    if celltype_colors is None:
        generated = sns.color_palette("tab20", n_colors=max(1, len(order)))
        colors: Mapping[str, object] = dict(zip(order, generated))
    else:
        colors = celltype_colors
        missing_colors = [celltype for celltype in order if celltype not in colors]
        if missing_colors:
            raise ValueError(f"celltype_colors is missing cell types: {missing_colors}")

    samples = [str(value) for value in (sample_order or counts.columns)]
    counts.columns = counts.columns.astype(str)
    missing_samples = sorted(set(samples) - set(counts.columns))
    if missing_samples:
        raise KeyError(f"sample_order contains unknown columns: {missing_samples}")

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
            raise KeyError(f"No {condition!r} metadata for samples: {missing_condition}")
        groups = list(condition_order or pd.unique([sample_conditions[s] for s in samples]))
        unknown_groups = sorted(set(sample_conditions[s] for s in samples) - set(groups))
        if unknown_groups:
            raise ValueError(f"condition_order omits {condition!r} values: {unknown_groups}")
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

    counts = counts.loc[:, samples].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = counts.sum(axis=0)
    if (totals <= 0).any():
        bad = totals.index[totals <= 0].tolist()
        raise ValueError(f"Samples with zero retained cells: {bad}")
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
        color = colors[celltype]
        if show_shading and len(samples) > 1:
            for i in range(len(samples) - 1):
                if condition_values is not None and condition_values[i] != condition_values[i + 1]:
                    continue
                ax.fill_between(
                    [x[i] + bar_width / 2, x[i + 1] - bar_width / 2],
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

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=rotation, ha="center", fontsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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


def _niche_order(labels, prefix):
    """Sort niche labels by their numeric suffix."""
    del prefix  # retained for compatibility with the original notebook helper
    return sorted(
        labels,
        key=lambda value: (
            int(str(value)[1:]) if str(value)[1:].isdigit() else 999,
            str(value),
        ),
    )


def _plot_spatial_categorical(
    ax: plt.Axes,
    sub: pd.DataFrame,
    col: str,
    palette: Mapping[str, object] | None = None,
    canonical_ct_labels: Sequence[str] | None = None,
) -> None:
    """Plot one categorical spatial scatter panel on an existing axis."""
    if palette is None and col == "final_CT":
        labels = sorted(sub[col].astype(str).unique())
        canonical = canonical_ct_labels or load_global_l2_class_names() or labels
        overrides = xenium_final_ct_rgba_overrides(labels, canonical_labels=canonical)
        if overrides:
            palette = {key: value[:3] for key, value in overrides.items()}

    if palette is not None:
        if col in ("CNiche", "TNiche"):
            labels = _niche_order(sub[col].astype(str).unique(), col[0])
        else:
            labels = sorted(sub[col].astype(str).unique())
        hue_order = [label for label in labels if label in palette]
        sns.scatterplot(
            data=sub,
            x="x_centroid",
            y="y_centroid",
            hue=col,
            hue_order=hue_order,
            palette={key: palette[key] for key in hue_order},
            s=2,
            alpha=0.7,
            ax=ax,
            linewidth=0,
        )
    else:
        sns.scatterplot(
            data=sub,
            x="x_centroid",
            y="y_centroid",
            hue=col,
            s=2,
            alpha=0.7,
            ax=ax,
            palette="tab20",
            linewidth=0,
        )
    ax.set_title(col, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(
        title=col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        markerscale=3,
    )


def plot_final_ct_by_lineage(
    df: pd.DataFrame,
    *,
    lineage_col: str = "final_lineage",
    ct_col: str = "final_CT",
    lineage_order: Sequence[str] | None = None,
    lineage_cmaps: Mapping[str, str] | None = None,
    default_cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 300,
    save_path: str | Path | None = None,
    skip_if_exists: bool = False,
    show: bool = True,
) -> plt.Figure | None:
    """Plot category counts within each lineage as a compact panel grid."""
    if lineage_col not in df.columns or ct_col not in df.columns:
        raise KeyError(f"DataFrame needs {lineage_col!r} and {ct_col!r}")

    plot_df = df.dropna(subset=[lineage_col, ct_col]).copy()
    present = set(plot_df[lineage_col].astype(str))
    order = list(lineage_order) if lineage_order is not None else sorted(present)
    panels = [label for label in order if label in present]
    panels.extend(sorted(present - set(panels)))
    if not panels:
        raise ValueError(f"No rows with valid {lineage_col}/{ct_col}")

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()
    cmap_by_lineage = lineage_cmaps or {}

    for ax, lineage in zip(axes_flat, panels):
        sub = plot_df.loc[plot_df[lineage_col].astype(str) == lineage]
        counts = sub[ct_col].value_counts().sort_values(ascending=False)
        colors = sns.color_palette(
            cmap_by_lineage.get(lineage, default_cmap),
            n_colors=len(counts),
        )
        x = range(len(counts))
        bars = ax.bar(x, counts.values, color=colors, edgecolor="none")
        ax.set_xticks(list(x))
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of cells")
        ax.set_title(lineage, fontsize=13)
        ax.set_ylim(0, counts.max() * 1.12 if len(counts) else 1)
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                format_cell_count_label(value),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        if not (skip_if_exists and out.is_file()):
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out}")
        else:
            print(f"Skip save, using existing: {out}")
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig
