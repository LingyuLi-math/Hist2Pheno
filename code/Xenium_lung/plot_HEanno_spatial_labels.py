## 2026.06.22: LLY, plot the spatial maps for incomplete cases, inverse the y-axis

"""Spatial scatter plots for HE-annotated Xenium cells (CNiche, TNiche, lineage).

Terminal usage (from repo root esccAI):

# Complete + Incomplete
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py

# Complete_Cases only
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \\
    --cases-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases

# Incomplete_Cases only
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \\
    --cases-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases

# 指定单个 sample 测试
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \
--cases-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases \
--sample TILD028LA

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_FIGURES_DIR = Path(
    "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/Xemiun/weiqin/"
    "SpatialPF-NGenetics/Spatial-PF-Processed/Annotation/HE_Annotations/"
    "LLY_annotation/Figures"
)
DEFAULT_LABELS_HE_PATH = (
    DEFAULT_FIGURES_DIR.parent
    / "DavisXenium_cells_partitioned_by_annotation.csv"
)

CNICHE_COLORS = {
    "C1": "#003399", "C2": "#666633", "C3": "#CC0033", "C4": "#99CC66",
    "C5": "#9999FF", "C6": "#66CCCC", "C7": "#FF9966", "C8": "#993366",
    "C9": "#996633", "C10": "#000000", "C11": "#66CCFF", "C12": "#CCCC00",
}
TNICHE_COLORS = {
    "T1": "#99FFCC", "T2": "#000000", "T3": "#808000", "T4": "#FFCC99",
    "T5": "#33CC33", "T6": "#993300", "T7": "#003300", "T8": "#0066CC",
    "T9": "#FF99FF", "T10": "#CC0066", "T11": "#330033", "T12": "#99CC00",
}
LINEAGE_COLORS = {
    "Epithelial": "#8103fb",
    "Immune": "#2adddc",
    "Endothelial": "#d4df8a",
    "Mesenchymal": "#f80505",
}
SUBLINEAGE_COLORS = {
    "Alveolar": "#FF7F0E",
    "Airway": "#1F77B4",
    "Myeloid": "#9467BD",
    "Lymphoid": "#8C564B",
    "Endothelial": LINEAGE_COLORS["Endothelial"],
    "Mesenchymal": LINEAGE_COLORS["Mesenchymal"],
}
NICHE_PALETTES = {
    "CNiche": CNICHE_COLORS,
    "TNiche": TNICHE_COLORS,
    "final_lineage": LINEAGE_COLORS,
    "final_sublineage": SUBLINEAGE_COLORS,
}


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        float(alpha),
    )


def xenium_lineage_rgba_overrides(labels) -> dict[str, tuple[float, float, float, float]]:
    """RGBA overrides for ``final_lineage`` / 4-class lineage (same as GT spatial maps)."""
    out = {}
    for lb in labels:
        key = str(lb)
        if key in LINEAGE_COLORS:
            out[key] = _hex_to_rgba(LINEAGE_COLORS[key])
    return out


def xenium_final_ct_rgba_overrides(
    labels,
    canonical_labels=None,
) -> dict[str, tuple[float, float, float, float]]:
    """RGBA overrides for ``final_CT`` / fine cell types (tab20 + tab20b, as GT final_CT panel).

    When ``canonical_labels`` is given (e.g. training ``class_names``), colors are fixed
    by alphabetical order in that universe so pred / true spatial panels stay consistent.
    """
    import matplotlib.pyplot as plt

    if canonical_labels is not None:
        universe = sorted({str(x) for x in canonical_labels})
    else:
        universe = sorted({str(x) for x in labels})
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    pool = np.vstack([colors_tab20, colors_tab20b])
    out = {}
    for i, name in enumerate(universe):
        rgba = pool[i % len(pool)]
        out[name] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    return out


def resolve_xenium_spatial_color_overrides(
    unique_labels,
    *,
    tier: str = "auto",
    canonical_labels=None,
) -> dict[str, tuple[float, float, float, float]] | None:
    """
    tier: ``lineage`` | ``ct`` | ``auto`` (lineage if all labels are in LINEAGE_COLORS else CT).
    """
    labels = [str(x) for x in unique_labels]
    if tier == "lineage" or (
        tier == "auto" and labels and set(labels) <= set(LINEAGE_COLORS.keys())
    ):
        overrides = xenium_lineage_rgba_overrides(labels)
        return overrides or None
    return xenium_final_ct_rgba_overrides(labels, canonical_labels=canonical_labels) or None

DEFAULT_SPATIAL_COLS = [
    "CNiche",
    "TNiche",
    "final_lineage",
    "final_sublineage",
    "final_CT",
]
DEFAULT_SUBPLOT_ROWS = 3
DEFAULT_SUBPLOT_COLS = 2

# 2x2 panel order matching paper-style figures (a)
LINEAGE_PANEL_ORDER = ["Endothelial", "Immune", "Epithelial", "Mesenchymal"]
LINEAGE_BAR_CMAPS = {
    "Endothelial": "YlOrBr",
    "Immune": "RdPu",
    "Epithelial": "Greens",
    "Mesenchymal": "PuBu",
}


def _niche_order(labels, prefix):
    del prefix  # kept for API compatibility with notebook helper
    return sorted(labels, key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)


def _plot_spatial_categorical(ax, sub, col, palette=None):
    if palette is not None:
        if col in ("CNiche", "TNiche"):
            labels = _niche_order(sub[col].astype(str).unique(), col[0])
        else:
            labels = sorted(sub[col].astype(str).unique())
        hue_order = [lb for lb in labels if lb in palette]
        sns.scatterplot(
            data=sub,
            x="x_centroid",
            y="y_centroid",
            hue=col,
            hue_order=hue_order,
            palette={k: palette[k] for k in hue_order},
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


def _format_cell_count_label(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(int(n))


def plot_final_ct_by_lineage(
    df: pd.DataFrame,
    *,
    lineage_col: str = "final_lineage",
    ct_col: str = "final_CT",
    lineage_order: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 300,
    save_path: str | Path | None = None,
    skip_if_exists: bool = False,
    show: bool = True,
) -> plt.Figure | None:
    """Bar plots of final_CT counts within each final_lineage (2x2 panels).

    Always renders and shows the figure in the notebook. If ``save_path`` is set,
    saves to disk only when the file is missing or ``skip_if_exists`` is False.

    In Jupyter, use ``show=True`` (default) and do not rely on the return value,
    otherwise the figure is displayed twice (``plt.show()`` + cell output).
    """
    if lineage_col not in df.columns or ct_col not in df.columns:
        raise KeyError(f"DataFrame needs {lineage_col!r} and {ct_col!r}")

    plot_df = df.dropna(subset=[lineage_col, ct_col]).copy()
    order = lineage_order or LINEAGE_PANEL_ORDER
    present = set(plot_df[lineage_col].astype(str))
    panels = [lb for lb in order if lb in present]
    extra = sorted(present - set(panels))
    panels = panels + extra
    if not panels:
        raise ValueError(f"No rows with valid {lineage_col}/{ct_col}")

    n_panels = len(panels)
    nrows = 2 if n_panels > 2 else 1
    ncols = 2 if n_panels > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, lineage in zip(axes_flat, panels):
        sub = plot_df.loc[plot_df[lineage_col].astype(str) == lineage]
        counts = sub[ct_col].value_counts().sort_values(ascending=False)
        cmap_name = LINEAGE_BAR_CMAPS.get(lineage, "viridis")
        colors = sns.color_palette(cmap_name, n_colors=len(counts))
        x = range(len(counts))
        bars = ax.bar(x, counts.values, color=colors, edgecolor="none")
        ax.set_xticks(list(x))
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of cells")
        # ax.set_title(lineage, fontsize=13, fontweight="bold")
        ax.set_title(lineage, fontsize=13)
        ax.set_ylim(0, counts.max() * 1.12 if len(counts) else 1)
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                _format_cell_count_label(int(val)),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        should_save = not (skip_if_exists and out.is_file())
        if should_save:
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


def _resolve_save_path(dataset_select: str, save_path: str | Path | None) -> Path:
    if save_path is None:
        out_dir = DEFAULT_FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{dataset_select}_spatial_maps.jpg"

    path = Path(save_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{dataset_select}_spatial_maps.jpg"


#########################################################################
## 2026.06.22: LLY, plot the spatial maps for incomplete cases, inverse the y-axis
#########################################################################
def plot_spatial_he_maps(
    dataset_select: str,
    labels_he_df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
    labels_he_path: str | Path | None = None,
    spatial_cols: list[str] | None = None,
    subplot_rows: int = DEFAULT_SUBPLOT_ROWS,
    subplot_cols: int = DEFAULT_SUBPLOT_COLS,
    figsize: tuple[float, float] = (14, 18),
    dpi: int = 150,
    invert_y: bool = True,    # because the high_res HE image is up to down, so we need to invert the y-axis
    show: bool = True,
) -> Path:
    """Plot spatial maps (default 3x2) for one sample and optionally save as JPEG.

    Parameters
    ----------
    dataset_select : str
        Sample id in ``labels_he_df['sample']`` (e.g. ``'VUHD113'``, ``'VUILD107MA'``).
    labels_he_df : pd.DataFrame | None
        HE annotation table. If ``None``, reads ``labels_he_path`` or ``DEFAULT_LABELS_HE_PATH``.
    labels_he_path : str | Path | None
        CSV path used when ``labels_he_df`` is not provided.
    save_path : str | Path | None
        Output ``.jpg`` file, or directory (default: ``DEFAULT_FIGURES_DIR``).
        If ``None``, writes ``{DEFAULT_FIGURES_DIR}/{dataset_select}_spatial_maps.jpg``.
    spatial_cols : list[str] | None
        Columns to plot (default: CNiche, TNiche, final_lineage, final_sublineage, final_CT).
    subplot_rows, subplot_cols : grid layout (default 3x2; hides unused panels).
    figsize, dpi : figure size and save resolution.
    invert_y : bool
        If True, flip the y-axis so coordinates match HE image orientation (y down).
    show : bool
        Call ``plt.show()`` when True.

    Returns
    -------
    Path
        Path to the saved JPEG file.
    """
    spatial_cols = spatial_cols or DEFAULT_SPATIAL_COLS
    if labels_he_df is None:
        csv_path = Path(labels_he_path or DEFAULT_LABELS_HE_PATH)
        labels_he_df = pd.read_csv(csv_path)
    labels_he = labels_he_df[labels_he_df["sample"] == dataset_select]
    plot_df = labels_he.dropna(subset=["x_centroid", "y_centroid"]).copy()
    print(f"{dataset_select}: n_cells={len(plot_df):,}")

    n_plots = len(spatial_cols)
    n_axes = subplot_rows * subplot_cols
    if n_plots > n_axes:
        raise ValueError(
            f"Need at least {n_plots} subplot panels, got {subplot_rows}x{subplot_cols}={n_axes}"
        )

    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
    axes_flat = axes.flatten()
    for ax, col in zip(axes_flat, spatial_cols):
        sub = plot_df.dropna(subset=[col])
        _plot_spatial_categorical(ax, sub, col, palette=NICHE_PALETTES.get(col))
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.suptitle(
        f"Spatial maps — {dataset_select} (n={len(plot_df):,})",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    if invert_y:
        for ax in axes_flat[:n_plots]:
            ax.invert_yaxis()

    out_path = _resolve_save_path(dataset_select, save_path)
    fig.savefig(out_path, format="jpg", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
)
DEFAULT_LABELS_HE_MATCH_PIXEL_PATH = (
    _DATA_ROOT
    / "Annotation/HE_Annotations/cells_partitioned_by_annotation_sample_match_with_pixel.csv"
)
COMPLETE_CASES = _DATA_ROOT / "Data/Complete_Cases"
INCOMPLETE_CASES = _DATA_ROOT / "Data/Incomplete_Cases"
DEFAULT_CASES_ROOTS = (COMPLETE_CASES, INCOMPLETE_CASES)
GT_SPATIAL_MAPS_SUFFIX = "_GT_spatial_maps.jpg"


def load_labels_he_df(labels_path: Path | str | None = None) -> pd.DataFrame:
    """Load per-cell HE annotation table (notebook cell: labels_HE_df)."""
    path = Path(labels_path or DEFAULT_LABELS_HE_MATCH_PIXEL_PATH)
    df = pd.read_csv(path)
    print("labels_HE_df samples:", sorted(df["sample"].unique()))
    print(f"rows: {len(df):,}")
    return df


def list_case_sample_dirs(cases_root: Path | str) -> list[str]:
    """Sample folder names under Complete_Cases or Incomplete_Cases."""
    root = Path(cases_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def plot_gt_spatial_maps_for_cases_root(
    labels_he_df: pd.DataFrame,
    cases_root: Path | str,
    *,
    spatial_cols: list[str] | None = None,
    subplot_rows: int = 3,
    subplot_cols: int = 2,
    figsize: tuple[float, float] = (14, 18),
    dpi: int = 300,
    invert_y: bool = True,
    show: bool = False,
    samples: list[str] | None = None,
) -> list[Path]:
    """Plot and save {sample}_GT_spatial_maps.jpg into each sample folder."""
    cases_root = Path(cases_root)
    sample_ids = samples or list_case_sample_dirs(cases_root)
    if not sample_ids:
        print(f"WARNING: no sample folders under {cases_root}, skip")
        return []

    spatial_cols = spatial_cols or DEFAULT_SPATIAL_COLS
    written: list[Path] = []
    print(f"Plotting {len(sample_ids)} sample(s) under {cases_root.name} ...")
    for dataset_select in sample_ids:
        save_path = cases_root / dataset_select / f"{dataset_select}{GT_SPATIAL_MAPS_SUFFIX}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out = plot_spatial_he_maps(
            dataset_select,
            labels_he_df=labels_he_df,
            spatial_cols=spatial_cols,
            subplot_rows=subplot_rows,
            subplot_cols=subplot_cols,
            figsize=figsize,
            dpi=dpi,
            invert_y=invert_y,
            save_path=save_path,
            show=show,
        )
        written.append(out)
        print(f"  {dataset_select}: {out}")
    return written


def plot_gt_spatial_maps_batch(
    *,
    labels_path: Path | str | None = None,
    cases_roots: Path | str | tuple[Path | str, ...] | list[Path | str] = DEFAULT_CASES_ROOTS,
    spatial_cols: list[str] | None = None,
    subplot_rows: int = 3,
    subplot_cols: int = 2,
    figsize: tuple[float, float] = (14, 18),
    dpi: int = 300,
    invert_y: bool = True,
    show: bool = False,
    samples: list[str] | None = None,
) -> list[Path]:
    """Mirror notebook: load labels CSV, then plot Complete + Incomplete cases."""
    labels_he_df = load_labels_he_df(labels_path)
    if isinstance(cases_roots, (str, Path)):
        roots = (Path(cases_roots),)
    else:
        roots = tuple(Path(p) for p in cases_roots)

    written: list[Path] = []
    for cases_root in roots:
        written.extend(
            plot_gt_spatial_maps_for_cases_root(
                labels_he_df,
                cases_root,
                spatial_cols=spatial_cols,
                subplot_rows=subplot_rows,
                subplot_cols=subplot_cols,
                figsize=figsize,
                dpi=dpi,
                invert_y=invert_y,
                show=show,
                samples=samples,
            )
        )
    print(f"Wrote {len(written)} GT spatial map(s)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot HE annotation spatial maps (CNiche, TNiche, lineage, final_CT) "
            "and save {sample}_GT_spatial_maps.jpg under each case folder."
        )
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_HE_MATCH_PIXEL_PATH,
        help="Per-cell HE table with pixel columns (default: sample_match_with_pixel.csv)",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Case root(s): Complete_Cases and/or Incomplete_Cases (repeatable). "
            f"Default: both {COMPLETE_CASES.name} + {INCOMPLETE_CASES.name}"
        ),
    )
    parser.add_argument("--sample", action="append", default=None, help="Limit to sample id(s)")
    parser.add_argument("--subplot-rows", type=int, default=3)
    parser.add_argument("--subplot-cols", type=int, default=2)
    parser.add_argument("--figsize", type=float, nargs=2, default=(14.0, 18.0), metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-invert-y", action="store_true", help="Do not flip y-axis for HE orientation")
    parser.add_argument("--show", action="store_true", help="Display each figure (default: save only)")
    args = parser.parse_args()

    cases_roots = tuple(args.cases_dir) if args.cases_dir else DEFAULT_CASES_ROOTS
    plot_gt_spatial_maps_batch(
        labels_path=args.labels_csv,
        cases_roots=cases_roots,
        subplot_rows=args.subplot_rows,
        subplot_cols=args.subplot_cols,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        invert_y=not args.no_invert_y,
        show=args.show,
        samples=args.sample,
    )


if __name__ == "__main__":
    main()