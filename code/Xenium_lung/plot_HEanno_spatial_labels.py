## 2026.06.22: LLY, plot the spatial maps for incomplete cases, inverse the y-axis

"""Spatial scatter plots for HE-annotated Xenium cells (CNiche, TNiche, lineage).

Terminal usage (from repo root esccAI):

# Complete + Incomplete
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py

# Complete_Cases only
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \\
    --cases-dir data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases

# Incomplete_Cases only
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \\
    --cases-dir data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases

# 指定单个 sample 测试
conda run -n SeededNTM python code/Xenium_lung/plot_HEanno_spatial_labels.py \
--cases-dir data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases \
--sample TILD028LA

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
from plotting_palettes import (  # noqa: E402
    CNICHE_COLORS,
    DEFAULT_L2_CLASS_NAMES_CSV,
    LINEAGE_COLORS,
    NICHE_PALETTES,
    SUBLINEAGE_COLORS,
    TNICHE_COLORS,
    Cell_Type_COLORS,
    _hex_to_rgba,
    load_global_l2_class_names,
    niche_palette_rgba_overrides,
    resolve_xenium_spatial_color_overrides,
    stardist_tier_rgba_overrides,
    xenium_final_ct_rgba_overrides,
    xenium_lineage_rgba_overrides,
)
from plotting_utils import (  # noqa: E402
    _format_cell_count_label,
    _niche_order,
    _plot_spatial_categorical,
    format_cell_count_label,
    plot_final_ct_by_lineage as _plot_final_ct_by_lineage,
)

DEFAULT_FIGURES_DIR = Path(
    "/home/lingyu/ssd2/Python/Hist2Pheno/data/Xemium/weiqin/"
    "SpatialPF-NGenetics/Spatial-PF-Processed/Annotation/HE_Annotations/"
    "LLY_annotation/Figures"
)
DEFAULT_LABELS_HE_PATH = (
    DEFAULT_FIGURES_DIR.parent
    / "DavisXenium_cells_partitioned_by_annotation.csv"
)



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
    """Compatibility wrapper preserving the legacy Xenium panel defaults."""
    return _plot_final_ct_by_lineage(
        df,
        lineage_col=lineage_col,
        ct_col=ct_col,
        lineage_order=lineage_order or LINEAGE_PANEL_ORDER,
        lineage_cmaps=LINEAGE_BAR_CMAPS,
        title=title,
        figsize=figsize,
        dpi=dpi,
        save_path=save_path,
        skip_if_exists=skip_if_exists,
        show=show,
    )


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
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
)
DEFAULT_LABELS_HE_MATCH_PIXEL_PATH = (
    _DATA_ROOT
    / "Annotation/HE_Annotations/cells_partitioned_by_annotation_sample_match_with_pixel.csv"
)
COMPLETE_CASES = _DATA_ROOT / "Data/Complete_Cases"
INCOMPLETE_CASES = _DATA_ROOT / "Data/Incomplete_Cases"
DEFAULT_CASES_ROOTS = (COMPLETE_CASES, INCOMPLETE_CASES)
GT_SPATIAL_MAPS_SUFFIX = "_GT_spatial_maps.jpg"

# --- Xenium lung: per-sample CSV paths / cell-count QC (notebook HEanno & ST sections) ---
GT_SUFFIX = "_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
STARDIST_SUFFIX = "_Float_prob0.01_nms_0.3.csv"
GT_X_COL = "X_pix_HE"
GT_Y_COL = "Y_pix_HE"
STARDIST_X_COL = "centroid_x"
STARDIST_Y_COL = "centroid_y"
MOESM5_XLSX_PATH = _DATA_ROOT / "Annotation/HE_Annotations/41588_2025_2080_MOESM5_ESM.xlsx"
MOESM5_PAPER_ST_SHEET = "Supplementary Table 2"


@dataclass(frozen=True)
class XeniumCountConfig:
    """Xenium lung paths and column names for GT / StarDist count & spatial loaders."""

    cases_root: Path
    gt_suffix: str = GT_SUFFIX
    stardist_suffix: str = STARDIST_SUFFIX
    gt_x_col: str = GT_X_COL
    gt_y_col: str = GT_Y_COL
    stardist_x_col: str = STARDIST_X_COL
    stardist_y_col: str = STARDIST_Y_COL
    tumor_id_suffix: str = "_all"


def xenium_cases_root(data_class: str = "Complete_Cases") -> Path:
    """``Data/{Complete_Cases|Incomplete_Cases}`` under the Xenium processed root."""
    return _DATA_ROOT / "Data" / data_class


def list_case_sample_dirs(cases_root: Path | str) -> list[str]:
    """Sample folder names under Complete_Cases or Incomplete_Cases."""
    root = Path(cases_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())

########################################################
## 2026.06.22: LLY, add the sample_csv_path function
##             clean Data_process_visual_xenium_all.ipynb
########################################################
def sample_csv_path(cases_root: Path, sample: str, suffix: str) -> Path:
    return Path(cases_root) / sample / f"{sample}{suffix}"


def gt_csv_path(config: XeniumCountConfig, sample: str) -> Path:
    return sample_csv_path(config.cases_root, sample, config.gt_suffix)


def stardist_csv_path(config: XeniumCountConfig, sample: str) -> Path:
    return sample_csv_path(config.cases_root, sample, config.stardist_suffix)


def count_csv_rows(csv_path: Path) -> int:
    return len(pd.read_csv(csv_path))


def load_spatial_coords(
    csv_path: Path,
    sample: str,
    *,
    x_col: str,
    y_col: str,
    tumor_id_suffix: str = "_all",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[x_col, y_col]).copy()
    df["TumorID"] = f"{sample}{tumor_id_suffix}"
    return df


def load_gt_coords(config: XeniumCountConfig, sample: str) -> pd.DataFrame:
    return load_spatial_coords(
        gt_csv_path(config, sample),
        sample,
        x_col=config.gt_x_col,
        y_col=config.gt_y_col,
        tumor_id_suffix=config.tumor_id_suffix,
    )


def load_stardist_coords(config: XeniumCountConfig, sample: str) -> pd.DataFrame:
    return load_spatial_coords(
        stardist_csv_path(config, sample),
        sample,
        x_col=config.stardist_x_col,
        y_col=config.stardist_y_col,
        tumor_id_suffix=config.tumor_id_suffix,
    )


def build_gt_stardist_counts_df(
    config: XeniumCountConfig,
    *,
    samples: Sequence[str] | None = None,
    skip_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-sample GT vs StarDist totals (``n_gt``, ``n_stardist``, ``ratio_stardist_gt``)."""
    sample_list = list(samples) if samples is not None else list_case_sample_dirs(config.cases_root)
    rows = []
    for sample in sample_list:
        gt_path = gt_csv_path(config, sample)
        sd_path = stardist_csv_path(config, sample)
        if not gt_path.is_file() or not sd_path.is_file():
            if verbose:
                missing = []
                if not gt_path.is_file():
                    missing.append("GT")
                if not sd_path.is_file():
                    missing.append("StarDist")
                print(f"SKIP {sample}: missing {' + '.join(missing)} CSV")
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"{sample}: GT={gt_path.is_file()}, StarDist={sd_path.is_file()}"
            )
        rows.append(
            {
                "sample": sample,
                "n_gt": count_csv_rows(gt_path),
                "n_stardist": count_csv_rows(sd_path),
            }
        )

    out = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if not out.empty:
        out["ratio_stardist_gt"] = out["n_stardist"] / out["n_gt"]
    return out


def load_paper_cell_counts_from_xlsx(
    xlsx_path: Path | str = MOESM5_XLSX_PATH,
    *,
    sheet_name: str = MOESM5_PAPER_ST_SHEET,
    header: int = 1,
) -> pd.DataFrame:
    """Paper Xenium totals from MOESM5 Supplementary Table 2 (Sample, Total Cells)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header)
    out = df[["Sample", "Total Cells"]].dropna(subset=["Sample"]).copy()
    out["Sample"] = out["Sample"].astype(str).str.strip()
    out["Total Cells"] = pd.to_numeric(out["Total Cells"], errors="coerce")
    out = out.dropna(subset=["Total Cells"]).astype({"Total Cells": int})
    return out.rename(columns={"Sample": "sample", "Total Cells": "n_paper_st"})


def build_paper_st_stardist_counts_df(
    config: XeniumCountConfig,
    paper_counts: pd.DataFrame,
    *,
    paper_sample_col: str = "sample",
    paper_count_col: str = "n_paper_st",
    skip_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-sample Paper ST vs StarDist totals."""
    paper_by_sample = paper_counts.set_index(paper_sample_col)[paper_count_col]
    rows = []
    for sample in list_case_sample_dirs(config.cases_root):
        if sample not in paper_by_sample.index:
            continue
        sd_path = stardist_csv_path(config, sample)
        if not sd_path.is_file():
            if verbose:
                print(f"SKIP {sample}: missing StarDist CSV")
            if skip_missing:
                continue
            raise FileNotFoundError(sd_path)
        rows.append(
            {
                "sample": sample,
                "n_paper_st": int(paper_by_sample.loc[sample]),
                "n_stardist": count_csv_rows(sd_path),
            }
        )

    out = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if not out.empty:
        out["ratio_stardist_paper"] = out["n_stardist"] / out["n_paper_st"]
        out["pct_stardist_of_paper"] = 100.0 * out["ratio_stardist_paper"]
    return out

########################################################
## 2026.07.02: LLY, add barplot for Paper ST vs StarDist counts
########################################################
def build_gt_stardist_paper_counts_df(
    config: XeniumCountConfig,
    paper_counts: pd.DataFrame,
    *,
    paper_sample_col: str = "sample",
    paper_count_col: str = "n_paper_st",
    skip_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-sample GT HE annotation, StarDist, and Paper ST totals."""
    gt_sd = build_gt_stardist_counts_df(config, skip_missing=skip_missing, verbose=verbose)
    if gt_sd.empty:
        return gt_sd

    paper_by_sample = paper_counts.set_index(paper_sample_col)[paper_count_col]
    out = gt_sd.copy()
    out["n_paper_st"] = out["sample"].map(paper_by_sample)
    missing_paper = out["n_paper_st"].isna()
    if missing_paper.any() and verbose:
        print(
            "SKIP (no Paper ST):",
            sorted(out.loc[missing_paper, "sample"].tolist()),
        )
    out = out.dropna(subset=["n_paper_st"]).astype({"n_paper_st": int})
    out["ratio_stardist_paper"] = out["n_stardist"] / out["n_paper_st"]
    cols = ["sample", "n_gt", "n_paper_st", "n_stardist", "ratio_stardist_gt", "ratio_stardist_paper"]
    return out[cols].sort_values("sample").reset_index(drop=True)
########################################################

def counts_df_for_gt_stardist_plots(
    counts_df: pd.DataFrame,
    *,
    gt_col: str = "n_gt",
    stardist_col: str = "n_stardist",
) -> pd.DataFrame:
    """Rename columns for ``Hist2Pheno_pkg.plot.plot_gt_stardist_*`` helpers."""
    rename = {}
    if gt_col != "n_gt" and gt_col in counts_df.columns:
        rename[gt_col] = "n_gt"
    if stardist_col != "n_stardist" and stardist_col in counts_df.columns:
        rename[stardist_col] = "n_stardist"
    return counts_df.rename(columns=rename)


def load_labels_he_df(labels_path: Path | str | None = None) -> pd.DataFrame:
    """Load per-cell HE annotation table (notebook cell: labels_HE_df)."""
    path = Path(labels_path or DEFAULT_LABELS_HE_MATCH_PIXEL_PATH)
    df = pd.read_csv(path)
    print("labels_HE_df samples:", sorted(df["sample"].unique()))
    print(f"rows: {len(df):,}")
    return df
########################################################

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