"""Reusable figure panels for external-val and Fig2/Fig3 notebooks.

Notebooks keep sample IDs, paths, and visual parameters (marker size, DPI,
figsize, fonts). Plotting logic lives here so cells stay short call sites.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "STARDIST_ROC_FILES",
    "GT_PRED_OBS_COLS",
    "PRED_OBS_COLS",
    "scatter_point_s",
    "load_rectangle_roi",
    "nuclei_in_rectangle",
    "crop_he_rectangle",
    "matched_pred_h5ad",
    "stardist_roc_dir",
    "load_stardist_roc_tier",
    "load_stardist_roc_tiers",
    "plot_roi_pred_spatial",
    "plot_pred_composition_stacked",
    "plot_gt_pred_spatial",
    "plot_auroc_gt_pred_composition",
    "plot_stardist_tier_roc",
    "load_stardist_polys",
    "plot_stardist_roi_boundary",
    "plot_annotated_roi_nuclei",
]

PRED_OBS_COLS = {
    "L1": "pred_lineage",
    "L12": "pred_sublineage",
    "L2": "pred_CT",
}
GT_PRED_OBS_COLS = [
    ("L1", "final_lineage", "pred_lineage"),
    ("L12", "final_sublineage", "pred_sublineage"),
    ("L2", "final_CT", "pred_CT"),
]
STARDIST_ROC_FILES = {
    "L1": (
        "validation_external_stardist_matched_AUROC_level1.csv",
        "validation_external_stardist_matched_AUROC_level1_class_names.csv",
        "final_lineage",
    ),
    "L12": (
        "validation_external_stardist_matched_AUROC_level12.csv",
        "validation_external_stardist_matched_AUROC_level12_class_names.csv",
        "final_sublineage",
    ),
    "L2": (
        "validation_external_stardist_matched_AUROC.csv",
        "validation_external_stardist_matched_AUROC_class_names.csv",
        "final_CT",
    ),
}


def scatter_point_s(marker_px: float, dpi: float) -> float:
    """Matplotlib scatter ``s`` from display pixels at ``dpi`` (points²)."""
    return (float(marker_px) * 72.0 / float(dpi)) ** 2


def load_rectangle_roi(tsv_path: str | Path, *, row: int = 0) -> dict[str, float]:
    """Read QuPath-style rectangle TSV with ``x_min_px`` … ``y_max_px``."""
    frame = pd.read_csv(tsv_path, sep="\t")
    r = frame.iloc[int(row)]
    return {
        "x_min_px": float(r["x_min_px"]),
        "y_min_px": float(r["y_min_px"]),
        "x_max_px": float(r["x_max_px"]),
        "y_max_px": float(r["y_max_px"]),
    }


def nuclei_in_rectangle(
    xy: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    """Boolean mask: nuclei with HE coords inside ``[x0,x1) × [y0,y1)``."""
    xy = np.asarray(xy, dtype=np.float64)
    return (
        (xy[:, 0] >= x0)
        & (xy[:, 0] < x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] < y1)
    )


def crop_he_rectangle(
    he_path: str | Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    """RGB crop of an HE image for the given native-pixel rectangle."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(he_path) as im:
        crop = im.convert("RGB").crop(
            (
                int(np.floor(x0)),
                int(np.floor(y0)),
                int(np.ceil(x1)),
                int(np.ceil(y1)),
            )
        )
    return np.asarray(crop)


def matched_pred_h5ad(
    data_root: str | Path,
    cancer: str,
    sample: str,
) -> Path:
    root = Path(data_root)
    return (
        root
        / "PanCancer/pred_all_spatial/matched"
        / cancer
        / sample
        / f"{sample}_matched_pred.h5ad"
    )


def stardist_roc_dir(
    data_root: str | Path,
    cancer: str,
    sample: str,
) -> Path:
    return Path(data_root) / "PanCancer/pred_all_spatial/stardist_roc" / f"{cancer}__{sample}"


def load_stardist_roc_tier(
    roc_dir: str | Path,
    pred_name: str,
    names_name: str,
    name_col: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``(class_names, y_true_idx, score[n, n_class])`` for one tier CSV."""
    roc_dir = Path(roc_dir)
    pred = pd.read_csv(roc_dir / pred_name)
    names = pd.read_csv(roc_dir / names_name).sort_values("class_index")
    classes = names[name_col].astype(str).tolist()
    prob_cols = sorted(
        [c for c in pred.columns if c.startswith("prob_")],
        key=lambda c: int(c.split("_")[1]),
    )
    y = pred["ground_truth_idx"].to_numpy(dtype=np.int64)
    score = pred[prob_cols].to_numpy(dtype=np.float64)
    if score.shape[1] != len(classes):
        raise RuntimeError(
            f"{roc_dir.name}/{pred_name}: {score.shape[1]} probs vs {len(classes)} classes"
        )
    return classes, y, score


def load_stardist_roc_tiers(
    roc_dir: str | Path,
    *,
    files: Mapping[str, tuple[str, str, str]] | None = None,
) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    files = STARDIST_ROC_FILES if files is None else files
    return {
        level: load_stardist_roc_tier(roc_dir, *spec)
        for level, spec in files.items()
    }


def _color_rgb(palette: Mapping[str, Any], name: str, fallback=(0.55, 0.55, 0.55, 1.0)):
    return palette.get(name, fallback)[:3]


def _present_order(palette: Mapping[str, Any], seen: set[str]) -> list[str]:
    order = [name for name in palette if name in seen]
    order += sorted(seen - set(order))
    return order


def plot_roi_pred_spatial(
    adata,
    he_path: str | Path,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    palettes: Mapping[str, Mapping[str, Any]],
    in_roi: np.ndarray | None = None,
    legend_mask: np.ndarray | None = None,
    obs_cols: Mapping[str, str] | None = None,
    marker_px: float = 5.0,
    dpi: float = 200,
    figsize: tuple[float, float] = (20.4, 14.82),
    title: str | None = None,
    legend_markersize: float = 13,
    legend_fontsize: float = 13,
    title_fontsize: float = 11,
    suptitle_fontsize: float = 12,
    save_path: str | Path | None = None,
    save_dpi: int = 300,
):
    """2×3 ROI panel: HE+points / points-only × L1 / L12 / L2 (pred only).

    ``legend_mask`` is a boolean over all nuclei. When set, each column legend
    lists only labels present in that mask (still drawn inside the ROI).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    obs_cols = PRED_OBS_COLS if obs_cols is None else obs_cols
    xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    if in_roi is None:
        in_roi = nuclei_in_rectangle(xy, x0, y0, x1, y1)
    in_roi = np.asarray(in_roi, dtype=bool)
    n = int(in_roi.sum())
    if n == 0:
        raise RuntimeError("No predicted nuclei inside the ROI box.")

    he_roi = crop_he_rectangle(he_path, x0, y0, x1, y1)
    roi_w, roi_h = he_roi.shape[1], he_roi.shape[0]
    xy_local = np.column_stack([xy[in_roi, 0] - x0, xy[in_roi, 1] - y0])
    point_s = scatter_point_s(marker_px, dpi)

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    for col, head in enumerate(("L1", "L12", "L2")):
        labels = adata.obs[obs_cols[head]].astype(str).to_numpy()[in_roi]
        palette = palettes[head]
        colors = np.asarray(
            [_color_rgb(palette, str(lab)) for lab in labels],
            dtype=np.float64,
        )
        for row, show_he in enumerate((True, False)):
            ax = axes[row, col]
            if show_he:
                ax.imshow(he_roi, extent=(0, roi_w, roi_h, 0), interpolation="bilinear")
            else:
                ax.set_facecolor("white")
            ax.scatter(
                xy_local[:, 0],
                xy_local[:, 1],
                c=colors,
                s=point_s,
                linewidths=0,
                rasterized=True,
                zorder=3,
            )
            ax.set_xlim(0, roi_w)
            ax.set_ylim(roi_h, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            suffix = "HE" if show_he else "points"
            ax.set_title(f"{head}  {suffix}  n={n:,}", fontsize=title_fontsize)

        # legend_mask is full-length; the legend lists only those nuclei (e.g. inside a TLS outline).
        if legend_mask is None:
            seen = set(map(str, labels.tolist()))
        else:
            legend_mask = np.asarray(legend_mask, dtype=bool)
            if legend_mask.shape != in_roi.shape:
                raise ValueError("legend_mask length must match the number of nuclei")
            seen = set(map(str, labels[legend_mask[in_roi]].tolist()))
        names = _present_order(palette, seen)
        ncol = 1 if len(names) <= 5 else (2 if len(names) <= 10 else 4)
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=legend_markersize,
                markerfacecolor=_color_rgb(palette, name, (0.5, 0.5, 0.5, 1)),
                markeredgecolor="none",
                label=name,
            )
            for name in names
        ]
        axes[1, col].legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            fontsize=legend_fontsize,
            frameon=False,
            ncol=ncol,
            handletextpad=0.35,
            columnspacing=1.0,
            labelspacing=0.4,
        )

    if title is None:
        title = f"QuPath ROI  [{x0:.0f}:{x1:.0f}]×[{y0:.0f}:{y1:.0f}]  (no GT)"
    fig.suptitle(title, fontsize=suptitle_fontsize)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.93, hspace=0.28)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    return fig, in_roi


def plot_pred_composition_stacked(
    obs: pd.DataFrame,
    palettes: Mapping[str, Mapping[str, Any]],
    *,
    obs_cols: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (12.0, 4.2),
    dpi: float = 200,
    title: str | None = None,
    bar_width: float = 0.55,
    pct_in_legend: bool = True,
    title_fontsize: float = 11,
    legend_fontsize_l1: float = 9,
    legend_fontsize_l2: float = 8,
    save_path: str | Path | None = None,
    save_dpi: int = 300,
    csv_path: str | Path | None = None,
):
    """1×3 100% stacked bars of predicted L1 / L12 / L2 composition."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    obs_cols = PRED_OBS_COLS if obs_cols is None else obs_cols
    n = len(obs)
    if n == 0:
        raise RuntimeError("Empty obs for composition plot.")

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    rows: list[dict[str, Any]] = []
    for ax, head in zip(axes, ("L1", "L12", "L2")):
        col = obs_cols[head]
        labels = obs[col].astype(str)
        palette = palettes[head]
        counts = labels.value_counts()
        order = _present_order(palette, set(counts.index.astype(str)))
        bottom = 0.0
        for name in order:
            frac = float(counts[name]) / float(n)
            ax.bar(
                0,
                frac,
                bottom=bottom,
                width=bar_width,
                color=_color_rgb(palette, name),
                edgecolor="none",
            )
            bottom += frac
            rows.append(
                {"head": head, "class": name, "n": int(counts[name]), "fraction": frac}
            )
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("fraction" if head == "L1" else "")
        ax.set_title(f"{head}  n={n:,}", fontsize=title_fontsize)
        handles = []
        for name in order:
            label = name
            if pct_in_legend:
                label = f"{name}  {100 * float(counts[name]) / n:.1f}%"
            handles.append(
                Patch(
                    facecolor=_color_rgb(palette, name),
                    edgecolor="none",
                    label=label,
                )
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            frameon=False,
            fontsize=legend_fontsize_l2 if head == "L2" else legend_fontsize_l1,
            ncol=1 if len(order) <= 6 else 2,
            handletextpad=0.3,
            labelspacing=0.3,
        )

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    comp_df = pd.DataFrame(rows)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(csv_path, index=False)
    return fig, comp_df


def plot_gt_pred_spatial(
    adata,
    palettes: Mapping[str, Mapping[str, Any]],
    *,
    columns: Sequence[tuple[str, str, str]] | None = None,
    point_s: float = 0.4,
    figsize_cm: tuple[float, float] = (16.4, 12.6),
    title: str | None = None,
    title_fontsize: float = 7,
    label_fontsize: float = 7,
    legend_fontsize: float = 5.5,
    legend_markersize: float = 4,
    height_ratios: Sequence[float] = (1.0, 1.0, 0.62),
    hspace: float = 0.08,
    wspace: float = 0.12,
    spatial_key: str = "spatial_HE",
):
    """3×3 GT vs Pred spatial panel (L1 / L12 / L2) with bottom legends."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = GT_PRED_OBS_COLS if columns is None else columns
    xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    def scatter_labels(ax, labels, palette):
        labels = np.asarray(labels, dtype=str)
        for name, color in palette.items():
            mask = labels == name
            if np.any(mask):
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=point_s,
                    c=[color],
                    linewidths=0,
                    rasterized=True,
                )
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].max(), xy[:, 1].min())
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54))
    gs = fig.add_gridspec(
        3, 3, height_ratios=list(height_ratios), hspace=hspace, wspace=wspace
    )
    for j, (head, gt_col, pred_col) in enumerate(columns):
        palette = palettes[head]
        for i, (col, row_name) in enumerate(
            ((gt_col, "Ground truth"), (pred_col, "Hist2Pheno"))
        ):
            ax = fig.add_subplot(gs[i, j])
            scatter_labels(ax, adata.obs[col].to_numpy(), palette)
            if i == 0:
                ax.set_title(head, fontsize=title_fontsize, pad=2)
            if j == 0:
                ax.set_ylabel(row_name, fontsize=label_fontsize)
        seen = set(adata.obs[gt_col].astype(str)) | set(adata.obs[pred_col].astype(str))
        present = [name for name in palette if name in seen]
        axL = fig.add_subplot(gs[2, j])
        axL.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=legend_markersize,
                markerfacecolor=palette[name],
                markeredgecolor="none",
                label=name,
            )
            for name in present
        ]
        axL.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            fontsize=legend_fontsize,
            ncol=2 if len(present) > 4 else 1,
            columnspacing=0.6,
            handletextpad=0.25,
            borderaxespad=0.0,
        )
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.98)
    return fig


def _draw_tier_roc(
    ax,
    classes: Sequence[str],
    y: np.ndarray,
    score: np.ndarray,
    palette: Mapping[str, Any],
    *,
    level: str,
    macro_fn,
    lw: float = 0.8,
    legend_fontsize: float | None = None,
    legend_loc: str = "lower right",
    legend_bbox=None,
    show_ylabel: bool = True,
    title_fontsize: float = 7,
):
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    macro = float(macro_fn(y, score, len(classes)))
    Y = label_binarize(y, classes=np.arange(len(classes)))
    if Y.ndim == 1:
        Y = np.column_stack([1 - Y, Y])
    rows = []
    for k, name in enumerate(classes):
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == len(Y):
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], score[:, k])
        class_auc = float(auc(fpr, tpr))
        ax.plot(
            fpr,
            tpr,
            color=_color_rgb(palette, name, (0.4, 0.4, 0.4, 1)),
            lw=lw,
            label=f"{name}  {class_auc:.2f}",
        )
        rows.append(
            {"level": level, "class": name, "auroc": class_auc, "n_positive": pos}
        )
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.4, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR")
    if show_ylabel:
        ax.set_ylabel("TPR")
    ax.set_title(f"{level}   macro {macro:.3f}", fontsize=title_fontsize, pad=2)
    ax.set_aspect("equal", adjustable="box")
    fs = legend_fontsize
    if fs is None:
        fs = 4.5 if level == "L2" else 5
    legend_kw = dict(
        frameon=False,
        fontsize=fs,
        loc=legend_loc,
        handlelength=1.0,
        borderaxespad=0.15,
    )
    if legend_bbox is not None:
        legend_kw["bbox_to_anchor"] = legend_bbox
        legend_kw.pop("borderaxespad", None)
    ax.legend(**legend_kw)
    return macro, rows


def _draw_gt_pred_composition_bar(
    ax,
    gt: pd.Series,
    pred: pd.Series,
    palette: Mapping[str, Any],
    *,
    level: str,
    bar_width: float = 0.62,
    show_ylabel: bool = True,
    title_fontsize: float = 7,
    legend_fontsize: float | None = None,
):
    from matplotlib.patches import Patch

    seen = set(gt.astype(str)) | set(pred.astype(str))
    order = _present_order(palette, seen)
    n = float(len(gt))
    bottoms = {"GT": 0.0, "Pred": 0.0}
    rows = []
    for name in order:
        color = _color_rgb(palette, name)
        for x, series, key in ((0, gt, "GT"), (1, pred, "Pred")):
            frac = float((series.astype(str) == name).sum()) / n
            ax.bar(
                x,
                frac,
                bottom=bottoms[key],
                width=bar_width,
                color=color,
                edgecolor="none",
                linewidth=0,
            )
            bottoms[key] += frac
            rows.append(
                {
                    "level": level,
                    "source": key,
                    "class": name,
                    "n": int((series.astype(str) == name).sum()),
                    "fraction": frac,
                }
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GT", "Pred"])
    ax.set_ylim(0, 1)
    if show_ylabel:
        ax.set_ylabel("fraction")
    ax.set_title(f"{level} composition", fontsize=title_fontsize, pad=2)
    fs = legend_fontsize
    if fs is None:
        fs = 4.5 if level == "L2" else 5
    handles = [
        Patch(facecolor=_color_rgb(palette, name), edgecolor="none", label=name)
        for name in order
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=fs,
        ncol=2 if len(order) > 4 else 1,
        columnspacing=0.5,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    return rows


def plot_auroc_gt_pred_composition(
    adata,
    roc_dir: str | Path,
    palettes: Mapping[str, Mapping[str, Any]],
    *,
    columns: Sequence[tuple[str, str, str]] | None = None,
    macro_fn=None,
    figsize_cm: tuple[float, float] = (16.4, 11.2),
    title: str | None = None,
    title_fontsize: float = 7,
    height_ratios: Sequence[float] = (1.15, 1.0),
    hspace: float = 0.42,
    wspace: float = 0.28,
    auroc_csv: str | Path | None = None,
    composition_csv: str | Path | None = None,
):
    """2×3 panel: per-class ROC (top) and GT vs Pred composition (bottom)."""
    import matplotlib.pyplot as plt

    if macro_fn is None:
        from uni_label_cv_helpers import macro_auc_ovr as macro_fn

    columns = GT_PRED_OBS_COLS if columns is None else columns
    tiers = load_stardist_roc_tiers(roc_dir)

    fig = plt.figure(figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54))
    gs = fig.add_gridspec(
        2, 3, height_ratios=list(height_ratios), hspace=hspace, wspace=wspace
    )
    auc_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []

    for j, (level, gt_col, pred_col) in enumerate(columns):
        classes, y, score = tiers[level]
        macro, rows = _draw_tier_roc(
            fig.add_subplot(gs[0, j]),
            classes,
            y,
            score,
            palettes[level],
            level=level,
            macro_fn=macro_fn,
            show_ylabel=(j == 0),
            title_fontsize=title_fontsize,
        )
        auc_rows.extend(rows)
        print(f"{level} macro-AUROC {macro:.3f}")

        comp_rows.extend(
            _draw_gt_pred_composition_bar(
                fig.add_subplot(gs[1, j]),
                adata.obs[gt_col].astype(str),
                adata.obs[pred_col].astype(str),
                palettes[level],
                level=level,
                show_ylabel=(j == 0),
                title_fontsize=title_fontsize,
            )
        )

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.01)
    auc_df = pd.DataFrame(auc_rows)
    comp_df = pd.DataFrame(comp_rows)
    if auroc_csv is not None:
        Path(auroc_csv).parent.mkdir(parents=True, exist_ok=True)
        auc_df.to_csv(auroc_csv, index=False)
    if composition_csv is not None:
        Path(composition_csv).parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(composition_csv, index=False)
    return fig, auc_df, comp_df


def plot_stardist_tier_roc(
    roc_dir: str | Path,
    palettes: Mapping[str, Mapping[str, Any]],
    *,
    macro_fn=None,
    figsize_cm: tuple[float, float] = (16.2, 5.6),
    title_fontsize: float = 7,
    legend_fontsize: float = 5,
    auroc_csv: str | Path | None = None,
):
    """1×3 per-class ROC for L1 / L12 / L2 (Figure 2d style)."""
    import matplotlib.pyplot as plt

    if macro_fn is None:
        from uni_label_cv_helpers import macro_auc_ovr as macro_fn

    tiers = load_stardist_roc_tiers(roc_dir)
    fig, axes = plt.subplots(
        1, 3, figsize=(figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    )
    auc_rows: list[dict[str, Any]] = []
    for ax, level in zip(axes, ("L1", "L12", "L2")):
        classes, y, score = tiers[level]
        if level == "L2":
            legend_loc, legend_bbox = "center left", (1.02, 0.5)
        else:
            legend_loc, legend_bbox = "lower right", None
        macro, rows = _draw_tier_roc(
            ax,
            classes,
            y,
            score,
            palettes[level],
            level=level,
            macro_fn=macro_fn,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            show_ylabel=True,
            title_fontsize=title_fontsize,
        )
        # fig2d titles omit explicit pad; keep macro in title via _draw_tier_roc
        ax.set_title(f"{level}   macro {macro:.3f}")
        auc_rows.extend(rows)
        print(f"{level} macro-AUROC {macro:.3f}")

    auc_df = pd.DataFrame(auc_rows)
    if auroc_csv is not None:
        Path(auroc_csv).parent.mkdir(parents=True, exist_ok=True)
        auc_df.to_csv(auroc_csv, index=False)
    return fig, auc_df


def load_stardist_polys(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a StarDist ``polys_prob*.pckl``.

    ``coord`` is ``(n, 2, n_rays)`` with axis 0 = y and axis 1 = x.
    ``points`` is ``(n, 2)`` as ``(y, x)``. Both are H&E pixels.
    """
    import pickle

    with Path(path).open("rb") as handle:
        packed = pickle.load(handle)
    coord = np.asarray(packed["coord"])
    points = np.asarray(packed["points"], dtype=np.float64)
    if coord.ndim != 3 or coord.shape[1] != 2 or points.shape != (coord.shape[0], 2):
        raise ValueError(f"{path} polys have unexpected shapes {coord.shape}, {points.shape}")
    return coord, points


def plot_stardist_roi_boundary(
    crop: np.ndarray,
    coord: np.ndarray,
    points: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    px_per_um: float,
    scale_um: float = 20.0,
    outline: str = "#3DDC00",
    center: str = "#F0A202",
    fig_w: float = 7.2,
    dpi: float = 120,
    point_s: float = 4.0,
    linewidth: float = 0.45,
    save_path: str | Path | None = None,
    save_dpi: int = 300,
):
    """H&E crop with StarDist boundaries and centers. Returns ``(fig, n_inside)``."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cy = points[:, 0]
    cx = points[:, 1]
    inside = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    n_in = int(inside.sum())
    height, width = crop.shape[:2]
    fig_h = fig_w * height / max(width, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    ax.imshow(crop)
    if n_in:
        poly_x = coord[inside, 1, :] - x0
        poly_y = coord[inside, 0, :] - y0
        verts = np.stack([poly_x, poly_y], axis=-1)
        verts = np.concatenate([verts, verts[:, :1, :]], axis=1)
        ax.add_collection(PolyCollection(
            verts,
            facecolors="none",
            edgecolors=outline,
            linewidths=linewidth,
            zorder=3,
        ))
        ax.scatter(
            cx[inside] - x0,
            cy[inside] - y0,
            s=point_s,
            c=center,
            linewidths=0,
            zorder=4,
        )
    bar = int(round(scale_um * px_per_um))
    yb = height - 36
    ax.plot([28, 28 + bar], [yb, yb], color="white", lw=2.4, solid_capstyle="butt", zorder=5)
    ax.text(28, yb - 10, f"{scale_um:.0f} µm", color="white", fontsize=8, va="bottom", ha="left", zorder=5)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    fig.tight_layout(pad=0.15)
    if save_path is not None:
        import matplotlib as mpl

        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest.with_suffix(".png"), dpi=save_dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return fig, n_in

## 2026.11.04 LLY add new function to plot annotated ROI nuclei
def plot_annotated_roi_nuclei(
    he: np.ndarray,
    ann: pd.DataFrame,
    xy: np.ndarray,
    labels: np.ndarray,
    palette: Mapping[str, Any],
    *,
    style: str = "dots",
    polys_path: str | Path | None = None,
    px_per_um: float,
    pad_um: float = 80.0,
    pad_x_um: float | None = None,
    pad_top_um: float | None = None,
    pad_bottom_um: float | None = None,
    reach_um: float = 25.0,
    dpi: int = 300,
    he_keep: float = 0.85,
    color_strength: float = 0.0,
    marker_px: float = 40.0,
    fill_linewidth: float = 0.35,
    corner_label: str = "Hyperplastic AECs",
    scale_um: float = 20.0,
    legend: bool = False,
    save_dir: str | Path | None = None,
    save_format: str = "png",   # 新增：可选 "png" 或 "pdf"
    show: bool = True,
) -> list:
    """One crop per annotation instance.

    ``style="dots"`` draws cell-type markers. ``style="fill"`` fills the
    StarDist polygon of each nucleus. ``polys_path`` is required for ``fill``.
    The corner label keeps the boxed 26 pt annotation. ``pad_x_um``,
    ``pad_top_um``, and ``pad_bottom_um`` override the shared ``pad_um``.
    Nucleus colors are the palette colors, drawn opaque.
    ``color_strength`` is ignored so an older call still runs.
    ``legend=True`` adds the cell types actually drawn, in palette order,
    in a margin to the right of the crop.

    Parameters
    ----------
    save_format:
        Output format. Can be "png" or "pdf".
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from scipy.spatial import cKDTree

    if style not in ("dots", "fill"):
        raise ValueError("style must be 'dots' or 'fill'")

    save_format = save_format.lower().lstrip(".")
    if save_format not in {"png", "pdf"}:
        raise ValueError("save_format must be 'png' or 'pdf'")

    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)

    poly_of = None
    coord = None

    if style == "fill":
        if polys_path is None:
            raise ValueError("style='fill' requires polys_path")
        coord, points = load_stardist_polys(polys_path)
        poly_xy = np.column_stack([points[:, 1], points[:, 0]])
        _, poly_of = cKDTree(poly_xy).query(xy, k=1)

    pad_x = int(round((pad_um if pad_x_um is None else pad_x_um) * px_per_um))
    pad_top = int(round((pad_um if pad_top_um is None else pad_top_um) * px_per_um))
    pad_bot = int(round((pad_um if pad_bottom_um is None else pad_bottom_um) * px_per_um))

    reach = reach_um * px_per_um
    point_s = (marker_px * 72 / dpi) ** 2

    height, width = he.shape[:2]
    suffix = "stardist_fill" if style == "fill" else "stardist_highlight"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    for name, group in ann.groupby("annotation_type_instance"):
        pts_ann = group[["X_pix_HE", "Y_pix_HE"]].to_numpy(dtype=np.float64)

        x0 = max(0, int(np.floor(pts_ann[:, 0].min()) - pad_x))
        x1 = min(width, int(np.ceil(pts_ann[:, 0].max()) + pad_x))
        y0 = max(0, int(np.floor(pts_ann[:, 1].min()) - pad_top))
        y1 = min(height, int(np.ceil(pts_ann[:, 1].max()) + pad_bot))

        crop = np.asarray(he[y0:y1, x0:x1])

        fade = (
            crop.astype(np.float32) * he_keep + 255 * (1 - he_keep)
        ).clip(0, 255).astype(np.uint8)

        inside = (
            (xy[:, 0] >= x0)
            & (xy[:, 0] < x1)
            & (xy[:, 1] >= y0)
            & (xy[:, 1] < y1)
        )

        dist, _ = cKDTree(pts_ann).query(
            xy[inside],
            distance_upper_bound=reach,
        )

        idx = np.flatnonzero(inside)[np.isfinite(dist)]

        hgt, wid = fade.shape[:2]
        lab = labels[idx]
        present = [cls for cls in palette if np.any(lab == cls)]
        legend_in = 0.0
        if legend and present:
            longest = max(len(str(cls)) for cls in present)
            ncol = 2 if len(present) > 18 else 1
            legend_in = (0.62 + 0.078 * longest) * ncol + 0.15

        fig_w = wid / dpi + legend_in
        fig = plt.figure(
            figsize=(fig_w, hgt / dpi),
            dpi=dpi,
            facecolor="white",
        )
        ax = fig.add_axes([0, 0, (wid / dpi) / fig_w, 1])

        ax.imshow(fade)

        if style == "dots":
            pts = xy[idx]

            for cls in palette:
                mask = lab == cls
                if not np.any(mask):
                    continue

                ax.scatter(
                    pts[mask, 0] - x0,
                    pts[mask, 1] - y0,
                    s=point_s,
                    c=[palette[cls][:3]],
                    linewidths=0,
                    zorder=3,
                )

        else:
            sel = coord[poly_of[idx]]
            verts = np.stack(
                [sel[:, 1, :] - x0, sel[:, 0, :] - y0],
                axis=-1,
            )
            verts = np.concatenate([verts, verts[:, :1, :]], axis=1)

            for cls in palette:
                mask = lab == cls
                if not np.any(mask):
                    continue

                rgb = tuple(palette[cls][:3])

                ax.add_collection(
                    PolyCollection(
                        verts[mask],
                        facecolors=[(*rgb, 1.0)],
                        edgecolors=[(*rgb, 1.0)],
                        linewidths=fill_linewidth,
                        antialiaseds=False,
                        zorder=3,
                    )
                )

        ax.set_xlim(0, wid)
        ax.set_ylim(hgt, 0)

        ax.plot(
            [0, wid, wid, 0, 0],
            [0, 0, hgt, hgt, 0],
            color="black",
            lw=1.4,
            zorder=5,
        )

        ax.set_axis_off()

        ax.text(
            0.02,
            0.08,
            corner_label,
            transform=ax.transAxes,
            fontsize=26,
            color="black",
            va="bottom",
            ha="left",
            zorder=6,
            bbox=dict(
                boxstyle="square,pad=0.3",
                fc="white",
                ec="black",
                lw=2.0,
            ),
        )

        bar = int(round(scale_um * px_per_um))
        ax.plot(
            [16, 16 + bar],
            [hgt - 18, hgt - 18],
            color="black",
            lw=3,
            solid_capstyle="butt",
            zorder=6,
        )

        if legend and present:
            leg_ax = fig.add_axes([(wid / dpi) / fig_w, 0, legend_in / fig_w, 1])
            leg_ax.set_axis_off()
            leg_ax.legend(
                handles=[
                    Line2D(
                        [0], [0], linestyle="none", marker="o", markersize=7,
                        markerfacecolor=tuple(palette[cls][:3]),
                        markeredgecolor="none",
                        label=str(cls),
                    )
                    for cls in present
                ],
                loc="center left",
                fontsize=8,
                frameon=False,
                ncol=2 if len(present) > 18 else 1,
                handletextpad=0.4,
                labelspacing=0.35,
                borderaxespad=0.4,
                title="Cell type",
                title_fontsize=9,
            )

        if save_dir is not None:
            out = save_dir / f"{name}_{suffix}.{save_format}"

            if save_format == "pdf":
                plt.rcParams["pdf.fonttype"] = 42
                plt.rcParams["ps.fonttype"] = 42
            fig.savefig(
                out,
                dpi=dpi,
                facecolor="white",
                format=save_format,
            )

            print(f"{name}: {len(idx)} nuclei on the annotation -> {out.name}")
        else:
            print(f"{name}: {len(idx)} nuclei on the annotation")

        if show:
            plt.show()

        figures.append(fig)

    return figures