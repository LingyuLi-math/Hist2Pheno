#!/usr/bin/env python3
"""Fold-0 view of the UNI2 A2 best model.

``best_mlp_gpu.pt`` is a copy of ``hce_group_fold_1.pt``. That checkpoint's
own validation sections are fold 1, so these notebooks do not show fold 1.
They show fold 0. Deployment predictions for all 896 training sections are in
``pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment/``.
PDAC (56) shows the top 5 by matched-cell L2 AUROC. GIST (110) keeps the
4 largest slides. HCC fold 0 is 8 and we show the 4 largest.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "code" / "Hist2Pheno_pkg"
_PAN = _REPO / "code" / "PanCancer"
for _p in (_PKG, _PAN, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plotting_palettes import get_palette  # noqa: E402

DATASETS = ("Lung", "BRCA", "CRC", "HCC", "PDAC", "GIST", "GBM")
# best_mlp_gpu.pt is hce_group_fold_1.pt, so the gallery is the other complete fold.
FOLD = 0
_UNI2 = "result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42"
PRED_ROOT = (
    _REPO
    / "data"
    / "PanCancer"
    / "pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment"
)
CKPT = (
    _REPO
    / "data"
    / "PanCancer"
    / _UNI2
    / "cross_dataset_cv"
    / "A2_adjacent_spatial_k8mean"
    / "best_mlp_gpu.pt"
)
FOLD_PLAN = _REPO / "data" / "PanCancer" / _UNI2 / "fold_plan.csv"
METRICS = PRED_ROOT / "metrics_per_section.csv"
FIG_DIR = PRED_ROOT / "internal_val_figures"

# How many fold-0 slides to draw in each notebook.
VIZ_N = {
    "Lung": 5,
    "BRCA": 1,
    "CRC": 1,
    "HCC": 4,
    "PDAC": 5,
    "GIST": 4,
    "GBM": 1,
}
# Kept first when they belong to FOLD.
PINNED = {
    "Lung": ("VUILD107MA",),
    "BRCA": ("rep1",),
    "CRC": ("P1CRC",),
    "GBM": ("P174511_Initial",),
}

L2_PALETTE = get_palette("pancancer", "fine")
L12_PALETTE = get_palette("pancancer", "intermediate")
L1_PALETTE = get_palette("pancancer", "coarse")
PALETTES = {"L1": L1_PALETTE, "L12": L12_PALETTE, "L2": L2_PALETTE}
OBS_GT = {"L1": "final_lineage", "L12": "final_sublineage", "L2": "final_CT"}
OBS_PR = {"L1": "pred_lineage", "L12": "pred_sublineage", "L2": "pred_CT"}


def fold0_table(dataset: str | None = None) -> pd.DataFrame:
    plan = pd.read_csv(FOLD_PLAN)
    plan["fold"] = plan["fold"].astype(int)
    out = plan.loc[plan["fold"] == FOLD].copy()
    if dataset:
        out = out.loc[out["dataset"] == dataset].copy()
    out = out.sort_values(["dataset", "n_keep"], ascending=[True, False]).reset_index(drop=True)
    return out


def pred_paths(dataset: str, sample: str) -> dict[str, Path]:
    matched = PRED_ROOT / "matched" / dataset / sample
    return {
        "pred": matched / f"{sample}_matched_pred.h5ad",
        "panel": matched / f"{sample}_panel_4x3.png",
        "star": PRED_ROOT
        / "stardist_all"
        / dataset
        / sample
        / f"{sample}_all_stardist_pred.h5ad",
        "roc": PRED_ROOT / "stardist_roc" / f"{dataset}__{sample}",
    }


def coverage(dataset: str | None = None) -> pd.DataFrame:
    rows = []
    for rec in fold0_table(dataset).itertuples(index=False):
        p = pred_paths(str(rec.dataset), str(rec.sample))
        rows.append(
            {
                "dataset": rec.dataset,
                "sample": rec.sample,
                "n_keep": int(rec.n_keep),
                "pred": p["pred"].is_file(),
                "panel": p["panel"].is_file(),
                "star": p["star"].is_file(),
            }
        )
    return pd.DataFrame(rows)


def viz_samples(dataset: str) -> list[str]:
    """Largest fold slides, plus PINNED, capped at VIZ_N[dataset]."""
    tab = fold0_table(dataset)
    if tab.empty:
        return []
    want = list(PINNED.get(dataset, ()))
    for sample in tab["sample"].astype(str):
        if sample not in want:
            want.append(sample)
    n = int(VIZ_N.get(dataset, 4))
    keep = [s for s in want if s in set(tab["sample"].astype(str))]
    return keep[:n]


def metrics_fold0(dataset: str | None = None) -> pd.DataFrame:
    if not METRICS.is_file():
        return pd.DataFrame()
    met = pd.read_csv(METRICS)
    f0 = fold0_table(dataset)
    keys = set(zip(f0["dataset"].astype(str), f0["sample"].astype(str)))
    # r.sample is DataFrame.sample; the column must be r["sample"].
    met = met.loc[
        met.apply(lambda r: (str(r["dataset"]), str(r["sample"])) in keys, axis=1)
    ].copy()
    return met.sort_values(["dataset", "macro_auc_L2_cells"], ascending=[True, False])


def rank_fold0(
    dataset: str | None = None,
    metric: str = "macro_auc_L2_cells",
) -> pd.DataFrame:
    """Fold ranks. ``rank_ds`` is within the cohort; ``rank_all`` is among every section in this fold."""
    all_m = metrics_fold0()
    if all_m.empty or metric not in all_m.columns:
        return pd.DataFrame()
    out = all_m.copy()
    out["rank_all"] = out[metric].rank(ascending=False, method="min").astype(int)
    out["n_all"] = len(out)
    parts = []
    for _, g in out.groupby("dataset", sort=False):
        g = g.copy()
        g["rank_ds"] = g[metric].rank(ascending=False, method="min").astype(int)
        g["n_ds"] = len(g)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    if dataset:
        out = out.loc[out["dataset"] == dataset].copy()
    return out.sort_values(["dataset", "rank_ds", "sample"]).reset_index(drop=True)


def rank_label(dataset: str, sample: str, metric: str = "macro_auc_L2_cells") -> str:
    tab = rank_fold0(dataset, metric)
    hit = tab.loc[tab["sample"].astype(str) == str(sample)]
    if hit.empty:
        return str(sample)
    r = hit.iloc[0]
    return (
        f"{sample}  L2 AUROC={float(r[metric]):.3f}  "
        f"#{int(r.rank_ds)}/{int(r.n_ds)} {dataset} fold {FOLD}  "
        f"(#{int(r.rank_all)}/{int(r.n_all)} all fold {FOLD})"
    )


def top_fold0_samples(
    dataset: str,
    n: int = 5,
    metric: str = "macro_auc_L2_cells",
) -> list[str]:
    tab = rank_fold0(dataset, metric)
    if tab.empty:
        return []
    return tab.sort_values(["rank_ds", "sample"])["sample"].astype(str).tolist()[: int(n)]


def export_paper_sections(
    dataset: str,
    samples: list[str],
    *,
    head: str = "L2",
    metric: str = "macro_auc_L2_cells",
    dpi: int = 300,
    composition_fmt: str = "png",
) -> Path:
    """Write GT/pred PNGs and composition figures, and copy the 4×3 panel."""
    import shutil

    import matplotlib.pyplot as plt

    ranks = rank_fold0(dataset, metric)
    out_dir = FIG_DIR / f"{dataset}_fold{FOLD}_top{len(samples)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(samples, 1):
        hit = ranks.loc[ranks["sample"].astype(str) == str(sample)]
        rank_ds = int(hit.iloc[0]["rank_ds"]) if len(hit) else i
        stem = f"{rank_ds:02d}_{sample}"
        src_panel = pred_paths(dataset, sample)["panel"]
        if src_panel.is_file():
            dest = out_dir / f"{stem}_panel_4x3.png"
            shutil.copy2(src_panel, dest)
            print(dest)
        fig = plot_spatial_gt_pred(dataset, sample, head, save=False)
        fig.savefig(
            out_dir / f"{stem}_spatial_{head}_gt_pred.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        fig = plot_composition(dataset, sample, head, save=False)
        comp = out_dir / f"{stem}_composition_{head}.{composition_fmt}"
        _save_figure(fig, comp, dpi=dpi)
        print(comp)
        plt.close(fig)
        if len(hit):
            rows.append(hit.iloc[0].to_dict())
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "ranks.csv", index=False)
    print(out_dir)
    return out_dir


def composition(dataset: str, sample: str, head: str = "L2") -> pd.DataFrame:
    import anndata as ad

    path = pred_paths(dataset, sample)["pred"]
    adata = ad.read_h5ad(path)
    gt = adata.obs[OBS_GT[head]].astype(str)
    pr = adata.obs[OBS_PR[head]].astype(str)
    names = list(PALETTES[head])
    present = [n for n in names if n in set(gt) or n in set(pr)]
    present += sorted((set(gt) | set(pr)) - set(present))
    return pd.DataFrame(
        {
            "class": present,
            "gt": [float((gt == n).mean()) for n in present],
            "pred": [float((pr == n).mean()) for n in present],
        }
    )


def _save_figure(fig, path: Path, *, dpi: int = 300) -> None:
    import matplotlib as mpl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pdf":
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_composition(
    dataset: str,
    sample: str,
    head: str = "L2",
    *,
    ax=None,
    save: bool = False,
    title: str | None = None,
    dpi: int = 300,
    fmt: str = "png",
):
    import matplotlib.pyplot as plt

    if fmt not in ("png", "pdf"):
        raise ValueError("fmt must be png or pdf")
    frac = composition(dataset, sample, head)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * len(frac) + 1.5), 3.4))
    else:
        fig = ax.figure
    x = np.arange(len(frac))
    w = 0.38
    colors = [PALETTES[head].get(c, (0.55, 0.55, 0.55, 1.0))[:3] for c in frac["class"]]
    ax.bar(x - w / 2, frac["gt"], w, color=colors, alpha=0.45, label="GT")
    ax.bar(x + w / 2, frac["pred"], w, color=colors, alpha=0.95, label="Hist2Pheno")
    ax.set_xticks(x)
    ax.set_xticklabels(frac["class"], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("fraction")
    ax.set_title(title or rank_label(dataset, sample))
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    if save:
        out = FIG_DIR / f"{dataset}__{sample}_composition_{head}.{fmt}"
        _save_figure(fig, out, dpi=dpi)
        print(out)
    return fig


## 2026.11.05 LLY add function to plot proportion sankey
def plot_proportion_sankey(
    dataset: str,
    sample: str,
    head: str = "L2",
    *,
    save: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (11.6, 7.4),
):
    """Three stacked proportion columns linked by cell type.

    Columns are Ground truth, Hist2Pheno on the matched cells, and Hist2Pheno
    on every StarDist nucleus. Each band keeps that cell type's palette color.
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath

    paths = pred_paths(dataset, sample)
    matched = ad.read_h5ad(paths["pred"])
    if not paths["star"].is_file():
        raise FileNotFoundError(paths["star"])
    star = ad.read_h5ad(paths["star"])
    series = (
        ("Ground truth", matched.obs[OBS_GT[head]].astype(str).to_numpy()),
        ("Hist2Pheno-match", matched.obs[OBS_PR[head]].astype(str).to_numpy()),
        ("Hist2Pheno-WSI", star.obs[OBS_PR[head]].astype(str).to_numpy()),
    )
    palette = PALETTES[head]
    seen = set(np.concatenate([lab for _, lab in series]))
    names = [n for n in palette if n in seen]
    names += sorted(seen - set(names))

    def _weights(lab: np.ndarray) -> list[float]:
        vc = pd.Series(lab).value_counts(normalize=True)
        return [float(vc.get(n, 0.0)) for n in names]

    def _stack(weights: list[float], y0: float = 0.03, y1: float = 0.94, gap: float = 0.007):
        kept = [(n, w) for n, w in zip(names, weights) if w > 0]
        usable = (y1 - y0) - gap * max(len(kept) - 1, 0)
        total = sum(w for _, w in kept)
        cursor = y1
        out = {}
        for n, w in kept:
            height = usable * (w / total)
            out[n] = (cursor - height, cursor)
            cursor -= height + gap
        return out

    columns = [(title, _stack(_weights(lab))) for title, lab in series]
    xs = ((0.06, 0.20), (0.43, 0.57), (0.80, 0.94))

    def _ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color):
        xm = (x0 + x1) / 2.0
        path = MPath(
            [
                (x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
                (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a),
            ],
            [
                MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY,
            ],
        )
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=0.45, zorder=1))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.axis("off")
    ax.set_title(f"{sample}   {head} cell-type proportion", fontsize=12, pad=6)
    for i in range(2):
        for name in names:
            if name not in columns[i][1] or name not in columns[i + 1][1]:
                continue
            ya, yb = columns[i][1][name]
            ya2, yb2 = columns[i + 1][1][name]
            color = palette.get(name, (0.55, 0.55, 0.55, 1.0))[:3]
            _ribbon(ax, xs[i][1], ya, yb, xs[i + 1][0], ya2, yb2, color)
    for (x0, x1), (title, box) in zip(xs, columns):
        ax.text((x0 + x1) / 2, 0.985, title, ha="center", va="bottom", fontsize=11, color="#222")
        for name, (ya, yb) in box.items():
            color = palette.get(name, (0.55, 0.55, 0.55, 1.0))[:3]
            ax.add_patch(Rectangle((x0, ya), x1 - x0, yb - ya, facecolor=color, edgecolor="white", linewidth=0.6, zorder=3))
            if yb - ya < 0.028:
                continue
            red, green, blue = color
            ink = "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.55 else "black"
            ax.text((x0 + x1) / 2, (ya + yb) / 2, name.replace("_", " "), ha="center", va="center", fontsize=7, color=ink, zorder=4)
    if save:
        out = FIG_DIR / f"{dataset}__{sample}_proportion_sankey_{head}.pdf"
        _save_figure(fig, out, dpi=dpi)
        print(out)
    return fig


def plot_spatial_gt_pred(
    dataset: str,
    sample: str,
    head: str = "L2",
    *,
    save: bool = False,
    title: str | None = None,
    dpi: int = 300,
):
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path = pred_paths(dataset, sample)["pred"]
    adata = ad.read_h5ad(path)
    xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    gt = adata.obs[OBS_GT[head]].astype(str).to_numpy()
    pr = adata.obs[OBS_PR[head]].astype(str).to_numpy()
    palette = PALETTES[head]
    names = [n for n in palette if n in set(gt) or n in set(pr)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, lab, title in ((axes[0], gt, "GT"), (axes[1], pr, "pretrained Hist2Pheno")):
        for name in names:
            mask = lab == name
            if not np.any(mask):
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=0.4,
                c=[palette[name][:3]],
                linewidths=0,
                rasterized=True,
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=palette[n][:3],
                markeredgecolor="none",
                label=n,
            )
            for n in names
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=6,
    )
    fig.suptitle(title or rank_label(dataset, sample), fontsize=9, y=1.02)
    fig.tight_layout()
    if save:
        out = FIG_DIR / f"{dataset}__{sample}_spatial_{head}_gt_pred.png"
        _save_figure(fig, out, dpi=dpi)
        print(out)
    return fig


def _scatter_classes(ax, xy, labels, palette, s: float) -> list[str]:
    labels = np.asarray(labels, dtype=str)
    present = {str(x) for x in labels}
    names = [n for n in palette if n in present]
    for name in names:
        mask = labels == name
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=s,
            c=[palette[name][:3]],
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return names

#############################################
# 2026.11.05 LLY add HE image background function
#############################################
def _he_background(he_image: str | Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """H&E array plus ``imshow`` extent in ``spatial_HE`` pixels.

    ``VUILD107MA-HE_lowres.jpg`` is not a uniform shrink of the TIFF
    (2514×2457 versus 19966×20710), so stretching it onto the slide shifts
    the tissue. Sample ``VUILD107MA-HE.tif`` on a uniform stride instead.
    ``X_pix_HE`` / ``Y_pix_HE`` in the coordinate table are these TIFF pixels.
    """
    path = Path(he_image)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            shape = tif.pages[0].shape
        height, width = int(shape[0]), int(shape[1])
        step = max(1, int(np.ceil(max(height, width) / 2500)))
        image = np.asarray(tifffile.memmap(path)[::step, ::step])
        n_r, n_c = image.shape[:2]
        # Coordinate x hits TIFF column x, the same frame as X_pix_HE.
        extent = (
            -0.5 * step,
            (n_c - 0.5) * step,
            (n_r - 0.5) * step,
            -0.5 * step,
        )
        return image, extent
    from PIL import Image

    image = np.asarray(Image.open(path))
    height, width = image.shape[:2]
    return image, (-0.5, width - 0.5, height - 0.5, -0.5)
#############################################

def save_celltype_and_roc(
    dataset: str,
    sample: str,
    *,
    dpi: int = 300,
    celltype_figsize: tuple[float, float] | None = None,
    stardist_figsize: tuple[float, float] | None = None,
    roc_figsize: tuple[float, float] = (4.0, 4.0),
    bar_figsize: tuple[float, float] | None = None,
    point_s: float = 0.45,
    he_image: str | Path | None = None,
    he_alpha: float = 0.8,
) -> dict[str, Path]:
    """Cell-type maps as a 300 dpi PNG; ROC in the pooled StarDist style as PDF.

    Written next to the existing 4x3 panel. ROC uses ``plot_multiclass_roc_curves``
    with ``pan_organ='pancancer'`` and a per-class AUROC bar, same as
    ``plot_pooled_stardist_tier_rocs``. ``he_image`` draws that H&E under the
    maps. ``he_alpha`` is its opacity (1 keeps the slide, lower fades it toward white).
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from panel_figures import load_stardist_roc_tiers

    paths = pred_paths(dataset, sample)
    matched = ad.read_h5ad(paths["pred"])
    star = ad.read_h5ad(paths["star"]) if paths["star"].is_file() else None
    xy = np.asarray(matched.obsm["spatial_HE"], dtype=np.float64)
    xy_star = None if star is None else np.asarray(star.obsm["spatial_HE"], dtype=np.float64)
    heads = ("L1", "L12", "L2")
    matched_rows = (
        ("GT", xy, {h: matched.obs[OBS_GT[h]].astype(str).to_numpy() for h in heads}, point_s),
        ("Hist2Pheno", xy, {h: matched.obs[OBS_PR[h]].astype(str).to_numpy() for h in heads}, point_s),
    )
    star_rows = ()
    if xy_star is not None:
        star_rows = (
            (
                "StarDist all",
                xy_star,
                {h: star.obs[OBS_PR[h]].astype(str).to_numpy() for h in heads},
                point_s * 0.4,
            ),
        )
    ## 2026.11.05 LLY add HE image background function
    he_bg = None if he_image is None else _he_background(he_image)
    if he_bg is not None and not 0.0 <= float(he_alpha) <= 1.0:
        raise ValueError("he_alpha must be between 0 and 1")

    def _render(rows, figsize, stem):
        n_leg = max(len(PALETTES[h]) for h in heads)
        extra_h = 0.18 * int(np.ceil(n_leg / 2)) + 0.35
        if figsize is None:
            figsize = (12.5, 3.3 * len(rows) + extra_h)
        fig, axes = plt.subplots(
            len(rows) + 1,
            3,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3.0] * len(rows) + [extra_h]},
        )
        seen = {h: set() for h in heads}
        for i, (row_name, row_xy, labs, s) in enumerate(rows):
            for j, head in enumerate(heads):
                ax = axes[i, j]

                ## 2026.11.05 LLY add HE image background function
                if he_bg is not None:
                    image, extent = he_bg
                    ax.imshow(
                        image,
                        extent=extent,
                        alpha=float(he_alpha),
                        zorder=0,
                        interpolation="bilinear",
                    )
                    ax.set_facecolor("white")
                    
                names = _scatter_classes(ax, row_xy, labs[head], PALETTES[head], s)
                seen[head].update(names)
                if j == 0:
                    ax.set_ylabel(row_name, fontsize=8)
                if i == 0:
                    ax.set_title(head, fontsize=11)
        frame = np.concatenate([row_xy for _, row_xy, _, _ in rows], axis=0)
        pad_x = (frame[:, 0].max() - frame[:, 0].min()) * 0.02
        pad_y = (frame[:, 1].max() - frame[:, 1].min()) * 0.02
        xlim = (frame[:, 0].min() - pad_x, frame[:, 0].max() + pad_x)
        ylim = (frame[:, 1].max() + pad_y, frame[:, 1].min() - pad_y)
        for i in range(len(rows)):
            for j in range(3):
                axes[i, j].set_xlim(*xlim)
                axes[i, j].set_ylim(*ylim)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        for j, head in enumerate(heads):
            axL = axes[-1, j]
            axL.axis("off")
            names = [n for n in PALETTES[head] if n in seen[head]]
            axL.legend(
                handles=[
                    Line2D(
                        [0], [0], linestyle="none", marker="o", markersize=6,
                        markerfacecolor=PALETTES[head][n][:3], markeredgecolor="none", label=n,
                    )
                    for n in names
                ],
                loc="upper center",
                fontsize=6.5,
                frameon=False,
                ncol=2 if len(names) > 6 else 1,
                columnspacing=0.7,
                handletextpad=0.3,
                borderaxespad=0.0,
            )
        fig.suptitle(sample, fontsize=12, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out_path = paths["panel"].with_name(stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    celltype = _render(matched_rows, celltype_figsize, f"{sample}_celltype.png")
    stardist = None
    if star_rows:
        stardist = _render(star_rows, stardist_figsize, f"{sample}_celltype_stardist.png")

    from uni_label_cv_helpers import plot_per_class_auc_bars, plot_stardist_tier_roc_pooled

    tiers = load_stardist_roc_tiers(paths["roc"])
    roc_titles = {
        "l2": "StarDist Level2 ROC",
        "l12": "StarDist Level12 ROC",
        "l1": "StarDist Level1 ROC",
    }
    roc_pdfs: dict[str, Path] = {}
    for tier in ("l2", "l12", "l1"):
        head = tier.upper() if tier != "l12" else "L12"
        names, y_true, y_score = tiers[head]
        pdf = paths["panel"].with_name(f"{sample}_roc_{tier}.pdf")
        info = plot_stardist_tier_roc_pooled(
            y_true,
            y_score,
            names,
            tier=tier,
            save_path=str(pdf),
            title=f"{roc_titles[tier]} — {sample}",
            pan_organ="pancancer",
            figsize=roc_figsize,
        )
        roc_pdfs[tier] = pdf
        per_class = None if info is None else info.get("per_class_auc")
        if per_class is not None and len(per_class):
            plot_per_class_auc_bars(
                per_class,
                pan_organ="pancancer",
                color_tier=tier,
                ylabel="cell type",
                title=f"Per-class one-vs-rest AUROC  ({head})",
                macro_auc=info.get("macro_auc"),
                figsize=bar_figsize,
                save_path=paths["panel"].with_name(f"{sample}_roc_{tier}_per_class_auc.pdf"),
                show=False,
            )
            macro = info.get("macro_auc", float("nan"))
            print(f"  {head} macro AUROC = {float(macro):.4f}")

    old_roc = paths["panel"].with_name(f"{sample}_roc.pdf")
    if old_roc.is_file():
        old_roc.unlink()

    out = {"panel": paths["panel"], "celltype": celltype, "stardist": stardist, "roc": roc_pdfs["l2"]}
    print(f"panel: {paths['panel']}")
    print(f"celltype: {celltype}")
    if stardist is not None:
        print(f"stardist: {stardist}")
    for tier, pdf in roc_pdfs.items():
        print(f"roc {tier}: {pdf}")
    return out


def cmd_status(dataset: str | None = None) -> None:
    cov = coverage(dataset)
    print(f"checkpoint  {'yes' if CKPT.is_file() else 'NO'}  {CKPT}")
    print(f"pred root   {PRED_ROOT}")
    print(f"fold {FOLD}      {len(cov)} sections")
    if cov.empty:
        return
    g = cov.groupby("dataset").agg(
        n=("sample", "size"),
        pred=("pred", "sum"),
        panel=("panel", "sum"),
    )
    print(g.to_string())
    missing = cov.loc[~cov["pred"]]
    if len(missing):
        print("missing pred:", ", ".join(f"{r.dataset}::{r.sample}" for r in missing.itertuples()))
    else:
        print(f"all fold-{FOLD} matched_pred.h5ad present — no GPU repredict needed")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("status", "fold0"))
    p.add_argument("--dataset", default=None, choices=list(DATASETS))
    args = p.parse_args(argv)
    if args.step == "status":
        cmd_status(args.dataset)
        return 0
    tab = fold0_table(args.dataset)
    print(tab[["dataset", "sample", "n_keep", "fold"]].to_string(index=False))
    print("viz subsets:")
    for ds in DATASETS if args.dataset is None else (args.dataset,):
        print(f"  {ds}: {viz_samples(ds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
