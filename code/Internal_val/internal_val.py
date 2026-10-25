#!/usr/bin/env python3
"""Fold-0 internal validation of the pan-cancer cross-section checkpoint.

``Data_process_visual_xenium_all.ipynb`` and the CODEX ``Data_process_visual_*``
notebooks show **per-dataset** models. This module points at the joint
pretrained weights:

  data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt

Predictions already live under ``data/PanCancer/pred_all_spatial/`` (896
sections, including all 182 fold-0 hold-outs). Notebooks only *display* a
subset. PDAC (56 fold-0) shows the top 5 by L2 AUROC. GIST (110) keeps 4
largest slides. HCC fold 0 is 8 and we show 4 (always including Fig. 2c
``kqr-83230``) plus a paper top-5 set.
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
FOLD = 0
PRED_ROOT = _REPO / "data" / "PanCancer" / "pred_all_spatial" / "oof"
CKPT = (
    _REPO
    / "data"
    / "PanCancer"
    / "result_all_spatial"
    / "cross_dataset_cv"
    / "D_emph_L2_spatial_pancancer"
    / "hce_group_fold_0.pt"
)
FOLD_PLAN = _REPO / "data" / "PanCancer" / "result_all_spatial" / "fold_plan.csv"
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
# Always keep these if they are in fold 0 (used in Fig. 2 / the lung notebook).
PINNED = {
    "Lung": ("VUILD107MA", "VUHD113"),
    "BRCA": ("rep1",),
    "CRC": ("P2CRC",),
    "HCC": ("kqr-83230_aligned_3bba12d6",),
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
        "roc": PRED_ROOT / "gt_matched_roc" / f"{dataset}__{sample}",
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
    """Largest fold-0 slides, plus PINNED, capped at VIZ_N[dataset]."""
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
    """Fold-0 ranks. ``rank_ds`` is within the cohort; ``rank_all`` is among 182."""
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
        f"#{int(r.rank_ds)}/{int(r.n_ds)} {dataset} fold 0  "
        f"(#{int(r.rank_all)}/{int(r.n_all)} all fold 0)"
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
) -> Path:
    """Write 300 dpi GT/pred + composition and copy the 4×3 panel for paper."""
    import shutil

    import matplotlib.pyplot as plt

    ranks = rank_fold0(dataset, metric)
    out_dir = FIG_DIR / f"{dataset}_fold0_top{len(samples)}"
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
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        fig = plot_composition(dataset, sample, head, save=False)
        fig.savefig(
            out_dir / f"{stem}_composition_{head}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
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


def plot_composition(
    dataset: str,
    sample: str,
    head: str = "L2",
    *,
    ax=None,
    save: bool = False,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

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
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / f"{dataset}__{sample}_composition_{head}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(out)
    return fig


def plot_spatial_gt_pred(
    dataset: str,
    sample: str,
    head: str = "L2",
    *,
    save: bool = False,
    title: str | None = None,
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
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / f"{dataset}__{sample}_spatial_{head}_gt_pred.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
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
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return names


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
) -> dict[str, Path]:
    """Cell-type maps as a 300 dpi PNG; ROC in the pooled StarDist style as PDF.

    Written next to the existing 4x3 panel. ROC uses ``plot_multiclass_roc_curves``
    with ``pan_organ='pancancer'`` and a per-class AUROC bar, same as
    ``plot_pooled_stardist_tier_rocs``.
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
    print(f"fold 0      {len(cov)} sections")
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
        print("all fold-0 matched_pred.h5ad present — no GPU repredict needed")


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
