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
PRED_ROOT = _REPO / "data" / "PanCancer" / "pred_all_spatial"
CKPT = (
    _REPO
    / "data"
    / "PanCancer"
    / "result_all_spatial"
    / "cross_dataset_cv"
    / "D_emph_L2_spatial_pancancer"
    / "best_mlp_gpu.pt"
)
FOLD_PLAN = _REPO / "data" / "PanCancer" / "fold_plan.csv"
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
