#!/usr/bin/env python3
"""Pan-cancer predict on unlabeled / Incomplete_Cases slides (no cell-type GT).

These are leftover regions from the seven training cohorts, not the Visium
External_val cancers. UNI StarDist-all h5ads already exist. This module only
loads those embeddings, applies ``D_emph_L2_spatial_pancancer``, and plots
L1 / L12 / L2. No TLS / SRI / TNI.

  Lung_Incomplete  20 Xenium Incomplete_Cases
  HCC_unlabel      8 CODEX HCC rest regions (no CODEX CSV)
  PDAC_unlabel     195 CODEX PDAC Incomplete_Cases

Lung Incomplete notebooks draw **all 20** slides, with the top 5 by matched-cell
L2 AUROC first. HCC / PDAC still use ``VIZ_N`` largest slides. ``predict``
defaults to that set. ``--all`` scores every slide.

  python -u code/External_val/unlabeled_external.py Lung_Incomplete status
  CUDA_DEVICE=0 python -u code/External_val/unlabeled_external.py HCC_unlabel predict
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_PKG = _REPO / "code" / "Hist2Pheno_pkg"
_PAN = _REPO / "code" / "PanCancer"
_PDAC = _REPO / "code" / "CODEX_pdac"
for _p in (_PKG, _PAN, _PDAC, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plotting_palettes import get_palette  # noqa: E402

HEADS = ("L1", "L12", "L2")
OBS_COL = {"L1": "pred_lineage", "L12": "pred_sublineage", "L2": "pred_CT"}
PALETTE_TIER = {"L1": "coarse", "L12": "intermediate", "L2": "fine"}
CKPT = (
    _REPO
    / "data"
    / "PanCancer"
    / "result_all_spatial"
    / "cross_dataset_cv"
    / "D_emph_L2_spatial_pancancer"
    / "best_mlp_gpu.pt"
)
SCALER = _REPO / "data" / "PanCancer" / "result_all_spatial" / "scaler.joblib"
PRED_ROOT = _REPO / "data" / "PanCancer" / "pred_unlabel"
LUNG_LEGACY = _REPO / "data" / "PanCancer" / "pred_all_spatial" / "incomplete_lung"
LUNG_CASES = (
    _REPO
    / "data"
    / "Xemium"
    / "weiqin"
    / "SpatialPF-NGenetics"
    / "Spatial-PF-Processed"
    / "Data"
    / "Incomplete_Cases"
)
HCC_HE = _REPO / "data" / "CODEX" / "HCC" / "Michael_data_transfer" / "s4769" / "HE"
HCC_REST = (
    _REPO
    / "data"
    / "CODEX"
    / "HCC"
    / "Michael_data_transfer"
    / "s4769"
    / "result_all_spatial"
    / "stardist_hcc_rest"
)
PDAC_ROOT = _REPO / "data" / "CODEX" / "HCC" / "Michael_data_transfer" / "s1167"
VIZ_N = {"Lung_Incomplete": 0, "HCC_unlabel": 5, "PDAC_unlabel": 5}
LUNG_RANKS = LUNG_LEGACY / "ranks_matched.csv"


@dataclass(frozen=True)
class Cohort:
    name: str
    feat_parent: Path
    pred_parent: Path


def _cohort(name: str) -> Cohort:
    key = name.strip()
    if key == "Lung_Incomplete":
        return Cohort(key, LUNG_CASES, LUNG_LEGACY)
    if key == "HCC_unlabel":
        return Cohort(key, HCC_HE, PRED_ROOT / key)
    if key == "PDAC_unlabel":
        return Cohort(key, PDAC_ROOT, PRED_ROOT / key)
    raise SystemExit(f"unknown cohort {name!r}; choose Lung_Incomplete, HCC_unlabel, PDAC_unlabel")


def _feat_h5ad(cohort: Cohort, sample: str) -> Path:
    return cohort.feat_parent / sample / f"{sample}_all_features_stardist.h5ad"


def pred_h5ad(cohort: Cohort, sample: str) -> Path:
    if cohort.name == "Lung_Incomplete":
        return cohort.pred_parent / sample / f"{sample}_all_stardist_pred.h5ad"
    return cohort.pred_parent / sample / f"{sample}_stardist_pred.h5ad"


def he_path(cohort: Cohort, sample: str) -> Path | None:
    root = cohort.feat_parent / sample
    if cohort.name == "Lung_Incomplete":
        tif = root / f"{sample}-HE.tif"
        return tif if tif.is_file() else None
    tif = root / "figures" / f"{sample}.tif"
    if tif.is_file():
        return tif
    zarr_dir = root / "he_img"
    return zarr_dir if zarr_dir.is_dir() else None


def discover_samples(name: str) -> list[str]:
    cohort = _cohort(name)
    if name == "Lung_Incomplete":
        rows = []
        for p in sorted(cohort.feat_parent.glob("*/*_all_features_stardist.h5ad")):
            if p.name.endswith("_matched_features_stardist.h5ad"):
                continue
            rows.append((p.stat().st_size, p.parent.name))
        return [s for _, s in sorted(rows, reverse=True)]
    if name == "HCC_unlabel":
        names = []
        if HCC_REST.is_dir():
            names = [
                p.name for p in HCC_REST.iterdir() if p.is_dir() and not p.name.startswith(".")
            ]
        rows = []
        for sample in names:
            feat = _feat_h5ad(cohort, sample)
            if feat.is_file():
                rows.append((feat.stat().st_size, sample))
        return [s for _, s in sorted(rows, reverse=True)]
    from match_codex_cells_with_pixel import list_incomplete_pdac_regions

    inc = list_incomplete_pdac_regions()
    rows = []
    for aid in inc["ACQUISITION_ID"].astype(str).str.strip():
        p = _feat_h5ad(cohort, aid)
        if p.is_file():
            rows.append((p.stat().st_size, aid))
    return [s for _, s in sorted(rows, reverse=True)]


def _matched_h5ad(sample: str) -> Path:
    return LUNG_CASES / sample / f"{sample}_matched_features.h5ad"


def rank_lung_incomplete(
    *,
    compute: bool = False,
    force: bool = False,
    cuda_device: str = "0",
    allow_cpu: bool = False,
) -> pd.DataFrame:
    """Matched-cell L1/L12/L2 scores for Incomplete_Cases that have GT.

    Three slides (TILD028LA, TILD049MA, VUHD038) have no matched h5ad.
    """
    if LUNG_RANKS.is_file() and not force:
        tab = pd.read_csv(LUNG_RANKS)
        return tab.sort_values(["rank_ds", "sample"], na_position="last").reset_index(drop=True)
    if not compute:
        return pd.DataFrame()

    from PanCancer_predict_visualize import load_matched, predict_block
    from pancancer_labels import ontology_class_names
    from uni_label_cv_helpers import macro_auc_ovr

    classes = ontology_class_names()
    model, scaler, device = _load_model(cuda_device, allow_cpu)
    rows = []
    for sample in discover_samples("Lung_Incomplete"):
        path = _matched_h5ad(sample)
        rec = {
            "sample": sample,
            "n_matched": 0,
            "acc_L1": np.nan,
            "acc_L12": np.nan,
            "acc_L2": np.nan,
            "macro_auc_L1": np.nan,
            "macro_auc_L12": np.nan,
            "macro_auc_L2": np.nan,
        }
        if not path.is_file():
            rows.append(rec)
            continue
        loaded = load_matched(path, "final_CT", classes)
        if loaded is None:
            rows.append(rec)
            continue
        got = predict_block(
            model, scaler, loaded["X"], loaded["xy"], device, use_spatial=True, spatial_k=8
        )
        rec["n_matched"] = int(loaded["X"].shape[0])
        for head in HEADS:
            y = np.asarray(loaded["gt"][head])
            pred = np.asarray(got["pred"][head])
            rec[f"acc_{head}"] = float((y == pred).mean())
            rec[f"macro_auc_{head}"] = float(
                macro_auc_ovr(y, got["probs"][head], len(classes[head]))
            )
        print(
            f"  rank {sample}: n={rec['n_matched']:,}  "
            f"L2 AUROC={rec['macro_auc_L2']:.3f}  acc={rec['acc_L2']:.3f}",
            flush=True,
        )
        rows.append(rec)
        del loaded, got
        gc.collect()
    tab = pd.DataFrame(rows)
    scored = tab["macro_auc_L2"].notna()
    tab["n_ds"] = int(scored.sum())
    tab["rank_ds"] = np.nan
    if scored.any():
        tab.loc[scored, "rank_ds"] = (
            tab.loc[scored, "macro_auc_L2"].rank(ascending=False, method="min").astype(int)
        )
    tab = tab.sort_values(["rank_ds", "sample"], na_position="last").reset_index(drop=True)
    LUNG_RANKS.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(LUNG_RANKS, index=False)
    print(f"wrote {LUNG_RANKS}", flush=True)
    return tab


def viz_samples(name: str, n: int | None = None) -> list[str]:
    """Lung: all Incomplete_Cases, top-5 L2 AUROC first. Others: largest ``VIZ_N``."""
    all_s = discover_samples(name)
    if name == "Lung_Incomplete":
        rank = rank_lung_incomplete(compute=False)
        if not rank.empty and "macro_auc_L2" in rank.columns:
            scored = rank.loc[rank["macro_auc_L2"].notna()].sort_values(
                "macro_auc_L2", ascending=False
            )
            top = scored["sample"].astype(str).tolist()
            rest = [s for s in all_s if s not in set(top)]
            ordered = top + rest
        else:
            ordered = all_s
        if n is None:
            return ordered
        return ordered[: int(n)]
    take = int(VIZ_N.get(name, 5) if n is None else n)
    if take <= 0:
        return all_s
    return all_s[:take]


def top_lung_incomplete(n: int = 5) -> list[str]:
    return viz_samples("Lung_Incomplete", n=int(n))


def coverage(name: str) -> pd.DataFrame:
    cohort = _cohort(name)
    rows = []
    for sample in discover_samples(name):
        feat = _feat_h5ad(cohort, sample)
        pred = pred_h5ad(cohort, sample)
        rows.append(
            {
                "sample": sample,
                "n_feat_mb": round(feat.stat().st_size / 1e6, 2) if feat.is_file() else 0.0,
                "feat": feat.is_file(),
                "pred": pred.is_file(),
                "he": he_path(cohort, sample) is not None,
            }
        )
    return pd.DataFrame(rows)


def _he_thumbnail(path: Path, max_side: int = 1400) -> tuple[np.ndarray, int, int] | None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if path.is_dir():
        import zarr

        z = zarr.open(str(path), mode="r")
        arr = np.asarray(z)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            im = Image.fromarray(arr, mode="RGB")
            im = im.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.BILINEAR,
            )
            arr = np.asarray(im)
        return arr, w, h
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".btf"}:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            arr = tif.series[0].asarray()
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        im = Image.fromarray(arr, mode="RGB")
        im = im.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.BILINEAR,
        )
        arr = np.asarray(im)
    return arr, w, h


def _palettes() -> dict:
    return {head: get_palette("pancancer", PALETTE_TIER[head]) for head in HEADS}


def composition_table(name: str, samples: list[str]) -> pd.DataFrame:
    import anndata as ad

    cohort = _cohort(name)
    rows = []
    for sample in samples:
        path = pred_h5ad(cohort, sample)
        adata = ad.read_h5ad(path)
        for head in HEADS:
            lab = adata.obs[OBS_COL[head]].astype(str)
            n = max(int(len(lab)), 1)
            for cls, cnt in lab.value_counts().items():
                rows.append(
                    {
                        "sample": sample,
                        "head": head,
                        "class": str(cls),
                        "n": int(cnt),
                        "fraction": float(cnt) / n,
                    }
                )
    return pd.DataFrame(rows)


def plot_spatial(
    name: str,
    samples: list[str],
    *,
    save: bool = False,
    figsize: tuple[float, float] = (17.2, 12.5),
    dpi: int = 160,
    save_dpi: int = 300,
    he_max_side: int = 1400,
    marker_px: float = 1.0,
):
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cohort = _cohort(name)
    palettes = _palettes()
    point_s = (float(marker_px) * 72.0 / dpi) ** 2
    fig_dir = cohort.pred_parent / "figures"
    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for sample in samples:
        adata = ad.read_h5ad(pred_h5ad(cohort, sample))
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        he = he_path(cohort, sample)
        thumb = _he_thumbnail(he, he_max_side) if he is not None else None
        if thumb is None:
            full_w = float(xy[:, 0].max()) if len(xy) else 1.0
            full_h = float(xy[:, 1].max()) if len(xy) else 1.0
            rgb = None
        else:
            rgb, full_w, full_h = thumb
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        for col, head in enumerate(HEADS):
            labels = adata.obs[OBS_COL[head]].astype(str).to_numpy()
            palette = palettes[head]
            for row, show_he in enumerate((True, False)):
                ax = axes[row, col]
                if show_he and rgb is not None:
                    ax.imshow(rgb, extent=(0, full_w, full_h, 0), interpolation="bilinear")
                else:
                    ax.set_facecolor("white")
                names = [n for n in palette if n in set(labels)]
                names += sorted(set(labels) - set(names))
                for cls in names:
                    mask = labels == cls
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        xy[mask, 0],
                        xy[mask, 1],
                        s=point_s,
                        c=[palette.get(cls, (0.55, 0.55, 0.55, 1.0))[:3]],
                        linewidths=0,
                        rasterized=True,
                    )
                ax.set_aspect("equal")
                ax.set_xlim(0, full_w)
                ax.set_ylim(full_h, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{head}  {'HE' if show_he else 'points'}  n={adata.n_obs:,}")
            seen = [n for n in palette if n in set(labels)]
            seen += sorted(set(labels) - set(seen))
            ncol = 1 if len(seen) <= 5 else 2 if len(seen) <= 10 else 4
            axes[1, col].legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        markersize=8,
                        markerfacecolor=palette.get(n, (0.5, 0.5, 0.5, 1))[:3],
                        markeredgecolor="none",
                        label=n,
                    )
                    for n in seen
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.06),
                fontsize=8,
                frameon=False,
                ncol=ncol,
            )
        fig.suptitle(f"{name}  {sample}  pan-cancer Hist2Pheno  (no GT)", fontsize=11)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18, top=0.93, hspace=0.28)
        if save:
            out = fig_dir / f"{sample}_spatial_L1_L12_L2.png"
            fig.savefig(out, dpi=save_dpi, bbox_inches="tight", facecolor="white")
            print(out)
        figures.append(fig)
    return figures


def plot_composition(
    name: str,
    samples: list[str],
    *,
    save: bool = False,
    dpi: int = 160,
    save_dpi: int = 300,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pancancer_labels import ontology_class_names

    palettes = _palettes()
    class_order = ontology_class_names()
    comp = composition_table(name, samples)
    fig_dir = _cohort(name).pred_parent / "figures"
    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for head in HEADS:
        sub = comp[comp["head"] == head]
        present = set(sub["class"].astype(str))
        palette = palettes[head]
        classes = [c for c in class_order[head] if c in present]
        classes += [c for c in palette if c in present and c not in classes]
        classes += sorted(present - set(classes))
        colors = [palette.get(c, (0.55, 0.55, 0.55, 1.0))[:3] for c in classes]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(classes) + 2), 5.4), dpi=dpi)
        x = np.arange(len(classes))
        width = 0.8 / max(len(samples), 1)
        for i, sample in enumerate(samples):
            ss = sub[sub["sample"] == sample].set_index("class")
            vals = [float(ss.loc[c, "fraction"]) if c in ss.index else 0.0 for c in classes]
            ax.bar(x + i * width, vals, width=width, color=colors)
        ax.set_xticks(x + width * (len(samples) - 1) / 2)
        ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("fraction of StarDist nuclei")
        ax.set_title(f"{name} predicted composition  {head}")
        if len(samples) > 1:
            ax.text(
                0.0,
                1.02,
                "left to right: " + "  ·  ".join(str(s).split("__")[-1][-24:] for s in samples),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
            )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=8,
                    markerfacecolor=col,
                    markeredgecolor="none",
                    label=cls,
                )
                for cls, col in zip(classes, colors)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            fontsize=8,
            frameon=False,
            ncol=4 if len(classes) > 10 else 2,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.38)
        if save:
            out = fig_dir / f"composition_{head}.png"
            fig.savefig(out, dpi=save_dpi, bbox_inches="tight", facecolor="white")
            print(out)
        figures.append(fig)
    return figures


def cmd_status(name: str, samples: list[str] | None = None) -> None:
    cov = coverage(name)
    viz = set(viz_samples(name))
    print(f"checkpoint  {'yes' if CKPT.is_file() else 'NO'}  {CKPT}")
    print(f"scaler      {'yes' if SCALER.is_file() else 'NO'}  {SCALER}")
    print(f"{name}: {len(cov)} slides with StarDist-all h5ad; viz={sorted(viz)}")
    if samples:
        cov = cov.loc[cov["sample"].isin(samples)]
    print(cov.to_string(index=False))
    missing = cov.loc[~cov["pred"], "sample"].tolist()
    if missing:
        print("missing pred:", ", ".join(missing[:12]), ("…" if len(missing) > 12 else ""))
    else:
        print("all listed pred h5ads present")
    if name == "Lung_Incomplete":
        rank = rank_lung_incomplete(compute=False)
        if rank.empty:
            print("Lung ranks missing — run: demo.sh rank")
        else:
            cols = [
                c
                for c in (
                    "rank_ds",
                    "n_ds",
                    "sample",
                    "n_matched",
                    "macro_auc_L1",
                    "macro_auc_L12",
                    "macro_auc_L2",
                    "acc_L2",
                )
                if c in rank.columns
            ]
            print("\nmatched-cell ranks (top 5 first in the notebook):")
            print(rank[cols].to_string(index=False))


def _load_model(cuda_device: str, allow_cpu: bool):
    import joblib
    from model import load_model_for_predict
    from PanCancer_predict_visualize import _device, setup_cuda

    setup_cuda(cuda_device)
    device = _device(allow_cpu)
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    if not SCALER.is_file():
        raise FileNotFoundError(SCALER)
    model = load_model_for_predict(
        str(_REPO),
        "PanCancer",
        "project_all_UNI",
        checkpoint_path=str(CKPT),
        device=device,
        hidden_dims=(768, 384, 192),
        parent_dir=True,
    )
    scaler = joblib.load(SCALER)
    return model, scaler, device


def cmd_predict(
    name: str,
    samples: list[str],
    *,
    cuda_device: str,
    force: bool,
    allow_cpu: bool,
) -> None:
    from PanCancer_predict_visualize import load_stardist_all, predict_block, write_star_h5ad
    from pancancer_labels import ontology_class_names

    cohort = _cohort(name)
    classes = ontology_class_names()
    todo = []
    for sample in samples:
        dest = pred_h5ad(cohort, sample)
        feat = _feat_h5ad(cohort, sample)
        if not feat.is_file():
            print(f"  skip {sample}: no feature h5ad", flush=True)
            continue
        if dest.is_file() and not force:
            print(f"  skip existing {sample}", flush=True)
            continue
        todo.append(sample)
    if not todo:
        print("nothing to predict")
        return
    model, scaler, device = _load_model(cuda_device, allow_cpu)
    print(f"predict {name}  n={len(todo)}  device={device}", flush=True)
    for sample in todo:
        feat = _feat_h5ad(cohort, sample)
        dest = pred_h5ad(cohort, sample)
        star = load_stardist_all(feat)
        if star is None:
            print(f"  FAIL {sample}: could not load {feat}", flush=True)
            continue
        got = predict_block(
            model, scaler, star["X"], star["xy"], device, use_spatial=True, spatial_k=8
        )
        write_star_h5ad(dest, star["obs_names"], star["xy"], got["pred"], classes)
        print(f"  {sample}: n={star['n']:,}  {dest}", flush=True)
        del star, got
        gc.collect()


def cmd_plot(name: str, samples: list[str], *, save: bool) -> None:
    import matplotlib.pyplot as plt

    for fig in plot_spatial(name, samples, save=save):
        plt.close(fig)
    for fig in plot_composition(name, samples, save=save):
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", choices=("Lung_Incomplete", "HCC_unlabel", "PDAC_unlabel"))
    p.add_argument("step", choices=("status", "dry-run", "predict", "plot", "rank"))
    p.add_argument("--sample", default="", help="one sample id")
    p.add_argument("--all", action="store_true", help="use every unlabeled slide, not the viz subset")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cuda-device", default=os.environ.get("CUDA_DEVICE", "0"))
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--save", action="store_true", help="write 300 dpi PNGs under pred_unlabel/figures")
    args = p.parse_args(argv)
    if args.sample:
        samples = [args.sample]
    elif args.all or args.step in {"status", "dry-run"}:
        samples = discover_samples(args.cohort)
    else:
        samples = viz_samples(args.cohort)
    if args.step in {"status", "dry-run"}:
        show = None if args.step == "status" and not args.sample else samples
        if args.step == "dry-run" and not args.sample and not args.all:
            show = viz_samples(args.cohort)
        cmd_status(args.cohort, show)
        if args.step == "dry-run":
            print("viz subset:", viz_samples(args.cohort))
        return 0
    if args.step == "rank":
        if args.cohort != "Lung_Incomplete":
            print("rank is only defined for Lung_Incomplete (matched GT)")
            return 1
        tab = rank_lung_incomplete(
            compute=True,
            force=args.force,
            cuda_device=args.cuda_device,
            allow_cpu=args.allow_cpu,
        )
        print(tab.to_string(index=False))
        return 0
    if args.step == "predict":
        cmd_predict(
            args.cohort,
            samples,
            cuda_device=args.cuda_device,
            force=args.force,
            allow_cpu=args.allow_cpu,
        )
        return 0
    cmd_plot(args.cohort, samples, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
