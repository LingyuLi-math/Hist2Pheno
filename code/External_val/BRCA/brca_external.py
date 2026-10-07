#!/usr/bin/env python3
"""BRCA external validation with the pan-cancer cross-section Hist2Pheno checkpoint.

One HE whole slide (GSM7782699), StarDist nuclei, no 10x tissue_positions /
scalefactors JSON. UNI crops are rescaled to 0.5 µm/px so patch_size=16 is an
8 µm field, matching the cohorts the checkpoint was trained on. There is no
cell-type ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PKG = _REPO / "code" / "Hist2Pheno_pkg"
_PAN = _REPO / "code" / "PanCancer"
for _p in (_PKG, _PAN, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BRCA_ROOT = _REPO / "data" / "Visium" / "BRCA"
HE_DIR = BRCA_ROOT / "HE_images"
STARDIST_ROOT = BRCA_ROOT / "StarDist_Segment"
UNI_ROOT = BRCA_ROOT / "project_all_UNI"
PRED_ROOT = BRCA_ROOT / "pred_pancancer_section"
FEATURE_SCRIPT = _REPO / "code" / "Image_feature_extraction.py"

CKPT = (
    _REPO
    / "data/PanCancer/result_all_spatial/cross_dataset_cv"
    / "D_emph_L2_spatial_pancancer/best_mlp_gpu.pt"
)
SCALER = _REPO / "data/PanCancer/result_all_spatial/scaler.joblib"

TARGET_UM_PER_PX = 0.5
PATCH_SIZE = 16
PATCH_UM = PATCH_SIZE * TARGET_UM_PER_PX  # 8 µm, same as pan-cancer UNI crops
SPATIAL_K = 8
HIDDEN_DIMS = (768, 384, 192)
HEADS = ("L1", "L12", "L2")
OBS_COL = {"L1": "pred_lineage", "L12": "pred_sublineage", "L2": "pred_CT"}
PALETTE_TIER = {"L1": "coarse", "L12": "intermediate", "L2": "fine"}


def discover_samples(only: str | None = None) -> list[str]:
    names = sorted(p.name for p in STARDIST_ROOT.iterdir() if p.is_dir()) if STARDIST_ROOT.is_dir() else []
    if only:
        if only not in names:
            raise SystemExit(f"sample {only!r} not under {STARDIST_ROOT}")
        return [only]
    if not names:
        raise SystemExit(f"no StarDist sample folders in {STARDIST_ROOT}")
    return names


def sample_paths(sample: str) -> dict[str, Path]:
    he = HE_DIR / f"{sample}.tif"
    csv_path = STARDIST_ROOT / sample / f"{sample}_Float_prob0.01_nms_0.3.csv"
    emb = UNI_ROOT / sample / "ImgEmbeddings_all_stardist" / f"sc_pth_{PATCH_SIZE}_{PATCH_SIZE}"
    feat = PRED_ROOT / sample / f"{sample}_stardist_features.h5ad"
    pred = PRED_ROOT / sample / f"{sample}_stardist_pred.h5ad"
    return {
        "he": he,
        "csv": csv_path,
        "emb": emb,
        "feat": feat,
        "pred": pred,
        "log": UNI_ROOT / sample / "logs",
    }


def ome_um_per_px(tif_path: Path) -> tuple[float, float]:
    import tifffile

    with tifffile.TiffFile(tif_path) as tif:
        desc = tif.pages[0].description or ""
        shape = tif.pages[0].shape
    import re

    mx = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', desc)
    my = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', desc)
    if not mx or not my:
        raise RuntimeError(f"no OME PhysicalSize in {tif_path}")
    return float(mx.group(1)), float(my.group(1))


def scale_for_mpp(mpp: float) -> float:
    """Factor that maps native pixels onto 0.5 µm/px. 1.0 means already there."""
    return float(mpp) / TARGET_UM_PER_PX


def geometry(sample: str) -> dict:
    paths = sample_paths(sample)
    if not paths["he"].is_file():
        raise FileNotFoundError(paths["he"])
    mpp_x, mpp_y = ome_um_per_px(paths["he"])
    if abs(mpp_x - mpp_y) > 1e-3:
        raise RuntimeError(f"{sample} anisotropic pixel size {mpp_x} vs {mpp_y}")
    scale = scale_for_mpp(mpp_x)
    native_patch_um = PATCH_SIZE * mpp_x
    return {
        "sample": sample,
        "mpp_um": mpp_x,
        "scale": scale,
        "scale_image": abs(scale - 1.0) > 1e-3,
        "patch_size_px": PATCH_SIZE,
        "patch_um_after_rescale": PATCH_UM,
        "patch_um_if_native_16px": native_patch_um,
        "he": str(paths["he"]),
        "csv": str(paths["csv"]),
        "embeddings": str(paths["emb"]),
    }


def _python() -> str:
    return sys.executable


def n_csv_rows(csv_path: Path) -> int:
    # header + rows; avoid loading the probability column into pandas twice
    with csv_path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def n_pth(emb_dir: Path) -> int:
    if not emb_dir.is_dir():
        return 0
    return sum(1 for p in emb_dir.glob("*.pth"))


def cmd_dry_run(samples: list[str]) -> None:
    print(f"target {TARGET_UM_PER_PX} µm/px, patch {PATCH_SIZE}px = {PATCH_UM:.1f} µm")
    print(f"checkpoint {CKPT} exists={CKPT.is_file()}")
    print(f"scaler     {SCALER} exists={SCALER.is_file()}")
    for sample in samples:
        geo = geometry(sample)
        paths = sample_paths(sample)
        n = n_csv_rows(paths["csv"]) if paths["csv"].is_file() else -1
        print(
            f"\n{sample}\n"
            f"  nuclei CSV rows : {n:,}\n"
            f"  OME µm/px       : {geo['mpp_um']:.6f}\n"
            f"  native 16px     : {geo['patch_um_if_native_16px']:.3f} µm"
            f"  (not the training field)\n"
            f"  scale_image     : {geo['scale_image']}  scale={geo['scale']:.6f}\n"
            f"  crop after scale: {geo['patch_um_after_rescale']:.1f} µm\n"
            f"  embeddings      : {n_pth(paths['emb']):,} .pth\n"
            f"  features h5ad   : {paths['feat'].is_file()}\n"
            f"  pred h5ad       : {paths['pred'].is_file()}"
        )


def cmd_features(samples: list[str], *, force: bool) -> None:
    for sample in samples:
        paths = sample_paths(sample)
        geo = geometry(sample)
        if not paths["csv"].is_file():
            raise FileNotFoundError(paths["csv"])
        n_rows = n_csv_rows(paths["csv"])
        have = n_pth(paths["emb"])
        if not force and have >= int(0.9 * n_rows):
            print(f"[features] skip {sample}: {have:,} .pth already (FORCE=1 to redo)", flush=True)
            continue
        paths["emb"].mkdir(parents=True, exist_ok=True)
        paths["log"].mkdir(parents=True, exist_ok=True)
        dataset = f"sc_{sample}"
        cmd = [
            _python(),
            "-u",
            str(FEATURE_SCRIPT),
            "--dataset",
            dataset,
            "--position_path",
            str(paths["csv"]),
            "--rawimage_path",
            str(paths["he"]),
            "--scale_image",
            "True" if geo["scale_image"] else "False",
            "--scale",
            f"{geo['scale']:.8f}",
            "--method",
            "UNI",
            "--patch_size",
            str(PATCH_SIZE),
            "--output_img",
            str(paths["emb"].parent / f"sc_pth_{PATCH_SIZE}_{PATCH_SIZE}_image"),
            "--output_pth",
            str(paths["emb"]),
            "--logging",
            str(paths["log"]) + "/",
            "--no_save_patch_images",
        ]
        print(
            f"[features] {sample}  mpp={geo['mpp_um']:.4f}  "
            f"scale={geo['scale']:.4f}  patch={PATCH_UM:.1f}µm  nuclei={n_rows:,}",
            flush=True,
        )
        subprocess.run(cmd, cwd=str(_REPO), check=True)
        meta = {
            **geo,
            "n_csv": n_rows,
            "n_pth": n_pth(paths["emb"]),
            "pth_prefix": dataset,
        }
        (paths["emb"].parent / "uni_geometry.json").write_text(json.dumps(meta, indent=2))
        print(f"[features] {sample} wrote {meta['n_pth']:,} embeddings", flush=True)


def _build_one_h5ad(sample: str, *, force: bool) -> Path:
    paths = sample_paths(sample)
    if paths["feat"].is_file() and not force:
        print(f"[h5ad] skip {sample}: {paths['feat']}", flush=True)
        return paths["feat"]
    if n_pth(paths["emb"]) == 0:
        raise FileNotFoundError(
            f"no UNI .pth under {paths['emb']}. Run: bash code/External_val/BRCA/demo.sh features"
        )
    from base import load_hist_embeddings, match_embeddings
    import anndata as ad

    raw = pd.read_csv(paths["csv"])
    if "centroid_x" not in raw.columns or "centroid_y" not in raw.columns:
        raise KeyError(f"{paths['csv']} needs centroid_x, centroid_y")
    width = max(8, len(str(len(raw))))
    raw.insert(0, "cell_id", [f"{sample}-stardist-{i:0{width}d}" for i in range(len(raw))])
    matching = raw.copy()
    matching["celltype"] = "unlabeled"
    prefix = f"sc_{sample}"
    print(f"[h5ad] load embeddings {paths['emb']}", flush=True)
    embeddings = load_hist_embeddings(
        paths["emb"],
        celltype_df=matching,
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=1.0,
        pth_prefix=prefix,
    )
    matched = match_embeddings(
        matching,
        embeddings,
        tolerance=1.0,
        coord_cols=("centroid_x", "centroid_y"),
    )
    if not matched:
        raise RuntimeError(f"matched 0 embeddings for {sample}; check pth prefix {prefix}")
    idx = np.asarray([int(item["idx"]) for item in matched], dtype=np.int64)
    obs = raw.iloc[idx].copy().reset_index(drop=True)
    obs.index = pd.Index(obs["cell_id"].astype(str), name="cell_id")
    X = np.vstack([np.asarray(item["embedding"], dtype=np.float32).reshape(-1) for item in matched])
    adata = ad.AnnData(X=X, obs=obs)
    xy = obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    adata.obsm["spatial"] = xy
    adata.obsm["spatial_HE"] = xy.copy()
    geo = geometry(sample)
    adata.uns["brca_uni"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in geo.items() if k != "scale_image"}
    adata.uns["brca_uni"]["scale_image"] = bool(geo["scale_image"])
    adata.uns["brca_uni"]["n_matched"] = int(adata.n_obs)
    adata.uns["brca_uni"]["n_csv"] = int(len(raw))
    paths["feat"].parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(paths["feat"])
    print(f"[h5ad] {sample}: {adata.n_obs:,}/{len(raw):,} nuclei, {adata.n_vars} features", flush=True)
    return paths["feat"]


def cmd_h5ad(samples: list[str], *, force: bool) -> None:
    for sample in samples:
        _build_one_h5ad(sample, force=force)


def _labels_from_index(idx: np.ndarray, names: list[str]) -> list[str]:
    out = []
    n = len(names)
    for i in idx:
        j = int(i)
        out.append(names[j] if 0 <= j < n else "")
    return out


def cmd_predict(samples: list[str], *, force: bool, allow_cpu: bool) -> None:
    import joblib
    import anndata as ad
    import torch

    from base import adata_X_to_dense, build_spatial_neighbor_index
    from model import load_model_for_predict
    from pancancer_labels import ontology_class_names
    from plot import mlp_collect_five_head_softmax_probs

    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    if not SCALER.is_file():
        raise FileNotFoundError(SCALER)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif allow_cpu:
        device = torch.device("cpu")
        print("WARNING: CUDA unavailable; predicting on CPU.", flush=True)
    else:
        raise RuntimeError("CUDA unavailable. Pass --allow-cpu.")

    classes = ontology_class_names()
    print(f"[predict] device={device}  ckpt={CKPT.name}  spatial_k={SPATIAL_K}", flush=True)
    model = load_model_for_predict(
        str(_REPO),
        "PanCancer",
        "project_all_UNI",
        checkpoint_path=str(CKPT),
        device=device,
        hidden_dims=HIDDEN_DIMS,
        parent_dir=True,
    )
    scaler = joblib.load(SCALER)
    prob_key = {"L1": "l1", "L12": "l12", "L2": "l2"}

    for sample in samples:
        paths = sample_paths(sample)
        if paths["pred"].is_file() and not force:
            print(f"[predict] skip {sample}: {paths['pred']}", flush=True)
            continue
        feat_path = paths["feat"] if paths["feat"].is_file() else _build_one_h5ad(sample, force=False)
        adata = ad.read_h5ad(feat_path)
        X = np.asarray(adata_X_to_dense(adata.X), dtype=np.float32)
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        Xs = np.asarray(scaler.transform(X), dtype=np.float32)
        nbr = build_spatial_neighbor_index(xy, k_neighbors=SPATIAL_K)
        probs = mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=nbr)
        obs = adata.obs.copy()
        for head in HEADS:
            pr = np.asarray(probs[prob_key[head]], dtype=np.float32)
            pred_i = np.argmax(pr, axis=1).astype(np.int64)
            obs[OBS_COL[head]] = _labels_from_index(pred_i, classes[head])
            obs[f"conf_{head}"] = pr.max(axis=1).astype(np.float32)
        pred = ad.AnnData(obs=obs)
        pred.obsm["spatial_HE"] = xy
        pred.uns["checkpoint"] = str(CKPT)
        pred.uns["scaler"] = str(SCALER)
        pred.uns["spatial_k"] = SPATIAL_K
        pred.uns["classes"] = classes
        pred.uns["note"] = (
            "No BRCA ground truth. Labels are pan-cancer L1/L12/L2 from the "
            "cross-section checkpoint applied to StarDist nuclei."
        )
        paths["pred"].parent.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(paths["pred"])
        _write_l2_probs(paths["pred"], probs["l2"], classes["L2"])
        print(f"[predict] {sample}: n={pred.n_obs:,} -> {paths['pred']}", flush=True)


########################################
# 2026.10.06 for hiatology niches index
########################################
def l2_probs_path(sample: str) -> Path:
    return sample_paths(sample)["pred"].with_name(f"{sample}_l2_probs.npz")


def _write_l2_probs(pred_h5ad: Path, probs_l2, class_names) -> Path:
    path = pred_h5ad.with_name(pred_h5ad.name.replace("_stardist_pred.h5ad", "_l2_probs.npz"))
    np.savez_compressed(
        path,
        probs=np.asarray(probs_l2, dtype=np.float32),
        class_names=np.asarray(list(class_names), dtype=object),
    )
    print(f"[probs] {path}  shape={tuple(np.asarray(probs_l2).shape)}", flush=True)
    return path


def ensure_l2_probs(sample: str, *, force: bool = False, allow_cpu: bool = False):
    """L2 softmax used by TLS / SRI / TNI. Cached next to the pred h5ad.

    Argmax labels in the pred h5ad are not enough: niche indices sum class
    probabilities, the same abundance the lung notebook reads from AUROC CSVs.
    """
    path = l2_probs_path(sample)
    if path.is_file() and not force:
        z = np.load(path, allow_pickle=True)
        names = [str(x) for x in z["class_names"].tolist()]
        print(f"[probs] cache {path}  {z['probs'].shape}", flush=True)
        return np.asarray(z["probs"], dtype=np.float32), names

    import joblib
    import anndata as ad
    import torch

    from base import adata_X_to_dense, build_spatial_neighbor_index
    from model import load_model_for_predict
    from pancancer_labels import ontology_class_names
    from plot import mlp_collect_five_head_softmax_probs

    paths = sample_paths(sample)
    if not paths["feat"].is_file():
        raise FileNotFoundError(paths["feat"])
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA unavailable. Pass --allow-cpu.")
    classes = ontology_class_names()
    model = load_model_for_predict(
        str(_REPO),
        "PanCancer",
        "project_all_UNI",
        checkpoint_path=str(CKPT),
        device=device,
        hidden_dims=HIDDEN_DIMS,
        parent_dir=True,
    )
    scaler = joblib.load(SCALER)
    adata = ad.read_h5ad(paths["feat"])
    X = np.asarray(adata_X_to_dense(adata.X), dtype=np.float32)
    xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    Xs = np.asarray(scaler.transform(X), dtype=np.float32)
    nbr = build_spatial_neighbor_index(xy, k_neighbors=SPATIAL_K)
    got = mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=nbr)
    probs = np.asarray(got["l2"], dtype=np.float32)
    if paths["pred"].is_file():
        pred = ad.read_h5ad(paths["pred"])
        if pred.n_obs != probs.shape[0]:
            raise RuntimeError(f"{sample}: probs {probs.shape[0]} != pred {pred.n_obs}")
        pred_names = [str(x) for x in pred.uns["classes"]["L2"]]
        if pred_names != list(classes["L2"]):
            raise RuntimeError(f"{sample}: L2 class order drifted from the pred h5ad")
        agree = float(np.mean(np.array(pred.obs["pred_CT"].astype(str)) == np.array(classes["L2"])[probs.argmax(1)]))
        print(f"[probs] {sample} argmax agrees with pred_CT: {agree:.4f}", flush=True)
    _write_l2_probs(paths["pred"], probs, classes["L2"])
    return probs, list(classes["L2"])


def cmd_probs(samples: list[str], *, force: bool, allow_cpu: bool) -> None:
    for sample in samples:
        ensure_l2_probs(sample, force=force, allow_cpu=allow_cpu)
########################################

def _palette():
    from plotting_palettes import get_palette

    return {head: get_palette("pancancer", PALETTE_TIER[head]) for head in HEADS}


def _he_thumbnail(he_path: Path, max_side: int = 1800) -> tuple[np.ndarray, int, int]:
    import tifffile
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with tifffile.TiffFile(he_path) as tif:
        arr = tif.series[0].asarray()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        im = Image.fromarray(arr, mode="RGB")
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
        arr = np.asarray(im)
    return arr, w, h


def _scatter_on_he(ax, thumb, full_w, full_h, xy, labels, palette, title: str) -> None:
    ax.imshow(thumb, extent=(0, full_w, full_h, 0), interpolation="bilinear")
    ax.set_xlim(0, full_w)
    ax.set_ylim(full_h, 0)
    colors = [palette.get(str(lab), (0.55, 0.55, 0.55, 1.0)) for lab in labels]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.35, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")


def composition_table(samples: list[str]) -> pd.DataFrame:
    import anndata as ad

    rows = []
    for sample in samples:
        path = sample_paths(sample)["pred"]
        if not path.is_file():
            raise FileNotFoundError(path)
        obs = ad.read_h5ad(path).obs
        n = len(obs)
        for head, col in OBS_COL.items():
            counts = obs[col].astype(str).value_counts()
            for name, k in counts.items():
                rows.append(
                    {
                        "sample": sample,
                        "head": head,
                        "class": name,
                        "n": int(k),
                        "fraction": float(k) / float(n),
                        "mean_conf": float(obs.loc[obs[col].astype(str) == name, f"conf_{head}"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def cmd_plot(samples: list[str]) -> None:
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = PRED_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    comp = composition_table(samples)
    comp_path = PRED_ROOT / "composition.csv"
    comp.to_csv(comp_path, index=False)
    print(f"[plot] {comp_path}", flush=True)
    palettes = _palette()

    for sample in samples:
        paths = sample_paths(sample)
        adata = ad.read_h5ad(paths["pred"])
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        thumb, full_w, full_h = _he_thumbnail(paths["he"])
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
        for ax, head in zip(axes, HEADS):
            labels = adata.obs[OBS_COL[head]].astype(str).to_numpy()
            _scatter_on_he(
                ax,
                thumb,
                full_w,
                full_h,
                xy,
                labels,
                palettes[head],
                f"{head}  n={adata.n_obs:,}",
            )
            names = list(dict.fromkeys(labels.tolist()))
            handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    markerfacecolor=palettes[head].get(n, (0.5, 0.5, 0.5, 1))[:3],
                    markeredgecolor="none",
                    label=n,
                )
                for n in names
            ]
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                fontsize=6,
                frameon=False,
                ncol=2 if len(names) > 8 else 1,
            )
        fig.suptitle(
            f"BRCA {sample}  pan-cancer cross-section Hist2Pheno  (no GT)",
            fontsize=12,
        )
        fig.tight_layout()
        out = fig_dir / f"{sample}_spatial_L1_L12_L2.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}", flush=True)

    from pancancer_labels import ontology_class_names

    class_order = ontology_class_names()
    for head in HEADS:
        sub = comp[comp["head"] == head]
        present = set(sub["class"].astype(str))
        classes = [c for c in class_order[head] if c in present]
        classes += sorted(present - set(classes))
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(classes) + 2), 4.2))
        x = np.arange(len(classes))
        width = 0.8 / max(len(samples), 1)
        for i, sample in enumerate(samples):
            ss = sub[sub["sample"] == sample].set_index("class")
            vals = [float(ss.loc[c, "fraction"]) if c in ss.index else 0.0 for c in classes]
            ax.bar(x + i * width, vals, width=width, label=sample)
        ax.set_xticks(x + width * (len(samples) - 1) / 2)
        ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("fraction of StarDist nuclei")
        ax.set_title(f"BRCA predicted composition  {head}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"composition_{head}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}", flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "step",
        choices=("dry-run", "features", "h5ad", "predict", "probs", "plot"),
    )
    p.add_argument("--sample", default="", help="one sample id; default all StarDist folders")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    samples = discover_samples(args.sample or None)
    if args.step == "dry-run":
        cmd_dry_run(samples)
    elif args.step == "features":
        cmd_features(samples, force=args.force)
    elif args.step == "h5ad":
        cmd_h5ad(samples, force=args.force)
    elif args.step == "predict":
        cmd_predict(samples, force=args.force, allow_cpu=args.allow_cpu)
    elif args.step == "probs":
        cmd_probs(samples, force=args.force, allow_cpu=args.allow_cpu)
    elif args.step == "plot":
        cmd_plot(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
