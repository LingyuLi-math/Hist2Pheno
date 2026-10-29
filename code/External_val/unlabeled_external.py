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
UNI2_CKPT = (
    _REPO
    / "data"
    / "PanCancer"
    / "result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42"
    / "cross_dataset_cv"
    / "A2_adjacent_spatial_k8mean"
    / "best_mlp_gpu.pt"
)
UNI2_CACHE = _REPO / "data" / "PanCancer" / "uni2_aligned"
SCALER = _REPO / "data" / "PanCancer" / "result_all_spatial" / "scaler.joblib"
PRED_ROOT = _REPO / "data" / "PanCancer" / "pred_unlabel"
UNI2_PRED = _REPO / "data" / "PanCancer" / "pred_uni2_A2_bestmlp_paired_v2_seed42" / "unlabel"
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
PDAC_COVERSLIPS = ("c003", "c005", "c007")
# 3 cores / coverslip for the notebook. c003/c007 = largest. c005 = immune /
# stroma / mixed among the already-scored large cores (not three CD8-high twins).
PDAC_REPRESENTATIVE = {
    "c003": (
        "Charvill-94_c003_v001_r001_reg146",
        "Charvill-94_c003_v001_r001_reg141",
        "Charvill-94_c003_v001_r001_reg147",
    ),
    "c005": (
        "Charvill-94_c005_v001_r001_reg075",
        "Charvill-94_c005_v001_r001_reg047",
        "Charvill-94_c005_v001_r001_reg037",
    ),
    "c007": (
        "Charvill-94_c007_v001_r001_reg062",
        "Charvill-94_c007_v001_r001_reg081",
        "Charvill-94_c007_v001_r001_reg069",
    ),
}


@dataclass(frozen=True)
class Cohort:
    name: str
    feat_parent: Path
    pred_parent: Path


def _uni2_ckpt() -> Path:
    """Default is the paired_v2 A2 weight. ``UNI2_CKPT`` selects another weight.

    2026.10.28: HCC_unlabel A2/A3 set this to that run's paired_v2 best_mlp.
    """
    raw = os.environ.get("UNI2_CKPT", "").strip()
    if not raw:
        return UNI2_CKPT
    path = Path(raw)
    return path if path.is_absolute() else (_REPO / path)


def _prob_folder(path: Path) -> Path:
    """2026.10.28, same cutoff rule as external_validation.prediction_dirname."""
    from external_validation import prediction_dirname

    return path.with_name(prediction_dirname(path.name))


def _uni2_pred_root() -> Path:
    """``UNI2_PRED_DIRNAME`` is a folder under data/PanCancer. Slides go in unlabel/."""
    raw = os.environ.get("UNI2_PRED_DIRNAME", "").strip()
    from external_validation import prediction_dirname

    name = prediction_dirname(raw or UNI2_PRED.name)
    return _REPO / "data" / "PanCancer" / name / "unlabel"


def _ckpt(embedder: str) -> Path:
    if embedder=="uni":
        from external_validation import uni_overrides
        return uni_overrides(_REPO)["checkpoint"]
    if embedder=="virchow2":
        from external_validation import virchow2_overrides
        return virchow2_overrides(_REPO)["checkpoint"]
    if embedder == "uni2":
        return _uni2_ckpt()
    if embedder != "uni":
        raise SystemExit(f"unknown embedder {embedder!r}. Choose uni or uni2")
    return CKPT


def _cohort(name: str, embedder: str = "uni") -> Cohort:
    key = name.strip()
    if embedder=="uni":
        from external_validation import uni_overrides, prediction_dirname
        pred=_REPO/"data/PanCancer"/prediction_dirname(uni_overrides(_REPO)["pred_dirname"])/"unlabel"/key
    elif embedder=="virchow2":
        from external_validation import virchow2_overrides, prediction_dirname
        pred=_REPO/"data/PanCancer"/prediction_dirname(virchow2_overrides(_REPO)["pred_dirname"])/"unlabel"/key
    elif embedder == "uni2":
        pred = _uni2_pred_root() / key
    elif key == "Lung_Incomplete":
        pred = _prob_folder(LUNG_LEGACY)
    elif key in {"HCC_unlabel", "PDAC_unlabel"}:
        pred = _prob_folder(PRED_ROOT) / key
    else:
        raise SystemExit(f"unknown cohort {name!r}; choose Lung_Incomplete, HCC_unlabel, PDAC_unlabel")
    if key == "Lung_Incomplete":
        return Cohort(key, LUNG_CASES, pred)
    if key == "HCC_unlabel":
        return Cohort(key, HCC_HE, pred)
    if key == "PDAC_unlabel":
        return Cohort(key, PDAC_ROOT, pred)
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


def _lung_ranks_path(embedder: str = "uni") -> Path:
    if embedder in ("uni","virchow2"):return _cohort("Lung_Incomplete",embedder).pred_parent/"ranks_matched.csv"
    if embedder == "uni2":
        return _uni2_pred_root() / "Lung_Incomplete" / "ranks_matched.csv"
    return LUNG_RANKS


def _legacy_rank_lung_incomplete(
    *,
    compute: bool = False,
    force: bool = False,
    cuda_device: str = "0",
    allow_cpu: bool = False,
    embedder: str = "uni",
) -> pd.DataFrame:
    """Matched-cell L1/L12/L2 scores for Incomplete_Cases that have GT.

    Three slides (TILD028LA, TILD049MA, VUHD038) have no matched h5ad.
    """
    ranks_path = _lung_ranks_path(embedder)
    if False and ranks_path.is_file() and not force:  # 2026.10.28, legacy ranks lack provenance; recompute explicitly
        tab = pd.read_csv(ranks_path)
        return tab.sort_values(["rank_ds", "sample"], na_position="last").reset_index(drop=True)
    if not compute:
        return pd.DataFrame()

    from PanCancer_predict_visualize import load_matched, predict_block
    from pancancer_labels import ontology_class_names
    from uni_label_cv_helpers import macro_auc_ovr

    classes = ontology_class_names()
    model, scaler, device = _load_model(cuda_device, allow_cpu, embedder=embedder)
    classes=model.checkpoint_metadata["head_classes"]
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
        try:
            loaded = load_matched(
                path,
                "final_CT",
                classes,
                embedder=embedder,
                uni2_cache_dir=UNI2_CACHE if embedder in ("uni","uni2","virchow2") else None,
            )
        except FileNotFoundError as exc:
            print(f"  rank {sample}: {exc}", flush=True)
            rows.append(rec)
            continue
        if loaded is None:
            rows.append(rec)
            continue
        got = predict_block(
            model, scaler, loaded["X"], loaded["xy"], device, use_spatial=True, spatial_k=None
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
    ranks_path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(ranks_path, index=False)
    print(f"wrote {ranks_path}", flush=True)
    return tab



def _rank_source(embedder):
    import anndata as ad
    import pipeline_provenance as pp
    from PanCancer_train_validate_cv_UNIlabel import _uni2_embedding_dir
    files=[_ckpt(embedder),Path(str(_ckpt(embedder))+".artifact.pt")];tokens=[]
    for sample in discover_samples("Lung_Incomplete"):
        h=_matched_h5ad(sample)
        if not h.is_file():continue
        files.append(h);a=ad.read_h5ad(h,backed="r")
        try:
            if embedder in ("uni","virchow2"):
                from embedding_backend import embedding_dir
                tokens.append(embedding_dir(h,a.uns,embedder))
            elif embedder=="uni2":tokens.append(_uni2_embedding_dir(h,a.uns))
            elif a.uns.get("embedding_dir"):tokens.append(Path(a.uns["embedding_dir"]))
        finally:a.file.close()
    return {"inputs":pp.capture_inputs(files,tokens,UNI2_CACHE),"embedder":embedder,
            "unit":os.environ.get("HIST2PHENO_INFERENCE_COORDINATES_UNIT","pixel"),
            "mpp":os.environ.get("HIST2PHENO_INFERENCE_MPP"),"implementation":pp.implementations(),
            "ranking_code":pp.sha256(__file__)}


def rank_lung_incomplete(*,compute=False,force=False,cuda_device="0",allow_cpu=False,embedder="uni"):
    import pipeline_provenance as pp
    path=_lung_ranks_path(embedder);marker=path.with_suffix(".pipeline.json")
    with pp.file_lock(str(path)+".lock"):
        try:source=_rank_source(embedder)
        except (OSError,RuntimeError):
            if not compute:return pd.DataFrame()
            raise
        stamp=pp.fingerprint(source)
        if not force and pp.complete(marker,stamp,[path]):return pd.read_csv(path)
        if not compute:return pd.DataFrame()
        result=_legacy_rank_lung_incomplete(compute=True,force=True,cuda_device=cuda_device,allow_cpu=allow_cpu,embedder=embedder)
        if _rank_source(embedder)!=source:raise RuntimeError("Ranking inputs changed")
        pp.publish(marker,stamp,[path],source=source)
        return result

def viz_samples(name: str, n: int | None = None, embedder: str = "uni") -> list[str]:
    """Lung: all Incomplete_Cases, top-5 L2 AUROC first. Others: largest ``VIZ_N``."""
    all_s = discover_samples(name)
    if name == "Lung_Incomplete":
        rank = rank_lung_incomplete(compute=False, embedder=embedder)
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


def coverslip_of(sample: str) -> str:
    for part in str(sample).split("_"):
        if part.startswith("c") and part[1:].isdigit():
            return part
    return "unknown"


def representative_pdac_samples() -> list[str]:
    """Nine Incomplete_Cases: 3 per coverslip (c003 / c005 / c007)."""
    out: list[str] = []
    for cs in PDAC_COVERSLIPS:
        out.extend(PDAC_REPRESENTATIVE[cs])
    return out


def mixed_rectangle_roi(
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    win_frac: float = 0.40,
    step_frac: float = 0.18,
    min_n: int = 350,
) -> tuple[float, float, float, float]:
    """Axis-aligned window with several predicted classes (for TMA zoom)."""
    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    span = np.minimum(x1 - x0, y1 - y0)
    win = float(max(400.0, win_frac * span))
    step = float(max(80.0, step_frac * win))
    best = None
    xs = np.arange(x0, max(x0 + 1.0, x1 - win), step)
    ys = np.arange(y0, max(y0 + 1.0, y1 - win), step)
    if len(xs) == 0:
        xs = np.array([x0])
    if len(ys) == 0:
        ys = np.array([y0])
    for xa in xs:
        for ya in ys:
            hit = (
                (xy[:, 0] >= xa)
                & (xy[:, 0] < xa + win)
                & (xy[:, 1] >= ya)
                & (xy[:, 1] < ya + win)
            )
            n = int(hit.sum())
            if n < min_n:
                continue
            vc = pd.Series(labels[hit]).value_counts(normalize=True)
            ncls = int((vc > 0.04).sum())
            top = float(vc.iloc[0])
            score = ncls + 0.15 * np.log1p(n) - (1.6 if top > 0.72 else 0.0)
            rec = (score, n, float(xa), float(ya), float(xa + win), float(ya + win))
            if best is None or rec[0] > best[0]:
                best = rec
    if best is None:
        return float(x0), float(y0), float(x1), float(y1)
    return best[2], best[3], best[4], best[5]


def coverage(name: str, embedder: str = "uni") -> pd.DataFrame:
    from PanCancer_predict_visualize import _stardist_uni2_dir

    cohort = _cohort(name, embedder)
    rows = []
    for sample in discover_samples(name):
        feat = _feat_h5ad(cohort, sample)
        pred = pred_h5ad(cohort, sample)
        rows.append(
            {
                "sample": sample,
                "n_feat_mb": round(feat.stat().st_size / 1e6, 2) if feat.is_file() else 0.0,
                "feat": feat.is_file(),
                "uni2": _stardist_uni2_dir(feat) is not None,
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


def _save_png_pdf(fig, path: Path, *, dpi: int) -> Path:
    """PNG preview plus Type 42 PDF so labels stay editable."""
    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    path = Path(path)
    fig.savefig(
        path.with_suffix(".png"),
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )
    pdf = path.with_suffix(".pdf")
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    return pdf


def composition_table(name: str, samples: list[str], embedder: str = "uni") -> pd.DataFrame:
    cmd_predict(name,samples,cuda_device=os.environ.get("CUDA_DEVICE","0"),force=False,allow_cpu=True,embedder=embedder)
    import anndata as ad
    import pipeline_provenance as pp

    cohort = _cohort(name, embedder)
    rows = []
    for sample in samples:
        path = pred_h5ad(cohort, sample)
        from pipeline_provenance import read_verified_prediction
        adata = read_verified_prediction(path)
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
    embedder: str = "uni",
    save: bool = False,
    figsize: tuple[float, float] = (17.2, 12.5),
    dpi: int = 160,
    save_dpi: int = 300,
    he_max_side: int = 1400,
    marker_px: float = 1.0,
):
    cmd_predict(name,samples,cuda_device=os.environ.get("CUDA_DEVICE","0"),force=False,allow_cpu=True,embedder=embedder)
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cohort = _cohort(name, embedder)
    palettes = _palettes()
    point_s = (float(marker_px) * 72.0 / dpi) ** 2
    fig_dir = cohort.pred_parent / "figures"
    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for sample in samples:
        from pipeline_provenance import read_verified_prediction
        adata = read_verified_prediction(pred_h5ad(cohort,sample))
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
            out = _save_png_pdf(
                fig, fig_dir / f"{sample}_spatial_L1_L12_L2.png", dpi=save_dpi
            )
            print(out)
        figures.append(fig)
    return figures


def plot_composition(
    name: str,
    samples: list[str],
    *,
    embedder: str = "uni",
    save: bool = False,
    dpi: int = 160,
    save_dpi: int = 300,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pancancer_labels import ontology_class_names

    palettes = _palettes()
    class_order = ontology_class_names()
    comp = composition_table(name, samples, embedder)
    fig_dir = _cohort(name, embedder).pred_parent / "figures"
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
            out = _save_png_pdf(
                fig, fig_dir / f"composition_{head}.png", dpi=save_dpi
            )
            print(out)
        figures.append(fig)
    return figures


def cmd_status(name: str, samples: list[str] | None = None, embedder: str = "uni") -> None:
    ckpt = _ckpt(embedder)
    cov = coverage(name, embedder)
    print(f"embedder    {embedder}")
    print(f"pred root   {_cohort(name, embedder).pred_parent}")
    viz = set(viz_samples(name, embedder=embedder))
    print(f"checkpoint  {'yes' if ckpt.is_file() else 'NO'}  {ckpt}")
    if embedder == "uni":
        print(f"scaler      {'yes' if SCALER.is_file() else 'NO'}  {SCALER}")
    else:
        art = Path(str(ckpt) + ".artifact.pt")
        print(f"scaler      {'yes' if art.is_file() else 'NO'}  {art}")
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
        rank = rank_lung_incomplete(compute=False, embedder=embedder)
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


def _load_model(cuda_device: str, allow_cpu: bool, embedder: str = "uni"):
    from model import load_model_for_predict
    from PanCancer_predict_visualize import _device, setup_cuda

    ckpt = _ckpt(embedder)
    setup_cuda(cuda_device)
    device = _device(allow_cpu)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model = load_model_for_predict(
        str(_REPO),
        "PanCancer",
        "project_all_Virchow2" if embedder=="virchow2" else ("project_all_UNI2" if embedder == "uni2" else "project_all_UNI"),
        checkpoint_path=str(ckpt),
        device=device,
        hidden_dims=None,
        parent_dir=True,
    )
    from checkpoint_artifact import checkpoint_scaler
    scaler = checkpoint_scaler(ckpt)
    from embedding_backend import validate_model
    validate_model(model,scaler,embedder)
    return model, scaler, device


def _legacy_cmd_predict(
    name: str,
    samples: list[str],
    *,
    cuda_device: str,
    force: bool,
    allow_cpu: bool,
    embedder: str = "uni",
) -> None:
    from PanCancer_predict_visualize import (
        _stardist_uni2_dir,
        load_stardist_all,
        predict_block,
        write_star_h5ad,
    )
    from pancancer_labels import ontology_class_names

    cohort = _cohort(name, embedder)
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
    if embedder == "uni2":
        os.environ["HIST2PHENO_CONTEXT_CACHE"] = str(cohort.pred_parent / "context_cache")
    model, scaler, device = _load_model(cuda_device, allow_cpu, embedder=embedder)
    classes=model.checkpoint_metadata.get("head_classes",classes)
    print(
        f"predict {name}  embedder={embedder}  n={len(todo)}  device={device}\n"
        f"  ckpt {_ckpt(embedder)}",
        flush=True,
    )
    for sample in todo:
        feat = _feat_h5ad(cohort, sample)
        dest = pred_h5ad(cohort, sample)
        uni2_dir = None
        if embedder == "uni2":
            uni2_dir = _stardist_uni2_dir(feat)
            if uni2_dir is None:
                print(f"  skip {sample}: no UNI2 sc_pth_14_14 next to {feat.parent}", flush=True)
                continue
        star = load_stardist_all(
            feat,
            embedder=embedder,
            uni2_cache_dir=UNI2_CACHE if embedder in ("uni","uni2","virchow2") else None,
            uni2_dir=uni2_dir,
        )
        if star is None:
            print(f"  FAIL {sample}: could not load {feat}", flush=True)
            continue
        got = predict_block(
            model, scaler, star["X"], star["xy"], device, use_spatial=True, spatial_k=None
        )
        write_star_h5ad(dest, star["obs_names"], star["xy"], got["pred"], classes)
        print(f"  {sample}: n={star['n']:,}  {dest}", flush=True)
        del star, got
        gc.collect()



def cmd_predict(name,samples,*,cuda_device,force,allow_cpu,embedder="uni"):
    # 2026.10.28, no existence-only skip / 无标签分支复用同一套输入/模型/概率事务
    from PanCancer_predict_visualize import predict_unlabeled_file
    import pipeline_provenance as pp
    from types import SimpleNamespace
    cohort=_cohort(name,embedder)
    model,scaler,device=_load_model(cuda_device,allow_cpu,embedder=embedder)
    classes=model.checkpoint_metadata["head_classes"]
    failures=[]
    for sample in samples:
        try:
            feat=_feat_h5ad(cohort,sample)
            if not feat.is_file():raise FileNotFoundError(feat)
            model.inference_coordinate_unit=os.environ.get("HIST2PHENO_INFERENCE_COORDINATES_UNIT","pixel")
            model.inference_mpp=model.checkpoint_metadata.get("section_mpp",{}).get(f"{name}::{sample}")
            rec=SimpleNamespace(dataset=name,sample=sample,section_id=f"{name}::{sample}",stardist_all=feat)
            predict_unlabeled_file(rec,pred_h5ad(cohort,sample),model=model,scaler=scaler,device=device,
                classes=classes,embedder=embedder,cache_dir=UNI2_CACHE,force=force)
        except Exception as exc:
            failures.append((sample,str(exc)));print(f"FAIL {sample}: {exc}",flush=True)
    if failures:raise RuntimeError(f"Unlabeled prediction failed: {failures}")

def cmd_plot(name: str, samples: list[str], *, save: bool, embedder: str = "uni") -> None:
    import matplotlib.pyplot as plt

    for fig in plot_spatial(name, samples, embedder=embedder, save=save):
        plt.close(fig)
    for fig in plot_composition(name, samples, embedder=embedder, save=save):
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort", choices=("Lung_Incomplete", "HCC_unlabel", "PDAC_unlabel"))
    p.add_argument("step", choices=("status", "dry-run", "predict", "plot", "rank"))
    p.add_argument("--sample", default="", help="one sample id")
    p.add_argument("--all", action="store_true", help="use every unlabeled slide, not the viz subset")
    p.add_argument("--force", action="store_true")
    p.add_argument("--embedder", choices=("uni", "uni2", "virchow2"), default="uni")
    p.add_argument("--cuda-device", default=os.environ.get("CUDA_DEVICE", "0"))
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--save", action="store_true", help="write 300 dpi PNGs under pred_unlabel/figures")
    args = p.parse_args(argv)
    if args.sample:
        samples = [args.sample]
    elif args.all or args.step in {"status", "dry-run"}:
        samples = discover_samples(args.cohort)
    else:
        samples = viz_samples(args.cohort, embedder=args.embedder)
    if args.step in {"status", "dry-run"}:
        show = None if args.step == "status" and not args.sample else samples
        if args.step == "dry-run" and not args.sample and not args.all:
            show = viz_samples(args.cohort, embedder=args.embedder)
        cmd_status(args.cohort, show, embedder=args.embedder)
        if args.step == "dry-run":
            print("viz subset:", viz_samples(args.cohort, embedder=args.embedder))
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
            embedder=args.embedder,
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
            embedder=args.embedder,
        )
        return 0
    cmd_plot(args.cohort, samples, save=args.save, embedder=args.embedder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
