#!/usr/bin/env python3
"""Predict pan-cancer L1/L12/L2 on every section and write 4×3 spatial+ROC panels.

Uses ``data/PanCancer/result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt``.

Outputs under ``data/PanCancer/pred_all_spatial/``:

  matched/{dataset}/{sample}/{sample}_matched_pred.h5ad
  matched/{dataset}/{sample}/{sample}_panel_4x3.png
  stardist_roc/{dataset}__{sample}/validation_external_stardist_matched_AUROC*.csv
  stardist_all/{dataset}/{sample}/{sample}_all_stardist_pred.h5ad
  metrics_per_section.csv

  conda activate SeededNTM
  python -u code/PanCancer/PanCancer_predict_visualize.py --datasets BRCA --max-sections 1
  python -u code/PanCancer/PanCancer_predict_visualize.py --include-incomplete-lung --skip-existing
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
for _p in (_PKG_DIR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base import adata_X_to_dense, build_spatial_neighbor_index, encode_labels_with_class_names  # noqa: E402
from model import load_model_for_predict  # noqa: E402
from plot import mlp_collect_five_head_softmax_probs  # noqa: E402
from uni_label_cv_helpers import (  # noqa: E402
    STARDIST_AUROC_TIER_SPECS,
    _dataframe_from_tier_scores,
    macro_auc_ovr,
)

from pancancer_cv import ok_mask  # noqa: E402
from pancancer_labels import map_native_series, ontology_class_names  # noqa: E402
from pancancer_paths import (  # noqa: E402
    COHORT_ORDER,
    DEFAULT_CKPT,
    DEFAULT_DATA_ROOT,
    DEFAULT_PRED_ROOT,
    discover_crc_predict_extras,
    discover_h5ads,
    discover_lung_incomplete,
    stardist_all_h5ad_for_matched,
)
from pancancer_viz import palette_for, plot_section_panel  # noqa: E402

_REPO_PREFIXES = (
    "/nobackup2/users/lingyu/Python/Hist2Pheno",
    "/home/lingyu/ssd2/Python/Hist2Pheno",
)


def resolve_repo_path(path) -> Path:
    """Prefer an existing file; remap nobackup2 ↔ ssd2 if needed."""
    p = Path(str(path))
    if p.exists():
        return p
    s = str(p)
    repo = str(_REPO_ROOT)
    for old in _REPO_PREFIXES:
        if s.startswith(old):
            q = Path(repo + s[len(old) :])
            if q.exists():
                return q
    return p

DEFAULT_HIDDEN = (768, 384, 192)
DEFAULT_SPATIAL_K = 8
HEADS = ("L1", "L12", "L2")
PROB_KEY = {"L1": "l1", "L12": "l12", "L2": "l2"}
TIER_CSV = {"L2": "l2", "L1": "l1", "L12": "l12"}


def setup_cuda(cuda_device: str) -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def _device(allow_cpu: bool):
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if allow_cpu:
        print("WARNING: CUDA unavailable; CPU.", flush=True)
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable. Pass --allow-cpu.")


def _spatial_xy(adata) -> np.ndarray:
    if "spatial_HE" in adata.obsm:
        return np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float64)
    raise KeyError("missing obsm spatial")


def _native_series(obs: pd.DataFrame, native_col: str) -> pd.Series:
    if native_col in obs.columns:
        return obs[native_col]
    if "celltype" in obs.columns:
        return obs["celltype"]
    raise KeyError(f"no native column among {list(obs.columns)}")


def load_matched(h5ad_path: Path, native_col: str, classes: dict[str, list[str]]) -> dict | None:
    import anndata as ad

    adata = ad.read_h5ad(resolve_repo_path(h5ad_path))
    native = _native_series(adata.obs, native_col)
    mapped = map_native_series(native)
    keep = mapped.notna().to_numpy()
    if not np.any(keep):
        return None
    idx = np.flatnonzero(keep)
    triples = mapped.to_numpy()[keep]
    y_l1 = np.asarray([t[0] for t in triples], dtype=object)
    y_l12 = np.asarray([t[1] for t in triples], dtype=object)
    y_l2 = np.asarray([t[2] for t in triples], dtype=object)
    X = np.asarray(adata_X_to_dense(adata.X)[idx], dtype=np.float32)
    xy = _spatial_xy(adata)[idx]
    names = adata.obs_names.to_numpy()[idx]
    return {
        "X": X,
        "xy": xy,
        "obs_names": names,
        "gt": {
            "L1": encode_labels_with_class_names(y_l1, classes["L1"]),
            "L12": encode_labels_with_class_names(y_l12, classes["L12"]),
            "L2": encode_labels_with_class_names(y_l2, classes["L2"]),
        },
        "gt_str": {"L1": y_l1, "L12": y_l12, "L2": y_l2},
        "n_raw": int(adata.n_obs),
    }


def load_stardist_all(path: Path) -> dict | None:
    import anndata as ad

    path = resolve_repo_path(path)
    if not path.is_file():
        return None
    adata = ad.read_h5ad(path)
    X = np.asarray(adata_X_to_dense(adata.X), dtype=np.float32)
    xy = _spatial_xy(adata)
    return {"X": X, "xy": xy, "obs_names": adata.obs_names.to_numpy(), "n": int(adata.n_obs)}


def predict_block(model, scaler, X, xy, device, *, use_spatial: bool, spatial_k: int) -> dict:
    Xs = scaler.transform(X)
    nbr = None
    if use_spatial:
        nbr = build_spatial_neighbor_index(xy, k_neighbors=spatial_k)
    probs = mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=nbr)
    out_p, out_y = {}, {}
    for head, pk in PROB_KEY.items():
        pr = np.asarray(probs[pk], dtype=np.float64)
        out_p[head] = pr
        out_y[head] = np.argmax(pr, axis=1).astype(np.int64)
    return {"probs": out_p, "pred": out_y}


def _scaler_section_order(ok: pd.DataFrame, seed: int) -> list:
    """Round-robin sections across datasets so a capped fit is not one-cohort."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list] = {}
    for rec in ok.itertuples(index=False):
        buckets.setdefault(str(rec.dataset), []).append(rec)
    for ds, recs in buckets.items():
        order = np.arange(len(recs))
        rng.shuffle(order)
        buckets[ds] = [recs[i] for i in order]
    ds_order = [d for d in COHORT_ORDER if d in buckets] + sorted(
        d for d in buckets if d not in COHORT_ORDER
    )
    out = []
    while any(buckets[d] for d in ds_order):
        for d in ds_order:
            if buckets[d]:
                out.append(buckets[d].pop(0))
    return out


def fit_or_load_scaler(index: pd.DataFrame, scaler_path: Path, max_cells: int | None, seed: int):
    import joblib
    from sklearn.preprocessing import StandardScaler

    scaler_path = Path(scaler_path)
    if scaler_path.is_file():
        print(f"  scaler {scaler_path}", flush=True)
        return joblib.load(scaler_path)
    print("  fitting StandardScaler (partial_fit on matched cells) …", flush=True)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    n_seen = 0
    ok = index.loc[ok_mask(index)] if "ok" in index.columns else index
    import anndata as ad

    unlimited = max_cells is None or int(max_cells) <= 0
    cap = None if unlimited else int(max_cells)
    for rec in _scaler_section_order(ok, seed):
        if cap is not None and n_seen >= cap:
            break
        try:
            h5ad = resolve_repo_path(rec.h5ad)
            adata = ad.read_h5ad(h5ad)
            n = int(adata.n_obs)
            take = n if cap is None else min(n, cap - n_seen)
            if take <= 0:
                break
            X = np.asarray(adata_X_to_dense(adata.X), dtype=np.float64)
            if take < n:
                idx = np.sort(rng.choice(n, size=take, replace=False))
                X = X[idx]
            scaler.partial_fit(X)
            n_seen += take
            print(f"    scaler cells {n_seen:,}  ({rec.section_id})", flush=True)
            del adata, X
        except Exception as exc:
            print(f"    scaler skip {getattr(rec, 'section_id', rec)}: {exc}", flush=True)
        finally:
            gc.collect()
    if n_seen == 0:
        raise RuntimeError("StandardScaler fit saw 0 cells; check h5ad paths in section_index.csv")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  wrote {scaler_path}  n={n_seen:,}", flush=True)
    return scaler


def write_auroc_csvs(out_dir: Path, classes: dict[str, list[str]], gt, pred, probs, matched_path: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for head, tier in TIER_CSV.items():
        spec = STARDIST_AUROC_TIER_SPECS[tier]
        names = classes[head]
        df = _dataframe_from_tier_scores(
            y_enc=gt[head],
            probs=probs[head],
            class_names=names,
            gt_col=spec["gt_col"],
            pred_col=spec["pred_col"],
            matched_features_path=matched_path,
            gt_names_raw=None,
        )
        csv_path = out_dir / spec["csv_name"]
        df.to_csv(csv_path, index=False)
        pd.DataFrame({"class_index": np.arange(len(names)), spec["names_col"]: names}).to_csv(
            out_dir / spec["csv_name"].replace(".csv", "_class_names.csv"),
            index=False,
        )


def write_pred_h5ad(path: Path, obs_names, xy, gt_str, pred, classes) -> None:
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index(obs_names.astype(str), name="cell_id"))
    obs["final_lineage"] = gt_str["L1"]
    obs["final_sublineage"] = gt_str["L12"]
    obs["final_CT"] = gt_str["L2"]
    obs["pred_lineage"] = [classes["L1"][int(i)] if 0 <= int(i) < len(classes["L1"]) else "" for i in pred["L1"]]
    obs["pred_sublineage"] = [
        classes["L12"][int(i)] if 0 <= int(i) < len(classes["L12"]) else "" for i in pred["L12"]
    ]
    obs["pred_CT"] = [classes["L2"][int(i)] if 0 <= int(i) < len(classes["L2"]) else "" for i in pred["L2"]]
    adata = ad.AnnData(obs=obs)
    adata.obsm["spatial_HE"] = np.asarray(xy, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def write_star_h5ad(path: Path, obs_names, xy, pred, classes) -> None:
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index(obs_names.astype(str), name="cell_id"))
    obs["pred_lineage"] = [classes["L1"][int(i)] if 0 <= int(i) < len(classes["L1"]) else "" for i in pred["L1"]]
    obs["pred_sublineage"] = [
        classes["L12"][int(i)] if 0 <= int(i) < len(classes["L12"]) else "" for i in pred["L12"]
    ]
    obs["pred_CT"] = [classes["L2"][int(i)] if 0 <= int(i) < len(classes["L2"]) else "" for i in pred["L2"]]
    adata = ad.AnnData(obs=obs)
    adata.obsm["spatial_HE"] = np.asarray(xy, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def process_section(rec, *, model, scaler, device, classes, palettes, pred_root: Path, use_spatial, spatial_k, skip_existing, skip_figures):
    dataset, sample, section_id = rec.dataset, rec.sample, rec.section_id
    matched_dir = pred_root / "matched" / dataset / sample
    panel_path = matched_dir / f"{sample}_panel_4x3.png"
    if skip_existing and panel_path.is_file():
        print(f"  skip existing {section_id}", flush=True)
        return None

    h5ad = resolve_repo_path(rec.h5ad)
    loaded = load_matched(h5ad, rec.native_col, classes)
    if loaded is None:
        print(f"  skip {section_id}: 0 kept cells", flush=True)
        return None
    got = predict_block(
        model, scaler, loaded["X"], loaded["xy"], device, use_spatial=use_spatial, spatial_k=spatial_k
    )
    write_pred_h5ad(
        matched_dir / f"{sample}_matched_pred.h5ad",
        loaded["obs_names"],
        loaded["xy"],
        loaded["gt_str"],
        got["pred"],
        classes,
    )
    roc_dir = pred_root / "stardist_roc" / f"{dataset}__{sample}"
    write_auroc_csvs(roc_dir, classes, loaded["gt"], got["pred"], got["probs"], "memory")

    star_pred = None
    xy_star = None
    star_path = stardist_all_h5ad_for_matched(h5ad)
    star = load_stardist_all(star_path)
    if star is not None:
        star_got = predict_block(
            model, scaler, star["X"], star["xy"], device, use_spatial=use_spatial, spatial_k=spatial_k
        )
        star_pred = star_got["pred"]
        xy_star = star["xy"]
        write_star_h5ad(
            pred_root / "stardist_all" / dataset / sample / f"{sample}_all_stardist_pred.h5ad",
            star["obs_names"],
            star["xy"],
            star_pred,
            classes,
        )
        del star, star_got
        gc.collect()

    macros = {}
    if not skip_figures:
        macros = plot_section_panel(
            save_path=panel_path,
            xy_gt=loaded["xy"],
            xy_star=xy_star,
            classes=classes,
            palettes=palettes,
            gt=loaded["gt"],
            pred=got["pred"],
            probs=got["probs"],
            star=star_pred,
            title=section_id,
        )
    row = {
        "section_id": section_id,
        "dataset": dataset,
        "sample": sample,
        "n_matched": int(len(loaded["xy"])),
        "n_stardist_all": int(len(xy_star)) if xy_star is not None else 0,
        "macro_auc_L1": macros.get("L1", float("nan")),
        "macro_auc_L12": macros.get("L12", float("nan")),
        "macro_auc_L2": macros.get("L2", float("nan")),
        "panel": str(panel_path) if not skip_figures else "",
    }
    for head in HEADS:
        y, pr = loaded["gt"][head], got["probs"][head]
        row[f"macro_auc_{head}_cells"] = macro_auc_ovr(y, pr, len(classes[head]))
    print(
        f"  {section_id}: n={row['n_matched']:,}  L2 AUROC={row['macro_auc_L2_cells']:.3f}  "
        f"stardist_all={row['n_stardist_all']:,}",
        flush=True,
    )
    del loaded, got
    gc.collect()
    return row


def process_incomplete(rec, *, model, scaler, device, classes, palettes, pred_root, use_spatial, spatial_k, skip_existing, skip_figures):
    sample = rec.sample
    out_dir = pred_root / "incomplete_lung" / sample
    panel_path = out_dir / f"{sample}_panel_4x3.png"
    if skip_existing and panel_path.is_file():
        print(f"  skip existing LungIncomplete::{sample}", flush=True)
        return None
    star = load_stardist_all(Path(rec.stardist_all))
    if star is None:
        return None
    got = predict_block(
        model, scaler, star["X"], star["xy"], device, use_spatial=use_spatial, spatial_k=spatial_k
    )
    write_star_h5ad(
        out_dir / f"{sample}_all_stardist_pred.h5ad",
        star["obs_names"],
        star["xy"],
        got["pred"],
        classes,
    )
    if not skip_figures:
        plot_section_panel(
            save_path=panel_path,
            xy_gt=None,
            xy_star=star["xy"],
            classes=classes,
            palettes=palettes,
            gt=None,
            pred=got["pred"],
            probs=None,
            star=got["pred"],
            title=f"LungIncomplete::{sample}",
        )
    print(f"  LungIncomplete::{sample}: n={star['n']:,}", flush=True)
    return {
        "section_id": rec.section_id,
        "dataset": "LungIncomplete",
        "sample": sample,
        "n_matched": 0,
        "n_stardist_all": int(star["n"]),
        "macro_auc_L1_cells": float("nan"),
        "macro_auc_L12_cells": float("nan"),
        "macro_auc_L2_cells": float("nan"),
        "panel": str(panel_path) if not skip_figures else "",
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--index-csv", type=Path, default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max-sections", type=int, default=None)
    p.add_argument("--no-spatial-context", action="store_true", help="disable kNN fusion (checkpoint is spatial)")
    p.add_argument("--spatial-k", type=int, default=DEFAULT_SPATIAL_K)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--include-incomplete-lung", action="store_true")
    p.add_argument(
        "--no-crc-p5",
        action="store_true",
        help="do not add CRC P5CRC when it is absent from section_index.csv",
    )
    p.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help="default: data/PanCancer/result_all_spatial/scaler.joblib",
    )
    p.add_argument(
        "--scaler-max-cells",
        type=int,
        default=400_000,
        help="partial_fit cap; 0 = all matched cells in the scaler index",
    )
    p.add_argument(
        "--scaler-from-work-index",
        action="store_true",
        help="fit scaler only on the sections being predicted (for smoke; do not reuse as production)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_cuda(args.cuda_device)
    device = _device(args.allow_cpu)
    classes = ontology_class_names()
    palettes = palette_for(classes)
    pred_root = args.pred_root.expanduser().resolve()
    pred_root.mkdir(parents=True, exist_ok=True)
    (pred_root / "ontology_heads.json").write_text(json.dumps(classes, indent=2))

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    print("=" * 60, flush=True)
    print("Pan-cancer predict / visualize", flush=True)
    print(f"  ckpt:   {ckpt}", flush=True)
    print(f"  out:    {pred_root}", flush=True)
    print(f"  device: {device}  spatial={not args.no_spatial_context}", flush=True)
    print("=" * 60, flush=True)

    model = load_model_for_predict(
        str(_REPO_ROOT),
        "PanCancer",
        "project_all_UNI",
        checkpoint_path=str(ckpt),
        device=device,
        hidden_dims=DEFAULT_HIDDEN,
        parent_dir=True,
    )
    print(
        f"  model.use_spatial_context={getattr(model, 'use_spatial_context', None)}",
        flush=True,
    )

    idx_path = args.index_csv or (args.data_root / "section_index.csv")
    if idx_path.is_file():
        full_index = pd.read_csv(idx_path)
    else:
        full_index = pd.DataFrame(discover_h5ads(None))
        full_index["ok"] = True
    if "ok" in full_index.columns:
        full_index = full_index.loc[ok_mask(full_index)].copy()
    index = full_index
    if args.datasets:
        index = index[index["dataset"].isin(args.datasets)].copy()
    want_crc = (not args.datasets) or ("CRC" in args.datasets)
    if want_crc and not args.no_crc_p5:
        have = set(index["sample"].astype(str)) if len(index) and "sample" in index.columns else set()
        extra = discover_crc_predict_extras(have)
        if extra:
            extra_df = pd.DataFrame(extra)
            extra_df["ok"] = True
            index = pd.concat([index, extra_df], ignore_index=True)
            print(
                f"  + CRC predict extras (not in CV index): {[r['sample'] for r in extra]}",
                flush=True,
            )
    if args.max_sections is not None:
        index = index.head(int(args.max_sections)).copy()
    print(f"  sections: {len(index)}", flush=True)

    scaler_path = args.scaler_path or (args.data_root / "result_all_spatial" / "scaler.joblib")
    scaler_index = index if args.scaler_from_work_index else full_index
    scaler = fit_or_load_scaler(scaler_index, scaler_path, args.scaler_max_cells, args.seed)

    rows = []
    for rec in index.itertuples(index=False):
        try:
            row = process_section(
                rec,
                model=model,
                scaler=scaler,
                device=device,
                classes=classes,
                palettes=palettes,
                pred_root=pred_root,
                use_spatial=not args.no_spatial_context,
                spatial_k=args.spatial_k,
                skip_existing=args.skip_existing,
                skip_figures=args.skip_figures,
            )
            if row:
                rows.append(row)
        except Exception as exc:
            import traceback

            print(f"  FAIL {getattr(rec, 'section_id', rec)}: {exc}", flush=True)
            traceback.print_exc()

    if args.include_incomplete_lung:
        for rec in discover_lung_incomplete():
            rec = argparse.Namespace(**rec)
            try:
                row = process_incomplete(
                    rec,
                    model=model,
                    scaler=scaler,
                    device=device,
                    classes=classes,
                    palettes=palettes,
                    pred_root=pred_root,
                    use_spatial=not args.no_spatial_context,
                    spatial_k=args.spatial_k,
                    skip_existing=args.skip_existing,
                    skip_figures=args.skip_figures,
                )
                if row:
                    rows.append(row)
            except Exception as exc:
                import traceback

                print(f"  FAIL {rec.section_id}: {exc}", flush=True)
                traceback.print_exc()

    if rows:
        met = pd.DataFrame(rows)
        met_path = pred_root / "metrics_per_section.csv"
        if met_path.is_file():
            prev = pd.read_csv(met_path)
            met = pd.concat([prev, met], ignore_index=True)
            if "section_id" in met.columns:
                met = met.drop_duplicates("section_id", keep="last")
        met.to_csv(met_path, index=False)
        print(f"\nWrote {met_path}  n={len(met)}", flush=True)
        for ds, sub in met.groupby("dataset"):
            print(
                f"  {ds:16s}  n={len(sub):4d}  mean L2 AUROC={sub['macro_auc_L2_cells'].mean():.3f}",
                flush=True,
            )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
