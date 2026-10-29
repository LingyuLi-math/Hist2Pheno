#!/usr/bin/env python3
"""Predict pan-cancer L1/L12/L2 on every section and write 4×3 spatial+ROC panels.

2026.10.29: selected --embedder defaults to its own A2 best checkpoint/artifact.
Legacy paths below are historical examples; outputs default to pred_<embedder>_deployment.

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
from checkpoint_artifact import read_artifact, checkpoint_scaler, digest, file_digest
from feature_integrity import feature_manifest, finite_rows, atomic_json, output_manifest, completed_run
import pipeline_provenance as pp
from evaluation_audit import EvaluationRun
from spatial_context import inference_runtime
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


def _stardist_uni2_dir(h5ad_path: Path, embedder="uni2") -> Path | None:
    """All-nuclei UNI2 tokens, not the GT crop directory."""
    if embedder in ("uni","virchow2"):
        from embedding_backend import embedding_dir
        try:return embedding_dir(h5ad_path,{},embedder,stardist=True)
        except FileNotFoundError:return None
    parent = Path(h5ad_path).parent
    sample = parent.name
    rel = Path("ImgEmbeddings_all_stardist") / "sc_pth_14_14"
    for cand in (parent / f"{sample}_project_all_UNI2" / rel, parent / "project_all_UNI2" / rel):
        if __import__("embedding_store").available(cand):
            return cand
    return None


def _feature_matrix(h5ad_path: Path, adata, row_idx, *, embedder: str, uni2_cache_dir: Path | None, uni2_dir: Path | None = None):
    """Rows of ``row_idx``. UNI2 drops rows with no finite token and returns that mask."""
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if embedder in ("uni","virchow2"):
        from embedding_backend import aligned
        if uni2_cache_dir is None:raise ValueError("virchow2 requires a cache directory")
        matrix=aligned(h5ad_path,adata,uni2_cache_dir,embedder,directory=uni2_dir)
        present=finite_rows(matrix,row_idx)
        return np.array(matrix[row_idx[present]],dtype=np.float32,copy=True),present
    if embedder != "uni2":
        raise ValueError("embedder must be uni or uni2")
    if uni2_cache_dir is None:
        raise ValueError("uni2 embedder requires uni2_cache_dir")
    from PanCancer_train_validate_cv_UNIlabel import _aligned_uni2_matrix

    if uni2_dir is not None:
        adata.uns["embedding_dir"] = str(uni2_dir)
    aligned = _aligned_uni2_matrix(Path(h5ad_path), adata, Path(uni2_cache_dir))
    present = finite_rows(aligned, row_idx)
    kept = row_idx[present]
    return np.array(aligned[kept], dtype=np.float32, copy=True), present


def load_matched(h5ad_path: Path, native_col: str, classes: dict[str, list[str]], *, embedder: str = "uni", uni2_cache_dir: Path | None = None) -> dict | None:
    import anndata as ad

    adata = ad.read_h5ad(resolve_repo_path(h5ad_path))
    pp.ground_truth_sources(adata,uni2_cache_dir)
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
    X, present = _feature_matrix(
        h5ad_path, adata, idx, embedder=embedder, uni2_cache_dir=uni2_cache_dir
    )
    if not np.all(present):
        n_miss = int((~present).sum())
        print(f"  {h5ad_path.parent.name}: drop {n_miss:,} kept cells with no UNI2 .pth", flush=True)
        idx = idx[present]
        y_l1, y_l12, y_l2 = y_l1[present], y_l12[present], y_l2[present]
    if len(idx) == 0:
        return None
    xy = _spatial_xy(adata)[idx]
    names = adata.obs_names.to_numpy()[idx]
    for head,values in (("L1",y_l1),("L12",y_l12),("L2",y_l2)):
        unknown=set(values)-set(classes[head])
        if unknown:raise ValueError(f"Latest GT contains classes absent from checkpoint {head}: {sorted(unknown)}")
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
        "native_col_used":str(native.name),
    }


def load_stardist_all(path: Path, *, embedder: str = "uni", uni2_cache_dir: Path | None = None, uni2_dir: Path | None = None, min_prob: float = 0.) -> dict | None:
    import anndata as ad

    path = resolve_repo_path(path)
    if not path.is_file():
        return None
    adata = ad.read_h5ad(path)
    row_idx = np.arange(int(adata.n_obs), dtype=np.int64)
    if not np.isfinite(min_prob) or not 0<=min_prob<=1:raise ValueError("Probability cutoff must be in [0,1]")
    if min_prob>0:
        if "probability" not in adata.obs:raise ValueError("Probability cutoff requested but feature h5ad has no probability column; rebuild from current StarDist CSV")
        values=np.asarray(adata.obs["probability"],dtype=float)
        if not np.isfinite(values).all():raise ValueError("Non-finite StarDist probabilities")
        row_idx=row_idx[values>=min_prob]
    X, present = _feature_matrix(
        path, adata, row_idx, embedder=embedder, uni2_cache_dir=uni2_cache_dir, uni2_dir=uni2_dir
    )
    if len(X) == 0:
        return None
    xy = _spatial_xy(adata)[row_idx[present]]
    names = adata.obs_names.to_numpy()[row_idx[present]]
    return {"X": X, "xy": xy, "obs_names": names, "n": int(len(names))}


def predict_block(model,scaler,X,xy,device,*,use_spatial: bool,spatial_k=None) -> dict:
    # 2026.10.28, explicit input units and no ignored K / 真实坐标单位与参数冲突检查
    pp.finite_matrix(X)
    Xs=np.asarray(scaler.transform(X),dtype=np.float32);pp.finite_matrix(Xs,"scaled features")
    request=None
    if not getattr(model,"use_spatial_context",False) and spatial_k is not None:
        raise ValueError("--spatial-k has no effect on a nonspatial checkpoint")
    if getattr(model,"use_spatial_context",False):
        if not use_spatial:raise ValueError("Spatial checkpoint requires context; use a nonspatial model for ablation")
        if spatial_k is not None and int(spatial_k)!=model.spatial_fusion.config.k:
            raise ValueError("--spatial-k conflicts with checkpoint; retrain for a K ablation")
        physical_xy,cfg=pp.inference_coordinates(model,xy,
            unit=getattr(model,"inference_coordinate_unit",None),mpp=getattr(model,"inference_mpp",None))
        request={"runtime":inference_runtime(X,physical_xy,scaler,cfg,model.spatial_fusion.mode)}
    probs=mlp_collect_five_head_softmax_probs(model,Xs,device,neighbor_index=request)
    classes=model.checkpoint_metadata['head_classes'];pp.validate_probabilities(probs,classes,len(X))
    out_p={h:np.asarray(probs[k],dtype=np.float32) for h,k in PROB_KEY.items()}
    return {"probs":out_p,"pred":{h:p.argmax(1).astype(np.int64) for h,p in out_p.items()}}


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
    pp.atomic_h5ad(adata,path)


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
    pp.atomic_h5ad(adata,path)



def _token_stamp(h5ad, cache_dir):
    if not Path(h5ad).is_file():
        return None
    import anndata as ad
    from PanCancer_train_validate_cv_UNIlabel import _uni2_embedding_dir
    data = ad.read_h5ad(h5ad, backed="r")
    try:
        directory = _uni2_embedding_dir(Path(h5ad), data.uns)
        return feature_manifest(directory, cache_dir)
    finally:
        data.file.close()


def _prediction_source(h5ad,star_path,model,classes,embedder,cache_dir,star_tokens,*,identity,scaler):
    h5ad,star_path=resolve_repo_path(h5ad),resolve_repo_path(star_path)
    tokens=[]
    if embedder in ("uni","uni2","virchow2"):
        if h5ad==star_path and star_tokens is not None:tokens.append(star_tokens)
        else:
            import anndata as ad
            from PanCancer_train_validate_cv_UNIlabel import _uni2_embedding_dir
            data=ad.read_h5ad(h5ad,backed="r")
            try:
                if embedder in ("uni","virchow2"):
                    from embedding_backend import embedding_dir
                    tokens.append(embedding_dir(h5ad,data.uns,embedder))
                else:tokens.append(_uni2_embedding_dir(h5ad,data.uns))
            finally:data.file.close()
        if star_tokens is not None and str(star_tokens) not in set(map(str,tokens)):tokens.append(star_tokens)
    optional=[]
    if embedder in ("uni","uni2","virchow2"):
        parent=star_path.parent;sample=parent.name
        patch=16 if embedder=="uni" else 14
        rel=Path("ImgEmbeddings_all_stardist")/f"sc_pth_{patch}_{patch}"
        project={"uni":"UNI","uni2":"UNI2","virchow2":"Virchow2"}[embedder]
        optional=[parent/f"{sample}_project_all_{project}"/rel,parent/f"project_all_{project}"/rel]
    files=[h5ad,star_path]
    import anndata as ad
    for file in set((h5ad,star_path)):
        if file.is_file():
            header=ad.read_h5ad(file,backed="r")
            try:files.extend(pp.ground_truth_sources(header,cache_dir))
            finally:header.file.close()
    index_source=getattr(model,"evaluation_index_path",None)
    if index_source:files.append(index_source)
    if embedder in ("uni","uni2","virchow2"):
        from embedding_backend import validate_tokens
        for directory in tokens:validate_tokens(model,directory,embedder)
    data_inputs=pp.capture_inputs(files,tokens,cache_dir,optional_tokens=optional)
    checkpoint=getattr(model,"checkpoint_source_path",None)
    if checkpoint:files.extend([checkpoint,str(checkpoint)+".artifact.pt"])
    return {"inputs":pp.capture_inputs(files,tokens,cache_dir,optional_tokens=optional),"data_inputs":data_inputs,"identity":identity,
        "model":pp.model_descriptor(model,scaler,classes),"embedder":embedder,
        "stardist_prob_min":float(os.environ.get("STAR_PROB_MIN","0")),
        "implementation":pp.implementations(),"predict_implementation":pp.sha256(Path(__file__)),
        "coordinate_unit":getattr(model,"inference_coordinate_unit",None),
        "mpp":getattr(model,"inference_mpp",None) or os.environ.get("HIST2PHENO_INFERENCE_MPP"),"scope":getattr(model,"evaluation_scope",None)}


def _prediction_stamp(*args,**kwargs):return pp.fingerprint(_prediction_source(*args,**kwargs))

def process_section(rec, *, model, scaler, device, classes, palettes, pred_root: Path, use_spatial, spatial_k, skip_existing, skip_figures, embedder="uni", uni2_cache_dir=None):
    dataset, sample, section_id = rec.dataset, rec.sample, rec.section_id
    matched_dir = pred_root / "matched" / dataset / sample
    panel_path = matched_dir / f"{sample}_panel_4x3.png"
    # 2026.10.24, bind predictions to input/checkpoint / 旧图不能代表新模型已预测
    # Historical stamp expanded below / 保留原注释，新增特征与 StarDist 输入版本
    marker = matched_dir/"prediction_run.json"
    h5ad = resolve_repo_path(rec.h5ad)
    star_path = stardist_all_h5ad_for_matched(h5ad)
    star_out = pred_root / "stardist_all" / str(dataset) / str(sample) / f"{sample}_all_stardist_pred.h5ad"
    uni2_star_ready = _stardist_uni2_dir(star_path,embedder) if embedder in ("uni","uni2","virchow2") else None
    # A finished matched-cell run must not hide a later StarDist UNI2 extract.
    # 2026.10.26, content-bound completion / 新增 token、覆盖 token、丢失输出均重算
    identity={"section_id":str(section_id),"dataset":str(dataset),"sample":str(sample),"native_col":str(rec.native_col)}
    source=_prediction_source(h5ad,star_path,model,classes,embedder,uni2_cache_dir,uni2_star_ready,identity=identity,scaler=scaler)
    stamp=pp.fingerprint(source)
    required = [str((matched_dir/f"{sample}_matched_pred.h5ad").relative_to(pred_root))]
    if not skip_figures:
        required.append(str(panel_path.relative_to(pred_root)))
    if uni2_star_ready is not None and star_path.is_file():
        required.append(str(star_out.relative_to(pred_root)))
    run = completed_run(marker, stamp, pred_root, required)
    if skip_existing and run:
        print(f"  skip verified existing {section_id}", flush=True)
        return dict(run["row"],prediction_status="verified_cache")
    atomic_json(marker,{"version":2,"state":"running","stamp":stamp,"outputs":[]})
    loaded = load_matched(h5ad, rec.native_col, classes, embedder=embedder, uni2_cache_dir=uni2_cache_dir)
    if loaded is None:
        print(f"  skip {section_id}: 0 kept cells", flush=True)
        raise ValueError(f"No valid matched cells: {section_id}")
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
    star = None
    uni2_star = uni2_star_ready if embedder in ("uni","uni2","virchow2") else None
    if embedder in ("uni","uni2","virchow2") and uni2_star is None:
        print(f"  {section_id}: no StarDist UNI2 tokens; skip all-nuclei map", flush=True)
    elif star_path.is_file() or embedder == "uni":
        star = load_stardist_all(
            star_path, embedder=embedder, uni2_cache_dir=uni2_cache_dir, uni2_dir=uni2_star
        )
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
        "evaluation_scope":getattr(model,"evaluation_scope","unspecified"),
        "checkpoint_sha256":getattr(model,"checkpoint_weight_sha256",None),
        "run_id":stamp,
        "requested_native_col":str(rec.native_col),"native_col_used":loaded["native_col_used"],
        "mapping_sha256":pp.fingerprint(pp.mapping_descriptor()),"model_family":pp.checkpoint_family(model.checkpoint_metadata),
        "scaler_sha256":pp.scaler_signature(scaler),"embedder":embedder,"prediction_status":"computed",
        "row_identity_sha256":pp.row_signature(loaded["obs_names"],loaded["xy"]),
        "gt_sha256":pp.fingerprint(loaded["obs_names"],loaded["gt_str"]),
        "input_sha256":pp.fingerprint(source["data_inputs"]),
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
    # 2026.10.26, verify every required output before publishing completion / 完整输出清单
    outputs = [matched_dir / f"{sample}_matched_pred.h5ad", *sorted(roc_dir.glob("*.csv"))]
    if not skip_figures:
        outputs.append(panel_path)
    if xy_star is not None:
        outputs.append(star_out)
    if _prediction_stamp(rec.h5ad,star_path,model,classes,embedder,uni2_cache_dir,uni2_star_ready,identity=identity,scaler=scaler) != stamp:
        raise RuntimeError("Prediction inputs changed during run; retry after extraction completes")
    atomic_json(marker, {"version":2,"stamp":stamp,
        "checkpoint_sha256":getattr(model,"checkpoint_weight_sha256",None),
        "metric_source":"GT-centered matched cells; not StarDist-center ROC",
        "outputs":output_manifest(outputs,pred_root),"row":row,"source":source})
    del loaded, got
    gc.collect()
    return row


def process_incomplete(rec, *, model, scaler, device, classes, palettes, pred_root, use_spatial, spatial_k, skip_existing, skip_figures, embedder="uni", uni2_cache_dir=None):
    sample = rec.sample
    out_dir = pred_root / "incomplete_lung" / sample
    panel_path = out_dir / f"{sample}_panel_4x3.png"
    # 2026.10.26, incomplete lung also uses verified manifests / 换模型或无图模式均正确处理
    star_path = Path(rec.stardist_all)
    uni2_star = _stardist_uni2_dir(star_path,embedder) if embedder in ("uni","uni2","virchow2") else None
    if embedder in ("uni","uni2","virchow2") and uni2_star is None:
        print(f"  LungIncomplete::{sample}: no StarDist UNI2 tokens", flush=True)
        return None
    model.evaluation_scope="unlabeled_deployment_not_oof"
    identity={"section_id":str(rec.section_id),"dataset":"LungIncomplete","sample":str(sample),"native_col":None}
    source=_prediction_source(star_path,star_path,model,classes,embedder,uni2_cache_dir,uni2_star,identity=identity,scaler=scaler)
    stamp=pp.fingerprint(source)
    marker = out_dir / "prediction_run.json"
    required = [str((out_dir/f"{sample}_all_stardist_pred.h5ad").relative_to(pred_root))]
    if not skip_figures:
        required.append(str(panel_path.relative_to(pred_root)))
    run = completed_run(marker, stamp, pred_root, required)
    if skip_existing and run:
        return dict(run["row"],prediction_status="verified_cache")
    star = load_stardist_all(star_path, embedder=embedder, uni2_cache_dir=uni2_cache_dir, uni2_dir=uni2_star)
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
    row = {
        "section_id": rec.section_id,
        "dataset": "LungIncomplete",
        "evaluation_scope":"unlabeled_deployment_not_oof",
        "run_id":stamp,
        "mapping_sha256":pp.fingerprint(pp.mapping_descriptor()),"model_family":pp.checkpoint_family(model.checkpoint_metadata),
        "scaler_sha256":pp.scaler_signature(scaler),"embedder":embedder,
        "checkpoint_sha256":getattr(model,"checkpoint_weight_sha256",None),
        "sample": sample,
        "n_matched": 0,
        "n_stardist_all": int(star["n"]),
        "macro_auc_L1_cells": float("nan"),
        "macro_auc_L12_cells": float("nan"),
        "macro_auc_L2_cells": float("nan"),
        "panel": str(panel_path) if not skip_figures else "",
    }
    outputs = [out_dir/f"{sample}_all_stardist_pred.h5ad"]
    if not skip_figures:
        outputs.append(panel_path)
    if _prediction_stamp(star_path,star_path,model,classes,embedder,uni2_cache_dir,uni2_star,identity=identity,scaler=scaler) != stamp:
        raise RuntimeError("Incomplete lung input changed while predicting")
    atomic_json(marker, {"version":2,"stamp":stamp,"row":row,"outputs":output_manifest(outputs,pred_root),"source":source})
    return row




def predict_unlabeled_file(rec,dest,*,model,scaler,device,classes,embedder,cache_dir,force=False):
    import anndata as ad
    dest=Path(dest);feature=resolve_repo_path(rec.stardist_all)
    minimum=float(os.environ.get("STAR_PROB_MIN","0"))
    if not np.isfinite(minimum) or not 0<=minimum<=1:raise ValueError("STAR_PROB_MIN must be in [0,1]")
    directory=_stardist_uni2_dir(feature,embedder) if embedder in ("uni","uni2","virchow2") else None
    if embedder in ("uni","uni2","virchow2") and directory is None:raise FileNotFoundError("No current UNI2 StarDist tokens")
    model.evaluation_scope="unlabeled_deployment_not_oof"
    identity={"section_id":str(rec.section_id),"dataset":str(rec.dataset),"sample":str(rec.sample),"native_col":None}
    marker=dest.with_suffix(".pipeline.json");probfile=dest.with_name(str(rec.sample)+"_l2_probs.npz")
    with pp.file_lock(str(dest)+".lock"):
        source=_prediction_source(feature,feature,model,classes,embedder,cache_dir,directory,identity=identity,scaler=scaler)
        stamp=pp.fingerprint(source)
        if not force and pp.complete(marker,stamp,[dest,probfile]):return
        pp.atomic_json(marker,{"version":pp.VERSION,"state":"running","stamp":stamp,"outputs":{}})
        data=load_stardist_all(feature,embedder=embedder,uni2_cache_dir=cache_dir,uni2_dir=directory,min_prob=minimum)
        if data is None:raise ValueError("No valid unlabeled cells")
        rows=pp.row_signature(data["obs_names"],data["xy"])
        got=predict_block(model,scaler,data["X"],data["xy"],device,use_spatial=True,spatial_k=None)
        obs=pd.DataFrame(index=pd.Index(data["obs_names"].astype(str),name="cell_id"))
        for head,col in (("L1","pred_lineage"),("L12","pred_sublineage"),("L2","pred_CT")):
            obs[col]=np.asarray(classes[head])[got["pred"][head]]
        result=ad.AnnData(obs=obs);result.obsm["spatial_HE"]=data["xy"]
        result.uns.update(pipeline_run_id=stamp,row_identity_sha256=rows,classes=classes,
                          checkpoint_sha256=model.checkpoint_weight_sha256,stardist_prob_min=minimum)
        if _prediction_stamp(feature,feature,model,classes,embedder,cache_dir,directory,identity=identity,scaler=scaler)!=stamp:
            raise RuntimeError("Unlabeled inputs changed during inference")
        pp.atomic_h5ad(result,dest)
        pp.atomic_npz(probfile,probs=got["probs"]["L2"],class_names=np.asarray(classes["L2"],dtype=str),
                      obs_names=np.asarray(data["obs_names"]).astype(str),row_identity_sha256=np.asarray(rows),pipeline_run_id=np.asarray(stamp))
        pp.publish(marker,stamp,[dest,probfile],source=source,row_signature=rows)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--index-csv", type=Path, default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="restrict to the actual fold_plan entry; best_mlp may come from any fold",
    )
    p.add_argument("--max-sections", type=int, default=None)
    p.add_argument("--no-spatial-context", action="store_true", help="require nonspatial checkpoint; spatial models cannot disable context")
    p.add_argument("--spatial-k", type=int, default=None)
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
        default=None,
        help="partial_fit cap; 0 = all matched cells in the scaler index",
    )
    p.add_argument(
        "--scaler-from-work-index",
        action="store_true",
        help="fit scaler only on the sections being predicted (for smoke; do not reuse as production)",
    )
    p.add_argument("--embedder", choices=("uni", "uni2", "virchow2"), default="uni")
    p.add_argument("--coordinates-unit",choices=("pixel","um"),default=os.environ.get("HIST2PHENO_INFERENCE_COORDINATES_UNIT","pixel"))
    p.add_argument("--inference-mpp",type=float,default=None)
    p.add_argument("--deployment",action="store_true",help="Use one selected checkpoint for all sections; these are NOT OOF metrics")
    args=p.parse_args(argv)
    if args.scaler_path is not None or args.scaler_max_cells is not None or args.scaler_from_work_index:
        p.error("Legacy scaler flags are unsupported: inference always uses checkpoint artifact scaler")
    if args.inference_mpp is not None and (not np.isfinite(args.inference_mpp) or args.inference_mpp<=0):
        p.error("--inference-mpp must be finite and positive")
    if args.coordinates_unit=="um" and args.inference_mpp not in (None,1.):
        p.error("Coordinates already in um must not be scaled by another mpp")
    return args


def resolve_encoder_defaults(args):
    # 2026.10.29, selected encoder defaults to its own A2 artifact, never legacy UNI.
    from external_validation import uni_overrides, uni2_overrides, virchow2_overrides
    loaders={"uni":uni_overrides,"uni2":uni2_overrides,"virchow2":virchow2_overrides}
    if args.checkpoint is None:args.checkpoint=loaders[args.embedder](_REPO_ROOT)["checkpoint"]
    if args.pred_root is None:args.pred_root=args.data_root/f"pred_{args.embedder}_deployment"
    return args


def main(argv=None) -> int:
    args=parse_args(argv)
    with pp.file_lock(args.pred_root.expanduser().resolve()/".evaluation.lock"):
        return _main(args)


def _main(args) -> int:
    setup_cuda(args.cuda_device)
    device = _device(args.allow_cpu)
    classes = ontology_class_names()
    palettes = palette_for(classes)
    pred_root = args.pred_root.expanduser().resolve()
    pred_root.mkdir(parents=True, exist_ok=True)

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    print("=" * 60, flush=True)
    print("Pan-cancer predict / visualize", flush=True)
    print(f"  ckpt:   {ckpt}", flush=True)
    print(f"  out:    {pred_root}", flush=True)
    print(f"  device: {device}  spatial={not args.no_spatial_context}  embedder={args.embedder}", flush=True)
    print("=" * 60, flush=True)

    model = load_model_for_predict(
        str(_REPO_ROOT),
        "PanCancer",
        "project_all_UNI",
        checkpoint_path=str(ckpt),
        device=device,
        hidden_dims=None,
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
    if args.fold is not None:
        plan_path = ckpt.parents[2] / "fold_plan.csv"
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        plan = pd.read_csv(plan_path)
        keys = set(
            zip(
                plan.loc[plan["fold"].astype(int) == int(args.fold), "dataset"].astype(str),
                plan.loc[plan["fold"].astype(int) == int(args.fold), "sample"].astype(str),
            )
        )
        index = index[
            index.apply(lambda r: (str(r.dataset), str(r.sample)) in keys, axis=1)
        ].copy()
        print(f"  fold={int(args.fold)}  {len(index)} sections from {plan_path.name}", flush=True)
    want_crc = (not args.datasets) or ("CRC" in args.datasets)
    if want_crc and not args.no_crc_p5 and args.fold is None and args.deployment:
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

    uni2_cache_dir = args.data_root / (args.embedder+"_aligned")
    # Exact training transform; implicit global/scaler refitting is disabled.
    scaler = checkpoint_scaler(ckpt)
    from embedding_backend import validate_model
    validate_model(model,scaler,args.embedder)
    meta = read_artifact(ckpt,required=True)["metadata"]
    if meta.get("head_classes") and all(meta["head_classes"].values()):
        classes = meta["head_classes"]
        palettes = palette_for(classes)
    atomic_json(pred_root / "ontology_heads.json", classes)
    if args.fold is not None and meta.get("fold") != int(args.fold) and args.deployment:
        raise ValueError("Requested held-out fold does not match selected checkpoint metadata")
    fold_lookup = {}
    if not args.deployment and meta.get("fold_mode")=="cell_stratified":
        raise ValueError("Per-cell OOF is in oof_predictions.npz; section-wide cell-stratified predictions require --deployment and are not OOF")
    if not args.deployment:
        plan = pd.read_csv(ckpt.parents[2]/"fold_plan.csv")
        fold_lookup = dict(zip(plan.section_id.astype(str),plan.fold.astype(int)))
        if not set(index.section_id.astype(str)).issubset(fold_lookup):
            raise ValueError("OOF prediction needs every section in fold_plan; use --deployment for new sections")
        pred_root = pred_root/"oof"
        pred_root.mkdir(parents=True,exist_ok=True)

    if index.empty:raise ValueError("No selected sections")
    if index.section_id.duplicated().any() or index.duplicated(["dataset","sample"]).any():
        raise ValueError("Duplicate sections/output paths")
    family=pp.checkpoint_family(meta)
    context={"scope":"deployment_not_oof" if args.deployment else "outer_fold_oof",
             "mapping_sha256":pp.fingerprint(pp.mapping_descriptor()),"model_family":family,"embedder":args.embedder,
             "checkpoint":str(ckpt),"coordinates_unit":args.coordinates_unit,"inference_mpp":args.inference_mpp}
    audit=EvaluationRun(pred_root,index[["section_id","dataset","sample","native_col","h5ad"]].to_dict("records"),context=context)
    failures = []
    for rec in index.itertuples(index=False):
        try:
            if not args.deployment:
                fold_ckpt = ckpt.parent/f"hce_group_fold_{fold_lookup[str(rec.section_id)]}.pt"
                model = load_model_for_predict(str(_REPO_ROOT),"PanCancer","project_all_UNI",checkpoint_path=str(fold_ckpt),device=device,hidden_dims=None,parent_dir=True)
                scaler = checkpoint_scaler(fold_ckpt)
                fold_meta = read_artifact(fold_ckpt,required=True)["metadata"]
                if pp.checkpoint_family(fold_meta)!=family:raise ValueError("Mixed OOF checkpoint protocols")
                if fold_meta.get("head_classes") != classes:
                    raise ValueError("Fold checkpoint class order mismatch")
                if fold_meta.get("fold") != fold_lookup[str(rec.section_id)]: raise ValueError("Fold artifact mismatch")
                if str(rec.section_id) not in fold_meta.get("held_out_sections",[]): raise ValueError("Section not held out by this artifact")
            model.evaluation_index_path=str(idx_path.resolve()) if idx_path.is_file() else None
            model.inference_coordinate_unit=args.coordinates_unit
            model.inference_mpp=args.inference_mpp or model.checkpoint_metadata.get("section_mpp",{}).get(str(rec.section_id))
            model.evaluation_scope="deployment_not_oof" if args.deployment else "outer_fold_oof"
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
                embedder=args.embedder,
                uni2_cache_dir=uni2_cache_dir,
            )
            if row:audit.success(rec.section_id,row)
            else:raise ValueError("No verified matched result")
        except Exception as exc:
            import traceback

            audit.failure(rec.section_id,exc)
            failures.append(str(getattr(rec,"section_id",rec)))
            print(f"  FAIL {getattr(rec, 'section_id', rec)}: {exc}", flush=True)
            traceback.print_exc()

    met=audit.finish()

    if args.include_incomplete_lung:
        # New unlabeled sections use the explicit deployment artifact, never the last OOF fold.
        model=load_model_for_predict(str(_REPO_ROOT),"PanCancer","project_all_UNI",checkpoint_path=str(ckpt),device=device,hidden_dims=None,parent_dir=True)
        scaler=checkpoint_scaler(ckpt)
        incomplete=list(discover_lung_incomplete())
        requests=[dict(r,dataset="LungIncomplete") for r in incomplete]
        uncontext=dict(context,scope="unlabeled_deployment_not_oof")
        uaudit=EvaluationRun(args.pred_root.expanduser().resolve()/"unlabeled_deployment",requests,context=uncontext)
        for rec in incomplete:
            rec = argparse.Namespace(**rec)
            try:
                model.inference_coordinate_unit=args.coordinates_unit
                model.inference_mpp=args.inference_mpp or model.checkpoint_metadata.get("section_mpp",{}).get(str(rec.section_id))
                row = process_incomplete(
                    rec,
                    model=model,
                    scaler=scaler,
                    device=device,
                    classes=classes,
                    palettes=palettes,
                    pred_root=args.pred_root.expanduser().resolve()/"unlabeled_deployment",
                    use_spatial=not args.no_spatial_context,
                    spatial_k=args.spatial_k,
                    skip_existing=args.skip_existing,
                    skip_figures=args.skip_figures,
                    embedder=args.embedder,
                    uni2_cache_dir=uni2_cache_dir,
                )
                if row:
                    uaudit.success(rec.section_id,row)
                else:uaudit.failure(rec.section_id,"No valid external features")
            except Exception as exc:
                import traceback

                uaudit.failure(rec.section_id,exc)
                failures.append(str(rec.section_id))
                print(f"  FAIL {rec.section_id}: {exc}", flush=True)
                traceback.print_exc()

        uaudit.finish()

    # 2026.10.28, never concatenate older metrics / 当前 CSV 仅是本次成功结果
    if not met.empty:
        for ds,sub in met.groupby("dataset"):
            print(f"{ds}: current sections={len(sub)}, mean L2 AUROC={sub.macro_auc_L2_cells.mean():.3f}",flush=True)
    print(f"Done; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
