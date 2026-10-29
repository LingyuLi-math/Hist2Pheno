#!/usr/bin/env python3
"""Pan-cancer Hist2Pheno: 3-head (L2/L12/L1) model on 7 public cohorts.

Labels come from ``PanCancerCellType.xlsx`` sheet ``celltype_pancancer``.

Default CV is 5-fold **cross-dataset section** holdout (each HE section is
wholly train or val). Pass ``--group-splitter cell_stratified`` to mix cells
across sections with StratifiedKFold on L1|L12|L2; those runs write a separate
folder ``data/PanCancer/result_cell_stratified_spatial/`` so they do not
overwrite the section-CV checkpoint.

  conda activate SeededNTM
  python -u code/PanCancer/prepare_pancancer_index.py
  python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py --steps index
  python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \\
      --datasets BRCA GBM --max-cells-per-section 5000 --steps prepare train he_validate
  python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \\
      --use-spatial-context --spatial-k 8 --steps prepare train he_validate
  python -u code/PanCancer/PanCancer_train_validate_cv_UNIlabel.py \\
      --group-splitter cell_stratified --use-spatial-context --spatial-k 8 \\
      --steps prepare train he_validate
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
for _p in (_PKG_DIR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from base import (  # noqa: E402
    adata_X_to_dense,
    encode_labels_with_class_names,
    evaluate,
    infer_pth_prefix_from_dir,
    load_hist_embeddings,
    match_embeddings,
    prepare_data_leave_one_group_out,
)
from checkpoint_artifact import copy_checkpoint, digest, file_digest, read_artifact
from spatial_context import SpatialConfig
from hierarchy_loss import HierarchyConfig
from model import run_group_kfold_cv_with_oof_report  # noqa: E402
from uni_label_cv_helpers import (  # noqa: E402
    plot_he_validate_level2_minimal,
    save_pooled_internal_validation_metrics,
)
from plot import plot_confusion_matrix, plot_multiclass_roc_curves  # noqa: E402

from pancancer_cv import (  # noqa: E402
    CELL_STRAT_ABLATION_TAG,
    CELL_STRAT_RESULT_DIR,
    assign_cell_stratified_folds,
    assign_section_folds,
    cache_path_for,
    fold_plan_frame,
    is_cell_stratified_splitter,
    load_existing_folds,
    ok_mask,
    summarize_cell_stratified_folds,
)
from pancancer_labels import (  # noqa: E402
    hierarchy_l2_to_l1,
    hierarchy_maps,
    map_native_series,
    ontology_class_names,
)
from pancancer_paths import (  # noqa: E402
    COHORT_ORDER,
    DEFAULT_DATA_ROOT,
    discover_h5ads,
)

PAN_ORGAN = None  # mixed 7-cohort L2 names; do not reuse a single-organ palette
DEFAULT_ABLATION_TAG = "D_emph_L2_spatial_pancancer"
DEFAULT_SAVE_RESULT = "result_all_spatial"
DEFAULT_CV_METRIC = "three_tier_auc_sum"
DEFAULT_SPATIAL_K = 8
DEFAULT_SPATIAL_MODE = "mean"
DEFAULT_TRAIN_BATCH_SIZE = 4096
DEFAULT_HIDDEN = (768, 384, 192)


def _train_loader_kwargs(
    seed: int,
    train_batch_size: int,
    *,
    train_balance_sampler: bool = False,
    train_balance_sampler_mode: str = "hierarchical_l1_l12_l2",
) -> dict:
    return {
        "seed": seed,
        "train_balance_sampler": bool(train_balance_sampler),
        "train_balance_sampler_mode": train_balance_sampler_mode,
        "num_workers_cuda": 0,
        "batch_size_cuda": train_batch_size,
        "pin_memory_cuda": False,
    }


def resolve_cell_stratified_outputs(
    group_splitter: str,
    save_result: str,
    ablation_tag: str,
) -> tuple[str, str]:
    """Keep cell-stratified checkpoints out of ``result_all_spatial``."""
    if not is_cell_stratified_splitter(group_splitter):
        return save_result, ablation_tag
    if save_result == DEFAULT_SAVE_RESULT:
        print(
            f"  cell_stratified: --save-result {DEFAULT_SAVE_RESULT!r} "
            f"→ {CELL_STRAT_RESULT_DIR!r} (do not overwrite section CV)",
            flush=True,
        )
        save_result = CELL_STRAT_RESULT_DIR
    if ablation_tag == DEFAULT_ABLATION_TAG:
        print(
            f"  cell_stratified: --ablation-tag {DEFAULT_ABLATION_TAG!r} "
            f"→ {CELL_STRAT_ABLATION_TAG!r}",
            flush=True,
        )
        ablation_tag = CELL_STRAT_ABLATION_TAG
    return save_result, ablation_tag


def setup_cuda(cuda_device: str) -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def setup_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda:0")
    if allow_cpu:
        print("WARNING: CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    raise RuntimeError("CUDA unavailable. Pass --allow-cpu-train to proceed on CPU.")


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # 2026.10.26, fail on unsupported nondeterministic kernels


def _uni2_embedding_dir(h5ad_path: Path, uns) -> Path:
    """Sibling of the UNI directory recorded on the matched h5ad.

    Lung stores ``{sample}_project_all_UNI``. The other cohorts store
    ``project_all_UNI``. UNI2 patch_size=14 writes ``sc_pth_14_14``.
    """
    recorded = str(uns.get("embedding_dir", "") or "")
    candidates: list[Path] = []
    if recorded:
        swapped = recorded
        if "project_all_UNI2" not in swapped:
            swapped = swapped.replace("project_all_UNI", "project_all_UNI2")
        swapped = swapped.replace("sc_pth_16_16", "sc_pth_14_14")
        candidates.append(Path(swapped))
    parent = h5ad_path.parent
    sample = parent.name
    rel = Path("ImgEmbeddings_all") / "sc_pth_14_14"
    candidates.append(parent / f"{sample}_project_all_UNI2" / rel)
    candidates.append(parent / "project_all_UNI2" / rel)
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if __import__("embedding_store").available(cand):
            return cand
    raise FileNotFoundError(
        f"No UNI2 sc_pth_14_14 next to {h5ad_path}. Looked at: "
        + ", ".join(str(c) for c in candidates)
    )


# 2026.10.26, lock whole cache transaction / 防止首次并发生成相互覆盖
from feature_integrity import feature_manifest, finite_rows, file_lock, atomic_json


def _aligned_uni2_matrix(h5ad_path: Path, adata, cache_dir: Path) -> np.ndarray:
    cache_dir = Path(cache_dir)
    key = digest(str(Path(h5ad_path).resolve()))
    with file_lock(cache_dir / f"{key}.lock"):
        return _aligned_uni2_matrix_locked(Path(h5ad_path), adata, cache_dir)


def _aligned_uni2_matrix_locked(h5ad_path: Path, adata, cache_dir: Path) -> np.ndarray:
    """H5ad-row-aligned UNI2 matrix. Unmatched rows are NaN. Cached on disk."""
    import json

    uni2_dir = _uni2_embedding_dir(h5ad_path, adata.uns)
    stat = h5ad_path.stat()
    uni2_stat = uni2_dir.stat()
    stamp = {
        "h5ad": str(h5ad_path.resolve()),
        "h5ad_size": int(stat.st_size),
        "h5ad_mtime_ns": int(stat.st_mtime_ns),
        "n_obs": int(adata.n_obs),
        "uni2_dir": str(uni2_dir.resolve()),
        # Changes while an extract is still writing .pth files, so a mid-run cache is not reused.
        "uni2_dir_mtime_ns": int(uni2_stat.st_mtime_ns),
        "feature_manifest": feature_manifest(uni2_dir, cache_dir),
        "alignment_version": 3,
        "alignment_implementation":digest(file_digest(Path(__file__)),file_digest(_PKG_DIR/"base.py"),file_digest(_PKG_DIR/"pipeline_provenance.py")),
        "pth_prefix": str(adata.uns.get("pth_prefix") or ""),
        "match_tolerance": float(adata.uns.get("match_tolerance", 1.0)),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = digest(stamp["h5ad"])
    npy_path = cache_dir / f"{key}.npy"
    stamp_path = cache_dir / f"{key}.json"
    if npy_path.is_file() and stamp_path.is_file():
        try:
            old = json.loads(stamp_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if old == stamp:
            cached = np.load(npy_path, mmap_mode="r")
            print(
                f"  UNI2 cache {h5ad_path.parent.name}: {cached.shape}",
                flush=True,
            )
            if cached.ndim == 2 and cached.shape[0] == adata.n_obs:
                return cached
            raise RuntimeError(f"Malformed aligned UNI2 cache: {npy_path}")
    prefix = str(adata.uns.get("pth_prefix") or "") or infer_pth_prefix_from_dir(uni2_dir)
    if not prefix:
        raise RuntimeError(f"Cannot infer UNI2 .pth prefix in {uni2_dir}")
    tolerance = float(adata.uns.get("match_tolerance", 1.0))
    xy = _spatial_xy(adata)
    cells = pd.DataFrame(
        {
            "centroid_x": xy[:, 0],
            "centroid_y": xy[:, 1],
            "celltype": "cell",
        }
    )
    print(f"  UNI2 match {h5ad_path.parent.name} ← {uni2_dir}", flush=True)
    embeddings = load_hist_embeddings(
        uni2_dir,
        celltype_df=cells,
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=tolerance,
        pth_prefix=prefix,
        strict=True,
    )
    matched = match_embeddings(
        cells,
        embeddings,
        tolerance=tolerance,
        coord_cols=("centroid_x", "centroid_y"),
    )
    del embeddings
    if not matched:
        raise RuntimeError(f"UNI2 matched 0 cells for {h5ad_path}")
    dim = int(np.asarray(matched[0]["embedding"]).reshape(-1).shape[0])
    aligned = np.full((adata.n_obs, dim), np.nan, dtype=np.float32)
    for item in matched:
        aligned[int(item["idx"])] = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
    del matched
    tmp = cache_dir / f"{key}.tmp.npy"
    # Ensure extraction was not changed while matching / 不发布混合版本缓存
    if feature_manifest(uni2_dir, cache_dir) != stamp["feature_manifest"]:
        raise RuntimeError("UNI2 tokens changed during alignment; finish extraction and retry")
    np.save(tmp, aligned)
    os.replace(tmp, npy_path)
    atomic_json(stamp_path, stamp)
    n_hit = int(finite_rows(aligned).sum())
    print(
        f"  UNI2 aligned {n_hit:,}/{adata.n_obs:,} rows, dim={dim}",
        flush=True,
    )
    return aligned


def _spatial_xy(adata) -> np.ndarray:
    if "spatial_HE" in adata.obsm:
        return np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=np.float64)
    raise KeyError("h5ad missing obsm['spatial_HE'] / obsm['spatial']")


def _native_series(obs: pd.DataFrame, native_col: str) -> pd.Series:
    if native_col in obs.columns:
        return obs[native_col]
    if "celltype" in obs.columns:
        return obs["celltype"]
    raise KeyError(f"no native column {native_col!r} in {list(obs.columns)}")


def _write_label_cache(cache_path: Path, *, keep_idx, y, y_l1, y_l12, xy, n_raw: int, version_stamp="") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd,tmp=tempfile.mkstemp(prefix=cache_path.name,suffix=".npz",dir=cache_path.parent);os.close(fd)
    np.savez_compressed(
        tmp,
        keep_idx=np.asarray(keep_idx, dtype=np.int64),
        y=np.asarray(y, dtype=object),
        y_l1=np.asarray(y_l1, dtype=object),
        y_l12=np.asarray(y_l12, dtype=object),
        xy=np.asarray(xy, dtype=np.float64),
        n_raw=np.int64(n_raw),
        version_stamp=np.asarray(version_stamp),
    )
    os.replace(tmp,cache_path)


def load_section_arrays(
    h5ad_path: Path,
    *,
    native_col: str = "final_CT",
    max_cells: int | None = None,
    rng: np.random.Generator | None = None,
    cache_path: Path | None = None,
    embedder: str = "uni",
    uni2_cache_dir: Path | None = None,
) -> dict | None:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    try:
        n_raw = int(adata.n_obs)
        # 2026.10.24, invalidation includes features/obs order/ontology / 防止陈旧标签
        import pancancer_labels as _labels
        ontology_file = _PKG_DIR / "dataset" / "PanCancerCellType.xlsx"
        stamp = digest(file_digest(h5ad_path), file_digest(ontology_file),
            file_digest(_PKG_DIR / "dataset" / "build_pancancer_celltype_v1.py"), native_col,
            __import__("pipeline_provenance").fingerprint(__import__("pipeline_provenance").mapping_descriptor()))
        cached = None
        cache_valid = False
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=True) as check:
                    cache_valid = "version_stamp" in check and str(check["version_stamp"]) == stamp
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                cache_valid=False

        if cache_valid:
            cached = np.load(cache_path, allow_pickle=True)
            idx = np.asarray(cached["keep_idx"], dtype=np.int64)
            y = np.asarray(cached["y"], dtype=object)
            y_l1 = np.asarray(cached["y_l1"], dtype=object)
            y_l12 = np.asarray(cached["y_l12"], dtype=object)
            xy = np.asarray(cached["xy"], dtype=np.float64)
            if len(idx) == 0:
                print(f"  skip {h5ad_path.parent.name}: 0 kept cells (cache)", flush=True)
                return None
        else:
            native = _native_series(adata.obs, native_col)
            mapped = map_native_series(native)
            keep = mapped.notna().to_numpy()
            if not np.any(keep):
                print(f"  skip {h5ad_path.parent.name}: 0 kept cells", flush=True)
                return None
            idx = np.flatnonzero(keep)
            triples = mapped.to_numpy()[keep]
            y_l1 = np.asarray([t[0] for t in triples], dtype=object)
            y_l12 = np.asarray([t[1] for t in triples], dtype=object)
            y = np.asarray([t[2] for t in triples], dtype=object)
            xy = _spatial_xy(adata)[idx]
            if cache_path is not None:
                _write_label_cache(
                    cache_path, keep_idx=idx, y=y, y_l1=y_l1, y_l12=y_l12, xy=xy, n_raw=n_raw, version_stamp=stamp
                )
        X_uni2 = None
        if embedder in ("uni","uni2","virchow2"):
            if uni2_cache_dir is None:
                raise ValueError("uni2 embedder requires uni2_cache_dir")
            if embedder in ("uni","virchow2"):
                from embedding_backend import aligned
                X_uni2=aligned(h5ad_path,adata,uni2_cache_dir,embedder)
            else:
                X_uni2 = _aligned_uni2_matrix(h5ad_path, adata, uni2_cache_dir)
            present = finite_rows(X_uni2, idx)
            n_miss = int((~present).sum())
            if n_miss:
                print(
                    f"  {h5ad_path.parent.name}: drop {n_miss:,} kept cells with no UNI2 .pth",
                    flush=True,
                )
                idx, y, y_l1, y_l12, xy = (
                    idx[present], y[present], y_l1[present], y_l12[present], xy[present],
                )
            if len(idx) == 0:
                print(f"  skip {h5ad_path.parent.name}: 0 UNI2 cells", flush=True)
                return None
        if max_cells is not None and len(idx) > int(max_cells):
            rng = rng or np.random.default_rng(42)
            pick = np.sort(rng.choice(len(idx), size=int(max_cells), replace=False))
            idx, y, y_l1, y_l12, xy = idx[pick], y[pick], y_l1[pick], y_l12[pick], xy[pick]
        if X_uni2 is not None:
            X = np.array(X_uni2[idx], dtype=np.float32, copy=True)
        else:
            X = np.asarray(adata_X_to_dense(adata.X)[idx], dtype=np.float32)
        return {
            "X": X,
            "cell_ids":adata.obs_names.to_numpy()[idx],
            "y": y,
            "y_l1": y_l1,
            "y_l12": y_l12,
            "xy": xy,
            "n": int(len(y)),
        }
    finally:
        if "cached" in locals() and cached is not None:cached.close()
        del adata
        gc.collect()


def _reencode_fixed(cv: dict, classes: dict[str, list[str]]) -> dict:
    y2 = encode_labels_with_class_names(cv["y_f"], classes["L2"])
    y1 = encode_labels_with_class_names(cv["y_level1_f"], classes["L1"])
    y12 = encode_labels_with_class_names(cv["y_level12_f"], classes["L12"])
    ok = (y2 >= 0) & (y1 >= 0) & (y12 >= 0)
    if not np.all(ok):
        n_bad = int((~ok).sum())
        print(f"  drop {n_bad} cells with unknown pan-cancer codes", flush=True)
        for key in list(cv):
            val = cv[key]
            if isinstance(val, np.ndarray) and val.shape[:1] == (ok.shape[0],):
                cv[key] = val[ok]
        y2, y1, y12 = y2[ok], y1[ok], y12[ok]
    cv["y_encoded_f"] = y2
    cv["y_level1_encoded_f"] = y1
    cv["y_level12_encoded_f"] = y12
    cv["class_names"] = np.asarray(classes["L2"])
    cv["class_names_level1"] = np.asarray(classes["L1"])
    cv["class_names_level12"] = np.asarray(classes["L12"])
    cv["child_to_parent"] = hierarchy_l2_to_l1(classes)
    return cv


def prepare_pooled_cv_data(
    index: pd.DataFrame,
    classes: dict[str, list[str]],
    *,
    max_cells_per_section: int | None = None,
    seed: int = 42,
    min_l2_samples: int = 1,
    cache_dir: Path | None = None,
    cv_k: int = 5,
    group_splitter: str = "cross_dataset_section",
    fold_of: dict[str, int] | None = None,
    embedder: str = "uni",
    uni2_cache_dir: Path | None = None,
) -> dict:
    ok = index.loc[ok_mask(index)].copy()
    if ok.empty:
        raise FileNotFoundError(
            "No usable sections in the index. Run prepare_pancancer_index.py first."
        )
    rng = np.random.default_rng(seed)
    X_parts, y_parts, y1_parts, y12_parts, xy_parts, g_parts = [], [], [], [], [], []
    sample_names = []
    cell_id_parts = []
    datasets = []
    n_obs = {}
    gi = 0
    encoder_protocol = None
    for rec_i, rec in enumerate(ok.itertuples(index=False)):
        cache_path = cache_path_for(rec.section_id, cache_dir) if cache_dir is not None else None
        if embedder in ("uni","uni2","virchow2"):
            import anndata as ad
            from embedding_backend import embedding_dir, protocol
            header=ad.read_h5ad(rec.h5ad,backed="r")
            try:current_protocol=protocol(embedding_dir(Path(rec.h5ad),header.uns,embedder))
            finally:header.file.close()
            if encoder_protocol is not None and current_protocol!=encoder_protocol:
                raise ValueError("Training sections use different encoder weights/preprocessing")
            encoder_protocol=current_protocol
        arr = load_section_arrays(
            Path(rec.h5ad),
            native_col=rec.native_col,
            max_cells=max_cells_per_section,
            rng=rng,
            cache_path=cache_path,
            embedder=embedder,
            uni2_cache_dir=uni2_cache_dir,
        )
        if arr is None:
            continue
        sample_names.append(rec.section_id)
        datasets.append(rec.dataset)
        n_obs[rec.section_id] = arr["n"]
        X_parts.append(arr["X"])
        cell_id_parts.append(arr["cell_ids"])
        y_parts.append(arr["y"])
        y1_parts.append(arr["y_l1"])
        y12_parts.append(arr["y_l12"])
        xy_parts.append(arr["xy"])
        g_parts.append(np.full(arr["n"], gi, dtype=np.int64))
        gi += 1
        del arr
        if (rec_i + 1) % 25 == 0 or rec_i + 1 == len(ok):
            print(f"  loaded {gi}/{len(ok)} kept sections, {sum(p.shape[0] for p in X_parts):,} cells", flush=True)
    if not X_parts:
        raise RuntimeError("All sections had 0 kept pan-cancer cells.")
    print("  stacking arrays …", flush=True)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    y1 = np.concatenate(y1_parts)
    y12 = np.concatenate(y12_parts)
    groups = np.concatenate(g_parts)
    xy = np.vstack(xy_parts)
    del X_parts, y_parts, y1_parts, y12_parts, xy_parts, g_parts
    gc.collect()
    print(
        f"  stacked {X.shape[0]:,} × {X.shape[1]}  (~{X.nbytes / 1e9:.1f} GB embeddings)",
        flush=True,
    )
    cv = prepare_data_leave_one_group_out(
        X,
        y,
        y1,
        groups=groups,
        min_l2_samples=min_l2_samples,
        y_level12=y12,
        y_level3=None,
        y_level4=None,
        X_coords=xy,
    )
    del X, y, y1, y12, groups, xy
    gc.collect()
    cv["embedder"] = embedder
    cv["embedding_protocol"] = encoder_protocol
    cv["cell_id_f"] = np.concatenate(cell_id_parts)[cv["mask"]]
    cv = _reencode_fixed(cv, classes)
    g2s = {i: s for i, s in enumerate(sample_names)}
    cv["sample_names"] = sample_names
    cv["sample_to_group"] = {s: i for i, s in enumerate(sample_names)}
    cv["group_to_sample"] = g2s
    cv["n_obs_per_sample"] = n_obs
    cv["dataset_f"] = np.asarray([datasets[int(g)] for g in cv["groups_f"]], dtype=object)
    if is_cell_stratified_splitter(group_splitter):
        cv["fold_id_f"] = assign_cell_stratified_folds(
            cv["y_level1_f"],
            cv["y_level12_f"],
            cv["y_f"],
            n_splits=cv_k,
            seed=seed,
        )
        cv["fold_mode"] = "cell_stratified"
        print(
            "  fold assignment: cell_stratified StratifiedKFold on L1|L12|L2 "
            "(sections mixed; every fold keeps all abundant cell types)",
            flush=True,
        )
    else:
        if fold_of is None:
            fold_of = assign_section_folds(sample_names, datasets, n_splits=cv_k, seed=seed)
        else:
            missing = [s for s in sample_names if s not in fold_of]
            if missing:
                fold_of = assign_section_folds(
                    sample_names,
                    datasets,
                    n_splits=cv_k,
                    seed=seed,
                    existing=fold_of,
                )
        cv["fold_id_f"] = np.asarray(
            [fold_of[g2s[int(g)]] for g in cv["groups_f"]], dtype=np.int64
        )
        cv["fold_mode"] = "section"
    print(
        f"  Pooled: {cv['X_f'].shape[0]:,} cells, "
        f"{len(cv['class_names'])} L2 / {len(cv['class_names_level12'])} L12 / "
        f"{len(cv['class_names_level1'])} L1, "
        f"{len(np.unique(cv['groups_f']))} sections, "
        f"{len(np.unique(cv['fold_id_f']))} preassigned folds "
        f"(mode={cv['fold_mode']})",
        flush=True,
    )
    return cv


@dataclass
class PooledCtx:
    data_root: Path
    save_result: str
    ablation_tag: str
    datasets: list[str]
    device: torch.device
    seed: int
    cv_k: int
    patience: int
    max_epochs: int
    train_batch_size: int
    hidden_dims: tuple[int, ...]
    resume_from_checkpoints: bool
    use_spatial_context: bool
    spatial_k: int
    spatial_mode: str
    max_cells_per_section: int | None
    group_splitter: str = "cross_dataset_section"
    val_selection_metric: str = DEFAULT_CV_METRIC
    cv_selection_metric: str = DEFAULT_CV_METRIC
    hce_w1: float = 1.0
    hce_w2: float = 1.0
    hce_w12: float = 2.0
    hce_w_l12head: float = 1.0
    hce_w_l3: float = 0.0
    hce_w_l4: float = 0.0
    train_group_frac: float = 0.8
    train_balance_sampler: bool = False
    train_balance_sampler_mode: str = "hierarchical_l1_l12_l2"
    class_weight_mode: str = "balanced"
    hierarchy_options: dict = field(default_factory=lambda: vars(HierarchyConfig()))
    spatial_options: dict = field(default_factory=dict)
    patient_map: Path | None = None
    mpp_map: Path | None = None
    inner_val_fraction: float = 0.2
    embedder: str = "uni"
    split_seed: int = 42  # 2026.10.26, fixed splits across training seeds / 固定 cross-section 拆分
    g: dict = field(default_factory=dict)

    @property
    def result_dir(self) -> Path:
        return self.data_root / self.save_result

    @property
    def ckpt_dir(self) -> Path:
        return self.result_dir / "cross_dataset_cv" / self.ablation_tag

    def result_fig(self, stem: str) -> str:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        return str(self.result_dir / stem)


def load_or_build_index(ctx: PooledCtx, index_csv: Path | None) -> pd.DataFrame:
    path = index_csv or (ctx.data_root / "section_index.csv")
    if path.is_file():
        idx = pd.read_csv(path)
        if ctx.datasets:
            idx = idx[idx["dataset"].isin(ctx.datasets)].copy()
        print(f"  index {path}  rows={len(idx)}", flush=True)
        return idx
    print("  section_index.csv missing — discovering h5ads", flush=True)
    rows = discover_h5ads(ctx.datasets or None)
    df = pd.DataFrame(rows)
    df["ok"] = True
    return df


def step_index(ctx: PooledCtx) -> None:
    from prepare_pancancer_index import main as index_main

    argv = ["--out-dir", str(ctx.data_root), "--cv-k", str(ctx.cv_k), "--seed", str(ctx.split_seed)]
    if ctx.datasets:
        argv.extend(["--datasets", *ctx.datasets])
    index_main(argv)


def step_prepare(ctx: PooledCtx, index_csv: Path | None = None) -> None:
    __import__("pipeline_provenance").mapping_descriptor()  # 2026.10.28, refresh long-lived-kernel ontology first
    classes = ontology_class_names()
    ctx.data_root.mkdir(parents=True, exist_ok=True)
    ctx.result_dir.mkdir(parents=True, exist_ok=True)
    (ctx.data_root / "ontology_heads.json").write_text(
        json.dumps({"L1": classes["L1"], "L12": classes["L12"], "L2": classes["L2"]}, indent=2)
    )
    index = load_or_build_index(ctx, index_csv)
    plan_path = ctx.result_dir / "fold_plan.csv"
    existing = load_existing_folds(plan_path)
    plan = fold_plan_frame(
        index,
        n_splits=ctx.cv_k,
        seed=ctx.split_seed,
        existing=existing or None,
    )
    # 2026.10.24, optional patient-grouped outer folds / 患者级隔离需真实元数据
    patient_lookup = None
    if ctx.patient_map is not None:
        from sklearn.model_selection import GroupKFold
        patients = pd.read_csv(ctx.patient_map, dtype=str)
        if not {"section_id","patient_id"}.issubset(patients) or patients.section_id.duplicated().any() or patients.patient_id.isna().any():
            raise ValueError("patient map needs unique section_id and nonempty patient_id")
        patient_lookup = dict(zip(patients.section_id, patients.patient_id))
        plan["patient_id"] = plan.section_id.map(patient_lookup)
        if plan.patient_id.isna().any(): raise ValueError("patient map missing indexed sections")
        for fold,(_,va) in enumerate(GroupKFold(ctx.cv_k).split(plan, groups=plan.patient_id)):
            plan.loc[plan.index[va],"fold"] = fold
    plan.to_csv(plan_path, index=False)
    fold_of = {
        str(s): int(f)
        for s, f in zip(plan["section_id"], plan["fold"])
        if pd.notna(f)
    }
    print(f"\n[Prepare] {index['dataset'].nunique() if len(index) else 0} datasets", flush=True)
    if not plan.empty:
        print(pd.crosstab(plan["fold"], plan["dataset"]).reindex(columns=COHORT_ORDER).fillna(0).astype(int), flush=True)
    cv = prepare_pooled_cv_data(
        index,
        classes,
        max_cells_per_section=ctx.max_cells_per_section,
        seed=ctx.split_seed,
        cache_dir=ctx.data_root / "label_cache",
        cv_k=ctx.cv_k,
        group_splitter=ctx.group_splitter,
        fold_of=fold_of,
        embedder=ctx.embedder,
        uni2_cache_dir=ctx.data_root / (ctx.embedder+"_aligned"),
    )
    if is_cell_stratified_splitter(ctx.group_splitter):
        cell_plan = summarize_cell_stratified_folds(
            cv["y_level1_f"], cv["y_level12_f"], cv["y_f"], cv["fold_id_f"]
        )
        cell_plan_path = ctx.data_root / "fold_plan_cell_stratified.csv"
        cell_plan.to_csv(cell_plan_path, index=False)
        ctx.result_dir.mkdir(parents=True, exist_ok=True)
        cell_plan.to_csv(ctx.result_dir / "fold_plan_cell_stratified.csv", index=False)
        print(f"  wrote {cell_plan_path}", flush=True)
        l2 = cell_plan[cell_plan["level"] == "L2"]
        wide = l2.pivot_table(index="fold", columns="class", values="n", fill_value=0)
        print(wide.astype(int), flush=True)
    cv["hierarchy_maps"] = hierarchy_maps(classes)
    cv["hierarchy_config"] = ctx.hierarchy_options
    cv["spatial_config"] = ctx.spatial_options or vars(SpatialConfig(k=ctx.spatial_k))
    cv["inner_val_fraction"] = ctx.inner_val_fraction
    cv["seed"] = ctx.seed
    cv["split_seed"] = ctx.split_seed
    cv["paired_initialization"] = True
    if is_cell_stratified_splitter(ctx.group_splitter) and ctx.inner_val_fraction:
        raise ValueError("cell_stratified is a development ablation: pass --inner-val-fraction 0 explicitly")
    if patient_lookup is not None:
        cv["patient_f"] = np.asarray([patient_lookup[cv["group_to_sample"][int(g)]] for g in cv["groups_f"]])
    if ctx.mpp_map is not None:
        table = pd.read_csv(ctx.mpp_map)
        if not {"section_id","mpp_um"}.issubset(table) or table.section_id.duplicated().any():
            raise ValueError("mpp map needs unique section_id,mpp_um")
        lookup = dict(zip(table.section_id, table.mpp_um))
        scale = np.asarray([lookup[cv["group_to_sample"][int(g)]] for g in cv["groups_f"]],float)
        if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError("Invalid per-section mpp")
        cv["section_mpp"] = {str(k):float(v) for k,v in lookup.items()}
        cv["X_coords_f"] *= scale[:,None]
        cv["spatial_config"]["coordinates_um"] = True
        cv["spatial_config"]["mpp"] = None
    # No global scaler fit: exact fold scaler is loaded from the selected artifact.
    scaler = None
    ctx.g.update(
        {
            "cv_data": cv,
            "scaler": scaler,
            "class_names": cv["class_names"],
            "class_names_level1": cv["class_names_level1"],
            "class_names_level12": cv["class_names_level12"],
            "USE_THREE_HEAD": True,
            "USE_FIVE_HEAD": False,
            "input_dim": int(cv["X_f"].shape[1]),
        }
    )
    print(f"  input_dim={ctx.g['input_dim']}", flush=True)


def step_train(ctx: PooledCtx) -> None:
    if "cv_data" not in ctx.g:
        step_prepare(ctx)
    cv = ctx.g["cv_data"]
    ctx.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[Train] splitter={ctx.group_splitter} k={ctx.cv_k} "
        f"sections={len(np.unique(cv['groups_f']))} "
        f"tag={ctx.ablation_tag} spatial={ctx.use_spatial_context} "
        f"balance_sampler={ctx.train_balance_sampler} "
        f"({ctx.train_balance_sampler_mode if ctx.train_balance_sampler else '-'})",
        flush=True,
    )
    lp = run_group_kfold_cv_with_oof_report(
        device=ctx.device,
        cv_data=cv,
        scaler=ctx.g["scaler"],
        class_names=list(ctx.g["class_names"]),
        evaluate=evaluate,
        hce_w1=ctx.hce_w1,
        hce_w2=ctx.hce_w2,
        hce_w12=ctx.hce_w12,
        hce_w_l12head=ctx.hce_w_l12head,
        hce_w_l3=ctx.hce_w_l3,
        hce_w_l4=ctx.hce_w_l4,
        n_splits=ctx.cv_k,
        train_group_frac=ctx.train_group_frac,
        patience=ctx.patience,
        max_epochs=ctx.max_epochs,
        loader_kwargs=_train_loader_kwargs(
            ctx.seed,
            ctx.train_batch_size,
            train_balance_sampler=ctx.train_balance_sampler,
            train_balance_sampler_mode=ctx.train_balance_sampler_mode,
        ),
        resume_from_checkpoints=ctx.resume_from_checkpoints,
        group_checkpoint_dir=str(ctx.ckpt_dir),
        hidden_dims=ctx.hidden_dims,
        random_state=ctx.split_seed,
        val_selection_metric=ctx.val_selection_metric,
        cv_selection_metric=ctx.cv_selection_metric,
        class_weight_mode=ctx.class_weight_mode,
        use_spatial_context=ctx.use_spatial_context,
        spatial_k=ctx.spatial_k,
        spatial_mode=ctx.spatial_mode,
        group_splitter=ctx.group_splitter,
    )
    dest = ctx.ckpt_dir / "best_mlp_gpu.pt"
    copy_checkpoint(lp["best_fold"]["checkpoint"], dest)
    # 2026.10.24, preserve all three OOF heads and exact row identity / 真正 OOF
    summary = lp["group_summary"]
    row_ids=summary["oof_row_indices"]
    if not np.array_equal(np.sort(row_ids),np.arange(len(cv["X_f"]))):
        raise RuntimeError("Incomplete or repeated OOF coverage; inspect skipped folds")
    if cv.get("fold_mode") != "cell_stratified":
        actual={}
        for fold in summary["folds"]:
            metadata=read_artifact(fold["checkpoint"],required=True)["metadata"]
            for section in metadata["held_out_sections"]:
                if section in actual: raise RuntimeError("Repeated section holdout is not unique k-fold OOF")
                actual[section]=int(fold["fold"])
        plan=pd.read_csv(ctx.result_dir/"fold_plan.csv")
        plan["fold"]=plan.section_id.map(actual)
        if plan.fold.isna().any():raise RuntimeError("Missing actual section fold in artifact")
        plan.to_csv(ctx.result_dir/"fold_plan.csv",index=False)
        cv["fold_id_f"]=np.asarray([actual[cv["group_to_sample"][int(g)]] for g in cv["groups_f"]],dtype=int)
    arrays = {"row_indices":row_ids}
    for head, data in summary["oof_heads"].items():
        arrays[head+"_labels"] = data["labels"]
        arrays[head+"_probs"] = data["probs"]
    np.savez_compressed(ctx.result_dir/"oof_predictions.npz", **arrays)
    pd.DataFrame({"row_index":np.arange(len(cv["groups_f"])),
        "section_id":[cv["group_to_sample"][int(g)] for g in cv["groups_f"]],
        "fold":cv["fold_id_f"],"cell_id":cv["cell_id_f"],"patient_id":cv.get("patient_f",None)}).to_csv(ctx.result_dir/"oof_rows.csv",index=False)
    print(f"  Best fold={lp['best_fold']['fold']}  {dest}", flush=True)
    ctx.g.update(lp)
    ctx.g["LP"] = lp
    ctx.g["BEST_MLP_CHECKPOINT"] = str(dest)
    ctx.g["model"] = lp["model"]
    fold_rows = []
    for fr in (lp.get("group_summary") or {}).get("folds") or []:
        row = {k: v for k, v in fr.items() if not isinstance(v, np.ndarray)}
        fold_rows.append(row)
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(ctx.ckpt_dir / "group_cv_folds.csv", index=False)


def step_he_validate(ctx: PooledCtx) -> None:
    if "LP" not in ctx.g:
        raise RuntimeError("Run --steps train first.")
    g = ctx.g
    rf = ctx.result_fig
    val_labels = g.get("val_labels")
    val_preds = g.get("val_preds")
    if val_labels is None or val_preds is None:
        raise RuntimeError("Missing OOF predictions from GroupKFold.")
    macro_auc = plot_he_validate_level2_minimal(
        plot_confusion_matrix,
        plot_multiclass_roc_curves,
        rf,
        g["class_names"],
        val_labels,
        val_preds,
        val_probs_l2=g.get("val_probs_l2"),
        title_prefix=(
            "Pan-cancer cell-stratified OOF Level2"
            if is_cell_stratified_splitter(ctx.group_splitter)
            else "Pan-cancer OOF Level2"
        ),
        pan_organ=PAN_ORGAN,
    )
    n_sec = int(len(np.unique(g["cv_data"]["groups_f"])))
    save_pooled_internal_validation_metrics(
        rf("validation_internal_metrics.csv"),
        macro_auc_l2=macro_auc,
        n_samples=n_sec,
        n_cells=int(g["cv_data"]["X_f"].shape[0]),
        n_folds=ctx.cv_k,
        train_group_frac=ctx.train_group_frac,
        best_fold=g["best_fold"],
        group_summary=g.get("group_summary") or g["logo_summary"],
    )
    print(f"  OOF L2 macro AUROC={macro_auc:.4f}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--save-result", default=DEFAULT_SAVE_RESULT)
    p.add_argument("--ablation-tag", default=DEFAULT_ABLATION_TAG)
    p.add_argument("--datasets", nargs="*", default=None, help=f"Subset of {COHORT_ORDER}")
    p.add_argument("--index-csv", type=Path, default=None)
    p.add_argument(
        "--steps",
        nargs="+",
        default=["prepare", "train", "he_validate"],
        help="index | prepare | train | he_validate | all",
    )
    p.add_argument("--cv-k", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    p.add_argument("--max-cells-per-section", type=int, default=None)
    p.add_argument(
        "--group-splitter",
        default="cross_dataset_section",
        help=(
            "cross_dataset_section (default; each HE section wholly train or val) | "
            "cell_stratified (mix cells; StratifiedKFold on L1|L12|L2; "
            "writes result_cell_stratified_spatial/) | "
            "group_kfold | shuffle"
        ),
    )
    p.add_argument("--use-spatial-context", action="store_true")
    p.add_argument("--spatial-k", type=int, default=DEFAULT_SPATIAL_K)
    p.add_argument("--spatial-mode", default=DEFAULT_SPATIAL_MODE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42, help="Fixed outer/inner splits and cell subsampling; independent of training seed")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--allow-cpu-train", action="store_true")
    p.add_argument("--no-resume-from-checkpoints", action="store_true")
    p.add_argument(
        "--train-balance-sampler",
        action="store_true",
        help="WeightedRandomSampler on the train split (on by default for cell_stratified).",
    )
    p.add_argument(
        "--no-train-balance-sampler",
        action="store_true",
        help="Disable the train sampler even for cell_stratified.",
    )
    p.add_argument(
        "--train-balance-sampler-mode",
        default="hierarchical_l1_l12_l2",
        help="global_l2 | hierarchical_l1_l2 | hierarchical_l1_l12_l2 (default).",
    )
    # 2026.10.24, independent ablation switches / 每一项均可独立关闭
    p.add_argument("--class-weight-mode",choices=("none","balanced","class_balanced"),default=None)
    p.add_argument("--hierarchy", choices=("adjacent","legacy"), default="adjacent")
    for name, default in vars(HierarchyConfig()).items():
        p.add_argument("--"+name.replace("_","-"),type=float,default=default)
    p.add_argument("--legacy-l2-l1-weight",type=float,default=2.0)
    p.add_argument("--spatial-radii-um",type=float,nargs="*",default=[])
    p.add_argument("--spatial-mpp",type=float,default=None)
    p.add_argument("--spatial-coordinates-um",action="store_true")
    p.add_argument("--spatial-weighting",choices=("uniform","gaussian"),default="uniform")
    p.add_argument("--spatial-sigma-um",type=float,default=16.)
    p.add_argument("--spatial-gate",action="store_true")
    p.add_argument("--spatial-geometry",action="store_true")
    p.add_argument("--context-cache-dir",type=Path,default=Path("/tmp/hist2pheno_context"))
    p.add_argument("--context-chunk-size",type=int,default=1024)
    p.add_argument("--patient-map",type=Path,default=None)
    p.add_argument("--mpp-map",type=Path,default=None)
    p.add_argument("--inner-val-fraction",type=float,default=0.2)
    p.add_argument(
        "--embedder",
        choices=("uni", "uni2", "virchow2"),
        default="uni",
        help="uni reads h5ad.X (1024-d). uni2 rematches UNI2 (1536-d); virchow2 rematches project_all_Virchow2/sc_pth_14_14 (1280-d).",
    )
    return p.parse_args(argv)


def resolve_steps(raw: list[str]) -> set[str]:
    if "all" in raw:
        return {"index", "prepare", "train", "he_validate"}
    return set(raw)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_cuda(args.cuda_device)
    setup_seed(args.seed)
    device = setup_device(args.allow_cpu_train)
    steps = resolve_steps(args.steps)
    save_result, ablation_tag = resolve_cell_stratified_outputs(
        args.group_splitter, args.save_result, args.ablation_tag
    )
    if args.no_train_balance_sampler:
        train_balance = False
    elif args.train_balance_sampler:
        train_balance = True
    else:
        train_balance = is_cell_stratified_splitter(args.group_splitter)
    if args.spatial_mode == "attention" and (len(args.spatial_radii_um)>1 or args.spatial_geometry):
        raise ValueError("Attention supports one scale without geometry; use mean for multiscale/geometry")
    if args.group_splitter == "shuffle":
        raise ValueError("PanCancer OOF CLI requires unique folds; use cross_dataset_section or group_kfold")
    if args.patient_map is not None and args.group_splitter != "cross_dataset_section":
        raise ValueError("Patient-map requires the default cross_dataset_section preassigned splitter")
    spatial_options = vars(SpatialConfig(k=args.spatial_k, radii_um=tuple(args.spatial_radii_um),
        mpp=args.spatial_mpp, coordinates_um=args.spatial_coordinates_um or args.mpp_map is not None,
        weighting=args.spatial_weighting, sigma_um=args.spatial_sigma_um,
        gate=args.spatial_gate,geometry=args.spatial_geometry,cache_dir=str(args.context_cache_dir),chunk_size=args.context_chunk_size))
    ctx = PooledCtx(
        class_weight_mode=args.class_weight_mode or ("none" if train_balance else "balanced"),
        hierarchy_options=vars(HierarchyConfig(**{k:getattr(args,k) for k in vars(HierarchyConfig())})) if args.hierarchy=="adjacent" else {},
        spatial_options=spatial_options, patient_map=args.patient_map, mpp_map=args.mpp_map,
        inner_val_fraction=args.inner_val_fraction, embedder=args.embedder, hce_w12=args.legacy_l2_l1_weight,
        data_root=args.data_root.expanduser().resolve(),
        save_result=save_result,
        ablation_tag=ablation_tag,
        datasets=list(args.datasets) if args.datasets else [],
        device=device,
        seed=args.seed,
        split_seed=args.split_seed,
        cv_k=args.cv_k,
        patience=args.patience,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        hidden_dims=DEFAULT_HIDDEN,
        resume_from_checkpoints=not args.no_resume_from_checkpoints,
        use_spatial_context=args.use_spatial_context,
        spatial_k=args.spatial_k,
        spatial_mode=args.spatial_mode,
        max_cells_per_section=args.max_cells_per_section,
        group_splitter=args.group_splitter,
        train_balance_sampler=train_balance,
        train_balance_sampler_mode=args.train_balance_sampler_mode,
    )
    ctx.data_root.mkdir(parents=True, exist_ok=True)
    print("=" * 60, flush=True)
    print("Pan-cancer Hist2Pheno  (L2 / L12 / L1)", flush=True)
    print(f"  data_root:  {ctx.data_root}", flush=True)
    print(f"  datasets:   {ctx.datasets or COHORT_ORDER}", flush=True)
    print(f"  embedder:   {ctx.embedder}", flush=True)
    print(f"  splitter:   {ctx.group_splitter}", flush=True)
    print(f"  results:    {ctx.result_dir}", flush=True)
    print(f"  tag:        {ctx.ablation_tag}", flush=True)
    print(f"  steps:      {sorted(steps)}", flush=True)
    print(f"  device:     {ctx.device}", flush=True)
    print(
        f"  sampler:    {ctx.train_balance_sampler}"
        + (f" ({ctx.train_balance_sampler_mode})" if ctx.train_balance_sampler else ""),
        flush=True,
    )
    print("=" * 60, flush=True)
    try:
        if "index" in steps:
            step_index(ctx)
        if "prepare" in steps or "train" in steps or "he_validate" in steps:
            if "cv_data" not in ctx.g:
                step_prepare(ctx, args.index_csv)
        if "train" in steps:
            step_train(ctx)
        if "he_validate" in steps:
            if "LP" not in ctx.g:
                step_train(ctx)
            step_he_validate(ctx)
        print("\nDone.", flush=True)
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
