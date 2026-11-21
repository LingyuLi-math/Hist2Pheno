#!/usr/bin/env python3
"""Five-fold mean of the five 2026-11-20 arms on NPC patient1.

tile_local reads the centroid token. tile_local_global and tile_local_global_full
read that token plus the 112 µm class token (full keeps all 1536 of the class
token). tile_local_global_k8 uses the same 3072-d matrix with eight neighbours.
l12_links_1_mpp05 reads the existing 14 px UNI2 tokens. An arm whose ensemble
h5ad is already on disk is skipped.

GPU: set CUDA_VISIBLE_DEVICES to a free card. Do not use a card that is training.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SAMPLE = "20210809-C-AH4199551"
NPC = REPO / "data" / "Visium" / "NPC"
FEAT = (
    NPC / "pred_uni2_fold0_l12_links_1_ivsafe_seed42" / SAMPLE
    / f"{SAMPLE}_stardist_features.h5ad"
)
EMB = (
    NPC / "project_all_UNI2" / SAMPLE / "ImgEmbeddings_all_stardist" / "sc_pth_14_14"
)
PAN = REPO / "data" / "PanCancer"
ARMS = (
    {
        "name": "tile_local",
        "kind": "local",
        "result": PAN / "result_uni2_ablation_tile_local_stratsec_new1066axisfix_ivsafe_paired_v2_seed42",
        "cv": "tile_local",
        "short": "tile_local_stratsec_ivsafe_seed42",
    },
    {
        "name": "tile_local_global",
        "kind": "local_global",
        "result": PAN / "result_uni2_ablation_tile_local_global_stratsec_new1066axisfix_ivsafe_paired_v2_seed42",
        "cv": "tile_local_global",
        "short": "tile_local_global_stratsec_ivsafe_seed42",
    },
    {
        "name": "tile_local_global_full",
        "kind": "local_global",
        "result": PAN / "result_uni2_ablation_tile_local_global_full_stratsec_new1066axisfix_ivsafe_paired_v2_seed42",
        "cv": "tile_local_global_full",
        "short": "tile_local_global_full_stratsec_ivsafe_seed42",
    },
    {
        "name": "tile_local_global_k8",
        "kind": "local_global",
        "result": PAN / "result_uni2_ablation_tile_local_global_k8_stratsec_new1066axisfix_ivsafe_paired_v2_seed42",
        "cv": "tile_local_global_k8",
        "short": "tile_local_global_k8_stratsec_ivsafe_seed42",
    },
    {
        "name": "l12_links_1_mpp05",
        "kind": "uni2",
        "result": PAN / "result_uni2_ablation_l12_links_1_mpp05_stratsec_new1066axisfix_ivsafe_paired_v2_seed42",
        "cv": "l12_links_1_mpp05",
        "short": "l12_links_1_mpp05_stratsec_ivsafe_seed42",
    },
)


def _paths():
    sys.path[:0] = [
        str(REPO / "code"),
        str(REPO / "code" / "Hist2Pheno_pkg"),
        str(REPO / "code" / "External_val"),
    ]


def tile_matrix(local: np.ndarray, global_: np.ndarray, kind: str) -> np.ndarray:
    if kind == "local":
        return np.asarray(local, dtype=np.float32)
    if kind == "local_global":
        return np.hstack([np.asarray(local, dtype=np.float32), np.asarray(global_, dtype=np.float32)])
    raise ValueError(kind)


def ensure_tiles(xy_h5ad: Path, mpp: float, device) -> tuple[np.ndarray, np.ndarray]:
    import anndata as ad
    import pipeline_provenance as pp
    import tile_context as tc

    stub_dir = NPC / "tile_external" / SAMPLE
    stub = stub_dir / f"{SAMPLE}.h5ad"
    src = ad.read_h5ad(xy_h5ad, backed="r")
    try:
        xy = np.asarray(src.obsm["spatial_HE"], dtype=np.float64)
        names = np.asarray(src.obs_names.astype(str))
    finally:
        src.file.close()
    root = NPC / "tile_context"
    out = tc.section_dir(root, stub)
    if not (out / "local.npy").is_file() or not (out / "global.npy").is_file():
        import torch
        stub_dir.mkdir(parents=True, exist_ok=True)
        kept = ad.AnnData(np.zeros((len(xy), 1), np.float32))
        kept.obs_names = names
        kept.obsm["spatial_HE"] = xy
        kept.uns["embedding_dir"] = str(EMB)
        kept.write_h5ad(stub)
        enc = pp.file_record(tc.uni2_weight_file(), root / ".provenance")["sha256"]
        model = tc.load_uni2(device)
        print(f"extracting tiles for {SAMPLE} on {device}", flush=True)
        tc.extract_section(stub, mpp, root, model, device, batch_size=16, amp="bf16", encoder_sha=enc)
        del model
        torch.cuda.empty_cache()
    local = np.load(out / "local.npy", mmap_mode="r")
    glob = np.load(out / "global.npy", mmap_mode="r")
    if len(local) != len(names):
        raise RuntimeError(f"tile rows {len(local)} != nuclei {len(names)}")
    return local, glob


def predict_fold(X, xy, names, obs, ckpt: Path, dest: Path, mpp: float, device) -> None:
    import anndata as ad
    import torch
    from checkpoint_artifact import checkpoint_scaler
    from model import load_model_for_predict
    from plot import mlp_collect_five_head_softmax_probs
    from pipeline_provenance import checkpoint_family, inference_coordinates

    dest.mkdir(parents=True, exist_ok=True)
    pred_path = dest / f"{SAMPLE}_stardist_pred.h5ad"
    npz_path = dest / f"{SAMPLE}_l2_probs.npz"
    if pred_path.is_file() and npz_path.is_file():
        print(f"  skip {ckpt.parent.name} fold {ckpt.stem}", flush=True)
        return
    model = load_model_for_predict(
        str(REPO), "PanCancer", "project_all_UNI2",
        checkpoint_path=str(ckpt), device=device, hidden_dims=None, parent_dir=True,
    )
    scaler = checkpoint_scaler(ckpt)
    if X.shape[1] != int(scaler.n_features_in_):
        raise ValueError(f"{ckpt.name}: features {X.shape[1]} != scaler {scaler.n_features_in_}")
    Xs = np.asarray(scaler.transform(X), dtype=np.float32)
    neighbors = None
    if getattr(model, "use_spatial_context", False):
        xy_u, cfg = inference_coordinates(model, xy, unit="pixel", mpp=float(mpp))
        cfg["cache_dir"] = str(dest / "context_cache")
        neighbors = {"xy": xy_u, "config_override": cfg}
    probs = mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=neighbors)
    classes = (model.checkpoint_metadata or {}).get("head_classes") or {}
    if not all(classes.get(k) for k in ("L1", "L12", "L2")):
        from pancancer_labels import ontology_class_names
        classes = ontology_class_names()
    family = checkpoint_family(model.checkpoint_metadata)
    y2 = np.asarray(probs["l2"]).argmax(1)
    y12 = np.asarray(probs["l12"]).argmax(1)
    y1 = np.asarray(probs["l1"]).argmax(1)
    out_obs = obs.copy()
    out_obs["pred_CT"] = np.asarray(classes["L2"])[y2]
    out_obs["pred_sublineage"] = np.asarray(classes["L12"])[y12]
    out_obs["pred_lineage"] = np.asarray(classes["L1"])[y1]
    out_obs["conf_L2"] = np.asarray(probs["l2"]).max(1)
    out_obs["conf_L12"] = np.asarray(probs["l12"]).max(1)
    out_obs["conf_L1"] = np.asarray(probs["l1"]).max(1)
    pred = ad.AnnData(obs=out_obs)
    pred.obsm["spatial_HE"] = xy
    pred.write_h5ad(pred_path)
    np.savez_compressed(
        npz_path,
        probs=np.asarray(probs["l2"], dtype=np.float32),
        class_names=np.asarray(classes["L2"], dtype=str),
        obs_names=np.asarray(names, dtype=str),
        l1_probs=np.asarray(probs["l1"], dtype=np.float32),
        l1_class_names=np.asarray(classes["L1"], dtype=str),
        l12_probs=np.asarray(probs["l12"], dtype=np.float32),
        l12_class_names=np.asarray(classes["L12"], dtype=str),
        checkpoint_family=np.asarray(family),
    )
    del model
    torch.cuda.empty_cache()
    print(f"  wrote {pred_path.parent.name}", flush=True)


def main() -> None:
    _paths()
    import anndata as ad
    import torch
    from fold_ensemble_external import ensemble

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible. Set CUDA_VISIBLE_DEVICES to a free GPU.")
    device = torch.device("cuda:0")
    header = ad.read_h5ad(FEAT, backed="r")
    try:
        xy = np.asarray(header.obsm["spatial_HE"], dtype=np.float64).copy()
        names = np.asarray(header.obs_names.astype(str)).copy()
        X_uni2 = np.asarray(header.X, dtype=np.float32)
        obs = header.obs.copy()
    finally:
        header.file.close()
    mpp = 0.4650793650793653
    print(f"{SAMPLE}: {len(names):,} nuclei, uni2 {X_uni2.shape}, device {device}", flush=True)
    local = glob = None
    if any(arm["kind"] != "uni2" for arm in ARMS):
        local, glob = ensure_tiles(FEAT, mpp, device)
        bad = ~np.isfinite(np.asarray(local[:, 0], dtype=np.float32))
        if bad.any():
            raise RuntimeError(f"{int(bad.sum())} nuclei have no tile feature")

    for arm in ARMS:
        out = NPC / f"pred_uni2_ens5_{arm['short']}" / SAMPLE
        if (out / f"{SAMPLE}_stardist_pred.h5ad").is_file() and (out / f"{SAMPLE}_l2_probs.npz").is_file():
            print(f"skip {arm['name']}: ensemble already written", flush=True)
            continue
        X = X_uni2 if arm["kind"] == "uni2" else tile_matrix(local, glob, arm["kind"])
        members = []
        for fold in range(5):
            ckpt = arm["result"] / "cross_dataset_cv" / arm["cv"] / f"hce_group_fold_{fold}.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            dest = NPC / f"pred_uni2_fold{fold}_{arm['short']}" / SAMPLE
            print(f"{arm['name']} fold {fold}", flush=True)
            predict_fold(X, xy, names, obs, ckpt, dest, mpp, device)
            members.append(dest)
        print(f"ensemble -> {out.parent.name}", flush=True)
        ensemble(SAMPLE, members, out, consistency="l2_parent")
    print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("STAR_PROB_MIN", "0")
    main()
