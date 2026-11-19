"""2026.11.17 synthetic checks of fold_ensemble_external (no data needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_ensemble_external as fe  # noqa: E402
import pipeline_provenance as pp  # noqa: E402

S = "slideA"
from pancancer_labels import hierarchy_maps, ontology_class_names  # noqa: E402

_C = ontology_class_names()
L1, L12, L2 = list(_C["L1"]), list(_C["L12"]), list(_C["L2"])
_M = hierarchy_maps(_C)
A, B = np.asarray(_M["l2_to_l12"]), np.asarray(_M["l12_to_l1"])


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def _member(root, k, n=200, names=None, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed + k)
    d = root / f"pred_uni2_fold{k}_run" / S
    d.mkdir(parents=True)
    names = names if names is not None else np.array([f"c{i}" for i in range(n)])
    p2 = _softmax(rng.normal(0, 1, (n, len(L2)))).astype(np.float32)
    p12 = _softmax(rng.normal(0, 1, (n, len(L12)))).astype(np.float32)
    p1 = _softmax(rng.normal(0, 1, (n, len(L1)))).astype(np.float32)
    obs = pd.DataFrame(index=names)
    obs["pred_CT"] = np.array(L2)[p2.argmax(1)]
    a = ad.AnnData(obs=obs); a.obsm["spatial_HE"] = rng.uniform(0, 100, (n, 2))
    a.uns["checkpoint"] = f"fold{k}.pt"; a.uns["stardist_prob_min"] = 0.0
    a.write_h5ad(d / f"{S}_stardist_pred.h5ad")
    np.savez_compressed(d / f"{S}_l2_probs.npz", probs=p2, class_names=np.array(L2), obs_names=names.astype(str),
                        l1_probs=p1, l1_class_names=np.array(L1), l12_probs=p12, l12_class_names=np.array(L12),
                        checkpoint_family=np.array("fam"))
    return d, (p1, p12, p2)


def test_ensemble_mean_and_report(tmp_path):
    import anndata as ad
    members = [_member(tmp_path, k) for k in range(5)]
    out = tmp_path / "pred_uni2_ens5_run" / S
    rep = fe.ensemble(S, [m for m, _ in members], out)
    mean = np.mean([p[2] for _, p in members], axis=0)
    with np.load(out / f"{S}_l2_probs.npz") as z:
        assert np.allclose(z["probs"], mean, atol=1e-6)
        assert list(z["class_names"]) == L2
    pred = ad.read_h5ad(out / f"{S}_stardist_pred.h5ad")
    assert (pred.obs["pred_CT"].to_numpy() == np.array(L2)[mean.argmax(1)]).all()
    assert pred.obs["ensemble_agree_L2"].between(0, 1).all()
    assert len(rep) == 6 and rep.iloc[-1]["member"].startswith("ENSEMBLE")
    assert np.isclose(rep[[f"frac_{c}" for c in L2]].sum(axis=1), 1).all()
    # 2026.11.18, default l2_parent: L12 / L1 are the ontology parents of pred_CT
    y2 = mean.argmax(1)
    assert (pred.obs["pred_sublineage"].to_numpy() == np.array(L12)[A[y2]]).all()
    assert (pred.obs["pred_lineage"].to_numpy() == np.array(L1)[B[A[y2]]]).all()
    assert "pred_sublineage_head" in pred.obs and "pred_lineage_head" in pred.obs


def test_ensemble_rejects_different_nuclei(tmp_path):
    a, _ = _member(tmp_path, 0)
    b, _ = _member(tmp_path, 1, names=np.array([f"x{i}" for i in range(200)]))
    with pytest.raises(ValueError):
        fe.ensemble(S, [a, b], tmp_path / "out" / S)


def test_seed_features_links_and_marker(tmp_path):
    src = tmp_path / "m0" / S; src.mkdir(parents=True)
    feat = src / f"{S}_stardist_features.h5ad"; feat.write_bytes(b"features")
    pp.publish(src / f"{S}_stardist_features.source.json", "STAMP", [feat], n_cells=1)
    dst = tmp_path / "m1" / S
    assert fe.seed_features(S, [src, dst]) == [str(dst / f"{S}_stardist_features.h5ad")]
    assert pp.complete(dst / f"{S}_stardist_features.source.json", "STAMP", [dst / f"{S}_stardist_features.h5ad"])
    assert fe.seed_features(S, [src, dst]) == []            # already seeded
    feat.write_bytes(b"changed")                             # source no longer matches its marker
    with pytest.raises(RuntimeError):
        fe.seed_features(S, [src, tmp_path / "m2" / S])


def test_consistency_modes_and_check(tmp_path):
    import anndata as ad
    members = [_member(tmp_path, k) for k in range(3)]
    dirs = [m for m, _ in members]
    chk = fe.check(S, dirs)
    assert len(chk) == 3 and (chk["L12_head_vs_L2_parent"] > 0).all()   # random heads disagree
    for mode in ("fused", "independent"):
        out = tmp_path / f"ens_{mode}" / S
        fe.ensemble(S, dirs, out, consistency=mode)
        obs = ad.read_h5ad(out / f"{S}_stardist_pred.h5ad").obs
        parent_ok = (obs["pred_sublineage"].to_numpy() == np.array(L12)[A[[L2.index(c) for c in obs["pred_CT"]]]])
        assert parent_ok.all() if mode == "fused" else not parent_ok.all()
    # fused with zero head weights == l2_parent
    p1 = np.mean([p[0] for _, p in members], 0); p12 = np.mean([p[1] for _, p in members], 0)
    p2 = np.mean([p[2] for _, p in members], 0)
    a = fe.consistent_labels(p1, p12, p2, A, B, mode="fused", w12=0, w1=0)
    b = fe.consistent_labels(p1, p12, p2, A, B, mode="l2_parent")
    assert all((x == y).all() for x, y in zip(a, b))


def test_rebuild_archives_derived_files(tmp_path):
    members = [_member(tmp_path, k) for k in range(2)]
    dirs = [m for m, _ in members]
    out = tmp_path / "pred_uni2_ens2_run" / S
    fe.ensemble(S, dirs, out)
    (out / f"{S}_niche_indices.h5ad").write_text("old")
    (out / f"{S}_confidence.npz").write_text("old")
    (out.parent / "figures").mkdir()
    (out.parent / "figures" / "a.pdf").write_text("old")
    fe.ensemble(S, dirs, out, consistency="fused")
    left = sorted(f.name for f in out.iterdir())
    assert left == sorted([f"{S}_stardist_pred.h5ad", f"{S}_l2_probs.npz", f"{S}_ensemble_report.csv"])
    prev = [d for d in out.parent.iterdir() if d.name.startswith("_previous_")]
    assert len(prev) == 1
    assert (prev[0] / S / f"{S}_niche_indices.h5ad").is_file() and (prev[0] / "figures" / "a.pdf").is_file()
