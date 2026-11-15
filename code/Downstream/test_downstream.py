"""2026.11.15 tests for code/Downstream (synthetic slides).

    python -m pytest -q code/Downstream/test_downstream.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slide_features as sf  # noqa: E402
import slide_fusion as fu  # noqa: E402


def test_region_pooling_means_and_counts():
    feats = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], np.float32)
    xy = np.array([[1, 1], [2, 2], [200, 5], [201, 6]], float)
    F, centres, counts = sf.region_features(feats, xy, region_um=100)
    assert counts.tolist() == [2, 2]
    assert np.allclose(F[0, :2], [0.5, 0.5]) and np.allclose(F[1, :2], [0.5, 0.5])
    assert np.allclose(F[:, 2], np.log1p(2))


def test_delong_matches_sklearn_auc_and_identity_p():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s1 = y + rng.normal(0, 1, 200); s2 = y + rng.normal(0, 2, 200)
    a1, a2, p = fu.delong_paired(y, s1, s2)
    assert abs(a1 - roc_auc_score(y, s1)) < 1e-9 and abs(a2 - roc_auc_score(y, s2)) < 1e-9
    assert fu.delong_paired(y, s1, s1)[2] == 1.0


def test_cindex_perfect_and_cox_gradient():
    import torch
    t = np.array([5, 4, 3, 2, 1.0]); e = np.ones(5)
    assert fu.c_index(t, e, -t) == 1.0
    r = torch.zeros(5, requires_grad=True)
    fu.cox_nll(r, torch.tensor(t), torch.tensor(e)).backward()
    assert torch.isfinite(r.grad).all()


def _write_cohort(tmp, n=60, task="binary", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = int(rng.random() < 0.5)
        he = rng.normal(0, 1, (30, 16)).astype(np.float32)
        ph = rng.normal(0, 1, (25, 10)).astype(np.float32)
        ph[:, 0] += 1.5 * y                      # signal only in the phenotype branch
        np.savez(tmp / f"he_{i}.npz", features=he)
        np.savez(tmp / f"ph_{i}.npz", region_features=ph)
        row = dict(patient_id=f"P{i}", slide_id=f"S{i}", uni2_npz=tmp / f"he_{i}.npz", pheno_npz=tmp / f"ph_{i}.npz")
        if task == "binary":
            row["label"] = y
        else:
            row["time"] = float(rng.exponential(10 / (1 + 2 * y))); row["event"] = int(rng.random() < 0.8)
        rows.append(row)
    return pd.DataFrame(rows)


def test_cv_runs_all_settings_and_finds_pheno_signal():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        man = _write_cohort(d)
        rep = fu.run_cv(man, "binary", d / "out", epochs=8, d=32)
        assert set(rep["metrics"]) == {"he", "pheno", "both"}
        assert rep["metrics"]["pheno"] > 0.75 and rep["metrics"]["he"] < 0.7
        oof = pd.read_csv(d / "out" / "oof_predictions.csv")
        assert oof[["he", "pheno", "both"]].notna().all().all() and oof["fold"].nunique() == 5


def test_cv_survival_runs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        man = _write_cohort(d, task="survival")
        rep = fu.run_cv(man, "survival", d / "out", epochs=4, d=32)
        assert all(0 <= v <= 1 for v in rep["metrics"].values())
        assert "both_vs_he" in rep["comparisons"]
