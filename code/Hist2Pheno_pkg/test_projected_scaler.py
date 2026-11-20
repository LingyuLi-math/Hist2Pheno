"""2026.11.20 train-fold PCA scaler: fit only on training rows, checkpoint round trip, spatial scaling."""
import numpy as np
import pytest
import torch

import projected_scaler as ps


def _data(n=3000, keep=6, raw=10, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(3, raw))
    g = rng.normal(size=(n, 3)) @ basis + 0.01 * rng.normal(size=(n, raw)) + 5
    return np.hstack([rng.normal(size=(n, keep)) * 3 + 1, g]).astype(np.float32)


def test_matches_manual_pca_then_standard_scaler():
    from sklearn.preprocessing import StandardScaler
    X = _data()
    tr = np.arange(0, 2000)
    s = ps.ProjectedScaler(split=6, dim=3).fit_rows(X, tr)
    assert s.n_features_in_ == 16 and s.n_features_out_ == 9
    C = s.proj_components_
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-5) and float(s.proj_explained_) > 0.99
    Z = s.transform(X[tr])
    assert np.allclose(Z.mean(0), 0, atol=1e-3) and np.allclose(Z.std(0), 1, atol=1e-2)
    G = X[tr, 6:].astype(np.float64)
    _, _, vt = np.linalg.svd(G - G.mean(0), full_matrices=False)
    P = (G - G.mean(0)) @ vt[:3].T
    ref = StandardScaler().fit_transform(np.hstack([X[tr, :6], P]))
    # PCA components are defined up to sign
    got = Z.copy()
    for j in range(6, 9):
        if np.dot(got[:, j], ref[:, j]) < 0:
            got[:, j] *= -1
    assert np.allclose(got, ref, atol=2e-3)


def test_only_training_rows_enter_the_fit():
    X = _data()
    tr = np.arange(0, 2000)
    a = ps.ProjectedScaler(split=6, dim=3).fit_rows(X, tr)
    Y = X.copy()
    Y[2000:] *= 100                     # outer test fold changes, fit must not
    b = ps.ProjectedScaler(split=6, dim=3).fit_rows(Y, tr)
    assert np.allclose(a.proj_components_, b.proj_components_) and np.allclose(a.mean_, b.mean_)


def test_checkpoint_round_trip():
    import checkpoint_artifact as ca
    import pipeline_provenance as pp
    X = _data()
    s = ps.ProjectedScaler(split=6, dim=3).fit_rows(X, np.arange(2000))
    state = ca.scaler_state(s)
    assert "proj_components_" in state
    r = ca.restore_scaler({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in state.items()})
    assert isinstance(r, ps.ProjectedScaler)
    assert np.allclose(r.transform(X[:50]), s.transform(X[:50]), atol=1e-6)
    assert pp.scaler_signature(r) == pp.scaler_signature(s)
    from sklearn.preprocessing import StandardScaler
    plain = StandardScaler().fit(X)
    assert not isinstance(ca.restore_scaler(ca.scaler_state(plain)), ps.ProjectedScaler)


def test_scale_rows_raw_and_projected_widths_agree():
    X = _data()
    s = ps.ProjectedScaler(split=6, dim=3).fit_rows(X, np.arange(2000))
    raw = X[:24].reshape(8, 3, 16)
    a = ps.scale_rows(s, raw)
    b = ps.scale_rows(s, s.inverse_transform(s.transform(X[:24])).reshape(8, 3, 9))
    assert a.shape == (8, 3, 9) and np.allclose(a, b, atol=1e-4)
    # neighbour mean of raw rows == mean of projected rows (linear projection)
    m_raw = ps.scale_rows(s, X[:4].mean(0, keepdims=True))
    m_proj = s.transform(X[:4]).mean(0, keepdims=True)
    assert np.allclose(m_raw, m_proj, atol=1e-4)


@pytest.mark.parametrize("threshold_mb", ["256", "0"])
def test_scale_split_uses_env(monkeypatch, tmp_path, threshold_mb):
    import model
    X = _data().astype(np.float16)
    monkeypatch.setenv("HIST2PHENO_MEMMAP_THRESHOLD_MB", threshold_mb)
    monkeypatch.setenv("HIST2PHENO_ARRAY_CACHE", str(tmp_path))
    monkeypatch.setenv(ps.ENV, "split=6,dim=3")
    sc, tr, va, all_ = model._scale_split_float32(X, np.arange(2000), np.arange(2000, 3000), build_all=True)
    assert isinstance(sc, ps.ProjectedScaler) and tr.shape == (2000, 9) and va.shape == (1000, 9)
    assert all_.shape == (3000, 9)
    assert np.allclose(np.asarray(va, np.float32), sc.transform(X[2000:]), atol=5e-3)
    monkeypatch.delenv(ps.ENV)
    sc2, tr2, _, _ = model._scale_split_float32(X, np.arange(2000), np.arange(2000, 3000))
    assert type(sc2).__name__ == "StandardScaler" and tr2.shape == (2000, 16)


def test_bad_spec():
    with pytest.raises(ValueError):
        ps.parse_spec("split=6")
    import os
    os.environ[ps.ENV] = "split=6,dim=3"
    try:
        with pytest.raises(ValueError):
            ps.make_scaler(8)
    finally:
        del os.environ[ps.ENV]
