# 2026.11.16, batch-level loader must give the same batches as the old DataLoader
import os
import numpy as np
import torch


def _result(n=1000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    nv = n // 4
    return {
        "X_train_scaled": rng.normal(size=(n, d)).astype(np.float32),
        "X_test_scaled": rng.normal(size=(nv, d)).astype(np.float32),
        "y_train_encoded": rng.integers(0, 5, n), "y_test_encoded": rng.integers(0, 5, nv),
        "y_train_level1_encoded": rng.integers(0, 2, n), "y_test_level1_encoded": rng.integers(0, 2, nv),
        "y_train_level12_encoded": rng.integers(0, 3, n), "y_test_level12_encoded": rng.integers(0, 3, nv),
        "use_spatial_context": True,
        "y_encoded_f": np.zeros(n + nv, int), "y_level1_encoded_f": np.zeros(n + nv, int),
        "y_train": np.zeros(n), "y_test": np.zeros(nv), "y_train_level1": np.zeros(n), "y_test_level1": np.zeros(nv),
        "train_global_indices": np.arange(n), "val_global_indices": np.arange(n, n + nv),
    }


def _epochs(loaders, n_epochs=2):
    out = []
    for _ in range(n_epochs):
        for b in loaders["train_loader"]:
            gidx = b[-1]
            order = torch.argsort(gidx)
            out.append(tuple(t[order] for t in b))   # compare as sets of rows
        for b in loaders["val_loader"]:
            out.append(tuple(b))
    return out


def _check(sampler_flag):
    from base import loader_train_test
    kw = dict(batch_size_cpu=64, num_workers_cpu=0, seed=7, train_balance_sampler=sampler_flag,
              train_balance_sampler_mode="global_l2")
    os.environ["HIST2PHENO_ROW_BATCH"] = "0"
    old = loader_train_test(_result(), **kw)
    os.environ["HIST2PHENO_ROW_BATCH"] = "1"
    new = loader_train_test(_result(), **kw)
    assert len(old["train_loader"]) == len(new["train_loader"])
    assert old["train_loader"].batch_size == new["train_loader"].batch_size
    assert type(old["train_loader"].sampler).__name__ == type(new["train_loader"].sampler).__name__
    assert len(new["val_loader"].dataset) == len(old["val_loader"].dataset)
    a, b = _epochs(old), _epochs(new)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            assert torch.equal(u, v)


def test_row_batches_match_shuffle():
    _check(False)


def test_row_batches_match_weighted_sampler():
    _check(True)


def test_float16_store_is_lazy_and_cast_per_batch(monkeypatch):
    from base import loader_train_test
    monkeypatch.setenv("HIST2PHENO_ROW_BATCH", "1")
    r = _result()
    r["X_train_scaled"] = r["X_train_scaled"].astype(np.float16)
    r["X_test_scaled"] = r["X_test_scaled"].astype(np.float16)
    L = loader_train_test(r, batch_size_cpu=64, num_workers_cpu=0, seed=1)
    assert L["train_dataset"].tensors[0].dtype == torch.float16      # no float32 copy
    b = next(iter(L["train_loader"]))
    assert b[0].dtype == torch.float32 and b[0].shape == (64, 6)
    v = next(iter(L["val_loader_eval"]))
    assert v[0].dtype == torch.float32


def test_scaled_split_store_dtype(monkeypatch, tmp_path):
    from model import _scale_split_float32
    monkeypatch.setenv("HIST2PHENO_MEMMAP_THRESHOLD_MB", "0")
    monkeypatch.setenv("HIST2PHENO_ARRAY_CACHE", str(tmp_path))
    X = np.random.default_rng(0).normal(size=(300, 5)).astype(np.float32)
    tr, va = np.arange(200), np.arange(200, 300)
    monkeypatch.delenv("HIST2PHENO_STORE_DTYPE", raising=False)
    _, a32, _, _ = _scale_split_float32(X, tr, va)
    monkeypatch.setenv("HIST2PHENO_STORE_DTYPE", "float16")
    _, a16, b16, _ = _scale_split_float32(X, tr, va)
    assert a32.dtype == np.float32 and a16.dtype == np.float16 and b16.dtype == np.float16
    assert np.abs(a16.astype(np.float32) - a32).max() < 5e-3


def test_shared_pooled_features(tmp_path):
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[1] / "PanCancer"))
    from PanCancer_train_validate_cv_UNIlabel import _share_pooled_features
    X = np.random.default_rng(0).normal(size=(500, 7)).astype(np.float32)
    cv = {"X_f": X.copy(), "sample_names": ["a", "b"], "cell_id_f": np.arange(500).astype(object),
          "groups_f": np.repeat([0, 1], 250)}
    a = _share_pooled_features(dict(cv), tmp_path, "p")["X_f"]
    b = _share_pooled_features(dict(cv), tmp_path, "p")["X_f"]
    assert isinstance(a, np.memmap) and a.filename == b.filename
    assert np.array_equal(np.asarray(a), X) and not a.flags.writeable
    c = _share_pooled_features(dict(cv, X_f=X + 1), tmp_path, "p")["X_f"]
    assert c.filename != a.filename
