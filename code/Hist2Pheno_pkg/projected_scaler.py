#!/usr/bin/env python3
"""2026.11.20 Train-fold PCA of a feature block, folded into the fold scaler.
在每一折的训练细胞上拟合 PCA（只降维某一段特征，例如 tile 的 global class token），
并把它和 StandardScaler 合成一个 scaler，随 checkpoint 一起保存。

Why: ``tile_context.py fit-pca`` fits one PCA on nuclei from all sections, so
the projection has seen the images of every outer test fold (no labels, but
not strictly fold-clean). Here the PCA is fitted on the outer-train rows of
the fold only (the same rows as the StandardScaler), so nothing of the test
fold enters the features.
原来的全局 PCA 用了所有 section（包括测试折）的图像；这里只用该折训练细胞。

Input  x = [x_keep (first ``split`` columns) | x_proj (the rest)]
Output z = ([x_keep, (x_proj - proj_mean_) @ proj_components_.T] - mean_) / scale_

``mean_ / var_ / scale_`` are those of the output columns (after projection),
so this is the StandardScaler of the reduced features, as in the global-PCA
path (PCA first, then StandardScaler). ``n_features_in_`` is the raw width,
so ``X.shape[1] == scaler.n_features_in_`` keeps its meaning for raw inputs.
Because the projection is linear, the mean of projected neighbours equals the
projection of the raw neighbour mean (used by spatial_context.runtime_batch).
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.preprocessing import StandardScaler

ENV = "HIST2PHENO_FEATURE_PROJECTION"   # "split=1536,dim=256[,sample=200000]"; unset = plain StandardScaler
STATE_KEYS = ("proj_split_", "proj_mean_", "proj_components_", "proj_explained_")


def parse_spec(text: str | None) -> dict | None:
    if not text:
        return None
    spec = {"sample": 200_000}
    for part in str(text).split(","):
        k, _, v = part.partition("=")
        if k.strip() not in ("split", "dim", "sample"):
            raise ValueError(f"{ENV}: unknown key {k!r} (expected split, dim, sample)")
        spec[k.strip()] = int(v)
    if "split" not in spec or "dim" not in spec or spec["dim"] <= 0:
        raise ValueError(f"{ENV} needs split=<n> and dim=<k>, got {text!r}")
    return spec


def make_scaler(n_features: int):
    """StandardScaler, or ProjectedScaler when HIST2PHENO_FEATURE_PROJECTION is set."""
    spec = parse_spec(os.environ.get(ENV))
    if spec is None:
        return StandardScaler()
    if n_features <= spec["split"] or n_features - spec["split"] < spec["dim"]:
        raise ValueError(f"{ENV}={os.environ.get(ENV)!r} does not fit {n_features} input features")
    return ProjectedScaler(split=spec["split"], dim=spec["dim"], sample=spec["sample"])


class ProjectedScaler(StandardScaler):
    """StandardScaler applied after a train-only PCA of columns [split:]."""

    def __init__(self, split: int = 1536, dim: int = 256, sample: int = 200_000, seed: int = 0):
        super().__init__()
        self.split, self.dim, self.sample, self.seed = int(split), int(dim), int(sample), int(seed)

    # -- fit ---------------------------------------------------------------
    def fit_rows(self, X, rows, block: int = 8192):
        """Fit on X[rows] (X may be a memmap; rows are read in blocks)."""
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        if len(rows) < 2:
            raise ValueError("ProjectedScaler needs >= 2 training rows")
        rng = np.random.default_rng(self.seed)
        pick = rows if len(rows) <= self.sample else np.sort(rng.choice(rows, size=self.sample, replace=False))
        G = np.asarray(X[pick, self.split:], dtype=np.float64)
        mu = G.mean(0)
        G -= mu
        cov = (G.T @ G) / max(1, len(G) - 1)
        del G
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        self.proj_split_ = np.int64(self.split)
        self.proj_mean_ = mu.astype(np.float32)
        self.proj_components_ = np.ascontiguousarray(v[:, :self.dim].T, dtype=np.float32)
        self.proj_explained_ = np.float64(max(w[:self.dim].sum(), 0.0) / max(w.sum(), 1e-12))
        inner = StandardScaler()
        for start in range(0, len(rows), block):
            inner.partial_fit(self.project(X[rows[start:start + block]]))
        for k in ("mean_", "var_", "scale_", "n_samples_seen_"):
            setattr(self, k, getattr(inner, k))
        self.n_features_in_ = int(X.shape[1])
        return self

    def fit(self, X, y=None, sample_weight=None):
        return self.fit_rows(X, np.arange(len(X)))

    def partial_fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError("ProjectedScaler is fitted in one call (fit / fit_rows), not partial_fit")

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X).transform(X)

    # -- transform ---------------------------------------------------------
    @property
    def n_features_out_(self) -> int:
        return int(self.proj_split_) + int(self.proj_components_.shape[0])

    def project(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != int(self.proj_split_) + self.proj_components_.shape[1]:
            raise ValueError(f"ProjectedScaler expects {int(self.proj_split_) + self.proj_components_.shape[1]} "
                             f"raw features, got {X.shape}")
        s = int(self.proj_split_)
        return np.hstack([X[:, :s], (X[:, s:] - self.proj_mean_) @ self.proj_components_.T])

    def transform(self, X, copy=None):
        return ((self.project(X) - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, X, copy=None):
        """Back to the projected (unscaled) space, not to the raw features (PCA is not invertible)."""
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_

    def scale_any(self, c: np.ndarray) -> np.ndarray:
        """Scale raw rows (width n_features_in_) or projected rows (width n_features_out_); any leading dims."""
        c = np.asarray(c, dtype=np.float32)
        lead = c.shape[:-1]
        flat = c.reshape(-1, c.shape[-1])
        if flat.shape[1] == self.n_features_in_:
            out = self.transform(flat)
        elif flat.shape[1] == self.n_features_out_:
            out = ((flat - self.mean_) / self.scale_).astype(np.float32)
        else:
            raise ValueError(f"width {flat.shape[1]} is neither raw {self.n_features_in_} nor projected "
                             f"{self.n_features_out_}")
        return out.reshape(*lead, out.shape[-1])


def scale_rows(scaler, c: np.ndarray) -> np.ndarray:
    """Affine StandardScaler on any leading dims; ProjectedScaler handles raw or projected widths."""
    if getattr(scaler, "proj_components_", None) is not None:
        return scaler.scale_any(c)
    return (np.asarray(c, dtype=np.float32) - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32)


def restore(state: dict):
    s = ProjectedScaler(split=int(state["proj_split_"]), dim=int(np.asarray(state["proj_components_"]).shape[0]))
    for k, v in state.items():
        setattr(s, k, v)
    return s
