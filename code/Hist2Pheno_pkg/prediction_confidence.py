"""Per-nucleus cell-type confidence from Hist2Pheno softmax (UTOPIA-style).

Hist2Pheno already stores an L2 probability vector per nucleus
(``{sample}_l2_probs.npz``, same role as cell2location abundances). Spatial
maps then take ``argmax``. Morphologically similar types (B vs T vs Plasma,
CD4 vs CD8) keep a low softmax, so they disappear from the argmax map even
when the mass on the parent class (L12 ``B_Plasma`` / ``T_cell``) is real.

This module does **not** retrain the MLP. It scores the existing output:

1. Softmax diagnostics on each nucleus: max probability, margin, entropy,
   and a collapse to a coarser label when the top classes sit in one
   similarity family.
2. Conformal *presence* tests in the sense of Jin et al., bioRxiv
   2026.03.01.708850 (UTOPIA, https://github.com/kaitianjin/UTOPIA).
   Pan-cancer OOF cells (``oof_predictions.npz``) are the calibration set.
   For each class ``c`` the null is "``c`` is absent". Nuclei whose predicted
   ``P(c)`` is larger than the OOF false-positive scores get a small p-value.
   Raw conformal p-values convert to a ``[0, 1]`` confidence. When a slide's
   UNI2 StarDist embeddings are supplied, confidence is instead the same
   stretch of the BH p-value computed inside each histology cluster
   (UTOPIA ``get_adj_pvals``), with the medium batch-effect tolerance.
   ``correct_predictions_by_he`` also clusters the calibration embeddings
   and zeroes a class wherever that cluster's BH p-value is above the FDR.
   Presence is a separate calibrated FPR gate vs OOF non-class cells.

NPC has no nucleus GT, so conformal calibration must come from the training
OOF, not from the NPC slide. That is the out-of-subject setting in UTOPIA.

Example::

    from prediction_confidence import (
        annotate_nuclei,
        conformal_from_oof,
        load_l2_probs,
    )

    class_names, probs = load_l2_probs(pred_dir / f"{sample}_l2_probs.npz")
    table = annotate_nuclei(probs, class_names, margin_min=0.08)
    # table["label_plot"] is argmax, or the parent family, or "low_confidence"

    conf = conformal_from_oof(
        test_probs=probs,
        class_names=class_names,
        oof_npz=repo / "data/PanCancer/result_uni2_A2_.../oof_predictions.npz",
        classes=("B_cell", "B_Plasma"),
    )
    # conf["B_cell"]["confidence"], conf["B_cell"]["discover_fdr"]
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _confidence_cache_stamp(files, **parameters):
    """Bind derived confidence to actual inputs, row order and all options."""
    h = hashlib.sha256()
    for name in [*files, __file__]:
        with open(name, 'rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                h.update(block)
        h.update(b'\0')
    h.update(json.dumps(parameters, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def _atomic_confidence_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp, **payload)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)

# Pan-cancer L2 names that histology routinely confuses at 7–8 µm.
DEFAULT_SIMILAR_FAMILIES: dict[str, tuple[str, ...]] = {
    "B_Plasma": ("B_cell", "Plasma"),
    "T_cell": ("CD4_T", "CD8_T", "Tregs"),
    "lymphoid": ("B_cell", "Plasma", "CD4_T", "CD8_T", "Tregs", "NK"),
    "myeloid": ("DC", "Macrophage", "Neutrophil"),
    "epithelial": (
        "Tumor",
        "DCIS",
        "Epithelial",
        "Alveolar",
        "Injury_epithelial",
        "Myoepithelial",
    ),
}

# L2 → L12 for the published pan-cancer heads (matches ontology_heads.json).
DEFAULT_L2_TO_L12: dict[str, str] = {
    "DCIS": "Tumor",
    "Tumor": "Tumor",
    "Alveolar": "Epithelial",
    "Epithelial": "Epithelial",
    "Injury_epithelial": "Epithelial",
    "Myoepithelial": "Epithelial",
    "CD4_T": "T_cell",
    "CD8_T": "T_cell",
    "Tregs": "T_cell",
    "NK": "T_cell",
    "B_cell": "B_Plasma",
    "Plasma": "B_Plasma",
    "DC": "Myeloid",
    "Macrophage": "Myeloid",
    "Neutrophil": "Myeloid",
    "Fibroblast": "Stromal",
    "Myofibroblast": "Stromal",
    "Stromal": "Stromal",
    "Endothelial": "Endothelial",
    "Neural": "Neural",
}

DEFAULT_L12_TO_L1: dict[str, str] = {
    "Tumor": "Epithelial",
    "Epithelial": "Epithelial",
    "T_cell": "Immune",
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Endothelial",
    "Neural": "Neural",
}

UNLABELED = "low_confidence"

# UTOPIA converts BH p-values to a display score with these defaults
# (Jin et al. 2026, UTOPIA/inference/confidence.py).
UTOPIA_P_MIN = 0.01
UTOPIA_P_MAX = 0.5
# UTOPIA/img_process/histology_clustering.py and demo/utopia_config.
UTOPIA_N_CLUSTERS = 15
UTOPIA_PCA_DIM = 80
_BATCH_QUANTILES = {
    "weak": np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
    "medium": np.array([0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.995, 0.999]),
    "strong": np.array(
        [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999]
    ),
}


def load_l2_probs(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read ``{sample}_l2_probs.npz`` → ``(class_names, probs [n, C])``."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        if "probs" not in z.files:
            raise KeyError(f"{path} has no 'probs'; keys={list(z.files)}")
        probs = np.asarray(z["probs"], dtype=np.float32)
        if "class_names" in z.files:
            names = [str(x) for x in np.asarray(z["class_names"]).tolist()]
        else:
            names = [f"class_{i}" for i in range(probs.shape[1])]
    if probs.ndim != 2 or probs.shape[1] != len(names):
        raise ValueError(f"{path}: probs {probs.shape} vs {len(names)} class names")
    return names, probs


def load_oof_head(
    oof_npz: str | Path,
    *,
    head: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """OOF labels and probabilities. ``head`` is ``l2``, ``l12``, or ``l1_head``."""
    key = {"l2": "l2", "l12": "l12", "l1": "l1_head", "l1_head": "l1_head"}[head]
    path = Path(oof_npz)
    with np.load(path, allow_pickle=False) as z:
        labels = np.asarray(z[f"{key}_labels"])
        probs = np.asarray(z[f"{key}_probs"], dtype=np.float32)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError(f"{path}: labels {labels.shape} vs probs {probs.shape}")
    return labels, probs


def softmax_entropy(probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Normalized entropy in ``[0, 1]``. 0 = one-hot, 1 = uniform."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    n_cls = p.shape[1]
    ent = -(p * np.log(p)).sum(axis=1)
    return np.asarray(ent / np.log(n_cls), dtype=np.float32)


def softmax_margin(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(top1_index, top1_prob, top1-top2)``."""
    p = np.asarray(probs, dtype=np.float32)
    order = np.argpartition(p, -2, axis=1)[:, -2:]
    a = p[np.arange(len(p)), order[:, 0]]
    b = p[np.arange(len(p)), order[:, 1]]
    top_is_b = b >= a
    top1 = np.where(top_is_b, order[:, 1], order[:, 0]).astype(np.int64)
    p1 = np.where(top_is_b, b, a)
    p2 = np.where(top_is_b, a, b)
    return top1, p1.astype(np.float32), (p1 - p2).astype(np.float32)


def aggregate_group_probs(
    probs: np.ndarray,
    class_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> tuple[list[str], np.ndarray]:
    """Sum softmax mass inside each named group (L12-style presence)."""
    index = {name: i for i, name in enumerate(class_names)}
    cols = []
    kept = []
    for group, members in groups.items():
        idx = [index[m] for m in members if m in index]
        if not idx:
            continue
        cols.append(np.asarray(probs[:, idx], dtype=np.float64).sum(axis=1))
        kept.append(group)
    if not cols:
        raise ValueError("no group members overlap class_names")
    return kept, np.stack(cols, axis=1).astype(np.float32)


def annotate_nuclei(
    probs: np.ndarray,
    class_names: Sequence[str],
    *,
    margin_min: float = 0.08,
    max_prob_min: float = 0.25,
    similar_families: Mapping[str, Sequence[str]] | None = None,
    l2_to_l12: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-nucleus labels that do not rely on a raw argmax.

    Columns
    -------
    pred_CT
        Ordinary argmax L2 name (what the maps currently draw).
    conf_max, conf_margin, conf_entropy
        Softmax peak, gap to runner-up, normalized entropy.
    family
        Similarity family that contains *both* top-1 and top-2, else empty.
    label_plot
        ``pred_CT`` when the call is sharp; the family name when the two
        leaders are similar (B vs Plasma → ``B_Plasma``); ``low_confidence``
        when even the peak is weak.
    parent_L12
        Ontology parent of ``pred_CT``.
    """
    names = [str(x) for x in class_names]
    top1, p1, margin = softmax_margin(probs)
    entropy = softmax_entropy(probs)
    families = similar_families or DEFAULT_SIMILAR_FAMILIES
    parent = l2_to_l12 or DEFAULT_L2_TO_L12
    member_to_family: dict[str, str] = {}
    for fam, members in families.items():
        for member in members:
            member_to_family.setdefault(str(member), fam)

    order = np.argpartition(np.asarray(probs, dtype=np.float32), -2, axis=1)[:, -2:]
    top2 = np.where(order[:, 0] == top1, order[:, 1], order[:, 0])

    pred = [names[i] for i in top1]
    runner = [names[i] for i in top2]
    fam = []
    plot = []
    for i, (a, b, m, peak) in enumerate(zip(pred, runner, margin, p1)):
        shared = member_to_family.get(a)
        if shared is not None and member_to_family.get(b) == shared and m < margin_min:
            fam.append(shared)
            plot.append(shared if peak >= max_prob_min else "low_confidence")
        elif peak < max_prob_min or m < margin_min:
            fam.append(shared or "")
            plot.append("low_confidence")
        else:
            fam.append(shared or "")
            plot.append(a)

    return pd.DataFrame(
        {
            "pred_CT": pred,
            "runner_up": runner,
            "conf_max": p1,
            "conf_margin": margin,
            "conf_entropy": entropy,
            "family": fam,
            "label_plot": plot,
            "parent_L12": [parent.get(a, "") for a in pred],
        }
    )


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values, clipped to ``[0, 1]``."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = n * ranked / np.arange(1, n + 1, dtype=np.float64)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def confidence_from_p(
    p_adj: np.ndarray,
    *,
    p_min: float = UTOPIA_P_MIN,
    p_max: float = UTOPIA_P_MAX,
) -> np.ndarray:
    """Map BH p-values to ``[0, 1]`` the way UTOPIA displays confidence.

    ``p_adj = p_min`` → 1, ``p_adj = p_max`` → 0, log10 stretch in between.
    ``1 - p_adj`` is also returned by callers that want the raw FDR scale.
    """
    p = np.clip(np.asarray(p_adj, dtype=np.float64), 1e-12, 1.0)
    log = np.log(p) / np.log(10.0)
    lo = np.log(p_min) / np.log(10.0)
    hi = np.log(p_max) / np.log(10.0)
    scores = (-log + hi) / (-lo + hi)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def cluster_histology_embeddings(
    embeddings: np.ndarray,
    *,
    n_clusters: int = UTOPIA_N_CLUSTERS,
    pca_dim: int = UTOPIA_PCA_DIM,
    random_state: int = 42,
) -> np.ndarray:
    """UTOPIA histology clusters: PCA to ``pca_dim``, then KMeans.

    The input is one nucleus per row (UNI2 StarDist, 1536-d). Labels are
    ``0 .. n_clusters-1`` in the row order of ``embeddings``.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be [n, d], got {x.shape}")
    if n_clusters < 2 or x.shape[0] < n_clusters:
        raise ValueError(f"need at least {n_clusters} nuclei to cluster, got {x.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain non-finite values")
    if x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)
    labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(x)
    return np.asarray(labels, dtype=np.int32)


def match_histology_clusters(
    test_embeddings: np.ndarray,
    calib_embeddings: np.ndarray,
    *,
    n_clusters: int = UTOPIA_N_CLUSTERS,
    pca_dim: int = UTOPIA_PCA_DIM,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Put test and calibration nuclei in one H&E cluster space.

    PCA and KMeans are fit on every test nucleus plus an equal-sized draw
    of calibration nuclei, then applied to the full calibration matrix.
    Cluster ids are the UTOPIA histology match: a test nucleus is scored
    against calibration nuclei of the same morphology.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    test = np.asarray(test_embeddings, dtype=np.float64)
    calib = np.asarray(calib_embeddings, dtype=np.float64)
    if test.ndim != 2 or calib.ndim != 2 or test.shape[1] != calib.shape[1]:
        raise ValueError(
            f"test {getattr(test, 'shape', None)} and calibration {getattr(calib, 'shape', None)} "
            "embeddings must share a feature dimension"
        )
    if test.shape[0] < 2 or calib.shape[0] < 2:
        raise ValueError("need at least two test and two calibration embeddings")
    if not np.isfinite(test).all() or not np.isfinite(calib).all():
        raise ValueError("embeddings contain non-finite values")
    rng = np.random.default_rng(random_state)
    n_fit = min(calib.shape[0], test.shape[0])
    take = (
        np.arange(calib.shape[0])
        if n_fit == calib.shape[0]
        else rng.choice(calib.shape[0], size=n_fit, replace=False)
    )
    fit = np.vstack([test, calib[take]])
    if n_clusters < 2 or fit.shape[0] < n_clusters:
        raise ValueError(f"need at least {n_clusters} nuclei to fit the clusters")
    if fit.shape[1] > pca_dim:
        n_components = min(pca_dim, fit.shape[0] - 1, fit.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        fit_z = pca.fit_transform(fit)
        test_z = pca.transform(test)
        calib_z = pca.transform(calib)
    else:
        test_z = test
        calib_z = calib
        fit_z = fit
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(fit_z)
    return (
        kmeans.predict(test_z).astype(np.int32),
        kmeans.predict(calib_z).astype(np.int32),
    )


def histology_tolerance(
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    test_yhat: np.ndarray,
    null: float,
    magnitude: str | float = "medium",
) -> float:
    """UTOPIA ``compute_tolerance``. Non-positive. 0 when the null is rare.

    A negative tolerance shifts calibration conformity scores down, so a
    test slide whose scores sit higher than the calibration pool is not
    called significant just for that shift.
    """
    if isinstance(magnitude, (int, float)) and not isinstance(magnitude, bool):
        return float(magnitude)
    calib_yhat = np.asarray(calib_yhat, dtype=np.float64)
    calib_y = np.asarray(calib_y, dtype=np.float64)
    test_yhat = np.asarray(test_yhat, dtype=np.float64)
    if calib_yhat.size == 0 or test_yhat.size == 0:
        return 0.0
    absent_fraction = float(np.mean(calib_y <= null))
    if absent_fraction <= 0.1:
        return 0.0
    try:
        quantiles = _BATCH_QUANTILES[str(magnitude)]
    except KeyError as exc:
        raise ValueError("magnitude must be a number or weak/medium/strong") from exc
    calib_cut = float(np.quantile(calib_yhat, absent_fraction))
    test_cut = float(np.quantile(test_yhat, absent_fraction))
    calib_low = calib_yhat[calib_yhat <= calib_cut]
    test_low = test_yhat[test_yhat <= test_cut]
    if calib_low.size == 0 or test_low.size == 0:
        return 0.0
    shift = np.quantile(test_low, quantiles) - np.quantile(calib_low, quantiles)
    return float(-np.max(np.clip(shift, 0.0, None)))


def utopia_conformal_bh(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    test_yhat: np.ndarray,
    *,
    null: float = 0.0,
    tolerance: float = 0.0,
) -> np.ndarray:
    """UTOPIA ``conformal_pvals``: clip score, left rank, BH on these rows.

    The denominator counts every calibration row, including true positives
    (their conformity is ``+inf`` and never ranks below a finite test score).
    """
    calib_y = np.asarray(calib_y, dtype=np.float64)
    calib_yhat = np.asarray(calib_yhat, dtype=np.float64)
    test_yhat = np.asarray(test_yhat, dtype=np.float64)
    calib_scores = np.full(calib_y.shape[0], np.inf, dtype=np.float64)
    absent = calib_y <= null
    calib_scores[absent] = -calib_yhat[absent]
    order = np.sort(calib_scores + float(tolerance))
    n = order.size
    if n == 0:
        return np.ones(test_yhat.shape[0], dtype=np.float64)
    rank = np.searchsorted(order, -test_yhat, side="left")
    return bh_adjust((rank + 1).astype(np.float64) / (n + 1))


def cluster_adjusted_pvalues(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    test_yhat: np.ndarray,
    test_clusters: np.ndarray,
    *,
    calib_clusters: np.ndarray | None = None,
    null: float = 0.0,
    correct_batch_effect: bool = True,
    magnitude: str | float = "medium",
    min_calib_pts: int | None = None,
) -> np.ndarray:
    """UTOPIA ``get_adj_pvals``. BH is inside one histology cluster at a time.

    Nuclei whose cluster has fewer than ``max(200, 1% of the test cluster)``
    calibration rows stay at p = 1. ``calib_clusters is None`` reuses the
    same calibration pool in every cluster: the NPC OOF cells are not nuclei
    of this slide, so they are not split by this slide's StarDist clusters.
    The batch tolerance still sees that cluster's own test scores.
    """
    calib_y = np.asarray(calib_y, dtype=np.float64)
    calib_yhat = np.asarray(calib_yhat, dtype=np.float64)
    test_yhat = np.asarray(test_yhat, dtype=np.float64)
    test_clusters = np.asarray(test_clusters)
    if test_clusters.shape[0] != test_yhat.shape[0]:
        raise ValueError(
            f"test clusters {test_clusters.shape[0]} vs test scores {test_yhat.shape[0]}"
        )
    if calib_clusters is not None:
        calib_clusters = np.asarray(calib_clusters)
        if calib_clusters.shape[0] != calib_y.shape[0]:
            raise ValueError("calibration clusters do not match calibration rows")
    adjusted = np.ones(test_yhat.shape[0], dtype=np.float64)
    for cluster in np.unique(test_clusters):
        test_mask = test_clusters == cluster
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue
        if calib_clusters is None:
            cy, chat = calib_y, calib_yhat
        else:
            keep = calib_clusters == cluster
            cy, chat = calib_y[keep], calib_yhat[keep]
        need = max(200, int(np.ceil(0.01 * n_test))) if min_calib_pts is None else int(min_calib_pts)
        if chat.size < need:
            continue
        tolerance = (
            histology_tolerance(chat, cy, test_yhat[test_mask], null, magnitude)
            if correct_batch_effect
            else 0.0
        )
        adjusted[test_mask] = utopia_conformal_bh(
            cy, chat, test_yhat[test_mask], null=null, tolerance=tolerance
        )
    return adjusted


def conformal_pvalues(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    test_yhat: np.ndarray,
    *,
    null: float = 0.0,
) -> np.ndarray:
    """Conformal p-values for H0: true abundance ``<= null`` (UTOPIA clip score).

    Calibration cells with ``y <= null`` contribute nonconformity ``-yhat``.
    A test nucleus with a large ``yhat`` gets a small p-value.
    """
    calib_y = np.asarray(calib_y, dtype=np.float64)
    calib_yhat = np.asarray(calib_yhat, dtype=np.float64)
    test_yhat = np.asarray(test_yhat, dtype=np.float64)
    scores = np.full(calib_y.shape[0], np.inf, dtype=np.float64)
    absent = calib_y <= null
    scores[absent] = -calib_yhat[absent]
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.ones(test_yhat.shape[0], dtype=np.float64)
    calib_scores = np.sort(scores[finite])
    test_scores = -test_yhat
    n = calib_scores.size
    # P(calib_score <= test_score) with +1 Laplace correction.
    rank = np.searchsorted(calib_scores, test_scores, side="right")
    return (rank + 1).astype(np.float64) / (n + 1)


def _subsample_null_scores(
    present: np.ndarray,
    scores: np.ndarray,
    *,
    max_calib: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep all true positives; downsample the null pool used for conformal p-values."""
    y = np.asarray(present, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if max_calib <= 0 or y.size <= max_calib:
        return y, scores, np.arange(y.shape[0])
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y == 0)
    n_neg = min(len(neg), max(max_calib, 1))
    if n_neg < len(neg):
        neg = rng.choice(neg, size=n_neg, replace=False)
    keep = np.concatenate([pos, neg]) if pos.size else neg
    return y[keep], scores[keep], keep


def conformal_from_oof(
    test_probs: np.ndarray,
    class_names: Sequence[str],
    oof_npz: str | Path,
    *,
    classes: Iterable[str] | None = None,
    head: str = "l2",
    oof_class_names: Sequence[str] | None = None,
    null: float = 0.0,
    max_calib: int = 200_000,
    seed: int = 42,
    groups: Mapping[str, Sequence[str]] | None = None,
    fpr: float = 0.1,
    oof_labels: np.ndarray | None = None,
    oof_probs: np.ndarray | None = None,
    test_clusters: np.ndarray | None = None,
    calib_clusters: np.ndarray | None = None,
    batch_magnitude: str | float = "medium",
    correct_batch_effect: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Calibrate each requested class (or group) on pan-cancer OOF cells.

    ``classes`` defaults to every L2 name. Pass ``("B_cell",)`` or use
    ``groups={"B_Plasma": ("B_cell", "Plasma")}`` to test coarser targets —
    UTOPIA's result that TLS-level types are more confident than CD4/B alone.

    ``present`` is a calibrated FPR gate: ``score`` at least the
    ``(1-fpr)`` quantile of OOF cells whose true label is not ``c``.
    Without ``test_clusters``, ``confidence`` is the log-stretch of the
    raw conformal p, and ``p_adj`` is whole-slide BH (a diagnostic only).
    With ``test_clusters``, ``p_adj`` and ``confidence`` follow UTOPIA:
    clip-score p-values, the medium batch tolerance, and BH inside each
    histology cluster. ``calib_clusters`` restricts that null to calibration
    nuclei in the same H&E cluster. Without it, every cluster reuses the
    full calibration pool.
    """
    loaded_from_file = oof_labels is None or oof_probs is None
    if loaded_from_file:
        labels, oof_probs = load_oof_head(oof_npz, head=head)
    else:
        labels = np.asarray(oof_labels)
        oof_probs = np.asarray(oof_probs, dtype=np.float32)
    if oof_class_names is None and loaded_from_file:
        key = {'l2': 'l2', 'l12': 'l12', 'l1': 'l1_head', 'l1_head': 'l1_head'}[head]
        with np.load(oof_npz, allow_pickle=False) as stored:
            if key + '_class_names' in stored:
                oof_class_names = stored[key + '_class_names'].astype(str).tolist()
    # Legacy archives need an explicitly verified order or the historical fallback.
    oof_names = list(oof_class_names) if oof_class_names is not None else list(class_names)
    if oof_probs.shape[1] != len(oof_names):
        raise ValueError(
            f"OOF probs have {oof_probs.shape[1]} columns, {len(oof_names)} names"
        )
    test_names = [str(x) for x in class_names]
    if test_probs.shape[1] != len(test_names):
        raise ValueError("test_probs width does not match class_names")
    if test_clusters is not None:
        test_clusters = np.asarray(test_clusters)
        if test_clusters.shape[0] != test_probs.shape[0]:
            raise ValueError(
                f"test clusters {test_clusters.shape[0]} vs nuclei {test_probs.shape[0]}"
            )
    if calib_clusters is not None:
        calib_clusters = np.asarray(calib_clusters)
        if calib_clusters.shape[0] != labels.shape[0]:
            raise ValueError(
                f"calibration clusters {calib_clusters.shape[0]} vs calibration rows {labels.shape[0]}"
            )

    targets: dict[str, Sequence[str]] = {}
    if groups:
        targets.update({str(k): tuple(v) for k, v in groups.items()})
    wanted = list(classes) if classes is not None else list(test_names)
    for name in wanted:
        targets.setdefault(str(name), (str(name),))

    oof_index = {n: i for i, n in enumerate(oof_names)}
    test_index = {n: i for i, n in enumerate(test_names)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for target, members in targets.items():
        oof_cols = [oof_index[m] for m in members if m in oof_index]
        test_cols = [test_index[m] for m in members if m in test_index]
        if not oof_cols or not test_cols:
            continue
        oof_score = oof_probs[:, oof_cols].sum(axis=1)
        test_score = np.asarray(test_probs[:, test_cols], dtype=np.float64).sum(axis=1)
        present = np.isin(labels, np.asarray(oof_cols, dtype=np.int64))
        y, yhat, keep = _subsample_null_scores(
            present, oof_score, max_calib=max_calib, seed=seed
        )
        p = conformal_pvalues(y, yhat, test_score, null=null)
        if test_clusters is None:
            p_adj = bh_adjust(p)
            confidence = confidence_from_p(p)
        else:
            used = None if calib_clusters is None else calib_clusters[keep]
            p_adj = cluster_adjusted_pvalues(
                y,
                yhat,
                test_score,
                test_clusters,
                calib_clusters=used,
                null=null,
                magnitude=batch_magnitude,
                correct_batch_effect=correct_batch_effect,
            )
            confidence = confidence_from_p(p_adj)
        absent = y <= null
        null_scores = yhat[absent]
        if null_scores.size == 0:
            threshold = 1.0
            null_q = np.ones(4, dtype=np.float32)
        else:
            threshold = float(np.quantile(null_scores, 1.0 - fpr))
            null_q = np.quantile(null_scores, [0.5, 0.9, 0.95, 0.99]).astype(np.float32)
        present = test_score >= threshold
        out[target] = {
            "p": p.astype(np.float32),
            "p_adj": p_adj.astype(np.float32),
            "confidence": confidence,
            "discover_fdr": p_adj.astype(np.float32),
            "score": test_score.astype(np.float32),
            "threshold": np.float32(threshold),
            "null_quantiles": null_q,
            "present": present.astype(bool),
        }
    if not out:
        raise ValueError("none of the requested classes/groups match the OOF names")
    return out


def correct_predictions_by_he(
    test_probs: np.ndarray,
    class_names: Sequence[str],
    test_embeddings: np.ndarray,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    calib_embeddings: np.ndarray,
    *,
    n_clusters: int = UTOPIA_N_CLUSTERS,
    pca_dim: int = UTOPIA_PCA_DIM,
    fdr: float = 0.10,
    seed: int = 42,
    max_calib: int = 200_000,
    batch_magnitude: str | float = "medium",
    correct_batch_effect: bool = True,
    unlabeled: str = UNLABELED,
) -> dict[str, np.ndarray]:
    """Drop predicted mass that H&E-matched conformal inference does not support.

    Jin et al., bioRxiv 2026.03.01.708850 (UTOPIA). Test and calibration
    nuclei are clustered on their H&E embeddings. Inside each cluster, BH
    p-values use only the calibration nuclei of that morphology. A class
    probability is kept where ``p_adj <= fdr`` and set to 0 otherwise.
    Rows with no remaining mass are labeled ``low_confidence``.
    """
    names = [str(x) for x in class_names]
    test_probs = np.asarray(test_probs, dtype=np.float32)
    calib_probs = np.asarray(calib_probs, dtype=np.float32)
    if test_probs.ndim != 2 or test_probs.shape[1] != len(names):
        raise ValueError("test_probs width does not match class_names")
    if calib_probs.shape[1] != len(names):
        raise ValueError("calibration probabilities do not match class_names")
    if not 0.0 < float(fdr) <= 1.0:
        raise ValueError("fdr must be in (0, 1]")
    test_clusters, calib_clusters = match_histology_clusters(
        test_embeddings,
        calib_embeddings,
        n_clusters=n_clusters,
        pca_dim=pca_dim,
        random_state=seed,
    )
    packs = conformal_from_oof(
        test_probs,
        names,
        Path("."),
        classes=names,
        oof_labels=np.asarray(calib_labels),
        oof_probs=calib_probs,
        oof_class_names=names,
        test_clusters=test_clusters,
        calib_clusters=calib_clusters,
        max_calib=max_calib,
        seed=seed,
        fpr=fdr,
        batch_magnitude=batch_magnitude,
        correct_batch_effect=correct_batch_effect,
    )
    order = [name for name in names if name in packs]
    confidence = np.column_stack(
        [np.asarray(packs[name]["confidence"], dtype=np.float32) for name in order]
    )
    p_adj = np.column_stack(
        [np.asarray(packs[name]["p_adj"], dtype=np.float32) for name in order]
    )
    corrected = test_probs.copy()
    index = {name: i for i, name in enumerate(names)}
    for column, name in enumerate(order):
        keep = p_adj[:, column] <= float(fdr)
        corrected[~keep, index[name]] = 0.0
    peak = corrected.max(axis=1)
    top = corrected.argmax(axis=1)
    labels_out = np.array([names[i] for i in top], dtype=object)
    labels_out[peak <= 0] = unlabeled
    return {
        "test_clusters": test_clusters,
        "calib_clusters": calib_clusters,
        "confidence": confidence.astype(np.float32),
        "p_adj": p_adj.astype(np.float32),
        "corrected_probs": corrected.astype(np.float32),
        "corrected_label": labels_out,
    }


def hierarchy_groups(
    l2_names: Sequence[str] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """L12 / L1 member lists from the pan-cancer L2 names."""
    names = [str(x) for x in (l2_names if l2_names is not None else DEFAULT_L2_TO_L12)]
    l12: dict[str, list[str]] = {}
    for l2 in names:
        parent = DEFAULT_L2_TO_L12.get(l2)
        if parent:
            l12.setdefault(parent, []).append(l2)
    l1: dict[str, list[str]] = {}
    for l12_name, members in l12.items():
        parent = DEFAULT_L12_TO_L1.get(l12_name)
        if parent:
            l1.setdefault(parent, []).extend(members)
    return {
        "L12": {k: tuple(v) for k, v in l12.items()},
        "L1": {k: tuple(v) for k, v in l1.items()},
    }


def gated_confidence_matrix(
    conformal: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    """``present × confidence`` for each named class, shape ``[n, C]``."""
    names = list(conformal)
    cols = [
        np.asarray(conformal[n]["confidence"], dtype=np.float32)
        * np.asarray(conformal[n]["present"], dtype=np.float32)
        for n in names
    ]
    return names, np.stack(cols, axis=1)


def trusted_argmax(
    conformal: Mapping[str, Mapping[str, np.ndarray]],
    *,
    unlabeled: str = UNLABELED,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-nucleus winner of gated confidence. All-zero rows become ``unlabeled``."""
    names, mat = gated_confidence_matrix(conformal)
    peak = mat.max(axis=1)
    top = mat.argmax(axis=1)
    labels = np.array([names[i] for i in top], dtype=object)
    labels[peak <= 0] = unlabeled
    return labels, peak


def conformal_hierarchy(
    test_probs: np.ndarray,
    class_names: Sequence[str],
    oof_npz: str | Path,
    *,
    fpr: float = 0.1,
    max_calib: int = 200_000,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """L2 + rolled-up L12 / L1 conformal packs, loading the OOF file once."""
    labels, oof_probs = load_oof_head(oof_npz, head="l2")
    names = [str(x) for x in class_names]
    groups = hierarchy_groups(names)
    kw = dict(
        oof_npz=oof_npz,
        head="l2",
        fpr=fpr,
        max_calib=max_calib,
        seed=seed,
        oof_labels=labels,
        oof_probs=oof_probs,
    )
    with np.load(oof_npz, allow_pickle=False) as stored:
        if 'l2_class_names' in stored:
            kw['oof_class_names'] = stored['l2_class_names'].astype(str).tolist()
    return {
        "L2": conformal_from_oof(test_probs, names, classes=names, groups=None, **kw),
        "L12": conformal_from_oof(test_probs, names, classes=[], groups=groups["L12"], **kw),
        "L1": conformal_from_oof(test_probs, names, classes=[], groups=groups["L1"], **kw),
    }


def attach_to_obs(
    obs: pd.DataFrame,
    table: pd.DataFrame,
    *,
    conformal: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    fdr: float = 0.1,
) -> pd.DataFrame:
    """Copy confidence columns onto a pred h5ad ``obs`` (same row order)."""
    if len(obs) != len(table):
        raise ValueError(f"obs n={len(obs)} vs confidence table n={len(table)}")
    out = obs.copy()
    for col in (
        "conf_max",
        "conf_margin",
        "conf_entropy",
        "family",
        "label_plot",
        "parent_L12",
        "runner_up",
    ):
        if col in table.columns:
            out[col] = table[col].to_numpy()
    if conformal:
        for name, pack in conformal.items():
            key = name.replace(" ", "_")
            out[f"conf_{key}"] = pack["confidence"]
            out[f"p_{key}"] = pack["p"]
            out[f"p_adj_{key}"] = pack["p_adj"]
            present = pack["present"] if "present" in pack else pack["p_adj"] <= fdr
            out[f"present_{key}_fpr{fdr:.2f}"] = present
    return out


def write_confidence_npz(
    path: str | Path,
    table: pd.DataFrame,
    conformal: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> Path:
    """Save a sidecar next to ``*_l2_probs.npz`` (does not rewrite the pred h5ad)."""
    path = Path(path)
    payload: dict[str, object] = {
        "pred_CT": table["pred_CT"].to_numpy().astype(object),
        "label_plot": table["label_plot"].to_numpy().astype(object),
        "conf_max": table["conf_max"].to_numpy(dtype=np.float32),
        "conf_margin": table["conf_margin"].to_numpy(dtype=np.float32),
        "conf_entropy": table["conf_entropy"].to_numpy(dtype=np.float32),
    }
    if conformal:
        payload["conformal_json"] = json.dumps(sorted(conformal)).encode("utf-8")
        for name, pack in conformal.items():
            safe = name.replace(" ", "_")
            payload[f"confidence_{safe}"] = pack["confidence"]
            payload[f"p_{safe}"] = pack["p"]
            payload[f"p_adj_{safe}"] = pack["p_adj"]
            payload[f"score_{safe}"] = pack["score"]
            if "present" in pack:
                payload[f"present_{safe}"] = pack["present"]
            if "threshold" in pack:
                payload[f"threshold_{safe}"] = np.asarray(pack["threshold"])
            if "null_quantiles" in pack:
                payload[f"null_quantiles_{safe}"] = pack["null_quantiles"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


CONF_CMAP = "YlGn"
GRAY_OTHER = (0.78, 0.78, 0.78, 0.55)
HEADS = ("L1", "L12", "L2")


def highlight_labels(pred_ct: np.ndarray, keep: Sequence[str]) -> np.ndarray:
    """Keep listed L2 names; everything else becomes ``other``."""
    pred_ct = np.asarray(pred_ct, dtype=str)
    keep_set = {str(x) for x in keep}
    out = np.array(["other"] * len(pred_ct), dtype=object)
    for name in keep_set:
        out[pred_ct == name] = name
    return out


def gated_from_npz(npz, class_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``present``, ``confidence``, ``present × confidence`` for one class."""
    present = np.asarray(npz[f"present_{class_name}"], dtype=bool)
    conf = np.asarray(npz[f"confidence_{class_name}"], dtype=np.float32)
    return present, conf, (conf * present.astype(np.float32))


def conformal_npz_path(pred_h5ad: str | Path, suffix: str = "_confidence.npz") -> Path:
    pred_h5ad = Path(pred_h5ad)
    stem = pred_h5ad.name.replace("_stardist_pred.h5ad", "")
    return pred_h5ad.with_name(stem + suffix)


def ensure_conformal_npz(
    l2_probs: str | Path,
    oof_npz: str | Path,
    sidecar: str | Path,
    classes: Sequence[str],
    *,
    fpr: float = 0.1,
) -> Path:
    """Write missing ``present_*`` / ``confidence_*`` keys into ``sidecar``."""
    sidecar = Path(sidecar)
    wanted = [str(c) for c in classes]
    stamp = _confidence_cache_stamp([l2_probs, oof_npz], fpr=float(fpr))
    payload: dict[str, object] = {}
    missing = list(wanted)
    if sidecar.is_file():
        with np.load(sidecar, allow_pickle=True) as z:
            if 'input_signature' in z.files and str(z['input_signature'].item()) == stamp:
                payload = {k: z[k] for k in z.files}
                missing = [c for c in wanted if f"present_{c}" not in z.files]
    if missing:
        names, probs = load_l2_probs(l2_probs)
        # 2026.11.18, family names (B_Plasma, T_cell, ...) are groups, as in the CLI; passing
        # them as plain classes made a B_Plasma-only call fail ("none ... match the OOF names")
        groups = {k: v for k, v in DEFAULT_SIMILAR_FAMILIES.items() if k in missing}
        singles = [c for c in missing if c not in groups]
        packs = conformal_from_oof(probs, names, oof_npz, classes=singles, groups=groups or None, fpr=fpr)
        for name, pack in packs.items():
            payload[f"present_{name}"] = pack["present"]
            payload[f"confidence_{name}"] = pack["confidence"]
            payload[f"score_{name}"] = pack["score"]
            payload[f"threshold_{name}"] = np.asarray(pack["threshold"])
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload['input_signature'] = np.asarray(stamp)
        _atomic_confidence_npz(sidecar, **payload)
    return sidecar


def load_stardist_embeddings(
    features_h5ad: str | Path,
    obs_names: Sequence[str],
) -> np.ndarray:
    """UNI2 matrix ``X`` from ``*_stardist_features.h5ad``, in ``obs_names`` order."""
    import h5py

    features_h5ad = Path(features_h5ad)
    with h5py.File(features_h5ad, "r") as handle:
        raw = handle["obs"]["cell_id"][:]
        ids = np.array(
            [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw]
        )
        matrix = np.asarray(handle["X"][:], dtype=np.float32)
    wanted = np.asarray(list(obs_names), dtype=str)
    if matrix.shape[0] != ids.shape[0]:
        raise ValueError(f"{features_h5ad}: X {matrix.shape} vs {ids.shape[0]} cell ids")
    if np.array_equal(ids, wanted):
        return matrix
    order = pd.Index(ids).get_indexer(wanted)
    if np.any(order < 0):
        raise KeyError(f"{features_h5ad} does not cover every probability nucleus")
    return matrix[order]


def ensure_cluster_conformal_npz(
    l2_probs: str | Path,
    oof_npz: str | Path,
    features_h5ad: str | Path,
    sidecar: str | Path,
    classes: Sequence[str],
    *,
    n_clusters: int = UTOPIA_N_CLUSTERS,
    pca_dim: int = UTOPIA_PCA_DIM,
    seed: int = 42,
    fpr: float = 0.1,
    batch_magnitude: str | float = "medium",
) -> Path:
    """Cluster UNI2 StarDist embeddings, then BH inside each cluster.

    The sidecar is separate from the raw-p ``*_confidence_all_l2.npz``.
    ``confidence_{class}`` is the log-stretch of the cluster-wise BH p-value.
    """
    sidecar = Path(sidecar)
    wanted = [str(c) for c in classes]
    names, probs = load_l2_probs(l2_probs)
    with np.load(l2_probs, allow_pickle=True) as stored:
        obs = (
            np.asarray(stored["obs_names"]).astype(str)
            if "obs_names" in stored.files
            else np.array([str(i) for i in range(len(probs))])
        )
    stamp = _confidence_cache_stamp(
        [l2_probs, oof_npz, features_h5ad], n_clusters=int(n_clusters),
        pca_dim=int(pca_dim), seed=int(seed), fpr=float(fpr),
        batch_magnitude=batch_magnitude, classes=wanted, class_names=names)
    if sidecar.is_file():
        with np.load(sidecar, allow_pickle=True) as stored:
            ready = "confidence_source" in stored.files and "he_cluster" in stored.files
            if ready:
                source = stored["confidence_source"]
                source_text = (
                    source.item() if getattr(source, "shape", None) == () else str(source)
                )
                ready = (
                    source_text == stamp
                    and len(stored["he_cluster"]) == len(probs)
                    and all(f"confidence_{c}" in stored.files for c in wanted)
                )
        if ready:
            return sidecar
    embeddings = load_stardist_embeddings(features_h5ad, obs)
    if len(embeddings) != len(probs):
        raise ValueError(f"embeddings {len(embeddings)} vs probabilities {len(probs)}")
    labels = cluster_histology_embeddings(
        embeddings, n_clusters=n_clusters, pca_dim=pca_dim, random_state=seed
    )
    counts = np.bincount(labels, minlength=n_clusters)
    print(
        "histology clusters "
        + ", ".join(f"{i}:{int(n)}" for i, n in enumerate(counts))
    )
    packs = conformal_from_oof(
        probs,
        names,
        oof_npz,
        classes=wanted,
        fpr=fpr,
        seed=seed,
        test_clusters=labels,
        batch_magnitude=batch_magnitude,
    )
    payload: dict[str, object] = {
        "obs_names": obs,
        "class_names": np.asarray(names, dtype=str),
        "he_cluster": labels.astype(np.int16),
        "n_clusters": np.int32(n_clusters),
        "pca_dim": np.int32(pca_dim),
        "confidence_source": np.array(stamp),
    }
    for name, pack in packs.items():
        payload[f"present_{name}"] = pack["present"]
        payload[f"confidence_{name}"] = pack["confidence"]
        payload[f"p_{name}"] = pack["p"]
        payload[f"p_adj_{name}"] = pack["p_adj"]
        payload[f"score_{name}"] = pack["score"]
        payload[f"threshold_{name}"] = np.asarray(pack["threshold"])
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    _atomic_confidence_npz(sidecar, **payload)
    return sidecar


def ensure_trusted_labels(
    l2_probs: str | Path,
    oof_npz: str | Path,
    cache: str | Path,
    *,
    fpr: float = 0.1,
) -> dict[str, np.ndarray]:
    """L1 / L12 / L2 gated-argmax labels, cached as ``*_confidence_hierarchy.npz``."""
    cache = Path(cache)
    names, probs = load_l2_probs(l2_probs)
    stamp = _confidence_cache_stamp([l2_probs, oof_npz], fpr=float(fpr), class_names=names)
    if cache.is_file():
        with np.load(cache, allow_pickle=True) as z:
            if ('input_signature' in z.files and str(z['input_signature'].item()) == stamp
                    and all(h in z.files for h in HEADS) and len(z["L2"]) == len(probs)):
                return {h: np.asarray(z[h], dtype=object) for h in HEADS}
    packs = conformal_hierarchy(probs, names, oof_npz, fpr=fpr)
    trusted: dict[str, np.ndarray] = {}
    peak: dict[str, np.ndarray] = {}
    for head in HEADS:
        trusted[head], peak[head] = trusted_argmax(packs[head])
    cache.parent.mkdir(parents=True, exist_ok=True)
    _atomic_confidence_npz(
        cache,
        input_signature=np.asarray(stamp),
        **{h: trusted[h] for h in HEADS},
        **{f"peak_{h}": peak[h].astype(np.float32) for h in HEADS},
    )
    return trusted


def _l2_rgba(name: str) -> tuple:
    try:
        try:
            from plotting_palettes import get_palette
        except ImportError:
            from Hist2Pheno_pkg.plotting_palettes import get_palette
        pal = get_palette("pancancer", "fine")
        color = pal.get(name)
        if color is not None:
            return tuple(color)
    except Exception:
        pass
    fallback = {
        "B_cell": (0.00, 0.62, 0.50, 1.0),
        "Plasma": (0.08, 0.26, 0.34, 1.0),
        "Tregs": (0.62, 0.70, 0.98, 1.0),
        "Tumor": (0.82, 0.05, 0.09, 1.0),
    }
    return fallback.get(str(name), (0.50, 0.50, 0.50, 1.0))


def _point_s(marker_px: float, dpi: float) -> float:
    return (float(marker_px) * 72.0 / float(dpi)) ** 2


def scatter_confidence(
    ax,
    thumb,
    full_w,
    full_h,
    xy,
    values,
    title: str,
    point_s: float,
    *,
    show_he: bool,
    min_show: float = 0.0,
    cmap: str = CONF_CMAP,
    gray=GRAY_OTHER,
) -> None:
    if show_he:
        ax.imshow(thumb, extent=(0, full_w, full_h, 0), interpolation="bilinear")
    else:
        ax.set_facecolor("white")
    ax.set_xlim(0, full_w)
    ax.set_ylim(full_h, 0)
    ax.scatter(xy[:, 0], xy[:, 1], c=[gray], s=point_s, linewidths=0, rasterized=True)
    values = np.asarray(values, dtype=np.float32)
    hi = values >= float(min_show)
    ax.scatter(
        xy[hi, 0],
        xy[hi, 1],
        c=values[hi],
        s=point_s,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)


def _dot_legend(ax, names: Sequence[str], palette: Mapping[str, tuple], *, markersize: float = 11):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=markersize,
            markerfacecolor=palette[n][:3],
            markeredgecolor="none",
            label=n,
        )
        for n in names
        if n in palette and n != "other"
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            fontsize=12,
            frameon=False,
            ncol=max(len(handles), 1),
        )


def _conf_colorbar(fig, *, left: float = 0.915, cmap: str = CONF_CMAP):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cax = fig.add_axes([left, 0.20, 0.012, 0.62])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
    cbar.set_label("confidence (raw conformal p)", fontsize=10)
    return cbar


def save_figure(fig, path: str | Path, *, dpi: int = 300) -> Path:
    """Write PDF (editable Type 42 text if rcParams set) and a PNG preview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        path.with_suffix(".png"),
        format="png",
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(path.with_suffix(".pdf"))
    return path.with_suffix(".pdf")


def _load_pred_xy(cohort, sample: str):
    from anndata import read_h5ad

    paths = cohort.sample_paths(sample)
    adata = read_h5ad(paths["pred"], backed="r")
    xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
    pred_ct = adata.obs["pred_CT"].astype(str).to_numpy()
    return paths, xy, pred_ct


def plot_bcell_presence_figure(
    cohort,
    sample: str,
    *,
    conf_npz: str | Path | None = None,
    oof_npz: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    marker_px: float = 1.2,
    he_max_side: int = 1400,
    save: bool = False,
    save_dpi: int = 300,
):
    """2×4: argmax B/Plasma, FPR presence, confidence, present×conf."""
    import matplotlib.pyplot as plt

    paths, xy, pred_ct = _load_pred_xy(cohort, sample)
    l2_probs = paths["pred"].with_name(f"{sample}_l2_probs.npz")
    sidecar = Path(conf_npz) if conf_npz is not None else conformal_npz_path(paths["pred"])
    if oof_npz is not None:
        ensure_conformal_npz(
            l2_probs, oof_npz, sidecar, ("B_cell", "Plasma", "B_Plasma"), fpr=0.10
        )
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    z = np.load(sidecar, allow_pickle=True)
    present, conf, gated = gated_from_npz(z, "B_cell")
    n_arg_b = int((pred_ct == "B_cell").sum())
    n_arg_p = int((pred_ct == "Plasma").sum())
    n_fpr = int(present.sum())
    n_hi = int((conf >= 0.5).sum())
    n_bp = int(np.asarray(z["present_B_Plasma"]).sum()) if "present_B_Plasma" in z.files else -1
    print(
        f"{sample}  n={len(pred_ct):,}  argmax B/Plasma={n_arg_b:,}/{n_arg_p:,}  "
        f"FPR<=0.10 B={n_fpr:,} ({100 * n_fpr / len(pred_ct):.2f}%)  "
        f"B_Plasma={n_bp:,}  conf>=0.5={n_hi:,}  present×conf n={int((gated > 0).sum()):,}",
        flush=True,
    )
    thumb, full_w, full_h = cohort._he_thumbnail(paths["he"], max_side=he_max_side)
    point_s = _point_s(marker_px, dpi)
    if figsize is None:
        figsize = (17.2 * 4.0 / 3.0, 6.6 * 1.9)
    pal_arg = {"B_cell": _l2_rgba("B_cell"), "Plasma": _l2_rgba("Plasma"), "other": GRAY_OTHER}
    pal_pres = {"B_cell": _l2_rgba("B_cell"), "other": GRAY_OTHER}
    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
    cat_panels = [
        (highlight_labels(pred_ct, ("B_cell", "Plasma")), pal_arg, f"argmax B/Plasma  n={n_arg_b + n_arg_p:,}"),
        (highlight_labels(np.where(present, "B_cell", "other"), ("B_cell",)), pal_pres, f"FPR≤0.10 B_cell  n={n_fpr:,}"),
    ]
    for col, (labels, palette, title) in enumerate(cat_panels):
        for row, show_he in enumerate((True, False)):
            cohort._scatter_on_he(
                axes[row, col],
                thumb,
                full_w,
                full_h,
                xy,
                labels,
                palette,
                f"{title}  {'HE' if show_he else 'points'}",
                point_s,
                show_he=show_he,
            )
        _dot_legend(axes[1, col], [k for k in palette if k != "other"], palette)
    for row, show_he in enumerate((True, False)):
        suffix = "HE" if show_he else "points"
        scatter_confidence(
            axes[row, 2], thumb, full_w, full_h, xy, conf,
            f"conf B_cell n(≥0.5)={n_hi:,}  {suffix}", point_s,
            show_he=show_he, min_show=0.15,
        )
        scatter_confidence(
            axes[row, 3], thumb, full_w, full_h, xy, gated,
            f"present×conf  n={n_fpr:,}  {suffix}", point_s,
            show_he=show_he, min_show=1e-6,
        )
    fig.suptitle(f"{getattr(cohort, 'name', '')} {sample}  B-cell presence vs argmax  (no GT)", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.14, top=0.90, hspace=0.32, wspace=0.08)
    _conf_colorbar(fig, left=0.915)
    if save:
        out = cohort.pred_root / "figures" / f"{sample}_spatial_B_confidence.png"
        save_figure(fig, out, dpi=save_dpi)
    return fig


def plot_class_argmax_vs_gated(
    cohort,
    sample: str,
    class_name: str,
    *,
    conf_npz: str | Path | None = None,
    oof_npz: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    marker_px: float = 1.2,
    he_max_side: int = 1400,
    save: bool = False,
    save_dpi: int = 300,
):
    """2×2: argmax of ``class_name`` vs present×confidence."""
    import matplotlib.pyplot as plt

    paths, xy, pred_ct = _load_pred_xy(cohort, sample)
    l2_probs = paths["pred"].with_name(f"{sample}_l2_probs.npz")
    sidecar = Path(conf_npz) if conf_npz is not None else conformal_npz_path(paths["pred"])
    if oof_npz is not None:
        ensure_conformal_npz(l2_probs, oof_npz, sidecar, (class_name,), fpr=0.10)
    z = np.load(sidecar, allow_pickle=True)
    present, conf, gated = gated_from_npz(z, class_name)
    n_arg = int((pred_ct == class_name).sum())
    n_gate = int(present.sum())
    print(f"{sample}  {class_name}: argmax={n_arg:,}  present×conf={n_gate:,}", flush=True)
    thumb, full_w, full_h = cohort._he_thumbnail(paths["he"], max_side=he_max_side)
    point_s = _point_s(marker_px, dpi)
    if figsize is None:
        figsize = (17.2 * 2.0 / 3.0, 6.6 * 1.9)
    color = _l2_rgba(class_name)
    pal = {class_name: color, "other": GRAY_OTHER}
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    labels = highlight_labels(pred_ct, (class_name,))
    for row, show_he in enumerate((True, False)):
        suffix = "HE" if show_he else "points"
        cohort._scatter_on_he(
            axes[row, 0], thumb, full_w, full_h, xy, labels, pal,
            f"argmax {class_name}  n={n_arg:,}  {suffix}", point_s, show_he=show_he,
        )
        scatter_confidence(
            axes[row, 1], thumb, full_w, full_h, xy, gated,
            f"present×conf {class_name}  n={n_gate:,}  {suffix}",
            point_s, show_he=show_he, min_show=1e-6,
        )
    _dot_legend(axes[1, 0], (class_name,), pal)
    fig.suptitle(f"{getattr(cohort, 'name', '')} {sample}  {class_name}  argmax vs present×conf", fontsize=12)
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.14, top=0.90, hspace=0.32, wspace=0.10)
    _conf_colorbar(fig, left=0.90)
    if save:
        out = cohort.pred_root / "figures" / f"{sample}_spatial_{class_name}_confidence.png"
        save_figure(fig, out, dpi=save_dpi)
    return fig


def plot_trusted_spatial(
    cohort,
    sample: str,
    *,
    oof_npz: str | Path,
    cache: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    marker_px: float = 1.2,
    he_max_side: int = 1400,
    save: bool = False,
    save_dpi: int = 300,
    compare: Sequence[str] = ("Tumor", "DCIS", "Epithelial", "DC", "B_cell", "Plasma", "CD4_T", UNLABELED),
):
    """2×3 L1/L12/L2 gated-argmax map (not a replacement for softmax argmax)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paths, xy, pred_ct = _load_pred_xy(cohort, sample)
    l2_probs = paths["pred"].with_name(f"{sample}_l2_probs.npz")
    cache_path = Path(cache) if cache is not None else conformal_npz_path(paths["pred"], "_confidence_hierarchy.npz")
    trusted = ensure_trusted_labels(l2_probs, oof_npz, cache_path, fpr=0.10)
    print(f"\n{sample}  argmax vs gated-argmax (L2)", flush=True)
    new = trusted["L2"].astype(str)
    for name in compare:
        n0 = int((pred_ct == name).sum()) if name != UNLABELED else 0
        n1 = int((new == name).sum())
        print(f"  {name:16s}  argmax={n0:6d}  trusted={n1:6d}", flush=True)
    thumb, full_w, full_h = cohort._he_thumbnail(paths["he"], max_side=he_max_side)
    point_s = _point_s(marker_px, dpi)
    if figsize is None:
        figsize = (17.2, 6.6 * 1.9)
    palettes = {h: dict(cohort._palette()[h]) for h in HEADS}
    for pal in palettes.values():
        pal[UNLABELED] = (0.72, 0.72, 0.72, 1.0)
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    for col, head in enumerate(HEADS):
        labels = trusted[head].astype(str)
        palette = palettes[head]
        n_lab = int((labels != UNLABELED).sum())
        for row, show_he in enumerate((True, False)):
            cohort._scatter_on_he(
                axes[row, col], thumb, full_w, full_h, xy, labels, palette,
                f"{head} trusted  {'HE' if show_he else 'points'}  n={n_lab:,}/{len(labels):,}",
                point_s, show_he=show_he,
            )
        seen = set(map(str, labels.tolist()))
        names = [n for n in palette if n in seen] + sorted(seen - set(palette))
        ncol = 1 if len(names) <= 5 else (2 if len(names) <= 10 else 4)
        handles = [
            Line2D(
                [0], [0], linestyle="none", marker="o", markersize=13,
                markerfacecolor=palette.get(n, (0.5, 0.5, 0.5, 1))[:3],
                markeredgecolor="none", label=n,
            )
            for n in names
        ]
        axes[1, col].legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
            fontsize=13, frameon=False, ncol=ncol,
            handletextpad=0.35, columnspacing=1.0, labelspacing=0.4,
        )
    fig.suptitle(
        f"{getattr(cohort, 'name', '')} {sample}  trusted gated-argmax  (present × conf, FPR≤0.10)",
        fontsize=12,
    )
    fig.subplots_adjust(bottom=0.22, top=0.93, hspace=0.28)
    if save:
        out = cohort.pred_root / "figures" / f"{sample}_spatial_L1_L12_L2_trusted.png"
        save_figure(fig, out, dpi=save_dpi)
    return fig


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("l2_probs", type=Path, help="{sample}_l2_probs.npz")
    p.add_argument(
        "--oof",
        type=Path,
        default=None,
        help="Pan-cancer oof_predictions.npz (enables conformal presence).",
    )
    p.add_argument(
        "--classes",
        nargs="*",
        default=["B_cell", "Plasma"],
        help="L2 names (and/or groups) to calibrate. Default: B_cell Plasma.",
    )
    p.add_argument(
        "--fdr",
        type=float,
        default=0.1,
        help="Calibrated FPR vs OOF non-class cells (not whole-slide BH).",
    )
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    names, probs = load_l2_probs(a.l2_probs)
    table = annotate_nuclei(probs, names)
    print(table["label_plot"].value_counts().to_string())
    conformal = None
    if a.oof is not None:
        groups = {k: v for k, v in DEFAULT_SIMILAR_FAMILIES.items() if k in a.classes}
        singles = [c for c in a.classes if c not in groups]
        conformal = conformal_from_oof(
            probs,
            names,
            a.oof,
            classes=singles,
            groups=groups or None,
            fpr=a.fdr,
        )
        n_slide = len(probs)
        for name, pack in conformal.items():
            q = pack["null_quantiles"]
            n_fpr = int(pack["present"].sum())
            n_bh = int((pack["p_adj"] <= a.fdr).sum())
            n_hi = int((pack["confidence"] >= 0.5).sum())
            print(
                f"{name}: max={float(pack['score'].max()):.3f}  "
                f"OOF-null p90/p95/p99={float(q[1]):.3f}/{float(q[2]):.3f}/{float(q[3]):.3f}  "
                f"gate={float(pack['threshold']):.3f}"
            )
            print(
                f"  FPR<={a.fdr:.2f}: {n_fpr:,}/{n_slide:,} "
                f"({100.0 * n_fpr / n_slide:.2f}%)  "
                f"conf>=0.5: {n_hi:,}  "
                f"BH-all FDR<={a.fdr:.2f}: {n_bh:,} (diagnostic; too strict on a WSI)"
            )
    dest = a.out or a.l2_probs.with_name(a.l2_probs.stem.replace("_l2_probs", "") + "_confidence.npz")
    write_confidence_npz(dest, table, conformal)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
