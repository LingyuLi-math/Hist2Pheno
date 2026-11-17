"""Cross-dataset section folds, cell-stratified folds, and on-disk label caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CELL_STRATIFIED_SPLITTERS = frozenset(
    {
        "cell_stratified",
        "stratified_cell",
        "cell_type_kfold",
        "mixed_cell",
    }
)
# 2026.11.16, whole sections held out, folds balanced on L2 composition
STRATIFIED_SECTION_SPLITTERS = frozenset({"stratified_section", "l2_stratified_section"})
PREASSIGNED_SPLITTERS = frozenset(
    {
        "preassigned",
        "stratified_group_kfold",
        "dataset_section",
        "cross_dataset_section",
    }
) | CELL_STRATIFIED_SPLITTERS | STRATIFIED_SECTION_SPLITTERS

CELL_STRAT_RESULT_DIR = "result_cell_stratified_spatial"
CELL_STRAT_ABLATION_TAG = "D_emph_L2_spatial_pancancer_cellstrat"


def is_cell_stratified_splitter(name: str) -> bool:
    return str(name).strip().lower() in CELL_STRATIFIED_SPLITTERS


def is_stratified_section_splitter(name: str) -> bool:
    return str(name).strip().lower() in STRATIFIED_SECTION_SPLITTERS


def splitter_family(name: str) -> str:
    """Name written to ``fold_plan.csv``; plans of different families never mix."""
    if is_stratified_section_splitter(name):
        return "stratified_section"
    if is_cell_stratified_splitter(name):
        return "cell_stratified"
    return "cross_dataset_section"


def cache_path_for(section_id: str, cache_dir: Path) -> Path:
    safe = str(section_id).replace("/", "_").replace("::", "__")
    return Path(cache_dir) / f"{safe}.npz"


def assign_section_folds(
    section_ids: list[str],
    datasets: list[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
    existing: dict[str, int] | None = None,
) -> dict[str, int]:
    """Map each section to a fold.

    Within a dataset, sections are shuffled then assigned round-robin, so every
    fold holds out ~1/k sections from **each** cohort. A cohort with fewer than
    ``n_splits`` sections appears in that many validation folds (CRC P1/P2/P5
    occupy three folds).

    ``existing`` freezes section→fold pairs already written to ``fold_plan.csv``.
    Sections absent from that map are placed on the least-filled fold of their
    own dataset (lowest fold index wins ties). Adding CRC P5 does not move the
    sections that were already assigned.
    """
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if len(section_ids) != len(datasets):
        raise ValueError("section_ids and datasets must be the same length")
    by_ds: dict[str, list[str]] = {}
    for sid, ds in zip(section_ids, datasets):
        by_ds.setdefault(str(ds), []).append(str(sid))
    if existing:
        out: dict[str, int] = {}
        for _ds, sids in by_ds.items():
            counts = {i: 0 for i in range(n_splits)}
            fresh: list[str] = []
            for sid in sids:
                if sid in existing:
                    fold = int(existing[sid])
                    if not 0 <= fold < n_splits:
                        raise ValueError("Existing fold plan incompatible with requested cv_k; use a new output/data directory")
                    out[sid] = fold
                    if 0 <= fold < n_splits:
                        counts[fold] += 1
                else:
                    fresh.append(sid)
            for sid in fresh:
                fold = min(counts, key=lambda f: (counts[f], f))
                out[sid] = int(fold)
                counts[fold] += 1
        return out
    rng = np.random.default_rng(int(seed))
    out = {}
    for _ds, sids in by_ds.items():
        order = np.asarray(sids, dtype=object)
        rng.shuffle(order)
        for i, sid in enumerate(order.tolist()):
            out[str(sid)] = int(i % n_splits)
    return out


def ok_mask(index: pd.DataFrame) -> pd.Series:
    if "ok" not in index.columns:
        return pd.Series(True, index=index.index)
    s = index["ok"]
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_existing_folds(path: Path) -> dict[str, int]:
    """Section→fold map from ``fold_plan.csv``, or empty if the file is absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    old = pd.read_csv(path)
    if "section_id" not in old.columns or "fold" not in old.columns:
        return {}
    out: dict[str, int] = {}
    for sid, fold in zip(old["section_id"], old["fold"]):
        if pd.isna(fold):
            continue
        out[str(sid)] = int(fold)
    return out


def fold_plan_frame(
    index: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    existing: dict[str, int] | None = None,
) -> pd.DataFrame:
    ok = index.loc[ok_mask(index)].copy()
    if ok.empty:
        return pd.DataFrame(columns=["section_id", "dataset", "fold", "n_keep"])
    mapping = assign_section_folds(
        ok["section_id"].astype(str).tolist(),
        ok["dataset"].astype(str).tolist(),
        n_splits=n_splits,
        seed=seed,
        existing=existing or None,
    )
    out = ok.copy()
    out["fold"] = out["section_id"].astype(str).map(mapping)
    cols = ["section_id", "dataset", "tech", "sample", "fold", "n_keep", "n_raw", "h5ad"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["fold", "dataset", "section_id"]).reset_index(drop=True)


def _collapse_rare_strata(keys: np.ndarray, n_splits: int) -> np.ndarray:
    """Merge strata with fewer than ``n_splits`` cells: L2 → L12 → L1 → ``_rare``."""
    keys = np.asarray(keys, dtype=object)
    ser = pd.Series(keys)
    counts = ser.map(ser.value_counts())
    rare = counts.to_numpy() < int(n_splits)
    if rare.any():
        parent12 = ser.str.rsplit("|", n=1).str[0].to_numpy()
        keys = np.where(rare, parent12, keys)
    ser = pd.Series(keys)
    counts = ser.map(ser.value_counts())
    rare = counts.to_numpy() < int(n_splits)
    if rare.any():
        parent1 = ser.str.split("|").str[0].to_numpy()
        keys = np.where(rare, parent1, keys)
    ser = pd.Series(keys)
    counts = ser.map(ser.value_counts())
    rare = counts.to_numpy() < int(n_splits)
    if rare.any():
        keys = np.where(rare, "_rare", keys)
    return keys


def assign_cell_stratified_folds(
    y_l1,
    y_l12,
    y_l2,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Per-cell StratifiedKFold on the joint L1|L12|L2 label.

    Sections are mixed: each fold's train and val sets both contain (as far as
    counts allow) every lineage / sublineage / cell type. L2 already encodes
    the ontology parent, so this is equivalent to stratifying on L2 when the
    hierarchy is one-parent-per-child.

    Strata with fewer than ``n_splits`` cells collapse to L12, then L1, then a
    leftover ``_rare`` bucket that is assigned round-robin.
    """
    from sklearn.model_selection import StratifiedKFold

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    y_l1 = np.asarray(y_l1)
    y_l12 = np.asarray(y_l12)
    y_l2 = np.asarray(y_l2)
    n = int(y_l2.shape[0])
    if y_l1.shape[0] != n or y_l12.shape[0] != n:
        raise ValueError("y_l1, y_l12, and y_l2 must have the same length")
    if n < n_splits:
        raise ValueError(f"Need at least n_splits={n_splits} cells, got {n}")

    composite = (
        pd.Series(y_l1).astype(str)
        + "|"
        + pd.Series(y_l12).astype(str)
        + "|"
        + pd.Series(y_l2).astype(str)
    ).to_numpy()
    y_strat = _collapse_rare_strata(composite, n_splits)

    fold_id = np.full(n, -1, dtype=np.int64)
    ser = pd.Series(y_strat)
    counts = ser.map(ser.value_counts()).to_numpy()
    usable = counts >= n_splits
    n_usable = int(usable.sum())
    if n_usable >= n_splits:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed)
        )
        idx = np.flatnonzero(usable)
        y_sub = y_strat[usable]
        dummy = np.zeros(idx.shape[0], dtype=np.int8)
        for fold, (_, va) in enumerate(skf.split(dummy, y_sub)):
            fold_id[idx[va]] = int(fold)

    leftover = np.flatnonzero(fold_id < 0)
    if leftover.size:
        rng = np.random.default_rng(int(seed) + 1)
        order = leftover.copy()
        rng.shuffle(order)
        fold_id[order] = np.arange(order.size, dtype=np.int64) % n_splits
    return fold_id


def summarize_cell_stratified_folds(
    y_l1,
    y_l12,
    y_l2,
    fold_id,
) -> pd.DataFrame:
    """Long table: level, fold, class, n (for ``fold_plan_cell_stratified.csv``)."""
    df = pd.DataFrame(
        {
            "fold": np.asarray(fold_id, dtype=np.int64),
            "L1": np.asarray(y_l1),
            "L12": np.asarray(y_l12),
            "L2": np.asarray(y_l2),
        }
    )
    parts = []
    for level, col in (("L1", "L1"), ("L12", "L12"), ("L2", "L2")):
        ct = pd.crosstab(df["fold"], df[col])
        long = ct.stack().reset_index()
        long.columns = ["fold", "class", "n"]
        long.insert(0, "level", level)
        parts.append(long)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["level", "fold", "class"]).reset_index(drop=True)


########################################################
# 2026.11.16, section-grouped folds balanced on L2 composition
# 按 section 整块留出，同时让每个 fold 的 L2 组成尽量接近全体
########################################################
def _fold_costs(F, M, S, inv_T, w_c, inv_n_ds, w_d, inv_S, k, size_weight):
    """Per-fold cost. F: fold×class counts, M: fold×dataset units, S: fold cells."""
    class_term = ((F * inv_T * k - 1.0) ** 2) @ w_c
    cohort_term = ((M * inv_n_ds * k - 1.0) ** 2) @ w_d
    size_term = size_weight * (S * inv_S * k - 1.0) ** 2
    return class_term + cohort_term + size_term


def assign_stratified_section_folds(
    section_ids,
    datasets,
    counts,
    *,
    n_splits: int = 5,
    seed: int = 42,
    existing: dict | None = None,
    units=None,
    cohort_weight: float = 1.0,
    size_weight: float = 0.5,
    swap_trials: int | None = None,
) -> dict[str, int]:
    """Map sections to folds so each fold holds out whole sections (or whole
    ``units``, e.g. patients) and each fold's L2 composition stays close to the
    pooled one.

    ``counts`` is sections × L2 classes (cell counts). The cost is the squared
    relative deviation of every fold from 1/k of each class (all classes
    weighted equally, so rare classes such as B_cell or Plasma matter as much
    as Tumor), of each cohort's units, and of the total cell count. Units are
    placed greedily, rarest first; then random swaps between two units of the
    same cohort are kept whenever they lower the cost (cohort counts per fold
    do not change). Sections in ``existing`` keep their fold.
    """
    k = int(n_splits)
    if k < 2:
        raise ValueError(f"n_splits must be >= 2, got {k}")
    sids = [str(s) for s in section_ids]
    ds = [str(d) for d in datasets]
    C = np.asarray(counts, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != len(sids) or len(ds) != len(sids):
        raise ValueError("section_ids, datasets and counts rows must align")
    if len(set(sids)) != len(sids):
        raise ValueError("duplicate section ids")
    unit_of = [str(u) for u in (units if units is not None else sids)]
    if len(unit_of) != len(sids):
        raise ValueError("units must align with section_ids")
    # Collapse sections into units (a unit never spans two folds).
    u_names = list(dict.fromkeys(unit_of))
    u_index = {u: i for i, u in enumerate(u_names)}
    U = np.zeros((len(u_names), C.shape[1]))
    np.add.at(U, [u_index[u] for u in unit_of], C)
    u_ds = {}
    for u, d in zip(unit_of, ds):
        if u_ds.setdefault(u, d) != d:
            raise ValueError(f"unit {u!r} spans two datasets")
    ds_names = sorted(set(ds))
    d_index = {d: i for i, d in enumerate(ds_names)}
    u_d = np.asarray([d_index[u_ds[u]] for u in u_names])
    u_size = U.sum(1)

    T = U.sum(0)
    present = T > 0
    inv_T = np.where(present, 1.0 / np.maximum(T, 1e-12), 0.0)
    w_c = present / max(int(present.sum()), 1)
    n_ds = np.bincount(u_d, minlength=len(ds_names)).astype(float)
    inv_n_ds = 1.0 / np.maximum(n_ds, 1.0)
    w_d = np.full(len(ds_names), float(cohort_weight) / max(len(ds_names), 1))
    inv_S = 1.0 / max(float(u_size.sum()), 1e-12)
    args = (inv_T, w_c, inv_n_ds, w_d, inv_S, k, float(size_weight))

    fold_u = np.full(len(u_names), -1, dtype=np.int64)
    frozen = np.zeros(len(u_names), dtype=bool)
    for sid, u in zip(sids, unit_of):
        if existing and sid in existing:
            f = int(existing[sid])
            if not 0 <= f < k:
                raise ValueError("Existing fold plan incompatible with requested cv_k; use a new output directory")
            i = u_index[u]
            if fold_u[i] not in (-1, f):
                raise ValueError(f"unit {u!r} already split across folds in the existing plan")
            fold_u[i] = f
            frozen[i] = True

    F = np.zeros((k, C.shape[1]))
    M = np.zeros((k, len(ds_names)))
    S = np.zeros(k)
    for i in np.flatnonzero(frozen):
        F[fold_u[i]] += U[i]; M[fold_u[i], u_d[i]] += 1; S[fold_u[i]] += u_size[i]

    rng = np.random.default_rng(int(seed))
    todo = np.flatnonzero(~frozen)
    share = (U[todo] * inv_T).max(1) if len(todo) else np.zeros(0)
    jitter = rng.random(len(todo))
    order = todo[np.lexsort((jitter, -u_size[todo], -share))]
    eye = np.eye(len(ds_names))
    for i in order:
        base = _fold_costs(F, M, S, *args)
        new = _fold_costs(F + U[i], M + eye[u_d[i]], S + u_size[i], *args)
        delta = new - base
        f = int(np.argmin(delta + 1e-12 * np.arange(k)))
        fold_u[i] = f
        F[f] += U[i]; M[f, u_d[i]] += 1; S[f] += u_size[i]

    movable = np.flatnonzero(~frozen)
    if len(movable) > 1:
        n_trials = int(swap_trials if swap_trials is not None else 60 * len(movable))
        by_ds = {d: movable[u_d[movable] == d] for d in range(len(ds_names))}
        cost = _fold_costs(F, M, S, *args)
        pick = rng.integers(0, len(movable), size=n_trials)
        for t in range(n_trials):
            a = movable[pick[t]]
            pool = by_ds[u_d[a]]
            if len(pool) < 2:
                continue
            b = pool[rng.integers(0, len(pool))]
            fa, fb = fold_u[a], fold_u[b]
            if fa == fb:
                continue
            diff = U[b] - U[a]
            ds_ = u_size[b] - u_size[a]
            Fa, Fb = F[fa] + diff, F[fb] - diff
            ca = _fold_costs(Fa[None], M[fa][None], np.array([S[fa] + ds_]), *args)[0]
            cb = _fold_costs(Fb[None], M[fb][None], np.array([S[fb] - ds_]), *args)[0]
            if ca + cb < cost[fa] + cost[fb] - 1e-12:
                F[fa], F[fb] = Fa, Fb
                S[fa] += ds_; S[fb] -= ds_
                cost[fa], cost[fb] = ca, cb
                fold_u[a], fold_u[b] = fb, fa
    return {sid: int(fold_u[u_index[u]]) for sid, u in zip(sids, unit_of)}


def section_label_counts(groups, labels, n_groups: int, n_classes: int) -> np.ndarray:
    """sections × classes cell counts from per-cell group ids and encoded labels."""
    g = np.asarray(groups, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    ok = (y >= 0) & (y < n_classes)
    flat = g[ok] * n_classes + y[ok]
    return np.bincount(flat, minlength=n_groups * n_classes).reshape(n_groups, n_classes)


def fold_balance_table(counts, fold_ids, class_names) -> pd.DataFrame:
    """fold × class share of each class held out in that fold (ideal 1/k),
    plus whether the fold's training side still has the class."""
    C = np.asarray(counts, dtype=np.float64)
    f = np.asarray(fold_ids, dtype=np.int64)
    k = int(f.max()) + 1
    F = np.zeros((k, C.shape[1]))
    np.add.at(F, f, C)
    T = F.sum(0)
    rows = []
    for fold in range(k):
        for j, name in enumerate(class_names):
            rows.append({
                "fold": fold, "class": str(name), "n_val": int(F[fold, j]),
                "share_of_class": float(F[fold, j] / T[j]) if T[j] > 0 else np.nan,
                "val_fraction_in_fold": float(F[fold, j] / F[fold].sum()) if F[fold].sum() > 0 else np.nan,
                "pooled_fraction": float(T[j] / T.sum()) if T.sum() > 0 else np.nan,
                "train_has_class": bool(T[j] - F[fold, j] > 0),
            })
    return pd.DataFrame(rows)


def fold_balance_score(counts, fold_ids) -> float:
    """Mean over present classes of the max |share-1/k| across folds (0 = perfect)."""
    C = np.asarray(counts, dtype=np.float64)
    f = np.asarray(fold_ids, dtype=np.int64)
    k = int(f.max()) + 1
    F = np.zeros((k, C.shape[1]))
    np.add.at(F, f, C)
    T = F.sum(0)
    ok = T > 0
    return float(np.abs(F[:, ok] / T[ok] - 1.0 / k).max(0).mean())


# ---------------------------------------------------------------------------
# 2026.11.17, inner validation split (early stopping) inside one outer fold.
# The old split is GroupShuffleSplit over sections: it ignores cohorts and
# classes, so a cohort with one training section in this fold (BRCA in folds
# 0/1) can lose that section, and with it every DCIS cell, to inner
# validation. "cohort_safe" keeps >=1 section of every cohort and every L2
# class of the outer training set in inner training.
# 内层验证集（early stopping）划分；cohort_safe 保证每个队列、每个 L2 类别都留在内层训练集。
INNER_VAL_MODES = ("random", "cohort_safe")


def inner_val_split(outer_tr, groups, *, fraction, seed, mode="random", datasets=None, labels=None):
    """Return ``(it, iv, report)``: positions into ``outer_tr`` for inner train /
    inner validation, plus a report (inner-val groups, L2 classes of the outer
    training set that inner training lost).

    ``mode="random"`` reproduces the previous GroupShuffleSplit exactly.
    ``mode="cohort_safe"`` draws ``round(fraction * n)`` groups per cohort, never
    the last group of a cohort, then moves inner-validation groups back to
    training until every class of the outer training set is in inner training.
    """
    outer_tr = np.asarray(outer_tr)
    g = np.asarray(groups)[outer_tr]
    lab = None if labels is None else np.asarray(labels)[outer_tr]
    mode = str(mode).strip().lower()
    if mode not in INNER_VAL_MODES:
        raise ValueError(f"inner_val mode must be one of {INNER_VAL_MODES}, got {mode!r}")
    if mode == "random":
        from sklearn.model_selection import GroupShuffleSplit

        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=int(seed))
                      .split(outer_tr, groups=g))
    else:
        if lab is None:
            raise ValueError("cohort_safe needs labels")
        # without cohort names every group counts as one cohort (class coverage still applies)
        ds = np.asarray(datasets)[outer_tr] if datasets is not None else np.full(len(g), "all", dtype=object)
        uniq, first = np.unique(g, return_index=True)
        cohort_of = dict(zip(uniq.tolist(), ds[first].tolist()))
        rng = np.random.default_rng(int(seed))
        val = set()
        for cohort in sorted(set(cohort_of.values()), key=str):
            units = sorted([u for u, c in cohort_of.items() if c == cohort], key=str)
            n_val = min(int(round(fraction * len(units))), len(units) - 1)
            if n_val > 0:
                order = rng.permutation(len(units))
                val.update(units[i] for i in order[:n_val])
        # class coverage: give back the inner-val group richest in each lost class
        while val:
            in_val = np.isin(g, list(val))
            lost = np.setdiff1d(np.unique(lab), np.unique(lab[~in_val]))
            if lost.size == 0:
                break
            c = lost[0]
            hit = in_val & (lab == c)
            cand, cnt = np.unique(g[hit], return_counts=True)
            back = cand[np.argmax(cnt)]
            val.discard(back.item() if isinstance(back, np.generic) else back)
        if not val:
            raise ValueError("cohort_safe inner validation is empty; lower n_splits or use mode='random'")
        in_val = np.isin(g, list(val))
        it, iv = np.flatnonzero(~in_val), np.flatnonzero(in_val)
    report = {"mode": mode, "inner_val_groups": sorted(np.unique(g[iv]).tolist(), key=str)}
    if lab is not None:
        report["lost_classes"] = np.setdiff1d(np.unique(lab), np.unique(lab[it])).tolist()
    return it, iv, report
