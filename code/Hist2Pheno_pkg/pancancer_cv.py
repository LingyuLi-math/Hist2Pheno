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
PREASSIGNED_SPLITTERS = frozenset(
    {
        "preassigned",
        "stratified_group_kfold",
        "dataset_section",
        "cross_dataset_section",
    }
) | CELL_STRATIFIED_SPLITTERS

CELL_STRAT_RESULT_DIR = "result_cell_stratified_spatial"
CELL_STRAT_ABLATION_TAG = "D_emph_L2_spatial_pancancer_cellstrat"


def is_cell_stratified_splitter(name: str) -> bool:
    return str(name).strip().lower() in CELL_STRATIFIED_SPLITTERS


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
