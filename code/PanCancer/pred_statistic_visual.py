"""Table helpers for Pred_statistic_visual_pancancer_all.ipynb.

The notebook already puts code/PanCancer on sys.path, then imports from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HEADS = (("L2", "l2"), ("L12", "l12"), ("L1", "l1_head"))

ABLATION_FOLD_COLUMNS = {
    "l2_macro_auc": "L2_auc",
    "l12_macro_auc": "L12_auc",
    "l1_head_macro_auc": "L1_auc",
    "three_tier_auc_sum": "sum_auc",
    "l2_macro_f1": "L2_f1",
    "l12_macro_f1": "L12_f1",
    "l1_macro_f1": "L1_f1",
    "best_epoch": "best_epoch",
}


def n_sample_dirs(root: Path) -> int:
    """Count per-section StarDist ROC directories under a prediction root."""
    roc = root / "stardist_roc"
    return len([p for p in roc.iterdir() if p.is_dir()]) if roc.is_dir() else 0


def load_fold_oof(path: Path, prefix: str) -> pd.DataFrame:
    """Fold metrics from a published UNI group_cv_folds.csv, indexed by fold."""
    df = pd.read_csv(path)
    keep = {
        "fold": "fold",
        "l1_head_macro_auc": f"{prefix}_OOF_L1",
        "l12_macro_auc": f"{prefix}_OOF_L12",
        "l2_macro_auc": f"{prefix}_OOF_L2",
        "three_tier_auc_sum": f"{prefix}_OOF_sum",
        "l2_macro_f1": f"{prefix}_OOF_L2_f1",
        "best_epoch": f"{prefix}_best_epoch",
    }
    out = df.loc[:, [c for c in keep if c in df.columns]].rename(columns=keep)
    return out.set_index("fold")


def load_per_class(root: Path, tier: str) -> pd.DataFrame | None:
    """Pooled StarDist per-class AUROC CSV, or None if that tier was not written."""
    path = root / f"roc_stardist_{tier}_cross_dataset_all_Complete_Cases_per_class_auc.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df.rename(columns={"auc_ovr": "auc"})[["class_name", "auc", "n_pos"]]


def fold_csv(spec: dict) -> Path:
    return spec["result"] / "cross_dataset_cv" / spec["tag"] / "group_cv_folds.csv"


def load_folds(path: Path, prefix: str) -> pd.DataFrame:
    """UNI2 ablation fold metrics, columns prefixed by architecture key."""
    df = pd.read_csv(path)
    keep = {src: f"{prefix}_{dst}" for src, dst in ABLATION_FOLD_COLUMNS.items() if src in df.columns}
    return df.loc[:, ["fold", *keep]].rename(columns=keep).set_index("fold")


def load_ontology_heads(path: Path) -> dict:
    return json.loads(path.read_text())


def ovr_auc(y, prob) -> np.ndarray:
    """One-vs-rest AUROC per column. Classes with no positive or no negative stay NaN."""
    y = np.asarray(y)
    n = prob.shape[1]
    auc = np.full(n, np.nan, dtype=np.float64)
    for c in range(n):
        positive = y == c
        n_pos = int(positive.sum())
        if n_pos == 0 or n_pos == y.size:
            continue
        auc[c] = roc_auc_score(positive, prob[:, c])
    return auc


def oof_labels(spec: dict, key: str = "l2_labels") -> np.ndarray:
    z = np.load(spec["result"] / "oof_predictions.npz", mmap_mode="r")
    order = np.argsort(z["row_indices"])
    labels = np.asarray(z[key][order])
    z.close()
    return labels


def oof_tables(spec: dict, class_names: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class and per-dataset OOF macro AUROC for one architecture."""
    rows = pd.read_csv(spec["result"] / "oof_rows.csv", low_memory=False)
    z = np.load(spec["result"] / "oof_predictions.npz", mmap_mode="r")
    order = np.argsort(z["row_indices"])
    if not np.array_equal(np.sort(z["row_indices"]), np.arange(len(rows))):
        raise ValueError(f"{spec['tag']} OOF row indices are not a complete 0..N-1 set")
    cohort = rows.section_id.str.split("::").str[0].to_numpy()[order]
    per_class, per_dataset = [], []
    for head, key in HEADS:
        y = z[f"{key}_labels"][order]
        prob = z[f"{key}_probs"][order]
        names = class_names[head]
        if prob.shape[1] != len(names):
            raise ValueError(f"{head} prob width {prob.shape[1]} != {len(names)} ontology names")
        auc = ovr_auc(y, prob)
        for i, name in enumerate(names):
            per_class.append({"head": head, "class_name": name, "n_pos": int((y == i).sum()), "auroc": auc[i]})
        for dataset in sorted(pd.unique(cohort)):
            mask = cohort == dataset
            d_auc = ovr_auc(y[mask], prob[mask])
            finite = d_auc[np.isfinite(d_auc)]
            per_dataset.append(
                {
                    "dataset": dataset,
                    "head": head,
                    "n_cells": int(mask.sum()),
                    "n_classes": int(finite.size),
                    "macro_auroc": float(finite.mean()) if finite.size else np.nan,
                }
            )
    z.close()
    return pd.DataFrame(per_class), pd.DataFrame(per_dataset)


def split_old_cache(out_dir: Path) -> None:
    """Split a combined A0/A1 cache CSV so later runs only score new architectures."""
    old_class = out_dir / "oof_per_class_auroc.csv"
    old_dataset = out_dir / "oof_per_dataset_auroc.csv"
    if old_class.is_file():
        old = pd.read_csv(old_class)
        if "arch" in old.columns:
            for key, sub in old.groupby("arch"):
                dest = out_dir / f"oof_per_class_{key}.csv"
                if not dest.is_file():
                    sub.drop(columns="arch").to_csv(dest, index=False)
    if old_dataset.is_file():
        old = pd.read_csv(old_dataset)
        if "arch" in old.columns:
            for key, sub in old.groupby("arch"):
                dest = out_dir / f"oof_per_dataset_{key}.csv"
                if not dest.is_file():
                    sub.drop(columns="arch").to_csv(dest, index=False)
