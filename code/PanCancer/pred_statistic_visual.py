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
    # 2026.10.26, align by true row identity / CSV 本已按原始行排列，禁止再次置换
    if rows.row_index.duplicated().any():
        raise ValueError("Duplicate row_index in oof_rows.csv")
    rows = rows.set_index("row_index").loc[np.asarray(z["row_indices"])[order]]
    cohort = rows.section_id.str.split("::").str[0].to_numpy()
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



# 2026.10.26, bind cached tables and cross-model comparison to exact inputs / 完整细胞身份
OOF_TABLE_VERSION = "aligned-cohort-v2"


def oof_identity(spec):
    rows = pd.read_csv(spec["result"] / "oof_rows.csv", dtype={"cell_id":str})
    if rows.row_index.duplicated().any():
        raise ValueError("Duplicate OOF row identity")
    columns = ["row_index", "section_id", "cell_id", "fold"]
    return rows.sort_values("row_index")[columns].reset_index(drop=True)


def cached_oof_tables(spec, class_names, out_dir, key):
    from checkpoint_artifact import digest, file_digest
    from feature_integrity import atomic_json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = digest(OOF_TABLE_VERSION, class_names,
        file_digest(spec["result"] / "oof_predictions.npz"),
        file_digest(spec["result"] / "oof_rows.csv"))
    class_path = out_dir / f"oof_per_class_{key}.csv"
    dataset_path = out_dir / f"oof_per_dataset_{key}.csv"
    marker = out_dir / f"oof_tables_{key}_v2.json"
    try:
        saved = json.loads(marker.read_text())
        valid = (saved["stamp"] == stamp and saved["class_sha"] == file_digest(class_path)
                 and saved["dataset_sha"] == file_digest(dataset_path))
    except (OSError, ValueError, KeyError):
        valid = False
    if valid:
        return pd.read_csv(class_path), pd.read_csv(dataset_path)
    classes_df, dataset_df = oof_tables(spec, class_names)
    classes_df.to_csv(class_path, index=False)
    dataset_df.to_csv(dataset_path, index=False)
    atomic_json(marker, {"stamp":stamp,"class_sha":file_digest(class_path),
                         "dataset_sha":file_digest(dataset_path)})
    return classes_df, dataset_df

# 2026.10.28, one best_mlp per architecture on every section / 部署比较，不是五折 OOF
PAIRED_V2_DEPLOYMENT = (
    {
        "arch": "A0",
        "label": "A0 legacy, no spatial",
        "pred": "pred_uni2_A0_legacy_nospatial_paired_v2_seed42_deployment",
        "ckpt": "result_uni2_A0_legacy_nospatial_paired_v2_seed42/cross_dataset_cv/A0_legacy_nospatial/best_mlp_gpu.pt",
    },
    {
        "arch": "A1",
        "label": "A1 adjacent, no spatial",
        "pred": "pred_uni2_A1_adjacent_nospatial_paired_v2_seed42_deployment",
        "ckpt": "result_uni2_A1_adjacent_nospatial_paired_v2_seed42/cross_dataset_cv/A1_adjacent_nospatial/best_mlp_gpu.pt",
    },
    {
        "arch": "A2",
        "label": "A2 adjacent, k=8 mean",
        "pred": "pred_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42_deployment",
        "ckpt": "result_uni2_A2_adjacent_spatial_k8mean_paired_v2_seed42/cross_dataset_cv/A2_adjacent_spatial_k8mean/best_mlp_gpu.pt",
    },
    {
        "arch": "A3",
        "label": "A3 adjacent, k=8 mean + gate",
        "pred": "pred_uni2_A3_bestmlp_paired_v2_seed42_deployment",
        "ckpt": "result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate/best_mlp_gpu.pt",
    },
)

_SECTION_AUC = {
    "L1": "macro_auc_L1_cells",
    "L12": "macro_auc_L12_cells",
    "L2": "macro_auc_L2_cells",
}


def load_paired_v2_deployment(repo: Path) -> pd.DataFrame:
    """Matched-cell metrics for each architecture's single best_mlp on all sections."""
    frames = []
    section_ids = None
    for spec in PAIRED_V2_DEPLOYMENT:
        pred = repo / "data" / "PanCancer" / spec["pred"]
        path = pred / "metrics_per_section.csv"
        ckpt = repo / "data" / "PanCancer" / spec["ckpt"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        df = pd.read_csv(path)
        if df["section_id"].duplicated().any():
            raise ValueError(f"{spec['arch']} has duplicate section_id rows")
        scopes = set(df["evaluation_scope"].astype(str))
        if scopes != {"deployment_not_oof"}:
            raise ValueError(f"{spec['arch']} evaluation_scope={sorted(scopes)}")
        if df["checkpoint_sha256"].nunique() != 1:
            raise ValueError(f"{spec['arch']} did not use one checkpoint")
        if df[list(_SECTION_AUC.values())].isna().any().any():
            raise ValueError(f"{spec['arch']} has a missing section AUROC")
        got = set(df["section_id"].astype(str))
        if section_ids is None:
            section_ids = got
        elif got != section_ids:
            raise ValueError(f"{spec['arch']} section ids do not match the other runs")
        out = df.copy()
        out.insert(0, "arch", spec["arch"])
        out.insert(1, "label", spec["label"])
        out.insert(2, "checkpoint", str(ckpt))
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def summarize_paired_v2_deployment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section-mean AUROC, plus L2 averaged inside each cancer."""
    l2 = df.pivot(index="section_id", columns="arch", values="macro_auc_L2_cells")
    l2 = l2[[spec["arch"] for spec in PAIRED_V2_DEPLOYMENT]]
    at_max = l2.eq(l2.max(axis=1), axis=0)
    unique_best = l2.idxmax(axis=1).where(at_max.sum(axis=1) == 1)
    wins = unique_best.value_counts()
    rows = []
    for spec in PAIRED_V2_DEPLOYMENT:
        sub = df.loc[df["arch"] == spec["arch"]]
        weights = sub["n_matched"].to_numpy()
        by_cancer = sub.groupby("dataset")["macro_auc_L2_cells"].mean()
        rows.append(
            {
                "arch": spec["arch"],
                "label": spec["label"],
                "checkpoint": sub["checkpoint"].iloc[0],
                "n_sections": int(sub["section_id"].nunique()),
                "n_matched": int(sub["n_matched"].sum()),
                "n_sections_without_stardist_tokens": int((sub["n_stardist_all"] == 0).sum()),
                "mean_section_L1": float(sub["macro_auc_L1_cells"].mean()),
                "mean_section_L12": float(sub["macro_auc_L12_cells"].mean()),
                "mean_section_L2": float(sub["macro_auc_L2_cells"].mean()),
                "cell_weighted_L2": float(np.average(sub["macro_auc_L2_cells"], weights=weights)),
                "mean_of_cancer_L2": float(by_cancer.mean()),
                "n_sections_unique_best_L2": int(wins.get(spec["arch"], 0)),
            }
        )
    overall = pd.DataFrame(rows)
    by_dataset = None
    for spec in PAIRED_V2_DEPLOYMENT:
        sub = df.loc[df["arch"] == spec["arch"]]
        grouped = sub.groupby("dataset").agg(
            n_sections=("section_id", "size"),
            n_matched=("n_matched", "sum"),
            **{spec["arch"]: ("macro_auc_L2_cells", "mean")},
        )
        if by_dataset is None:
            by_dataset = grouped
        else:
            by_dataset = by_dataset.join(grouped[[spec["arch"]]])
    archs = [spec["arch"] for spec in PAIRED_V2_DEPLOYMENT]
    by_dataset["best"] = by_dataset[archs].idxmax(axis=1)
    mean_row = {"n_sections": int(by_dataset["n_sections"].sum()), "n_matched": int(by_dataset["n_matched"].sum())}
    for arch in archs:
        mean_row[arch] = float(by_dataset[arch].mean())
    mean_row["best"] = max(archs, key=lambda arch: mean_row[arch])
    by_dataset = pd.concat([by_dataset, pd.DataFrame([mean_row], index=["mean of 7 cancers"])])
    return overall, by_dataset.reset_index(names="dataset")


def plot_paired_v2_l2(by_dataset: pd.DataFrame, out_pdf: Path):
    """Grouped bars of mean section L2 AUROC. The equal-cancer row is omitted."""
    import matplotlib.pyplot as plt

    plot_df = by_dataset.loc[by_dataset["dataset"] != "mean of 7 cancers"].copy()
    archs = [spec["arch"] for spec in PAIRED_V2_DEPLOYMENT]
    labels = [spec["label"] for spec in PAIRED_V2_DEPLOYMENT]
    x = np.arange(len(plot_df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, (arch, label) in enumerate(zip(archs, labels)):
        ax.bar(x + (i - 1.5) * width, plot_df[arch].to_numpy(), width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["dataset"].tolist())
    ax.set_ylabel("Mean section L2 AUROC")
    ax.set_xlabel("Cancer")
    ax.set_ylim(0.60, 0.92)
    ax.set_title("UNI2 paired_v2 best_mlp on all sections")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    return fig


def save_paired_v2_deployment(repo: Path, out_dir: Path | None = None):
    """Write the overall and per-cancer tables plus the L2 bar chart."""
    if out_dir is None:
        out_dir = repo / "data" / "PanCancer" / "uni2_A0_A3_paired_v2_seed42_deployment_compare"
    sections = load_paired_v2_deployment(repo)
    overall, by_dataset = summarize_paired_v2_deployment(sections)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_path = out_dir / "overall_section_mean.csv"
    dataset_path = out_dir / "l2_by_cancer.csv"
    overall.to_csv(overall_path, index=False)
    by_dataset.to_csv(dataset_path, index=False)
    fig = plot_paired_v2_l2(by_dataset, out_dir / "l2_by_cancer.pdf")
    missing = sections.loc[sections["n_stardist_all"] == 0, ["arch", "section_id"]].drop_duplicates()
    return {
        "sections": sections,
        "overall": overall,
        "by_dataset": by_dataset,
        "overall_path": overall_path,
        "dataset_path": dataset_path,
        "figure": fig,
        "missing_stardist": missing,
    }


def split_old_cache(out_dir: Path) -> None:
    """Split a combined A0/A1 cache CSV so later runs only score new architectures."""
    # 2026.10.26, correctness migration / 旧表可能含 cohort 错配，不再导入旧缓存
    return
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
