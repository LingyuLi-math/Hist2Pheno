## 2026.08.12 LLY, copy from Xenium_lung/xenium_uni_nb_helpers.py
## 2026.08.13, adjust the code to fit pan_organ and scheme_for_pan_organ and resolve_effective_scheme for CODEX ESCC and Xenium Lung dataset
## 2026.08.14, add functions to create symlinks for the StarDist sample dirs and analyze the StarDist macro AUROC by clinical groups


"""Shared cross-dataset UNI-label CV/notebook helpers.

Originally developed for Lung_train_validate_cv_UNIlabel1234.ipynb, including
its insample and StarDist tiers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Literal, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from plotting_palettes import (
    clinical_columns_for_pan_organ,
    clinical_group_order,
    scheme_for_pan_organ,
    default_spatial_point_size,
)

###############################################
# 2026.06.24 LLY, clean Lung_train_validate_cv_UNIlabel_all_clean.ipynb
##                create Lung_train_validate_cv_UNIlabel.py
############################################### 
# Figure / metrics filenames under .../{therapy_model}/result/
RESULT_FIG = {
    "conf_matrix_level2": "conf_matrix_level2.pdf",
    "conf_matrix_level1": "conf_matrix_level1.pdf",
    "conf_matrix_level1_L1head": "conf_matrix_level1_L1head.pdf",
    "conf_matrix_level12": "conf_matrix_level12.pdf",
    "conf_matrix_level3": "conf_matrix_level3.pdf",
    "conf_matrix_level4": "conf_matrix_level4.pdf",
    "f1_perclass_level2": "f1_perclass_level2.pdf",
    "f1_perclass_level1": "f1_perclass_level1.pdf",
    "f1_perclass_level1_L1head": "f1_perclass_level1_L1head.pdf",
    "f1_perclass_level12": "f1_perclass_level12.pdf",
    "f1_perclass_level3": "f1_perclass_level3.pdf",
    "f1_perclass_level4": "f1_perclass_level4.pdf",
    "celltype_valid_level2": "celltype_valid_level2.jpg",
    "celltype_true_level2": "celltype_true_level2.jpg",
    "celltype_valid_level1": "celltype_valid_level1.jpg",
    "celltype_true_level1": "celltype_true_level1.jpg",
    "celltype_valid_level12": "celltype_valid_level12.jpg",
    "celltype_true_level12": "celltype_true_level12.jpg",
    "celltype_valid_level3": "celltype_valid_level3.jpg",
    "celltype_true_level3": "celltype_true_level3.jpg",
    "celltype_valid_level4": "celltype_valid_level4.jpg",
    "celltype_true_level4": "celltype_true_level4.jpg",
    "validation_internal_metrics": "validation_internal_metrics.csv",
    "validation_external_stardist_matched_metrics": "validation_external_stardist_matched_metrics.csv",
    "stardist_pred_level2": "celltype_pred_stardist_level2.jpg",
    "stardist_true_level2": "celltype_true_stardist_level2.jpg",
    "stardist_pred_level1": "celltype_pred_stardist_level1.jpg",
    "stardist_true_level1": "celltype_true_stardist_level1.jpg",
    "stardist_pred_level12": "celltype_pred_stardist_level12.jpg",
    "stardist_true_level12": "celltype_true_stardist_level12.jpg",
    "stardist_pred_level3": "celltype_pred_stardist_level3.jpg",
    "stardist_true_level3": "celltype_true_stardist_level3.jpg",
    "stardist_pred_level4": "celltype_pred_stardist_level4.jpg",
    "stardist_true_level4": "celltype_true_stardist_level4.jpg",
    "acc_level2_stardist": "acc_level2_stardist.pdf",
    "acc_level1_stardist": "acc_level1_stardist.pdf",
    "acc_level1_stardist_L1head": "acc_level1_stardist_L1head.pdf",
    "acc_level12_stardist": "acc_level12_stardist.pdf",
    "acc_level3_stardist": "acc_level3_stardist.pdf",
    "acc_level4_stardist": "acc_level4_stardist.pdf",
    "conf_matrix_level2_stardist": "conf_matrix_level2_stardist.pdf",
    "conf_matrix_level1_stardist": "conf_matrix_level1_stardist.pdf",
    "conf_matrix_level12_stardist": "conf_matrix_level12_stardist.pdf",
    "conf_matrix_level3_stardist": "conf_matrix_level3_stardist.pdf",
    "conf_matrix_level4_stardist": "conf_matrix_level4_stardist.pdf",
    "f1_level2_stardist": "f1_level2_stardist.pdf",
    "f1_level1_stardist": "f1_level1_stardist.pdf",
    "f1_level1_stardist_L1head": "f1_level1_stardist_L1head.pdf",
    "f1_level12_stardist": "f1_level12_stardist.pdf",
    "f1_level3_stardist": "f1_level3_stardist.pdf",
    "f1_level4_stardist": "f1_level4_stardist.pdf",
    "roc_stardist_level2": "roc_stardist_level2.pdf",
    "roc_stardist_level2_from_AUROC_csv": "roc_stardist_level2_from_AUROC_csv.pdf",
    "roc_internal_level2_oof": "roc_internal_level2_oof.pdf",
    "roc_stardist_level1": "roc_stardist_level1.pdf",
    "roc_stardist_level1_L1head": "roc_stardist_level1_L1head.pdf",
    "roc_stardist_level12": "roc_stardist_level12.pdf",
    "roc_stardist_level3": "roc_stardist_level3.pdf",
    "roc_stardist_level4": "roc_stardist_level4.pdf",
}

# Backward-compatible name used by existing Xenium notebooks.
LUNG_FIG = RESULT_FIG


def make_result_fig(cases_root, therapy_data, therapy_model, save_result="result"):
    """Return ``(result_fig, result_dir)`` for a sample under *cases_root*."""
    result_dir = Path(cases_root) / therapy_data / therapy_model / save_result

    def result_fig(name, mkdir=True):
        if name not in LUNG_FIG:
            raise KeyError(f"Unknown figure key {name!r}. Add it to LUNG_FIG.")
        if mkdir:
            result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir / LUNG_FIG[name])

    return result_fig, result_dir


def discover_case_samples(cases_root: Path, sample: str | None) -> list[str]:
    if sample is not None:
        return [sample]
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")
    return sorted(
        p.name for p in cases_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

########################################################
# 2026.06.25 LLY: use h5ad files of Xenium_lung Complete_Cases 25 datasets for training and validation
########################################################
def discover_h5ad_case_samples(cases_root: Path, sample: str | None = None) -> list[str]:
    """Sample IDs that have pre-built ``{sample}_matched_features.h5ad``."""
    if sample is not None:
        return [sample]
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")
    out = []
    for p in sorted(cases_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / f"{p.name}_matched_features.h5ad").is_file():
            out.append(p.name)
    return out


########################################################
## 2026.07.05 LLY: Add the function to discover the StarDist all h5ad samples for Complete_Cases
########################################################
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"


def discover_stardist_all_h5ad_samples(cases_root: Path, sample: str | None = None) -> list[str]:
    """Sample IDs that have pre-built ``{sample}_all_features_stardist.h5ad``."""
    if sample is not None:
        h5ad = cases_root / sample / f"{sample}{STARDIST_ALL_H5AD_SUFFIX}"
        if not h5ad.is_file():
            raise FileNotFoundError(f"Missing StarDist-all h5ad: {h5ad}")
        return [sample]
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")
    out = []
    for p in sorted(cases_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / f"{p.name}{STARDIST_ALL_H5AD_SUFFIX}").is_file():
            out.append(p.name)
    return out
########################################################

def make_pooled_result_fig(data_root, save_result="result"):
    """Return ``(result_fig, result_dir)`` under ``Data/result/`` (pooled Complete_Cases runs)."""
    result_dir = Path(data_root) / save_result

    def result_fig(name, mkdir=True):
        if name not in LUNG_FIG:
            raise KeyError(f"Unknown figure key {name!r}. Add it to LUNG_FIG.")
        if mkdir:
            result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir / LUNG_FIG[name])

    return result_fig, result_dir


########################################################
## 2026.06.27 LLY: Add the function to load the all_features_stardist_label.h5ad
########################################################
STARDIST_ALL_LABEL_H5AD_SUFFIX = "_all_features_stardist_label.h5ad"
STARDIST_INCOMPLETE_RESULT_SUBDIR = "stardist_Incomplete_Cases"

HEAD_TO_CLASS_NAMES_KEY = {
    "l2": "class_names",
    "l1": "class_names_level1",
    "l12": "class_names_level12",
    "l3": "class_names_level3",
    "l4": "class_names_level4",
}
HEAD_DISPLAY_NAMES = {
    "l2": "L2 celltype (final_CT)",
    "l1": "L1 lineage (final_lineage)",
    "l12": "L12 sublineage (final_sublineage)",
    "l3": "L3 CNiche",
    "l4": "L4 TNiche",
}


def stardist_all_label_h5ad_path(data_root, sample, save_result="result"):
    """``Data/{save_result}/stardist/{sample}/{sample}_all_features_stardist_label.h5ad``."""
    _, result_dir = make_pooled_stardist_result_fig(data_root, sample, save_result)
    return result_dir / f"{sample}{STARDIST_ALL_LABEL_H5AD_SUFFIX}"

########################################################
## 2026.07.05 LLY: Add the function to get the StarDist all label h5ad path for Incomplete_Cases
########################################################
def stardist_incomplete_all_label_h5ad_path(data_root, sample, save_result="result"):
    """``Data/{save_result}/stardist_Incomplete_Cases/{sample}/{sample}_all_features_stardist_label.h5ad``."""
    result_dir = Path(data_root) / save_result / STARDIST_INCOMPLETE_RESULT_SUBDIR / sample
    return result_dir / f"{sample}{STARDIST_ALL_LABEL_H5AD_SUFFIX}"


########################################################
## 2026.08.20 LLY: HCC rest (no CODEX annotation) all-nuclei label path
########################################################
STARDIST_HCC_REST_RESULT_SUBDIR = "stardist_hcc_rest"


def stardist_hcc_rest_all_label_h5ad_path(data_root, sample, save_result="result"):
    """``s4769/{save_result}/stardist_hcc_rest/{sample}/{sample}_all_features_stardist_label.h5ad``."""
    result_dir = Path(data_root) / save_result / STARDIST_HCC_REST_RESULT_SUBDIR / sample
    return result_dir / f"{sample}{STARDIST_ALL_LABEL_H5AD_SUFFIX}"
########################################################


def load_stardist_label_head_predictions(
    h5ad_path,
    *,
    heads: Sequence[str] = ("l2", "l12", "l1"),
    obsm_key: str = "spatial_HE",
) -> dict:
    """Load argmax preds + coords from a StarDist ``*_label.h5ad``.

    Uses ``uns['pred_prob_class_names']`` and ``obs['{head}_prob_{j}']``.
    """
    import anndata as ad

    path = Path(h5ad_path)
    adata = ad.read_h5ad(path)
    names_map = adata.uns.get("pred_prob_class_names") or {}
    coords = None
    for key in (obsm_key, "spatial_HE", "spatial"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=np.float64)[:, :2]
            break
    if coords is None and {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        coords = adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    if coords is None:
        raise KeyError(f"{path.name}: no spatial coords in obsm or centroid_x/y.")
    tiers = {}
    for head in heads:
        if head not in names_map:
            raise KeyError(
                f"{path.name}: missing head {head!r} in pred_prob_class_names; "
                f"available={list(names_map)}"
            )
        class_names = [str(x) for x in names_map[head]]
        prob_cols = [f"{head}_prob_{j}" for j in range(len(class_names))]
        missing = [c for c in prob_cols if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"{path.name}: missing {missing[:4]}")
        probs = adata.obs[prob_cols].to_numpy(dtype=np.float64)
        tiers[head] = {
            "class_names": class_names,
            "pred_encoded": probs.argmax(axis=1).astype(np.int64),
        }
    return {
        "path": path,
        "coords": coords,
        "tiers": tiers,
        "n_obs": int(adata.n_obs),
    }


def plot_stardist_label_spatial_heads(
    rec: Mapping,
    *,
    sample: str,
    heads: Sequence[str],
    pan_organ: str,
    spatial_point_size: float | None = None,
    fig_size: tuple[float, float] = (10, 8),
    show: bool = False,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Write one pred-only spatial JPG per head next to the label h5ad."""
    from plot import plot_celltype_spatial_distribution, plot_tier_spatial_distribution

    if spatial_point_size is None:
        spatial_point_size = default_spatial_point_size(pan_organ, stardist_map=True)

    save_dir = Path(rec["path"]).parent
    n_obs = rec.get("n_obs")
    n_txt = f" (n={n_obs:,})" if n_obs is not None else ""
    prefix = title_prefix if title_prefix is not None else sample
    saved = {}
    for head in heads:
        out = save_dir / f"{sample}_stardist_pred_{head}.jpg"
        display = HEAD_DISPLAY_NAMES.get(head, str(head).upper())
        plot_tier_spatial_distribution(
            pred_encoded=rec["tiers"][head]["pred_encoded"],
            class_names=rec["tiers"][head]["class_names"],
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            save_path_pred=str(out),
            fig_size=fig_size,
            spatial_point_size=spatial_point_size,
            spatial_color_scheme=scheme_for_pan_organ(pan_organ, head=head),
            celltype_col=head,
            pan_organ=pan_organ,
            show=show,
            X_coords_matched=rec["coords"],
            title_pred=f"{prefix}  StarDist pred {display}{n_txt}",
        )
        print(f"  saved {out.name}")
        saved[head] = out
    return saved


def plot_stardist_label_spatial_overview(
    loaded: Mapping[str, Mapping],
    *,
    heads: Sequence[str],
    pan_organ: str,
    save_path=None,
    point_size: float | None = None,
    sample_labels: Mapping[str, str] | None = None,
    suptitle: str | None = None,
    show: bool = True,
):
    """Compact ``n_samples × n_heads`` pred-only spatial grid."""
    import matplotlib.pyplot as plt
    from plotting_palettes import resolve_palette

    if point_size is None:
        point_size = default_spatial_point_size(pan_organ, overview=True)

    samples = list(loaded)
    n_row, n_col = len(samples), len(heads)
    if n_row == 0:
        raise ValueError("loaded is empty")
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.0 * n_col, 3.2 * n_row), squeeze=False,
    )
    for i, sample in enumerate(samples):
        rec = loaded[sample]
        xy = rec["coords"]
        row_label = (sample_labels or {}).get(sample, sample)
        n_obs = rec.get("n_obs")
        for j, head in enumerate(heads):
            ax = axes[i, j]
            names = rec["tiers"][head]["class_names"]
            pred = rec["tiers"][head]["pred_encoded"]
            palette = resolve_palette(
                names,
                pan_organ=pan_organ,
                scheme=scheme_for_pan_organ(pan_organ, head=head),
            )
            colors = np.array(
                [palette.get(names[int(k)], (0.5, 0.5, 0.5, 1.0)) for k in pred]
            )
            ax.scatter(
                xy[:, 0], xy[:, 1], c=colors, s=point_size, alpha=0.7,
                linewidths=0, rasterized=True,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(HEAD_DISPLAY_NAMES.get(head, str(head).upper()), fontsize=10)
            if j == 0:
                n_txt = f"\nn={n_obs:,}" if n_obs is not None else ""
                ax.set_ylabel(f"{row_label}{n_txt}", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.002)
    fig.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Overview → {out}")
    # Jupyter auto-displays a returned Figure; close after show so the
    # notebook cell does not render the same overview twice.
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


def plot_stardist_label_spatial_maps(
    data_root,
    samples: Sequence[str],
    save_result: str,
    *,
    label_h5ad_path_fn,
    heads: Sequence[str],
    pan_organ: str,
    spatial_point_size: float | None = None,
    fig_size: tuple[float, float] = (10, 8),
    show: bool = False,
    title_prefix_fn=None,
    missing_error: str = "No label h5ads. Run the inference cell first.",
) -> dict[str, dict]:
    """Load label h5ads and write one pred-only spatial JPG per head."""
    loaded: dict[str, dict] = {}
    for sample in samples:
        path = Path(label_h5ad_path_fn(data_root, sample, save_result))
        if not path.is_file():
            print(f"  skip (missing label h5ad): {sample}")
            continue
        rec = load_stardist_label_head_predictions(path, heads=heads)
        rec["sample"] = sample
        prefix = title_prefix_fn(sample) if title_prefix_fn is not None else sample
        print(f"  {prefix}: n={rec['n_obs']:,}")
        plot_stardist_label_spatial_heads(
            rec,
            sample=sample,
            heads=heads,
            pan_organ=pan_organ,
            spatial_point_size=spatial_point_size,
            fig_size=fig_size,
            show=show,
            title_prefix=prefix,
        )
        loaded[sample] = rec
    if not loaded:
        raise FileNotFoundError(missing_error)
    return loaded


def attach_five_head_probs_to_adata_obs(adata, head_probs, g=None, *, cv_data=None):
    """
    Write five-head softmax probabilities into ``adata.obs`` as ``{head}_prob_{j}``.

    Class name order for each head is stored in ``adata.uns['pred_prob_class_names']``.
    """
    g = g or {}
    class_names_uns = {}
    for head, probs in head_probs.items():
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f"head_probs[{head!r}] must be 2D, got shape {probs.shape}")
        names_key = HEAD_TO_CLASS_NAMES_KEY.get(head)
        names = g.get(names_key) if names_key else None
        if names is None and cv_data is not None and names_key:
            names = cv_data.get(names_key)
        if names is not None:
            class_names_uns[head] = [str(x) for x in list(names)]
        for j in range(probs.shape[1]):
            adata.obs[f"{head}_prob_{j}"] = probs[:, j]
    if class_names_uns:
        adata.uns["pred_prob_class_names"] = class_names_uns
    return adata
########################################################

def make_pooled_stardist_result_fig(data_root, sample, save_result="result"):
    """Per-sample StarDist outputs under ``Data/result/stardist/{sample}/``."""
    result_dir = Path(data_root) / save_result / "stardist" / sample

    def result_fig(name, mkdir=True):
        if name not in LUNG_FIG:
            raise KeyError(f"Unknown figure key {name!r}. Add it to LUNG_FIG.")
        if mkdir:
            result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir / LUNG_FIG[name])

    return result_fig, result_dir


def prepare_pooled_cv_data_from_h5ads(
    cases_root: Path,
    samples: list[str],
    min_l2_samples: int = 2,
):
    """
    Load and concatenate pre-built HE matched h5ads; ``groups_f`` = sample index per cell.

    Uses global label encoders across all datasets (via ``prepare_data_leave_one_group_out``).
    """
    import anndata as ad

    from base import adata_X_to_dense, prepare_data_leave_one_group_out

    if not samples:
        raise ValueError("samples list is empty")

    sample_to_group = {s: i for i, s in enumerate(samples)}
    X_parts, y_parts, y_l1_parts = [], [], []
    y_l12_parts, y_l3_parts, y_l4_parts = [], [], []
    coord_parts, group_parts = [], []
    n_obs_per_sample = {}

    for sample in samples:
        h5ad_path = cases_root / sample / f"{sample}_matched_features.h5ad"
        if not h5ad_path.is_file():
            raise FileNotFoundError(f"Missing matched h5ad: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        n_obs = adata.n_obs
        n_obs_per_sample[sample] = n_obs
        X_parts.append(adata_X_to_dense(adata.X))
        y_parts.append(adata.obs["final_CT"].to_numpy())
        y_l1_parts.append(adata.obs["final_lineage"].to_numpy())
        y_l12_parts.append(adata.obs["final_sublineage"].to_numpy())
        y_l3_parts.append(adata.obs["CNiche"].to_numpy())
        y_l4_parts.append(adata.obs["TNiche"].to_numpy())
        coord_parts.append(np.asarray(adata.obsm["spatial_HE"], dtype=np.float64))
        group_parts.append(np.full(n_obs, sample_to_group[sample], dtype=np.int64))
        print(f"  {sample}: {n_obs:,} cells", flush=True)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    y_level1 = np.concatenate(y_l1_parts)
    y_level12 = np.concatenate(y_l12_parts)
    y_level3 = np.concatenate(y_l3_parts)
    y_level4 = np.concatenate(y_l4_parts)
    X_coords = np.vstack(coord_parts)
    groups = np.concatenate(group_parts)

    cv_data = prepare_data_leave_one_group_out(
        X,
        y,
        y_level1,
        groups=groups,
        min_l2_samples=min_l2_samples,
        y_level12=y_level12,
        y_level3=y_level3,
        y_level4=y_level4,
        X_coords=X_coords,
    )
    cv_data["sample_names"] = list(samples)
    cv_data["sample_to_group"] = sample_to_group
    cv_data["group_to_sample"] = {v: k for k, v in sample_to_group.items()}
    cv_data["n_obs_per_sample"] = n_obs_per_sample
    print(
        f"  Pooled: {cv_data['X_f'].shape[0]:,} cells × {cv_data['X_f'].shape[1]} features, "
        f"{len(cv_data['class_names'])} L2 classes",
        flush=True,
    )
    return cv_data


def plot_he_validate_level2_minimal(
    plot_confusion_matrix,
    plot_multiclass_roc_curves,
    result_fig,
    class_names,
    val_labels,
    val_preds,
    val_probs_l2=None,
    *,
    title_prefix="OOF Level2",
    roc_color_scheme="xenium_lung_fine",
    pan_organ=None,
):
    """Internal validation (minimal): L2 confusion matrix + macro AUROC only."""
    print(f"\n[{title_prefix}] confusion matrix (L2)", flush=True)
    plot_confusion_matrix(
        val_labels,
        val_preds,
        class_names,
        figsize=(10, 8),
        save_path=result_fig("conf_matrix_level2"),
    )
    if val_probs_l2 is None:
        print("  Skip ROC: no val_probs_l2", flush=True)
        return float("nan")
    n_classes = len(class_names)
    macro_auc = macro_auc_ovr(val_labels, val_probs_l2, n_classes)
    print(f"  L2 macro AUROC (OOF): {macro_auc:.4f}", flush=True)
    valid = (np.asarray(val_labels) >= 0) & (np.asarray(val_labels) < n_classes)
    if np.any(valid):
        eff_scheme = roc_color_scheme
        if pan_organ is not None and str(roc_color_scheme).strip().lower() in (
            "xenium_lung_fine",
            "xenium_ct",
        ):
            eff_scheme = scheme_for_pan_organ(pan_organ, head="l2")
        plot_multiclass_roc_curves(
            np.asarray(val_labels)[valid],
            np.asarray(val_probs_l2)[valid],
            class_names,
            figsize=(3.0, 3.0),
            max_curves=n_classes,
            save_path=result_fig("roc_internal_level2_oof"),
            title=f"{title_prefix} ROC",
            roc_color_scheme=eff_scheme,
            pan_organ=pan_organ,
        )
    return macro_auc


def save_pooled_internal_validation_metrics(
    metrics_csv_path,
    *,
    macro_auc_l2,
    n_samples,
    n_cells,
    n_folds,
    train_group_frac,
    best_fold,
    group_summary,
):
    """Write minimal internal-validation CSV (AUROC-focused; no acc/F1 columns)."""
    import pandas as pd

    row = {
        "mode": "cross_dataset_group_cv",
        "n_samples": int(n_samples),
        "n_cells": int(n_cells),
        "n_folds": int(n_folds),
        "train_group_frac": float(train_group_frac),
        "best_fold": int(best_fold["fold"]) if best_fold else -1,
        "oof_l2_macro_auc": float(macro_auc_l2),
        "cv_l2_macro_f1_mean": float(group_summary.get("l2_macro_f1_mean", float("nan"))),
    }
    df = pd.DataFrame([row])
    Path(metrics_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_csv_path, index=False)
    return df

###############################################

# (short, head_pred_key, labels_glob, preds_glob, names_glob, y_f_key in cv_data)
EXTRA_TIER_SPECS = (
    ("level12", "preds_l12", "val_labels_level12", "val_preds_level12", "class_names_level12", "y_level12_f"),
    ("level3", "preds_l3", "val_labels_level3", "val_preds_level3", "class_names_level3", "y_level3_f"),
    ("level4", "preds_l4", "val_labels_level4", "val_preds_level4", "class_names_level4", "y_level4_f"),
)

STARDIST_EXTRA_SPECS = (
    ("L12 sublineage", "preds_l12", "class_names_level12", "y_star_level12", "le_level12"),
    ("L3 CNiche", "preds_l3", "class_names_level3", "y_star_level3", "le_level3"),
    ("L4 TNiche", "preds_l4", "class_names_level4", "y_star_level4", "le_level4"),
)


def tier_class_names(names_key, cv_data=None, g=None):
    """Resolve class name array from globals or cv_data (never use ``or`` on numpy arrays)."""
    g = g or {}
    names = g.get(names_key)
    if names is None and cv_data is not None:
        names = cv_data.get(names_key)
    return names


###############################################################
# 2026.08.12 Adjust the function to use spatial-contex
# 2026.08.12 val_preds_level1 vs val_preds_level1_head - separate evaluation, on HCC
###############################################################
def ensure_lp_extra_insample_preds(
    model,
    scaler,
    cv_data,
    device,
    predict_all_label_heads,
    g=None,
    *,
    neighbor_index=None,
):
    """Populate in-sample predictions for extra heads from the best-fold model."""
    g = g if g is not None else {}
    if g.get("val_preds_level12") is not None:
        l1_head = g.get("val_preds_level1_head")
        if l1_head is not None and len(l1_head) == len(cv_data["y_level1_encoded_f"]):
            g["val_labels_level1_head"] = cv_data["y_level1_encoded_f"]
        return
    if not hasattr(model, "level12_head"):
        return
    if neighbor_index is None and g.get("spatial_neighbor_index") is not None:
        neighbor_index = g.get("spatial_neighbor_index")
    X_ins = scaler.transform(cv_data["X_f"])
    head_preds = predict_all_label_heads(
        model, X_ins, device, neighbor_index=neighbor_index
    )
    for short, pk in (("level12", "preds_l12"), ("level3", "preds_l3"), ("level4", "preds_l4")):
        yk = f"y_{short}_encoded_f"
        if yk in cv_data and pk in head_preds:
            g[f"val_preds_{short}"] = head_preds[pk]
            g[f"val_labels_{short}"] = cv_data[yk]
    if g.get("val_preds_level1_head") is None and "preds_l1" in head_preds:
        g["val_preds_level1_head"] = head_preds["preds_l1"]
        g["val_labels_level1_head"] = cv_data["y_level1_encoded_f"]


###############################################################
# 2026.08.12 add stardist prediction
###############################################################
def stardist_head_preds(
    model_star,
    scaler,
    X_star,
    device,
    predict_all_label_heads,
    *,
    neighbor_index=None,
):
    if not hasattr(model_star, "level12_head"):
        return None
    return predict_all_label_heads(
        model_star,
        scaler.transform(X_star),
        device,
        neighbor_index=neighbor_index,
    )


def encode_star_labels(y_raw, le):
    return le.transform(np.asarray(y_raw).astype(str))


def metrics_triplet(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_he_confusion_matrices(plot_confusion_matrix, result_fig, cv_data=None, g=None):
    """In-sample confusion matrices for L2/L1/L12/L3/L4 (+ L1 head if available)."""
    g = g or {}
    rows = [
        ("L2 celltype", "val_labels", "val_preds", "class_names", "conf_matrix_level2", (10, 8)),
        ("L1 lineage (L2→L1 agg)", "val_labels_level1", "val_preds_level1", "class_names_level1", "conf_matrix_level1", (4, 4)),
        ("L1 lineage (L1 head)", "val_labels_level1_head", "val_preds_level1_head", "class_names_level1", "conf_matrix_level1_L1head", (4, 4)),
        ("L12 sublineage", "val_labels_level12", "val_preds_level12", "class_names_level12", "conf_matrix_level12", (6, 5)),
        ("L3 CNiche", "val_labels_level3", "val_preds_level3", "class_names_level3", "conf_matrix_level3", (8, 7)),
        ("L4 TNiche", "val_labels_level4", "val_preds_level4", "class_names_level4", "conf_matrix_level4", (8, 7)),
    ]
    for title, lk, pk, nk, fig_key, figsize in rows:
        y_true, y_pred = g.get(lk), g.get(pk)
        if y_pred is None:
            print(f"Skip {title}: no {pk} (re-run ablation cell after reload model/base)")
            continue
        if y_true is None and lk == "val_labels_level1_head":
            # Backward compatibility for callers where both L1 predictions have
            # the same evaluation scope.
            y_true = g.get("val_labels_level1")
        if y_true is None:
            print(f"Skip {title}: no {lk}")
            continue
        if len(y_true) != len(y_pred):
            print(
                f"Skip {title}: {lk} and {pk} have different evaluation scopes "
                f"({len(y_true):,} vs {len(y_pred):,})"
            )
            continue
        names = tier_class_names(nk, cv_data, g)
        if names is None:
            print(f"Skip {title}: missing {nk}")
            continue
        print(f"\n{title}")
        plot_confusion_matrix(y_true, y_pred, names, figsize=figsize, save_path=result_fig(fig_key))
    l1_agg = g.get("val_preds_level1")
    l1_head = g.get("val_preds_level1_head")
    if l1_agg is not None and l1_head is not None:
        if len(l1_agg) == len(l1_head):
            print("L1 agg vs L1 head agreement:", (l1_agg == l1_head).mean())
        else:
            print(
                "Skip L1 agg vs L1 head agreement: different evaluation scopes "
                f"({len(l1_agg):,} OOF rows vs {len(l1_head):,} in-sample cells)"
            )


def _extra_tier_color_scheme(color_scheme, tier, *, pan_organ=None):
    """Resolve a shared override or a per-tier scheme mapping.

    When ``color_scheme`` is ``None``, uses the per-tier canonical defaults for
    ``pan_organ`` (defaults to Xenium lung for backward compatibility).
    """
    organ = pan_organ or "xenium_lung"
    default = scheme_for_pan_organ(organ, head=tier)
    if color_scheme is None:
        return default
    if isinstance(color_scheme, Mapping):
        return color_scheme.get(tier, default)
    return color_scheme


def plot_he_spatial_extra_tiers(
    plot_tier_spatial_distribution,
    plot_celltype_spatial_distribution,
    result_fig,
    therapy_data,
    X_coords_plot,
    cv_data,
    g=None,
    *,
    spatial_color_scheme=None,
    pan_organ=None,
):
    """Plot extra tiers using per-tier defaults, one scheme, or a tier mapping."""
    g = g or {}
    for tier, title, pk, nk, yk, fig_pred, fig_true in (
        ("l12", "L12 sublineage", "val_preds_level12", "class_names_level12", "y_level12_f", "celltype_valid_level12", "celltype_true_level12"),
        ("l3", "L3 CNiche", "val_preds_level3", "class_names_level3", "y_level3_f", "celltype_valid_level3", "celltype_true_level3"),
        ("l4", "L4 TNiche", "val_preds_level4", "class_names_level4", "y_level4_f", "celltype_valid_level4", "celltype_true_level4"),
    ):
        pred = g.get(pk)
        if pred is None:
            print(f"Skip {title} spatial: no {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        y_f = cv_data.get(yk) if cv_data is not None else None
        if names is None or y_f is None:
            print(f"Skip {title} spatial: missing {nk} or cv_data[{yk!r}]")
            continue
        print(f"\n{title}")
        plot_tier_spatial_distribution(
            pred_encoded=pred,
            class_names=names,
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            save_path_pred=result_fig(fig_pred),
            save_path_true=result_fig(fig_true),
            spatial_color_scheme=_extra_tier_color_scheme(
                spatial_color_scheme, tier, pan_organ=pan_organ
            ),
            pan_organ=pan_organ,
            title_pred=f"{therapy_data} predicted {title}",
            title_true=f"{therapy_data} ground truth {title}",
            fig_size=(10, 8),
            show=True,
            X_coords_matched=X_coords_plot,
            y_tier_f=y_f,
        )


def plot_he_f1_extra_tiers(plot_per_class_f1, result_fig, model, device, train_dataset, val_dataset, input_dim, best_epoch, g=None):
    g = g or {}
    for title, lk, pk, nk, fig_key, mname in (
        ("L12 sublineage", "val_labels_level12", "val_preds_level12", "class_names_level12", "f1_perclass_level12", "MLP-Level12"),
        ("L3 CNiche", "val_labels_level3", "val_preds_level3", "class_names_level3", "f1_perclass_level3", "MLP-Level3"),
        ("L4 TNiche", "val_labels_level4", "val_preds_level4", "class_names_level4", "f1_perclass_level4", "MLP-Level4"),
    ):
        y_true, y_pred = g.get(lk), g.get(pk)
        if y_pred is None:
            print(f"Skip {title} per-class F1: no {pk}")
            continue
        names = tier_class_names(nk, None, g)
        if names is None:
            print(f"Skip {title} per-class F1: missing {nk}")
            continue
        m = metrics_triplet(y_true, y_pred)
        print(f"\n{title}")
        plot_per_class_f1(
            y_true,
            y_pred,
            names,
            model=model,
            device=device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_dim=input_dim,
            best_epoch=best_epoch,
            test_acc=m["accuracy"],
            test_macro_f1=m["macro_f1"],
            test_weighted_f1=m["weighted_f1"],
            model_name=mname,
            y_sort_by="f1_asc",
            save_path=result_fig(fig_key),
        )


_STARDIST_Y_KEYS = ("y_star_level12", "y_star_level3", "y_star_level4")


def _stardist_extra_tier_tag(title: str) -> str:
    if "L12" in title:
        return "l12"
    if "L3" in title:
        return "l3"
    return "l4"


def stardist_tier_labels(head_preds, cv_data, g, tiers=None):
    g = g or {}
    tiers_set = None if tiers is None else {str(t).lower() for t in tiers}
    for (title, pk, nk, _yk, lek), y_star_key in zip(STARDIST_EXTRA_SPECS, _STARDIST_Y_KEYS):
        if tiers_set is not None and _stardist_extra_tier_tag(title) not in tiers_set:
            continue
        if pk not in head_preds:
            print(f"Skip {title}: missing {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        le = cv_data.get(lek) if cv_data is not None else None
        y_raw = g.get(y_star_key)
        if names is None or le is None or y_raw is None:
            print(f"Skip {title}: need {nk}, {lek}, {y_star_key}")
            continue
        y_true = encode_star_labels(y_raw, le)
        y_pred = np.asarray(head_preds[pk], dtype=np.int64)
        n = min(len(y_true), len(y_pred))
        yield title, y_true[:n], y_pred[:n], names


def plot_stardist_confusion_extra(plot_confusion_matrix, result_fig, head_preds, cv_data, g=None, tiers=None):
    g = g or {}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g, tiers=tiers):
        figsize = (6, 5) if "L12" in title else (8, 7)
        fig_key = "conf_matrix_level12_stardist" if "L12" in title else (
            "conf_matrix_level3_stardist" if "L3" in title else "conf_matrix_level4_stardist"
        )
        print(f"\n{title} (StarDist) confusion matrix")
        plot_confusion_matrix(y_true, y_pred, names, figsize=figsize, save_path=result_fig(fig_key))
        print(f"Saved: {result_fig(fig_key)}")


def plot_stardist_f1_extra(plot_per_class_f1, result_fig, model_star, device, head_preds, cv_data, g=None, tiers=None):
    g = g or {}
    key_map = {"L12": "f1_level12_stardist", "L3": "f1_level3_stardist", "L4": "f1_level4_stardist"}
    name_map = {"L12": "MLP-StarDist-Level12", "L3": "MLP-StarDist-Level3", "L4": "MLP-StarDist-Level4"}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g, tiers=tiers):
        tag = "L12" if "L12" in title else ("L3" if "L3" in title else "L4")
        m = metrics_triplet(y_true, y_pred)
        print(f"\n{title} (StarDist) per-class F1")
        plot_per_class_f1(
            y_true,
            y_pred,
            names,
            model=model_star,
            device=device,
            test_acc=m["accuracy"],
            test_macro_f1=m["macro_f1"],
            test_weighted_f1=m["weighted_f1"],
            model_name=name_map[tag],
            y_sort_by="f1_asc",
            save_path=result_fig(key_map[tag]),
        )
        print(f"Saved: {result_fig(key_map[tag])}")


def plot_stardist_acc_extra(plot_per_class_accuracy, result_fig, model_star, device, head_preds, cv_data, g=None, tiers=None):
    g = g or {}
    key_map = {"L12": "acc_level12_stardist", "L3": "acc_level3_stardist", "L4": "acc_level4_stardist"}
    name_map = {"L12": "MLP-StarDist-Level12", "L3": "MLP-StarDist-Level3", "L4": "MLP-StarDist-Level4"}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g, tiers=tiers):
        tag = "L12" if "L12" in title else ("L3" if "L3" in title else "L4")
        m = metrics_triplet(y_true, y_pred)
        print(f"\n{title} (StarDist): acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
        plot_per_class_accuracy(
            y_true,
            y_pred,
            names,
            model=model_star,
            device=device,
            test_acc=m["accuracy"],
            test_macro_f1=m["macro_f1"],
            test_weighted_f1=m["weighted_f1"],
            model_name=name_map[tag],
            y_sort_by="acc_asc",
            save_path=result_fig(key_map[tag]),
        )
        print(f"Saved: {result_fig(key_map[tag])}")


def plot_stardist_spatial_extra(
    plot_tier_spatial_distribution,
    plot_celltype_spatial_distribution,
    result_fig,
    therapy_data,
    X_coords_star,
    head_preds,
    cv_data,
    g=None,
    tiers=None,
    *,
    spatial_color_scheme=None,
    pan_organ=None,
):
    """Plot selected extra tiers using per-tier defaults or caller overrides."""
    g = g or {}
    fig_map = {"L12": ("stardist_pred_level12", "stardist_true_level12"), "L3": ("stardist_pred_level3", "stardist_true_level3"), "L4": ("stardist_pred_level4", "stardist_true_level4")}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g, tiers=tiers):
        tag = "L12" if "L12" in title else ("L3" if "L3" in title else "L4")
        y_key = f"y_star_{'level12' if tag == 'L12' else 'level3' if tag == 'L3' else 'level4'}"
        print(f"\n{title} (StarDist)")
        plot_tier_spatial_distribution(
            pred_encoded=y_pred,
            class_names=names,
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            save_path_pred=result_fig(fig_map[tag][0]),
            save_path_true=result_fig(fig_map[tag][1]),
            spatial_color_scheme=_extra_tier_color_scheme(
                spatial_color_scheme, tag.lower(), pan_organ=pan_organ
            ),
            pan_organ=pan_organ,
            title_pred=f"{therapy_data} StarDist pred {title}",
            title_true=f"{therapy_data} ground truth {title}",
            fig_size=(10, 8),
            show=True,
            X_coords_matched=X_coords_star,
            y_tier_f=g.get(y_key),
        )
        print(f"Saved: {result_fig(fig_map[tag][0])}")


###############################################################
# 2026.08.13, add plot_stardist_roc_extra for HCC dataset
###############################################################
def plot_stardist_roc_extra(
    plot_multiclass_roc_curves,
    result_fig,
    head_probs,
    cv_data,
    m,
    g=None,
    tiers=None,
    *,
    roc_color_scheme=None,
    pan_organ=None,
):
    """Plot selected extra-tier ROC curves with tier-aware color schemes."""
    g = g or {}
    tiers_set = None if tiers is None else {str(t).lower() for t in tiers}
    roc_specs = (
        ("L12 sublineage", "l12", "class_names_level12", "y_star_level12", "le_level12", "roc_stardist_level12", "StarDist Level12 ROC"),
        ("L3 CNiche", "l3", "class_names_level3", "y_star_level3", "le_level3", "roc_stardist_level3", "StarDist Level3 (CNiche) ROC"),
        ("L4 TNiche", "l4", "class_names_level4", "y_star_level4", "le_level4", "roc_stardist_level4", "StarDist Level4 (TNiche) ROC"),
    )
    for title, pk, nk, yk, lek, fig_key, roc_title in roc_specs:
        if tiers_set is not None and pk not in tiers_set:
            continue
        if pk not in head_probs:
            print(f"Skip {title} ROC: missing probs {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        le = cv_data.get(lek) if cv_data is not None else None
        y_raw = g.get(yk)
        if names is None or le is None or y_raw is None:
            print(f"Skip {title} ROC: need {nk}, {lek}, {yk}")
            continue
        y_true = encode_star_labels(y_raw, le)
        scores = head_probs[pk]
        mm = min(len(y_true), len(scores), m)
        print(f"\n{title} (StarDist) ROC")
        plot_multiclass_roc_curves(
            y_true[:mm],
            scores[:mm],
            names,
            figsize=(3.0, 3.0),
            max_curves=len(names),
            save_path=result_fig(fig_key),
            title=roc_title,
            roc_color_scheme=_extra_tier_color_scheme(
                roc_color_scheme, pk, pan_organ=pan_organ
            ),
            pan_organ=pan_organ,
        )
        print(f"Saved: {result_fig(fig_key)}")


def build_insample_tier_metrics(g=None, cv_data=None):
    """Mirror notebook insample_tiers + _num_classes for save_hce_validation_metrics."""
    g = g or {}

    def tri(lk, pk):
        yt, yp = g.get(lk), g.get(pk)
        if yt is None or yp is None:
            return None
        return metrics_triplet(yt, yp)

    insample = {
        "l2": tri("val_labels", "val_preds"),
        "l1": tri("val_labels_level1", "val_preds_level1"),
        "l1_head": tri("val_labels_level1", "val_preds_level1_head"),
        "l12": tri("val_labels_level12", "val_preds_level12"),
        "l3": tri("val_labels_level3", "val_preds_level3"),
        "l4": tri("val_labels_level4", "val_preds_level4"),
    }
    num_classes = {}
    if g.get("class_names") is not None:
        num_classes["l2"] = len(g["class_names"])
    if g.get("class_names_level1") is not None:
        num_classes["l1"] = len(g["class_names_level1"])
        num_classes["l1_head"] = len(g["class_names_level1"])
    for tier, cn in (("l12", "class_names_level12"), ("l3", "class_names_level3"), ("l4", "class_names_level4")):
        arr = tier_class_names(cn, cv_data, g)
        if arr is not None:
            num_classes[tier] = len(arr)
    return insample, num_classes


########################################################
## 2026.06.24 LLY, save Stardist predicted metrics to CSV file 
########################################################
def _stardist_l1_from_l2_triplet(
    matched_features_path,
    all_preds,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
):
    """Level-1 metrics by aggregating level-2 argmax preds (StarDist matched bundle)."""
    from base import load_matched_features_bundle

    loaded = load_matched_features_bundle(matched_features_path)
    if "y_level1" not in loaded:
        return None

    y_true_l1_raw = loaded["y_level1"]
    num_l2 = len(class_names)
    child_to_parent = np.full(num_l2, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent hierarchy mapping for level2 class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    preds_l2 = np.asarray(all_preds, dtype=np.int64)
    preds_l1 = child_to_parent[preds_l2]

    if np.asarray(y_true_l1_raw).dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array(
            [name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64
        )
    else:
        y_true_l1 = np.asarray(y_true_l1_raw, dtype=np.int64)

    n = min(len(y_true_l1), len(preds_l1))
    valid = y_true_l1[:n] >= 0
    if not np.any(valid):
        return None
    return metrics_triplet(y_true_l1[:n][valid], preds_l1[:n][valid])


def _stardist_l1_head_triplet(head_preds, matched_features_path, class_names_level1):
    """Level-1 metrics from the dedicated L1 logits head."""
    from base import load_matched_features_bundle

    if head_preds is None or "preds_l1" not in head_preds:
        return None
    loaded = load_matched_features_bundle(matched_features_path)
    if "y_level1" not in loaded:
        return None

    y_true_l1_raw = loaded["y_level1"]
    preds_l1 = np.asarray(head_preds["preds_l1"], dtype=np.int64)
    n_l1 = len(class_names_level1)

    if np.asarray(y_true_l1_raw).dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array(
            [name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64
        )
    else:
        y_true_l1 = np.asarray(y_true_l1_raw, dtype=np.int64)

    n = min(len(y_true_l1), len(preds_l1))
    valid = (y_true_l1[:n] >= 0) & (preds_l1[:n] >= 0) & (preds_l1[:n] < n_l1)
    if not np.any(valid):
        return None
    return metrics_triplet(y_true_l1[:n][valid], preds_l1[:n][valid])


def build_stardist_matched_tier_metrics(
    *,
    all_labels,
    all_preds,
    matched_features_path,
    class_names_star,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
    predict_all_label_heads=None,
    cv_data=None,
    g=None,
):
    """
    Compute per-tier accuracy / macro-F1 / weighted-F1 on StarDist-matched h5ad.

    Returns ``(insample, num_classes)`` in the same shape as ``build_insample_tier_metrics``
    so ``collect_hce_tier_metrics_extras`` can write ``insample_*`` CSV columns.
    """
    g = g or {}
    insample = {}
    num_classes = {}

    if class_names_star is not None:
        num_classes["l2"] = len(class_names_star)
    if class_names_level1 is not None:
        num_classes["l1"] = len(class_names_level1)
        num_classes["l1_head"] = len(class_names_level1)

    if all_labels is not None and all_preds is not None:
        labels = np.asarray(all_labels, dtype=np.int64)
        preds = np.asarray(all_preds, dtype=np.int64)
        max_idx = len(class_names_star) - 1
        valid = (
            (labels >= 0)
            & (labels <= max_idx)
            & (preds >= 0)
            & (preds <= max_idx)
        )
        if np.any(valid):
            insample["l2"] = metrics_triplet(labels[valid], preds[valid])

    l1_tri = _stardist_l1_from_l2_triplet(
        matched_features_path,
        all_preds,
        class_names_star,
        class_names_level1,
        y_encoded_f,
        y_level1_encoded_f,
    )
    if l1_tri is not None:
        insample["l1"] = l1_tri

    head_preds = None
    if (
        model_star is not None
        and scaler is not None
        and X_star is not None
        and device is not None
        and predict_all_label_heads is not None
    ):
        head_preds = stardist_head_preds(
            model_star, scaler, X_star, device, predict_all_label_heads
        )

    l1_head_tri = _stardist_l1_head_triplet(
        head_preds, matched_features_path, class_names_level1
    )
    if l1_head_tri is not None:
        insample["l1_head"] = l1_head_tri

    if head_preds is not None:
        for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g):
            if "L12" in title:
                insample["l12"] = metrics_triplet(y_true, y_pred)
                num_classes["l12"] = len(names)
            elif "L3" in title:
                insample["l3"] = metrics_triplet(y_true, y_pred)
                num_classes["l3"] = len(names)
            elif "L4" in title:
                insample["l4"] = metrics_triplet(y_true, y_pred)
                num_classes["l4"] = len(names)

    return insample, num_classes


def macro_auc_ovr(y_true, y_score, n_classes):
    """Multiclass macro AUROC: mean of evaluable one-vs-rest class AUCs."""
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = min(len(y_true), len(y_score))
    y_true, y_score = y_true[:n], y_score[:n]
    valid = (y_true >= 0) & (y_true < int(n_classes))
    if not np.any(valid):
        return float("nan")
    y_true = y_true[valid]
    y_score = y_score[valid]

    Y = label_binarize(y_true, classes=np.arange(int(n_classes)))
    aucs = []
    for k in range(int(n_classes)):
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == len(Y):
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], y_score[:, k], drop_intermediate=True)
        aucs.append(float(auc(fpr, tpr)))
    return float(np.mean(aucs)) if aucs else float("nan")


def _child_to_parent_map(class_names, y_encoded_f, y_level1_encoded_f):
    num_l2 = len(class_names)
    child_to_parent = np.full(num_l2, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent hierarchy mapping for level2 class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")
    return child_to_parent


def _encode_level1_labels(y_raw, class_names_level1):
    y_raw = np.asarray(y_raw)
    if y_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        return np.array([name_to_idx.get(str(v), -1) for v in y_raw], dtype=np.int64)
    return y_raw.astype(np.int64)


def collect_tier_auc_extras(tier_aucs, prefix="insample"):
    """Flat CSV columns: ``{prefix}_{tier}_macro_auc`` for each tier."""
    row = {}
    for tier, val in (tier_aucs or {}).items():
        if val is None:
            continue
        v = float(val)
        if np.isfinite(v):
            row[f"{prefix}_{tier}_macro_auc"] = v
    return row


def build_stardist_matched_tier_aucs(
    *,
    matched_features_path,
    class_names_star,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
    cv_data=None,
    g=None,
    probs_l2=None,
    head_probs=None,
):
    """
    Compute macro AUROC per tier on StarDist-matched data (same alignment as ROC plots).
    """
    from base import encode_labels_with_class_names, load_matched_features_bundle
    from plot import _aggregate_l2_probs_to_l1, mlp_collect_five_head_softmax_probs, mlp_collect_softmax_probs

    g = g or {}
    tier_aucs = {}
    loaded = load_matched_features_bundle(matched_features_path)

    x_arr = X_star
    if x_arr is None and "X" in loaded:
        x_arr = loaded["X"]

    if probs_l2 is None:
        if model_star is None or scaler is None or x_arr is None or device is None:
            print(
                "  ⚠ Skip macro AUROC: pass probs_l2 (run ROC cell) or ensure "
                "model_star + scaler + X_star/device are available.",
                flush=True,
            )
            return tier_aucs
        probs_l2 = mlp_collect_softmax_probs(model_star, scaler.transform(x_arr), device)

    probs_l2 = np.asarray(probs_l2, dtype=np.float64)
    m = len(probs_l2)

    if head_probs is not None:
        first_scores = next(iter(head_probs.values()), None)
        if first_scores is not None and len(first_scores) < m:
            print(
                "  ⚠ head_probs length < full StarDist rows; recomputing five-head softmax.",
                flush=True,
            )
            head_probs = None

    if "y" in loaded:
        y_enc = encode_labels_with_class_names(loaded["y"], class_names_star)
        m = min(m, len(y_enc))
        valid = (y_enc[:m] >= 0) & (y_enc[:m] < len(class_names_star))
        if np.any(valid):
            tier_aucs["l2"] = macro_auc_ovr(
                y_enc[:m][valid], probs_l2[:m][valid], len(class_names_star)
            )

    if "y_level1" in loaded:
        child_to_parent = _child_to_parent_map(
            class_names_star, y_encoded_f, y_level1_encoded_f
        )
        probs_l1 = _aggregate_l2_probs_to_l1(
            probs_l2[:m], child_to_parent, len(class_names_level1)
        )
        y_l1 = _encode_level1_labels(loaded["y_level1"][:m], class_names_level1)
        valid_l1 = (y_l1 >= 0) & (y_l1 < len(class_names_level1))
        if np.any(valid_l1):
            tier_aucs["l1"] = macro_auc_ovr(
                y_l1[valid_l1], probs_l1[valid_l1], len(class_names_level1)
            )

    if head_probs is None and model_star is not None and scaler is not None and x_arr is not None and device is not None:
        x_scaled = scaler.transform(x_arr)[:m]
        head_probs = mlp_collect_five_head_softmax_probs(model_star, x_scaled, device)

    if head_probs and "l1" in head_probs and "y_level1" in loaded:
        y_l1 = _encode_level1_labels(loaded["y_level1"][:m], class_names_level1)
        scores = np.asarray(head_probs["l1"], dtype=np.float64)[:m]
        valid = (y_l1 >= 0) & (y_l1 < len(class_names_level1))
        if np.any(valid):
            tier_aucs["l1_head"] = macro_auc_ovr(
                y_l1[valid], scores[valid], len(class_names_level1)
            )

    if head_probs:
        auc_specs = (
            ("l12", "l12", "class_names_level12", "y_star_level12", "le_level12"),
            ("l3", "l3", "class_names_level3", "y_star_level3", "le_level3"),
            ("l4", "l4", "class_names_level4", "y_star_level4", "le_level4"),
        )
        for tier, pk, nk, yk, lek in auc_specs:
            if pk not in head_probs:
                continue
            names = tier_class_names(nk, cv_data, g)
            le = cv_data.get(lek) if cv_data is not None else None
            y_raw = g.get(yk)
            if names is None or le is None or y_raw is None:
                continue
            y_true = encode_star_labels(y_raw, le)
            scores = np.asarray(head_probs[pk], dtype=np.float64)
            mm = min(len(y_true), len(scores), m)
            tier_aucs[tier] = macro_auc_ovr(y_true[:mm], scores[:mm], len(names))

    return tier_aucs


########################################################
## 2026.06.24 LLY, plot summary ROC from AUROC CSV of 25 datasets
########################################################
STARDIST_AUROC_TIER_SPECS: dict[str, dict[str, str]] = {
    "l2": {
        "csv_name": "validation_external_stardist_matched_AUROC.csv",
        "names_col": "final_CT",
        "gt_col": "ground_truth_final_CT",
        "pred_col": "predict_stardist_final_CT",
        "roc_color_scheme": "xenium_lung_fine",
        "concat_stem": "validation_external_stardist_matched_AUROC_all_samples",
        "pooled_title": "StarDist Level2 ROC — cross-dataset model (pooled cells)",
    },
    "l1": {
        "csv_name": "validation_external_stardist_matched_AUROC_level1.csv",
        "names_col": "final_lineage",
        "gt_col": "ground_truth_final_lineage",
        "pred_col": "predict_stardist_final_lineage",
        "roc_color_scheme": "xenium_lung_coarse",
        "concat_stem": "validation_external_stardist_matched_AUROC_level1_all_samples",
        "pooled_title": "StarDist Level1 ROC from L2 probs — cross-dataset model (pooled cells)",
    },
    "l12": {
        "csv_name": "validation_external_stardist_matched_AUROC_level12.csv",
        "names_col": "final_sublineage",
        "gt_col": "ground_truth_final_sublineage",
        "pred_col": "predict_stardist_final_sublineage",
        "roc_color_scheme": "xenium_lung_intermediate",
        "concat_stem": "validation_external_stardist_matched_AUROC_level12_all_samples",
        "pooled_title": "StarDist Level12 ROC — cross-dataset model (pooled cells)",
    },
    "l3": {
        "csv_name": "validation_external_stardist_matched_AUROC_level3.csv",
        "names_col": "CNiche",
        "gt_col": "ground_truth_CNiche",
        "pred_col": "predict_stardist_CNiche",
        "roc_color_scheme": "xenium_lung_CNiche",
        "concat_stem": "validation_external_stardist_matched_AUROC_level3_all_samples",
        "pooled_title": "StarDist Level3 (CNiche) ROC — cross-dataset model (pooled cells)",
    },
    "l4": {
        "csv_name": "validation_external_stardist_matched_AUROC_level4.csv",
        "names_col": "TNiche",
        "gt_col": "ground_truth_TNiche",
        "pred_col": "predict_stardist_TNiche",
        "roc_color_scheme": "xenium_lung_TNiche",
        "concat_stem": "validation_external_stardist_matched_AUROC_level4_all_samples",
        "pooled_title": "StarDist Level4 (TNiche) ROC — cross-dataset model (pooled cells)",
    },
}


def stardist_auroc_tier_spec(tier: str) -> dict[str, str]:
    key = str(tier).lower()
    if key not in STARDIST_AUROC_TIER_SPECS:
        raise ValueError(
            f"Unknown AUROC tier {tier!r}; choose from {sorted(STARDIST_AUROC_TIER_SPECS)}"
        )
    return STARDIST_AUROC_TIER_SPECS[key]


def stardist_tier_auroc_csv_path(result_fig, tier: str) -> str:
    """Per-sample AUROC CSV path under the StarDist result directory."""
    spec = stardist_auroc_tier_spec(tier)
    parent = Path(result_fig("roc_stardist_level2")).parent
    parent.mkdir(parents=True, exist_ok=True)
    return str(parent / spec["csv_name"])


def _auroc_csv_path_from_metrics(metrics_csv_path: str) -> str:
    """``..._metrics.csv`` -> ``..._AUROC.csv`` in the same directory."""
    p = Path(metrics_csv_path)
    if p.name.endswith("_metrics.csv"):
        return str(p.with_name(p.name.replace("_metrics.csv", "_AUROC.csv")))
    return str(p.with_name("validation_external_stardist_matched_AUROC.csv"))


def build_stardist_level2_auroc_table(
    *,
    matched_features_path,
    class_names_star,
    all_labels=None,
    all_preds=None,
    probs_l2=None,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
):
    """
    Per-cell Level2 table for ROC replay: ground truth / StarDist prediction labels
    plus ``prob_0`` … ``prob_{C-1}`` (softmax scores in ``class_names_star`` order).
    """
    import pandas as pd
    from base import encode_labels_with_class_names, load_matched_features_bundle
    from plot import mlp_collect_softmax_probs

    class_names = np.asarray(class_names_star)
    n_classes = len(class_names)
    loaded = load_matched_features_bundle(matched_features_path)

    if all_preds is None:
        raise ValueError("all_preds is required for StarDist AUROC table.")
    preds = np.asarray(all_preds, dtype=np.int64)

    if all_labels is not None:
        y_enc = np.asarray(all_labels, dtype=np.int64)
    elif "y" in loaded:
        y_enc = encode_labels_with_class_names(loaded["y"], class_names)
    else:
        y_enc = np.full(len(preds), -1, dtype=np.int64)

    x_arr = X_star
    if x_arr is None and "X" in loaded:
        x_arr = loaded["X"]

    if probs_l2 is None:
        if model_star is None or scaler is None or x_arr is None or device is None:
            raise ValueError(
                "Need probs_l2 or model_star + scaler + X_star/device for AUROC table."
            )
        probs_l2 = mlp_collect_softmax_probs(model_star, scaler.transform(x_arr), device)
    probs_l2 = np.asarray(probs_l2, dtype=np.float64)

    m = min(len(preds), len(y_enc), len(probs_l2))
    preds = preds[:m]
    y_enc = y_enc[:m]
    probs_l2 = probs_l2[:m]

    if "y" in loaded:
        gt_raw = np.asarray(loaded["y"][:m], dtype=object)
    else:
        gt_raw = np.array(
            [
                str(class_names[i]) if 0 <= i < n_classes else ""
                for i in y_enc
            ],
            dtype=object,
        )

    def _idx_to_name(idx: int) -> str:
        if 0 <= idx < n_classes:
            return str(class_names[idx])
        return ""

    pred_names = np.array([_idx_to_name(int(p)) for p in preds], dtype=object)
    gt_names = np.array(
        [
            str(g) if str(g) not in ("", "nan") else _idx_to_name(int(y))
            for g, y in zip(gt_raw, y_enc)
        ],
        dtype=object,
    )

    data = {}
    if str(matched_features_path).endswith(".h5ad"):
        import anndata as ad

        adata = ad.read_h5ad(str(matched_features_path))
        data["cell_id"] = adata.obs_names.to_numpy()[:m]

    data.update(
        {
            "ground_truth_final_CT": gt_names,
            "predict_stardist_final_CT": pred_names,
            "ground_truth_idx": y_enc,
            "predict_stardist_idx": preds,
        }
    )

    for j in range(n_classes):
        data[f"prob_{j}"] = probs_l2[:, j]

    return pd.DataFrame(data)


def _dataframe_from_tier_scores(
    *,
    y_enc: np.ndarray,
    probs: np.ndarray,
    class_names,
    gt_col: str,
    pred_col: str,
    matched_features_path,
    gt_names_raw=None,
):
    import pandas as pd

    class_names = np.asarray(class_names)
    n_classes = len(class_names)
    y_enc = np.asarray(y_enc, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    m = min(len(y_enc), len(probs))
    y_enc = y_enc[:m]
    probs = probs[:m]
    preds = np.argmax(probs, axis=1)

    def _idx_to_name(idx: int) -> str:
        if 0 <= idx < n_classes:
            return str(class_names[idx])
        return ""

    if gt_names_raw is not None:
        gt_names = np.asarray(gt_names_raw[:m], dtype=object)
        gt_names = np.array(
            [
                str(g) if str(g) not in ("", "nan") else _idx_to_name(int(y))
                for g, y in zip(gt_names, y_enc)
            ],
            dtype=object,
        )
    else:
        gt_names = np.array([_idx_to_name(int(y)) for y in y_enc], dtype=object)

    pred_names = np.array([_idx_to_name(int(p)) for p in preds], dtype=object)
    data: dict = {}
    if str(matched_features_path).endswith(".h5ad"):
        import anndata as ad

        adata = ad.read_h5ad(str(matched_features_path))
        data["cell_id"] = adata.obs_names.to_numpy()[:m]

    data.update(
        {
            gt_col: gt_names,
            pred_col: pred_names,
            "ground_truth_idx": y_enc,
            "predict_stardist_idx": preds,
        }
    )
    for j in range(n_classes):
        data[f"prob_{j}"] = probs[:, j]
    return pd.DataFrame(data)


def _build_stardist_level1_auroc_table_from_l2_probs(
    *,
    matched_features_path,
    class_names_star,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    probs_l2,
):
    from base import load_matched_features_bundle
    from plot import _aggregate_l2_probs_to_l1

    spec = stardist_auroc_tier_spec("l1")
    loaded = load_matched_features_bundle(matched_features_path)
    if "y_level1" not in loaded:
        raise ValueError(f"y_level1 not found in {matched_features_path}")
    child_to_parent = _child_to_parent_map(
        class_names_star, y_encoded_f, y_level1_encoded_f
    )
    probs_l2 = np.asarray(probs_l2, dtype=np.float64)
    m = len(probs_l2)
    probs_l1 = _aggregate_l2_probs_to_l1(
        probs_l2[:m], child_to_parent, len(class_names_level1)
    )
    y_l1 = _encode_level1_labels(loaded["y_level1"][:m], class_names_level1)
    return _dataframe_from_tier_scores(
        y_enc=y_l1,
        probs=probs_l1,
        class_names=class_names_level1,
        gt_col=spec["gt_col"],
        pred_col=spec["pred_col"],
        matched_features_path=matched_features_path,
        gt_names_raw=loaded["y_level1"][:m],
    )


def _build_stardist_extra_tier_auroc_table(
    tier: str,
    *,
    matched_features_path,
    head_probs,
    cv_data=None,
    g=None,
    m: int | None = None,
):
    g = g or {}
    tier_specs = {
        "l12": ("l12", "class_names_level12", "y_star_level12", "le_level12"),
        "l3": ("l3", "class_names_level3", "y_star_level3", "le_level3"),
        "l4": ("l4", "class_names_level4", "y_star_level4", "le_level4"),
    }
    pk, nk, yk, lek = tier_specs[tier]
    spec = stardist_auroc_tier_spec(tier)
    if pk not in head_probs:
        raise ValueError(f"Missing head_probs[{pk!r}] for tier {tier!r}.")
    names = tier_class_names(nk, cv_data, g)
    le = cv_data.get(lek) if cv_data is not None else None
    y_raw = g.get(yk)
    if names is None or le is None or y_raw is None:
        raise ValueError(f"Need {nk}, {lek}, {yk} for tier {tier!r} AUROC table.")
    scores = np.asarray(head_probs[pk], dtype=np.float64)
    mm = min(len(y_raw), len(scores), m if m is not None else len(scores))
    y_true = encode_star_labels(y_raw[:mm], le)
    return _dataframe_from_tier_scores(
        y_enc=y_true,
        probs=scores[:mm],
        class_names=names,
        gt_col=spec["gt_col"],
        pred_col=spec["pred_col"],
        matched_features_path=matched_features_path,
        gt_names_raw=y_raw[:mm],
    )


def _tier_class_names_from_table(tier, df, class_names_star, class_names_level1, cv_data, g):
    key = str(tier).lower()
    n_prob = sum(1 for c in df.columns if c.startswith("prob_"))
    if key == "l2":
        return np.asarray(class_names_star)[:n_prob]
    if key == "l1":
        return np.asarray(class_names_level1)[:n_prob]
    tier_name_keys = {
        "l12": "class_names_level12",
        "l3": "class_names_level3",
        "l4": "class_names_level4",
    }
    names = tier_class_names(tier_name_keys[key], cv_data, g)
    return np.asarray(names)[:n_prob]


def build_stardist_tier_auroc_table(
    tier: str,
    *,
    matched_features_path,
    class_names_star=None,
    class_names_level1=None,
    y_encoded_f=None,
    y_level1_encoded_f=None,
    all_labels=None,
    all_preds=None,
    probs_l2=None,
    head_probs=None,
    cv_data=None,
    g=None,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
    m: int | None = None,
):
    """Build per-cell AUROC replay table for StarDist tier ``l2|l1|l12|l3|l4``."""
    key = str(tier).lower()
    if key == "l2":
        if class_names_star is None:
            raise ValueError("class_names_star is required for L2 AUROC table.")
        return build_stardist_level2_auroc_table(
            matched_features_path=matched_features_path,
            class_names_star=class_names_star,
            all_labels=all_labels,
            all_preds=all_preds,
            probs_l2=probs_l2,
            model_star=model_star,
            scaler=scaler,
            X_star=X_star,
            device=device,
        )
    if key == "l1":
        if probs_l2 is None:
            raise ValueError("probs_l2 is required for L1 AUROC table.")
        return _build_stardist_level1_auroc_table_from_l2_probs(
            matched_features_path=matched_features_path,
            class_names_star=class_names_star,
            class_names_level1=class_names_level1,
            y_encoded_f=y_encoded_f,
            y_level1_encoded_f=y_level1_encoded_f,
            probs_l2=probs_l2 if m is None else probs_l2[:m],
        )
    if key in ("l12", "l3", "l4"):
        if head_probs is None:
            raise ValueError(f"head_probs is required for {key} AUROC table.")
        return _build_stardist_extra_tier_auroc_table(
            key,
            matched_features_path=matched_features_path,
            head_probs=head_probs,
            cv_data=cv_data,
            g=g,
            m=m,
        )
    raise ValueError(f"Unknown tier {tier!r}")


def save_stardist_external_auroc_tier_csv(
    tier: str,
    *,
    auroc_csv_path,
    matched_features_path,
    class_names_star=None,
    class_names_level1=None,
    y_encoded_f=None,
    y_level1_encoded_f=None,
    all_labels=None,
    all_preds=None,
    probs_l2=None,
    head_probs=None,
    cv_data=None,
    g=None,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
    m: int | None = None,
):
    """Write per-sample ``validation_external_stardist_matched_AUROC[_level*].csv``."""
    import pandas as pd

    spec = stardist_auroc_tier_spec(tier)
    df = build_stardist_tier_auroc_table(
        tier,
        matched_features_path=matched_features_path,
        class_names_star=class_names_star,
        class_names_level1=class_names_level1,
        y_encoded_f=y_encoded_f,
        y_level1_encoded_f=y_level1_encoded_f,
        all_labels=all_labels,
        all_preds=all_preds,
        probs_l2=probs_l2,
        head_probs=head_probs,
        cv_data=cv_data,
        g=g,
        model_star=model_star,
        scaler=scaler,
        X_star=X_star,
        device=device,
        m=m,
    )
    class_names = _tier_class_names_from_table(
        tier, df, class_names_star, class_names_level1, cv_data, g
    )
    out = Path(auroc_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_prob = sum(1 for c in df.columns if c.startswith("prob_"))
    tier_label = str(tier).upper()
    print(
        f"Saved StarDist {tier_label} AUROC table: {out} "
        f"({len(df):,} cells; prob_0..prob_{n_prob - 1} match class_names order)",
        flush=True,
    )
    names_path = out.with_name(out.stem + "_class_names.csv")
    pd.DataFrame(
        {
            "class_index": np.arange(len(class_names)),
            spec["names_col"]: class_names,
        }
    ).to_csv(names_path, index=False)
    print(f"  Class name order: {names_path}", flush=True)
    return df


def save_stardist_external_auroc_level2_csv(
    *,
    matched_features_path,
    class_names_star,
    all_labels=None,
    all_preds=None,
    auroc_csv_path,
    probs_l2=None,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
):
    """Write ``validation_external_stardist_matched_AUROC.csv`` for Level2 ROC replay."""
    return save_stardist_external_auroc_tier_csv(
        "l2",
        auroc_csv_path=auroc_csv_path,
        matched_features_path=matched_features_path,
        class_names_star=class_names_star,
        all_labels=all_labels,
        all_preds=all_preds,
        probs_l2=probs_l2,
        model_star=model_star,
        scaler=scaler,
        X_star=X_star,
        device=device,
    )


def plot_stardist_roc_from_auroc_csv(
    auroc_csv_path,
    class_names_star,
    *,
    save_path=None,
    title="StarDist Level2 ROC (from AUROC CSV)",
    roc_color_scheme="xenium_lung_fine",
    pan_organ=None,
    figsize=(3.0, 3.0),
    max_curves=None,
):
    """Replay ``plot_multiclass_roc_curves`` using a saved AUROC table."""
    import pandas as pd
    from plot import plot_multiclass_roc_curves

    df = pd.read_csv(auroc_csv_path)
    prob_cols = sorted(
        [c for c in df.columns if c.startswith("prob_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    class_names = np.asarray(class_names_star)
    if len(prob_cols) != len(class_names):
        raise ValueError(
            f"AUROC CSV has {len(prob_cols)} prob_* columns but "
            f"{len(class_names)} class names."
        )
    y_true = df["ground_truth_idx"].to_numpy(dtype=np.int64)
    y_score = df[prob_cols].to_numpy(dtype=np.float64)
    eff_scheme = roc_color_scheme
    if pan_organ is not None and str(roc_color_scheme).strip().lower() in (
        "xenium_lung_fine",
        "xenium_ct",
    ):
        eff_scheme = scheme_for_pan_organ(pan_organ, head="l2")
    return plot_multiclass_roc_curves(
        y_true,
        y_score,
        class_names,
        figsize=figsize,
        max_curves=max_curves or len(class_names),
        save_path=save_path,
        title=title,
        roc_color_scheme=eff_scheme,
        pan_organ=pan_organ,
    )


def discover_stardist_auroc_sample_paths(
    cases_root,
    *,
    model_suffix: str = "_project_all_UNI",
    auroc_name: str = "validation_external_stardist_matched_AUROC.csv",
    layout: str = "per_sample",
):
    """
    Return ``[(sample, auroc_csv, class_names_csv), ...]`` for samples with both files.

    layout
    ------
    ``per_sample`` (default): ``{cases_root}/{sample}/{sample}{model_suffix}/result/...``
    ``pooled_stardist``: cross-dataset outputs under ``{cases_root}/{sample}/...``
    (``cases_root`` = ``Data/result/stardist``).
    """
    cases_root = Path(cases_root)
    found = []
    missing = []
    for sample_dir in sorted(cases_root.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith(("_", ".")):
            continue
        sample = sample_dir.name
        if layout == "pooled_stardist":
            result_dir = sample_dir
        elif layout == "per_sample":
            result_dir = sample_dir / f"{sample}{model_suffix}" / "result"
        else:
            raise ValueError(
                "layout must be one of: per_sample, pooled_stardist"
            )
        auroc_csv = result_dir / auroc_name
        names_csv = result_dir / auroc_name.replace(".csv", "_class_names.csv")
        if auroc_csv.is_file() and names_csv.is_file():
            found.append((sample, auroc_csv, names_csv))
        elif layout == "per_sample":
            missing.append(sample)
    if layout == "pooled_stardist":
        missing = []
    return found, missing


def build_global_final_ct_class_names(sample_paths):
    """Sorted union of ``final_CT`` across per-sample ``*_class_names.csv`` files."""
    return build_global_tier_class_names(sample_paths, names_col="final_CT")


def build_global_tier_class_names(sample_paths, *, names_col: str):
    """Sorted union of class names across per-sample ``*_class_names.csv`` files."""
    import pandas as pd

    names = set()
    for _sample, _auroc, names_csv in sample_paths:
        sub = pd.read_csv(names_csv)
        col = names_col if names_col in sub.columns else sub.columns[-1]
        names.update(str(v) for v in sub[col].dropna().unique())
    return sorted(names)


def _align_sample_auroc_to_global(
    df,
    local_class_names,
    global_index,
    *,
    gt_col: str = "ground_truth_final_CT",
    pred_col: str | None = None,
):
    """Map one sample AUROC table to global label indices and ``prob_*`` columns."""
    import pandas as pd

    local_class_names = [str(c) for c in local_class_names]
    n = len(df)
    n_global = len(global_index)
    y_true = np.array(
        [global_index.get(str(g), -1) for g in df[gt_col]],
        dtype=np.int64,
    )
    y_score = np.zeros((n, n_global), dtype=np.float64)
    for local_j, cname in enumerate(local_class_names):
        prob_col = f"prob_{local_j}"
        if prob_col not in df.columns:
            continue
        global_j = global_index.get(cname)
        if global_j is not None:
            y_score[:, global_j] = df[prob_col].to_numpy(dtype=np.float64)

    pred_name_col = pred_col
    if pred_name_col is None:
        pred_name_col = gt_col.replace("ground_truth_", "predict_stardist_")
    out = pd.DataFrame(
        {
            gt_col: df[gt_col].astype(str),
            pred_name_col: df[pred_name_col].astype(str) if pred_name_col in df.columns else "",
            "ground_truth_global_idx": y_true,
        }
    )
    if "cell_id" in df.columns:
        out.insert(0, "cell_id", df["cell_id"].astype(str))
    for gj in range(n_global):
        out[f"prob_{gj}"] = y_score[:, gj]
    return out, y_true, y_score


def concat_stardist_auroc_tier_all_samples(
    cases_root,
    tier: str = "l2",
    *,
    out_dir=None,
    model_suffix: str = "_project_all_UNI",
    layout: str = "per_sample",
    save_concat_csv: bool = True,
):
    """
    Concatenate per-sample StarDist AUROC CSV files into a global class union.

    Tier keys: ``l2``, ``l1``, ``l12``, ``l3``, ``l4`` (see ``STARDIST_AUROC_TIER_SPECS``).
    """
    import pandas as pd

    spec = stardist_auroc_tier_spec(tier)
    sample_paths, missing = discover_stardist_auroc_sample_paths(
        cases_root,
        model_suffix=model_suffix,
        layout=layout,
        auroc_name=spec["csv_name"],
    )
    if not sample_paths:
        raise FileNotFoundError(
            f"No {spec['csv_name']} pairs under {cases_root}. "
            f"Run pooled StarDist with levels including {tier!r} first."
        )

    global_class_names = build_global_tier_class_names(
        sample_paths, names_col=spec["names_col"]
    )
    global_index = {name: i for i, name in enumerate(global_class_names)}

    chunks = []
    y_true_parts = []
    y_score_parts = []
    for sample, auroc_csv, names_csv in sample_paths:
        df = pd.read_csv(auroc_csv)
        names_df = pd.read_csv(names_csv)
        name_col = spec["names_col"] if spec["names_col"] in names_df.columns else names_df.columns[-1]
        local_names = names_df.sort_values("class_index")[name_col].tolist()
        aligned, y_true, y_score = _align_sample_auroc_to_global(
            df,
            local_names,
            global_index,
            gt_col=spec["gt_col"],
            pred_col=spec["pred_col"],
        )
        aligned.insert(0, "sample", sample)
        chunks.append(aligned)
        y_true_parts.append(y_true)
        y_score_parts.append(y_score)
        print(f"  {sample}: {len(df):,} cells, {len(local_names)} local classes", flush=True)

    concat_df = pd.concat(chunks, ignore_index=True)
    y_true = np.concatenate(y_true_parts)
    y_score = np.vstack(y_score_parts)

    if missing:
        print(
            f"  WARNING: {len(missing)} sample(s) missing AUROC CSV: {missing}",
            flush=True,
        )

    out_dir = Path(out_dir) if out_dir is not None else Path(cases_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_concat_csv:
        concat_path = out_dir / f"{spec['concat_stem']}_concat.csv"
        names_path = out_dir / f"{spec['concat_stem']}_class_names.csv"
        concat_df.to_csv(concat_path, index=False)
        pd.DataFrame(
            {
                "class_index": np.arange(len(global_class_names)),
                spec["names_col"]: global_class_names,
            }
        ).to_csv(names_path, index=False)
        print(f"Saved pooled AUROC concat: {concat_path} ({len(concat_df):,} rows)", flush=True)
        print(
            f"Saved global class names: {names_path} ({len(global_class_names)} classes)",
            flush=True,
        )

    return {
        "tier": str(tier).lower(),
        "concat_df": concat_df,
        "global_class_names": global_class_names,
        "y_true": y_true,
        "y_score": y_score,
        "sample_paths": sample_paths,
        "missing_samples": missing,
        "out_dir": out_dir,
        "spec": spec,
    }


def concat_stardist_auroc_level2_all_samples(
    cases_root,
    *,
    out_dir=None,
    model_suffix: str = "_project_all_UNI",
    layout: str = "per_sample",
    save_concat_csv: bool = True,
):
    """
    Concatenate per-sample ``validation_external_stardist_matched_AUROC.csv`` files
    into a **global** ``final_CT`` label space (class union).
    """
    return concat_stardist_auroc_tier_all_samples(
        cases_root,
        "l2",
        out_dir=out_dir,
        model_suffix=model_suffix,
        layout=layout,
        save_concat_csv=save_concat_csv,
    )


def plot_stardist_tier_roc_pooled(
    y_true,
    y_score,
    global_class_names,
    *,
    tier: str = "l2",
    save_path=None,
    title: str | None = None,
    roc_color_scheme: str | None = None,
    figsize=(6.0, 6.0),
    max_curves=None,
):
    """Multiclass OvR ROC on concatenated StarDist-matched cells across samples."""
    from plot import plot_multiclass_roc_curves

    spec = stardist_auroc_tier_spec(tier)
    class_names = np.asarray(global_class_names)
    return plot_multiclass_roc_curves(
        y_true,
        y_score,
        class_names,
        figsize=figsize,
        max_curves=max_curves or len(class_names),
        save_path=save_path,
        title=title or spec["pooled_title"],
        roc_color_scheme=roc_color_scheme or spec["roc_color_scheme"],
    )


def plot_stardist_level2_roc_pooled(
    y_true,
    y_score,
    global_class_names,
    *,
    save_path=None,
    title="StarDist Level2 ROC — all Complete_Cases (pooled cells)",
    roc_color_scheme="xenium_lung_fine",
    figsize=(6.0, 6.0),
    max_curves=None,
):
    """Multiclass OvR ROC on concatenated StarDist-matched cells across samples."""
    return plot_stardist_tier_roc_pooled(
        y_true,
        y_score,
        global_class_names,
        tier="l2",
        save_path=save_path,
        title=title,
        roc_color_scheme=roc_color_scheme,
        figsize=figsize,
        max_curves=max_curves,
    )


def plot_pooled_stardist_tier_roc(
    stardist_result_root,
    out_dir,
    tier: str,
    *,
    layout: str = "pooled_stardist",
    pan_organ: str | None = None,
    roc_color_scheme: str | None = None,
    figsize=(8.0, 8.0),
    show_barplot: bool = False,
    save_per_class_csv: bool = False,
    show: bool = True,
    save_concat_csv: bool = True,
):
    """
    Discover per-sample AUROC CSVs, pool across Complete_Cases, and plot pooled ROC.

    Writes ``roc_stardist_{tier}_cross_dataset_all_Complete_Cases_pooled.pdf`` under ``out_dir``.

    Tier keys: ``l2``, ``l1``, ``l12``, ``l3``, ``l4``.
    """
    import matplotlib.pyplot as plt

    spec = stardist_auroc_tier_spec(tier)
    tier_key = str(tier).lower()
    out_dir = Path(out_dir)
    print(f"\n=== {tier_key.upper()} | discovering {spec['csv_name']} ===", flush=True)
    pooled = concat_stardist_auroc_tier_all_samples(
        stardist_result_root,
        tier_key,
        out_dir=out_dir,
        layout=layout,
        save_concat_csv=save_concat_csv,
    )
    n_samples = len(pooled["sample_paths"])
    print(
        f"Pooled: {len(pooled['concat_df']):,} cells from {n_samples} samples; "
        f"{len(pooled['global_class_names'])} global {spec['names_col']} classes",
        flush=True,
    )

    pdf_name = f"roc_stardist_{tier_key}_cross_dataset_all_Complete_Cases_pooled.pdf"
    eff_scheme = roc_color_scheme
    if eff_scheme is None and pan_organ is not None:
        eff_scheme = scheme_for_pan_organ(pan_organ, head=tier_key)
    roc_info = plot_stardist_tier_roc_pooled(
        pooled["y_true"],
        pooled["y_score"],
        pooled["global_class_names"],
        tier=tier_key,
        save_path=str(out_dir / pdf_name),
        title=f"{spec['pooled_title']} (n={n_samples} Complete_Cases)",
        roc_color_scheme=eff_scheme,
        figsize=figsize,
        max_curves=len(pooled["global_class_names"]),
    )
    if show:
        plt.show()

    per_class = None
    if roc_info is not None and roc_info.get("per_class_auc") is not None:
        print(f"Pooled macro AUROC ({tier_key.upper()}): {roc_info['macro_auc']:.4f}", flush=True)
        per_class = roc_info["per_class_auc"].sort_values("auc_ovr", ascending=False)
        if save_per_class_csv:
            per_class_path = out_dir / (
                f"roc_stardist_{tier_key}_cross_dataset_all_Complete_Cases_per_class_auc.csv"
            )
            per_class.to_csv(per_class_path, index=False)

        if show_barplot and len(per_class):
            try:
                import seaborn as sns
            except ImportError:  # pragma: no cover
                sns = None
            if sns is not None:
                fig, ax = plt.subplots(figsize=(8, max(5, 0.22 * len(per_class))))
                plot_df = per_class.sort_values("auc_ovr", ascending=True)
                sns.barplot(data=plot_df, y="class_name", x="auc_ovr", ax=ax, color="steelblue")
                ax.set_xlim(0, 1)
                ax.set_xlabel("One-vs-rest AUROC")
                ax.set_ylabel(spec["names_col"])
                ax.set_title(f"Per-class AUROC ({tier_key.upper()}, cross-dataset pooled)")
                plt.tight_layout()
                if show:
                    plt.show()

    return {
        "pooled": pooled,
        "roc_info": roc_info,
        "per_class_auc": per_class,
        "tier": tier_key,
        "spec": spec,
    }

########################################################
# 2026.08.14 create symlinks for the StarDist sample dirs for Pred_statistic_visual_hcc_all.ipynb
########################################################
def symlink_stardist_sample_dirs(
    stardist_result_root,
    sample_ids,
    dest_root,
    *,
    overwrite: bool = True,
) -> Path:
    """Create ``dest_root/<sample>`` symlinks into ``stardist_result_root/<sample>``.

    Used when pooled ROC / clinical AUROC should run on a subset of regions
    (e.g. HCC ``Annotation == 'Y'``) without copying files.
    """
    result_root = Path(stardist_result_root)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    missing: list[str] = []
    samples = sorted(
        {
            str(s).strip()
            for s in sample_ids
            if s is not None and str(s).strip() and str(s).strip().lower() != "nan"
        }
    )
    for sample in samples:
        src = result_root / sample
        dst = dest_root / sample
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                continue
            dst.unlink()
        if src.is_dir():
            dst.symlink_to(src, target_is_directory=True)
            n_ok += 1
        else:
            missing.append(sample)
    print(f"Linked {n_ok}/{len(samples)} StarDist sample dirs under {dest_root}", flush=True)
    if missing:
        preview = missing[:8]
        extra = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8})"
        print(f"WARNING: missing {len(missing)} sample dir(s): {preview}{extra}", flush=True)
    return dest_root


def plot_pooled_stardist_tier_rocs(
    stardist_result_root,
    out_dir,
    tiers: Sequence[str],
    *,
    figsize: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    skip_missing: bool = True,
    **kwargs,
) -> dict:
    """Plot pooled StarDist ROC for each tier, skipping missing CSVs by default."""
    results: dict = {}
    for tier in tiers:
        tier_key = str(tier).lower()
        call_kw = dict(kwargs)
        if isinstance(figsize, Mapping):
            fig = figsize.get(tier_key, figsize.get(str(tier)))
            if fig is not None:
                call_kw["figsize"] = fig
        elif figsize is not None:
            call_kw["figsize"] = figsize
        try:
            results[tier_key] = plot_pooled_stardist_tier_roc(
                stardist_result_root, out_dir, tier_key, **call_kw
            )
        except FileNotFoundError as exc:
            if not skip_missing:
                raise
            print(f"Skip {tier_key.upper()}: {exc}", flush=True)
    return results
########################################################


########################################################
# 2026.07.04 - added AUROC for clinical comparison
########################################################
STARDIST_MACRO_AUROC_TIERS: tuple[str, ...] = ("l2", "l1", "l12", "l3", "l4")
STARDIST_CLINICAL_COLUMNS: tuple[str, ...] = ("Status", "Sample_Affect_Pairing")
STARDIST_TEST_METHODS: tuple[str, ...] = ("rank", "parametric")
STARDIST_TIER_DISPLAY: dict[str, str] = {
    "l2": "Level2 (final_CT)",
    "l1": "Level1 (lineage)",
    "l12": "Level12 (sublineage)",
    "l3": "Level3 (CNiche)",
    "l4": "Level4 (TNiche)",
}


def _ensure_xenium_lung_on_path() -> Path:
    """``plot_q4_clinical_comparison`` still lives in ``code/Xenium_lung``."""
    xenium_dir = Path(__file__).resolve().parent.parent / "Xenium_lung"
    if str(xenium_dir) not in sys.path:
        sys.path.insert(0, str(xenium_dir))
    return xenium_dir


def sample_macro_auroc_from_auroc_csv(auroc_csv_path) -> float:
    """Per-sample macro OvR AUROC from a saved StarDist AUROC table."""
    import pandas as pd

    df = pd.read_csv(auroc_csv_path)
    prob_cols = sorted(
        [c for c in df.columns if c.startswith("prob_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not prob_cols or "ground_truth_idx" not in df.columns:
        return float("nan")
    y_true = df["ground_truth_idx"].to_numpy(dtype=np.int64)
    y_score = df[prob_cols].to_numpy(dtype=np.float64)
    return macro_auc_ovr(y_true, y_score, len(prob_cols))


def build_stardist_sample_macro_auroc_table(
    stardist_result_root,
    tiers: Sequence[str] = STARDIST_MACRO_AUROC_TIERS,
    *,
    layout: str = "pooled_stardist",
) -> pd.DataFrame:
    """
    Sample-level macro AUROC for each StarDist tier from per-sample AUROC CSV files.

    Returns wide dataframe: ``Sample``, ``macro_auc_l2``, ``macro_auc_l1``, ...
    """
    import pandas as pd

    per_sample: dict[str, dict[str, float]] = {}
    for tier in tiers:
        spec = stardist_auroc_tier_spec(tier)
        sample_paths, _ = discover_stardist_auroc_sample_paths(
            stardist_result_root,
            layout=layout,
            auroc_name=spec["csv_name"],
        )
        if not sample_paths:
            print(f"  ⚠ No {spec['csv_name']} found for tier {tier}; skip.", flush=True)
            continue
        for sample, auroc_csv, _names_csv in sample_paths:
            per_sample.setdefault(str(sample), {})[str(tier).lower()] = float(
                sample_macro_auroc_from_auroc_csv(auroc_csv)
            )

    rows = []
    for sample in sorted(per_sample):
        row: dict = {"Sample": sample}
        for tier in tiers:
            row[f"macro_auc_{str(tier).lower()}"] = per_sample[sample].get(
                str(tier).lower(), float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def merge_stardist_auroc_with_clinical(
    auroc_summary: pd.DataFrame,
    clinical_info: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str = "Donor_ID",
    clinical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join per-sample macro AUROC to clinical metadata.

    Default (Xenium): ``Sample`` ↔ ``Donor_ID``.
    HCC: pass ``clinical_key="matched_he"``.
    """
    import pandas as pd

    out = auroc_summary.copy()
    if sample_col not in out.columns:
        raise KeyError(f"AUROC table missing {sample_col!r}")
    out[sample_col] = out[sample_col].astype(str)
    clinical = clinical_info.copy()
    if clinical_key not in clinical.columns:
        raise KeyError(f"Clinical table missing {clinical_key!r}")
    clinical[clinical_key] = clinical[clinical_key].astype(str)
    clinical = clinical.drop_duplicates(clinical_key)
    if sample_col == clinical_key:
        merged = out.merge(clinical, on=sample_col, how="left")
    else:
        merged = out.merge(clinical, left_on=sample_col, right_on=clinical_key, how="left")
    cols = list(clinical_columns or STARDIST_CLINICAL_COLUMNS)
    for col in cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged

########################################################
## 2026.08.14, add function to get the effective clinical columns for the CODEX HCC dataset
########################################################
def _effective_clinical_columns(
    clinical_columns: Sequence[str] | None,
    pan_organ: str | None,
) -> tuple[str, ...]:
    """Use organ defaults when the caller left the Xenium column tuple."""
    cols = tuple(clinical_columns) if clinical_columns is not None else STARDIST_CLINICAL_COLUMNS
    if pan_organ is None:
        return cols
    organ_cols = clinical_columns_for_pan_organ(pan_organ)
    if organ_cols and cols in (STARDIST_CLINICAL_COLUMNS, ()):
        return organ_cols
    return cols
########################################################

def test_metric_by_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    clinical_columns: Sequence[str],
    pan_organ: str | None = None,
    method: Literal["rank", "parametric"] = "rank",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    min_group_size: int = 2,
) -> pd.DataFrame:
    """Compare a sample-level metric across clinical groups.

    ``method="rank"``: Mann–Whitney U (2 groups) or Kruskal–Wallis (3+).
    ``method="parametric"``: Welch's t-test (2 groups) or one-way ANOVA (3+).
    ``pan_organ`` selects the x-axis group order (``xenium_lung`` / ``codex_hcc``).
    """
    from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind

    order_map = clinical_group_order(pan_organ or "xenium_lung")

    method_key = str(method).strip().lower()
    if method_key not in STARDIST_TEST_METHODS:
        raise ValueError(
            f"method must be one of {STARDIST_TEST_METHODS}, got {method!r}."
        )

    rows = []
    for group_col in clinical_columns:
        if group_col not in df.columns or metric_col not in df.columns:
            continue
        sub = df.dropna(subset=[group_col, metric_col]).copy()
        sub[group_col] = sub[group_col].astype(str)
        present = sub[group_col].unique().tolist()
        preset = [g for g in order_map.get(group_col, ()) if g in present]
        rest = sorted(g for g in present if g not in preset)
        label_order = preset + rest
        groups = []
        labels = []
        for label in label_order:
            vals = sub.loc[sub[group_col] == label, metric_col].astype(float).to_numpy()
            if len(vals) >= min_group_size:
                groups.append(vals)
                labels.append(str(label))
        if len(groups) < 2:
            rows.append(
                {
                    "clinical_variable": group_col,
                    "metric_col": metric_col,
                    "method": method_key,
                    "test": "insufficient_groups",
                    "p_value": float("nan"),
                    "statistic": float("nan"),
                    "n_groups": len(groups),
                    "group_labels": labels,
                }
            )
            continue
        if len(groups) == 2:
            if method_key == "parametric":
                if alternative == "two-sided":
                    stat, p = ttest_ind(
                        groups[0], groups[1], equal_var=False, alternative="two-sided"
                    )
                elif alternative == "greater":
                    stat, p = ttest_ind(
                        groups[1], groups[0], equal_var=False, alternative="greater"
                    )
                else:
                    stat, p = ttest_ind(
                        groups[1], groups[0], equal_var=False, alternative="less"
                    )
                test_name = "Welch t-test"
            elif alternative == "two-sided":
                stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                test_name = "Mann-Whitney U"
            elif alternative == "greater":
                stat, p = mannwhitneyu(groups[1], groups[0], alternative="greater")
                test_name = "Mann-Whitney U"
            else:
                stat, p = mannwhitneyu(groups[1], groups[0], alternative="less")
                test_name = "Mann-Whitney U"
        elif method_key == "parametric":
            stat, p = f_oneway(*groups)
            test_name = "ANOVA"
        else:
            stat, p = kruskal(*groups)
            test_name = "Kruskal-Wallis"
        medians = [float(np.median(g)) for g in groups]
        rows.append(
            {
                "clinical_variable": group_col,
                "metric_col": metric_col,
                "method": method_key,
                "test": test_name,
                "p_value": float(p),
                "statistic": float(stat),
                "n_groups": len(groups),
                "group_labels": labels,
                "group_median_q4": medians,
                "group_median_values": medians,
            }
        )
    import pandas as pd

    out = pd.DataFrame(rows)
    if len(out):
        out["neg_log10_p"] = -np.log10(out["p_value"].clip(lower=1e-300))
    return out


def plot_stardist_macro_auroc_by_clinical(
    merged_auc: pd.DataFrame,
    *,
    tiers: Sequence[str] = STARDIST_MACRO_AUROC_TIERS,
    clinical_columns: Sequence[str] = STARDIST_CLINICAL_COLUMNS,
    sample_col: str = "Sample",
    pan_organ: str | None = None,
    method: Literal["rank", "parametric"] = "rank",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    figsize: tuple[float, float] = (8.0, 4.0),
    legend_ncol: int = 2,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict:
    """Boxplots of per-sample macro AUROC by clinical group, with significance tests."""
    import matplotlib.pyplot as plt

    _ensure_xenium_lung_on_path()
    from histology_derived_niche_index import plot_q4_clinical_comparison

    clinical_columns = _effective_clinical_columns(clinical_columns, pan_organ)
    plot_df = merged_auc.copy()
    present_cols = tuple(c for c in clinical_columns if c in plot_df.columns)
    if not present_cols:
        raise KeyError(f"None of {list(clinical_columns)!r} found in merged_auc columns.")
    if "Sample" not in plot_df.columns and sample_col in plot_df.columns:
        plot_df["Sample"] = plot_df[sample_col].astype(str)

    save_dir = Path(save_dir) if save_dir is not None else None
    display_fn = None
    try:
        from IPython.display import display as display_fn  # type: ignore
    except ImportError:
        pass

    by_tier: dict = {}
    for tier in tiers:
        tier_key = str(tier).lower()
        metric_col = f"macro_auc_{tier_key}"
        if metric_col not in plot_df.columns or not np.isfinite(plot_df[metric_col]).any():
            print(f"  Skip {tier_key.upper()}: no finite {metric_col}", flush=True)
            continue

        stats = test_metric_by_clinical_groups(
            plot_df,
            metric_col=metric_col,
            clinical_columns=present_cols,
            pan_organ=pan_organ,
            method=method,
            alternative=alternative,
        )
        print(
            f"\n=== {tier_key.upper()} macro AUROC vs clinical groups "
            f"(method={method}) ===",
            flush=True,
        )
        if display_fn is not None:
            display_fn(stats)
        else:
            print(stats.to_string(index=False))

        tier_label = STARDIST_TIER_DISPLAY.get(tier_key, tier_key.upper())
        save_path = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if method == "rank" else f"_{method}"
            save_path = save_dir / (
                f"stardist_{tier_key}_macro_auroc_clinical_comparison{suffix}.pdf"
            )

        fig, _axes = plot_q4_clinical_comparison(
            plot_df,
            stats,
            q4_col=metric_col,
            clinical_columns=present_cols,
            pan_organ=pan_organ or "xenium_lung",
            figsize=figsize,
            legend_ncol=legend_ncol,
            ylabel=f"Macro AUROC ({tier_label})",
            suptitle=f"StarDist {tier_label} macro AUROC by clinical group",
            save_path=save_path,
        )
        if show:
            plt.show()
        by_tier[tier_key] = {"stats": stats, "fig": fig, "metric_col": metric_col}

    return {"merged": plot_df, "by_tier": by_tier, "method": method}

########################################################
# 2026.08.14 analyze the StarDist macro AUROC by clinical groups
########################################################
def analyze_stardist_macro_auroc_by_clinical(
    stardist_result_root,
    clinical_info,
    *,
    tiers: Sequence[str] = STARDIST_MACRO_AUROC_TIERS,
    layout: str = "pooled_stardist",
    clinical_columns: Sequence[str] = STARDIST_CLINICAL_COLUMNS,
    clinical_key: str = "Donor_ID",
    sample_col: str = "Sample",
    pan_organ: str | None = None,
    method: Literal["rank", "parametric"] = "rank",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    figsize: tuple[float, float] = (8.0, 4.0),
    legend_ncol: int = 2,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict:
    """
    Compare sample-level StarDist macro AUROC across clinical groups.

    Builds the per-sample AUROC table, joins clinical metadata, then plots
    boxplots with Mann–Whitney / Kruskal–Wallis (``method="rank"``) or
    Welch t / ANOVA (``method="parametric"``).
    ``pan_organ`` selects clinical columns and x-axis group order
    (``xenium_lung`` or ``codex_hcc``).
    """
    clinical_columns = _effective_clinical_columns(clinical_columns, pan_organ)
    auroc_df = build_stardist_sample_macro_auroc_table(
        stardist_result_root, tiers=tiers, layout=layout
    )
    if auroc_df.empty:
        raise FileNotFoundError(
            f"No per-sample AUROC CSVs under {stardist_result_root}. "
            "Run step_pooled_stardist for each tier first."
        )
    merged = merge_stardist_auroc_with_clinical(
        auroc_df,
        clinical_info,
        sample_col=sample_col,
        clinical_key=clinical_key,
        clinical_columns=clinical_columns,
    )
    plotted = plot_stardist_macro_auroc_by_clinical(
        merged,
        tiers=tiers,
        clinical_columns=clinical_columns,
        sample_col=sample_col,
        pan_organ=pan_organ,
        method=method,
        alternative=alternative,
        figsize=figsize,
        legend_ncol=legend_ncol,
        show=show,
        save_dir=save_dir,
    )
    return {
        "auroc_df": auroc_df,
        "merged": plotted["merged"],
        "by_tier": plotted["by_tier"],
        "method": method,
    }
########################################################

def save_stardist_external_validation_metrics(
    *,
    all_labels,
    all_preds,
    matched_features_path,
    class_names_star,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    therapy_data,
    metrics_csv_path,
    model_star=None,
    scaler=None,
    X_star=None,
    device=None,
    predict_all_label_heads=None,
    cv_data=None,
    g=None,
    logo_summary=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=2.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    probs_l2=None,
    head_probs=None,
):
    """Write ``validation_external_stardist_matched_metrics.csv`` with acc/F1 + macro AUROC."""
    from base import collect_hce_tier_metrics_extras, save_hce_validation_metrics

    g = g or {}
    stardist_tiers, stardist_num_classes = build_stardist_matched_tier_metrics(
        all_labels=all_labels,
        all_preds=all_preds,
        matched_features_path=matched_features_path,
        class_names_star=class_names_star,
        class_names_level1=class_names_level1,
        y_encoded_f=y_encoded_f,
        y_level1_encoded_f=y_level1_encoded_f,
        model_star=model_star,
        scaler=scaler,
        X_star=X_star,
        device=device,
        predict_all_label_heads=predict_all_label_heads,
        cv_data=cv_data,
        g=g,
    )
    tier_aucs = build_stardist_matched_tier_aucs(
        matched_features_path=matched_features_path,
        class_names_star=class_names_star,
        class_names_level1=class_names_level1,
        y_encoded_f=y_encoded_f,
        y_level1_encoded_f=y_level1_encoded_f,
        model_star=model_star,
        scaler=scaler,
        X_star=X_star if X_star is not None else g.get("X_star"),
        device=device,
        cv_data=cv_data,
        g=g,
        probs_l2=probs_l2,
        head_probs=head_probs,
    )
    if not tier_aucs:
        print("  ⚠ No macro AUROC computed (see warnings above).", flush=True)

    tier_extra = collect_hce_tier_metrics_extras(
        logo_summary=None,
        insample=stardist_tiers,
        num_classes=stardist_num_classes,
    )
    tier_extra.update(collect_tier_auc_extras(tier_aucs, prefix="insample"))

    l2_star = stardist_tiers.get("l2") or {}
    l1_star = stardist_tiers.get("l1") or {}
    if tier_aucs.get("l2") is not None:
        tier_extra["val_l2_macro_auc"] = tier_aucs["l2"]
    if tier_aucs.get("l1") is not None:
        tier_extra["val_l1_macro_auc"] = tier_aucs["l1"]

    mean_best_epoch = None
    if logo_summary is not None and logo_summary.get("folds"):
        mean_best_epoch = int(
            round(float(np.mean([f["best_epoch"] for f in logo_summary["folds"]])))
        )

    metrics_df, combined_df = save_hce_validation_metrics(
        val_acc=l2_star.get("accuracy", float("nan")),
        val_macro_f1=l2_star.get("macro_f1", float("nan")),
        val_weighted_f1=l2_star.get("weighted_f1", float("nan")),
        val_level1_acc=l1_star.get("accuracy", float("nan")),
        val_level1_macro_f1=l1_star.get("macro_f1", float("nan")),
        val_level1_weighted_f1=l1_star.get("weighted_f1", float("nan")),
        class_names=class_names_star,
        class_names_level1=class_names_level1,
        best_epoch=mean_best_epoch,
        hce_lambda=None,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        therapy_data=f"{therapy_data}_stardist_matched",
        metrics_csv_path=metrics_csv_path,
        extra_metrics=tier_extra,
    )

    auroc_csv_path = _auroc_csv_path_from_metrics(metrics_csv_path)
    if all_preds is not None:
        try:
            save_stardist_external_auroc_level2_csv(
                matched_features_path=matched_features_path,
                class_names_star=class_names_star,
                all_labels=all_labels,
                all_preds=all_preds,
                auroc_csv_path=auroc_csv_path,
                probs_l2=probs_l2,
                model_star=model_star,
                scaler=scaler,
                X_star=X_star if X_star is not None else g.get("X_star"),
                device=device,
            )
        except Exception as exc:
            print(f"  ⚠ Skip AUROC CSV: {exc}", flush=True)

    return metrics_df, combined_df, tier_aucs
