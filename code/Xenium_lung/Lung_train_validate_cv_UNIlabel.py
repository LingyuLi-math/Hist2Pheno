## 2026.02.11 using FineST env
## 2026.02.12 vilization cell type of each ROI
## 2026.03.04 woody_pro:: upload the HE-StarDist embedding and predict the cell type of NCRT_tumor1
## 2026.03.06 woody_pro:: clearn code
## 2026.03.13 update cell type level1: nanually make codex_meta_celltype_level1.csv
## 2026.03.16 use valis to make registration [dont try]
## 2026.03.18 delete two subtypes, adjust scale factor by batch
## 2026.03.19 Train and using NCRT_train.ipynb
## 2026.03.23 Adjust file name and path, then adjust model to use level1 and level2, make model_train_validate.ipynb
## 2026.04.04 woody_pro: SeededNTM; make processing for x, add cv in training for tumor1
## 2026.04.06 Adjust loss function
## 2026.04.14 LLY update using UNI method, based on LOGO CV
## 2026.05.11 Use codex_meta_celltype_level012.csv, run Data_process_visualization.ipynb and demo_level012.sh,
##            to get he_cell_coords/ and matched_features_tumor1_level012.npz
## 2026.05.11 add the prediction form Level1 directly
## 2026.05.21 copy form NCRT_train_validate_tumor1_cv_UNIlabel012.ipynb, adjust the code for Xenium data
## 2026.05.22 make the code clean, add the StarDist prediction, and adjust the code for Xenium data
## 2026.06.24 train on Complete_Cases, validate on Incomplete_Cases
## 2026.06.22 LLY CLI refactor from Lung_train_validate_cv_UNIlabel_all_clean.ipynb
## 2026.06.22 LLY cross-dataset mode: pool Complete_Cases h5ads, dataset-level 5-fold CV → Data/result/
## 2026.06.25 LLY use Xenium_lung Complete_Cases 25 datasets for training and validation
## 2026.06.26 LLY add --save-result option to resolve result folder name for cross-dataset and per-sample
##            update the code to use spatial context for model training and validation
## 2026.06.27 LLY add the StarDist predict on all nuclei (each dataset)
##            using transer_embedding_label_h5ad.py --steps stardist_all_h5ad
##            obtain the all_features_stardist.h5ad


## cd /home/lingyu/ssd2/Python/Collaborate/esccAI
## conda activate SeededNTM
## python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --sample VUILD107MA
## python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --cases-set complete
## conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --sample VUILD107MA --steps all


########################################################
# 2026.06.25 LLY: use Xenium_lung Complete_Cases 25 datasets for training and validation
########################################################
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# conda activate SeededNTM

# # 全流程：train + internal validate + 25 个 sample StarDist predict
# python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --mode cross-dataset --cases-set complete --save-result result_all_AUCwithweights

# # 仅 StarDist matched（需已训练 checkpoint + matched h5ad）
# python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --mode cross-dataset --steps stardist
#
# # 仅 StarDist all nuclei → {sample}_all_features_stardist_label.h5ad（需 all_features_stardist.h5ad）
# python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
#   --mode cross-dataset --cases-set complete \
#   --pooled-save-result result_all_spatial \
#   --use-spatial-context --spatial-k 8 --spatial-mode mean \
#   --ablation-tag D_emph_L2_spatial_bs4096 \
#   --steps stardist_all
########################################################

########################################################
# 2026.06.26 LLY: add spatial context for model training and validation
########################################################
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# conda activate SeededNTM

# python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
#   --mode cross-dataset --cases-set complete \
#   --use-spatial-context --spatial-k 8 --spatial-mode mean \
#   --pooled-save-result result_all_spatial \
#   --ablation-tag D_emph_L2_spatial_bs4096 \
#   --no-resume-from-checkpoints
########################################################


########################################################
# 2026.06.27 LLY: add StarDist all nuclei for model training and validation
########################################################
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# conda activate SeededNTM

# python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \
#   --mode cross-dataset --cases-set complete \
#   --pooled-save-result result_all_spatial \
#   --use-spatial-context --spatial-k 8 --spatial-mode mean \
#   --ablation-tag D_emph_L2_spatial_bs4096 \
#   --steps stardist_all
########################################################

#!/usr/bin/env python3
"""
Xenium lung: stratified K-fold HCE training + HE internal validation + StarDist external validation.

Mirrors ``Lung_train_validate_cv_UNIlabel_all_clean.ipynb``.  Assumes UNI embeddings exist under
``{sample}/{sample}_project_all_UNI/ImgEmbeddings_all/``; HE / StarDist h5ad can be built here
or pre-built via ``transer_embedding_label_h5ad.py``.

Terminal usage (from repo root ``esccAI``):

  # One sample, full pipeline
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --sample VUILD107MA

  # All samples under Complete_Cases
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --cases-set complete

  # Cross-dataset: pool 25 h5ads, 5-fold dataset CV, outputs under Data/result/
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --mode cross-dataset --cases-set complete

  # StarDist matched only (cross-dataset model already trained)
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --mode cross-dataset --steps stardist

  # StarDist all nuclei → label h5ad (no AUROC; needs all_features_stardist.h5ad per sample)
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \\
    --mode cross-dataset --cases-set complete \\
    --pooled-save-result result_all_spatial \\
    --use-spatial-context --spatial-k 8 --spatial-mode mean \\
    --ablation-tag D_emph_L2_spatial_bs4096 \\
    --steps stardist_all

  # Train only (skip plots / StarDist)
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --sample VUILD107MA --steps he_h5ad train

  # StarDist validation only (requires trained checkpoint + stardist h5ad)
  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py --sample VUILD107MA --steps stardist

CSV outputs under ``.../{sample}_project_all_UNI/result/``:

  - ``validation_internal_metrics.csv`` — HE in-sample acc/F1 (after ``he_validate``)
  - ``validation_external_stardist_matched_metrics.csv`` — StarDist acc/F1 + macro AUROC
    (``val_l2_macro_auc``, ``insample_l2_macro_auc``, … per tier)
  - ``validation_external_stardist_matched_AUROC.csv`` — per-cell Level2 labels + ``prob_*``
    for ROC replay; ``roc_stardist_level2_from_AUROC_csv.pdf`` plotted from that table

Cross-dataset ``--steps stardist_all`` (§5 notebook):

  - Input: ``Complete_Cases/{sample}/{sample}_all_features_stardist.h5ad``
    (from ``transer_embedding_label_h5ad.py --steps stardist_all_h5ad``)
  - Output: ``Data/{pooled_save_result}/stardist/{sample}/{sample}_all_features_stardist_label.h5ad``
  - ``adata.obs``: ``l2_prob_*``, ``l1_prob_*``, ``l12_prob_*``, ``l3_prob_*``, ``l4_prob_*``
  - ``adata.uns['pred_prob_class_names']``: class order per head
  - No matched AUROC CSV (most nuclei lack pathologist labels)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# Headless plotting for terminal runs.
os.environ.setdefault("MPLBACKEND", "Agg")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
print("Loading dependencies...", flush=True)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
_XENIUM_DIR = _REPO_ROOT / "code" / "Xenium_lung"
for _p in (_PKG_DIR, _XENIUM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import base  # noqa: E402
from base import (  # noqa: E402
    XENIUM_CELL_COORD_COLUMN_RENAME_FULL,
    CellTypeDataset,
    adata_X_to_dense,
    build_spatial_neighbor_index,
    collect_hce_tier_metrics_extras,
    encode_labels_with_class_names,
    evaluate,
    evaluate_and_plot_on_all_data,
    load_cell_pixcoords,
    load_matched_features_bundle,
    match_celltype2stardist,
    match_hist2cell_h5ad,
    prepare_data_from_matched_h5ad,
    save_hce_validation_metrics,
)
from model import (  # noqa: E402
    _build_spatial_neighbor_index_for_cv_data,
    get_select4_best_checkpoint_path,
    load_model_for_predict,
    predict_all_label_heads,
    run_group_kfold_cv_with_oof_report,
    run_stratified_kfold_cv_with_insample_report,
    sync_best_mlp_from_logo_fold,
)
from plot import (  # noqa: E402
    mlp_collect_five_head_softmax_probs,
    mlp_collect_softmax_probs,
    plot_celltype_spatial_distribution,
    plot_confusion_matrix,
    plot_level1_accuracy_from_level1_head,
    plot_level1_accuracy_from_level2_predictions,
    plot_level1_f1_from_level1_head,
    plot_level1_f1_from_level2_predictions,
    plot_level1_roc_from_level1_head,
    plot_level1_roc_from_level2_scores,
    plot_level1_spatial_distribution,
    plot_multiclass_roc_curves,
    plot_per_class_accuracy,
    plot_per_class_f1,
    plot_tier_spatial_distribution,
)
from xenium_uni_nb_helpers import (  # noqa: E402
    build_insample_tier_metrics,
    save_stardist_external_validation_metrics,
    plot_stardist_roc_from_auroc_csv,
    _auroc_csv_path_from_metrics,
    discover_case_samples,
    discover_h5ad_case_samples,
    ensure_lp_extra_insample_preds,
    make_pooled_result_fig,
    make_pooled_stardist_result_fig,
    make_result_fig,
    attach_five_head_probs_to_adata_obs,
    stardist_all_label_h5ad_path,
    plot_he_confusion_matrices,
    plot_he_f1_extra_tiers,
    plot_he_spatial_extra_tiers,
    plot_he_validate_level2_minimal,
    prepare_pooled_cv_data_from_h5ads,
    save_pooled_internal_validation_metrics,
    plot_stardist_acc_extra,
    plot_stardist_confusion_extra,
    plot_stardist_f1_extra,
    plot_stardist_roc_extra,
    plot_stardist_spatial_extra,
    stardist_head_preds,
    macro_auc_ovr,
)

DEFAULT_DATA_ROOT = (
    _REPO_ROOT
    / "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data"
)
DEFAULT_PYTHON_ROOT = Path("/home/lingyu/ssd2/Python/")
DEFAULT_ABLATION_TAG = "D_emph_L2"
DEFAULT_VAL_SELECTION_METRIC = "five_tier_auc_sum"
DEFAULT_CV_SELECTION_METRIC = "five_tier_auc_sum"
DEFAULT_SAVE_RESULT = "result"
DEFAULT_PER_SAMPLE_SAVE_RESULT = "result"
DEFAULT_POOLED_SAVE_RESULT = "result_all"
DEFAULT_SPATIAL_K = 8
DEFAULT_SPATIAL_MODE = "mean"
DEFAULT_TRAIN_BATCH_SIZE = 4096


def _train_loader_kwargs(seed: int, train_batch_size: int) -> dict:
    return {
        "seed": seed,
        "train_balance_sampler": False,
        "num_workers_cuda": 0,
        "batch_size_cuda": train_batch_size,
    }


def resolve_model_checkpoint(ctx: RunContext, explicit: str | None = None) -> str:
    """Prefer explicit path, then ``best_mlp_gpu.pt`` under cases_root, then k-fold ckpts."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    local_best = ctx.cases_root / ctx.therapy_data / ctx.therapy_model / "best_mlp_gpu.pt"
    if local_best.is_file():
        return str(local_best)

    ckpt_dir = (
        ctx.cases_root
        / ctx.therapy_data
        / ctx.therapy_model
        / "ablation_kfold_ckpts"
        / ctx.ablation_tag
    )
    folds = sorted(ckpt_dir.glob("hce_kfold_fold_*.pt"))
    if folds:
        print(
            f"  WARNING: {local_best.name} not found; using {folds[0].name}. "
            "Run --steps train to sync the best fold.",
            flush=True,
        )
        return str(folds[0])

    legacy = get_select4_best_checkpoint_path(
        str(ctx.python_root), ctx.therapy_data, ctx.therapy_model
    )
    if os.path.isfile(legacy):
        return legacy

    raise FileNotFoundError(
        f"No checkpoint under {ctx.cases_root / ctx.therapy_data / ctx.therapy_model}. "
        "Run --steps train first or pass --checkpoint-path."
    )


@dataclass
class RunContext:
    """Per-sample runtime state (mirrors notebook globals after each section)."""

    sample: str
    cases_root: Path
    python_root: Path
    therapy_data: str
    therapy_model: str
    save_result: str
    device: torch.device
    seed: int
    match_tolerance: float
    column_rename: dict
    force_rebuild_h5ad: bool
    input_dim: int | None
    hidden_dims: tuple[int, ...]
    cv_k: int
    stratify_target: str
    patience: int
    max_epochs: int
    train_batch_size: int
    resume_from_checkpoints: bool
    ablation_tag: str
    hce_w1: float
    hce_w2: float
    hce_w12: float
    hce_w_l12head: float
    hce_w_l3: float
    hce_w_l4: float
    build_stardist_h5ad: bool
    val_selection_metric: str = DEFAULT_VAL_SELECTION_METRIC
    cv_selection_metric: str = DEFAULT_CV_SELECTION_METRIC
    auto_cv_k: bool = True
    g: dict = field(default_factory=dict)

    @property
    def sample_dir(self) -> Path:
        return self.cases_root / self.therapy_data

    @property
    def matched_he_h5ad(self) -> Path:
        return self.sample_dir / f"{self.therapy_data}_matched_features.h5ad"

    @property
    def matched_stardist_h5ad(self) -> Path:
        return self.sample_dir / f"{self.therapy_data}_matched_features_stardist.h5ad"

    def result_fig(self, name: str) -> str:
        fn, _ = make_result_fig(
            self.cases_root, self.therapy_data, self.therapy_model, self.save_result
        )
        return fn(name)


@dataclass
class PooledRunContext:
    """Cross-dataset training on all Complete_Cases; outputs under ``Data/result/``."""

    data_root: Path
    cases_root: Path
    python_root: Path
    samples: list[str]
    device: torch.device
    seed: int
    save_result: str
    input_dim: int | None
    hidden_dims: tuple[int, ...]
    cv_k: int
    train_group_frac: float
    patience: int
    max_epochs: int
    train_batch_size: int
    resume_from_checkpoints: bool
    ablation_tag: str
    hce_w1: float
    hce_w2: float
    hce_w12: float
    hce_w_l12head: float
    hce_w_l3: float
    hce_w_l4: float
    val_selection_metric: str = DEFAULT_VAL_SELECTION_METRIC
    cv_selection_metric: str = DEFAULT_CV_SELECTION_METRIC
    use_spatial_context: bool = False
    spatial_k: int = DEFAULT_SPATIAL_K
    spatial_mode: str = DEFAULT_SPATIAL_MODE
    g: dict = field(default_factory=dict)

    @property
    def result_dir(self) -> Path:
        return self.data_root / self.save_result

    @property
    def ckpt_dir(self) -> Path:
        return self.result_dir / "cross_dataset_cv" / self.ablation_tag

    def result_fig(self, name: str) -> str:
        fn, _ = make_pooled_result_fig(self.data_root, self.save_result)
        return fn(name)

    def stardist_result_fig(self, sample: str):
        fn, _ = make_pooled_stardist_result_fig(self.data_root, sample, self.save_result)
        return fn


def setup_cuda(cuda_device: str) -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def setup_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda:0")
    if allow_cpu:
        print("WARNING: CUDA unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA unavailable. Use a CUDA PyTorch build or pass --allow-cpu-train."
    )


def setup_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def resolve_cases_root(args: argparse.Namespace) -> Path:
    if args.cases_root is not None:
        return args.cases_root.expanduser().resolve()
    sub = "Complete_Cases" if args.cases_set == "complete" else "Incomplete_Cases"
    return (DEFAULT_DATA_ROOT / sub).resolve()


def resolve_steps(raw: list[str]) -> set[str]:
    if "all" in raw:
        return {"he_h5ad", "train", "he_validate", "stardist"}
    return set(raw)


def _min_stratum_count(cv_data: dict, stratify_target: str) -> int:
    y_enc = np.asarray(cv_data["y_encoded_f"])
    y_l1_enc = np.asarray(cv_data["y_level1_encoded_f"])
    if stratify_target == "level2":
        y_strat = y_enc
    elif stratify_target == "level1":
        y_strat = y_l1_enc
    elif stratify_target == "joint":
        y_strat = np.array(
            [f"{int(a)}|{int(b)}" for a, b in zip(y_l1_enc, y_enc)],
            dtype=object,
        )
    else:
        raise ValueError(f"Unknown stratify_target: {stratify_target!r}")
    _, counts = np.unique(y_strat, return_counts=True)
    return int(np.min(counts)) if counts.size else 0


def resolve_stratified_cv_params(
    cv_data: dict,
    cv_k: int,
    stratify_target: str,
    *,
    auto_adjust: bool = True,
) -> tuple[int, str]:
    """
    Pick ``n_splits`` and ``stratify_target`` valid for StratifiedKFold.

    With ``auto_adjust=True`` (default), coarsens joint→level2→level1 and reduces
    ``cv_k`` to the smallest stratum count when rare L1|L2 pairs block k-fold.
    """
    if int(cv_k) < 2:
        raise ValueError(f"--cv-k must be >= 2, got {cv_k}")

    if stratify_target == "joint":
        targets = ["joint", "level2", "level1"]
    elif stratify_target == "level2":
        targets = ["level2", "level1"]
    elif stratify_target == "level1":
        targets = ["level1"]
    else:
        raise ValueError(f"Unknown stratify_target: {stratify_target!r}")

    if not auto_adjust:
        min_count = _min_stratum_count(cv_data, stratify_target)
        if min_count < cv_k:
            raise ValueError(
                f"Cannot run StratifiedKFold(n_splits={cv_k}): minimum class/group "
                f"count in stratification target is {min_count}. Reduce --cv-k, use "
                f"coarser --stratify-target, or omit --no-auto-cv-k."
            )
        return cv_k, stratify_target

    for target in targets:
        min_count = _min_stratum_count(cv_data, target)
        if min_count < 2:
            continue
        effective_k = min(cv_k, min_count)
        if effective_k != cv_k or target != stratify_target:
            print(
                f"  ⚠ Auto CV: n_splits {cv_k}→{effective_k}, "
                f"stratify {stratify_target!r}→{target!r} "
                f"(min stratum count={min_count})",
                flush=True,
            )
        return effective_k, target

    raise ValueError(
        "Cannot run StratifiedKFold: some strata have <2 cells even at level1. "
        "Filter rare labels or train on pooled samples."
    )


def step_he_h5ad(ctx: RunContext) -> None:
    print(f"\n[HE h5ad] {ctx.matched_he_h5ad.name}", flush=True)
    cell_coords = ctx.sample_dir / (
        f"{ctx.therapy_data}_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
    )
    emb_dir = ctx.sample_dir / ctx.therapy_model / "ImgEmbeddings_all/sc_pth_16_16"
    adata = match_hist2cell_h5ad(
        cell_coords_path=str(cell_coords),
        hist_embedding_dir=emb_dir,
        matched_h5ad_path=str(ctx.matched_he_h5ad),
        coord_cols=("X_pix_HE", "Y_pix_HE"),
        tolerance=ctx.match_tolerance,
        pth_prefix=ctx.therapy_data,
        column_rename=ctx.column_rename,
        auto_rename=False,
        force_rebuild=ctx.force_rebuild_h5ad,
    )
    ctx.g["adata"] = adata
    ctx.g["matched_features_path"] = str(ctx.matched_he_h5ad)
    print(f"  → {adata.n_obs:,} cells × {adata.n_vars} features", flush=True)


def step_prepare_cv(ctx: RunContext) -> None:
    adata = ctx.g["adata"]
    cv_data = prepare_data_from_matched_h5ad(
        adata,
        groups=np.arange(adata.n_obs, dtype=np.int64),
    )
    use_five_head = all(c in adata.obs.columns for c in ("final_sublineage", "CNiche", "TNiche"))
    if use_five_head:
        print(
            "  Five-head training: L2=final_CT, L1=final_lineage, "
            "L12=final_sublineage, L3=CNiche, L4=TNiche",
            flush=True,
        )

    scaler = StandardScaler()
    scaler.fit(cv_data["X_f"])

    ctx.g.update(
        {
            "cv_data": cv_data,
            "scaler": scaler,
            "class_names": cv_data["class_names"],
            "class_names_level1": cv_data["class_names_level1"],
            "class_names_level12": cv_data.get("class_names_level12"),
            "class_names_level3": cv_data.get("class_names_level3"),
            "class_names_level4": cv_data.get("class_names_level4"),
            "y_encoded_f": cv_data["y_encoded_f"],
            "y_level1_encoded_f": cv_data["y_level1_encoded_f"],
            "X_coords_plot": cv_data.get("X_coords_f"),
            "USE_FIVE_HEAD": use_five_head,
        }
    )
    inferred_dim = int(cv_data["X_f"].shape[1])
    if ctx.input_dim is not None and ctx.input_dim != inferred_dim:
        print(
            f"  ⚠ --input-dim={ctx.input_dim} differs from data dim {inferred_dim}; "
            "using data dimension.",
            flush=True,
        )
    ctx.g["input_dim"] = inferred_dim
    print(f"  input_dim={inferred_dim}", flush=True)


def step_train(ctx: RunContext) -> None:
    if "cv_data" not in ctx.g:
        step_he_h5ad(ctx)
        step_prepare_cv(ctx)

    cv_data = ctx.g["cv_data"]
    scaler = ctx.g["scaler"]
    class_names = ctx.g["class_names"]
    ckpt_dir = (
        ctx.cases_root
        / ctx.therapy_data
        / ctx.therapy_model
        / "ablation_kfold_ckpts"
        / ctx.ablation_tag
    )

    cv_k, stratify_target = resolve_stratified_cv_params(
        cv_data,
        ctx.cv_k,
        ctx.stratify_target,
        auto_adjust=ctx.auto_cv_k,
    )
    print(
        f"\n[Train] stratified_kfold tag={ctx.ablation_tag} k={cv_k} "
        f"stratify={stratify_target!r}",
        flush=True,
    )
    lp = run_stratified_kfold_cv_with_insample_report(
        device=ctx.device,
        cv_data=cv_data,
        scaler=scaler,
        class_names=class_names,
        evaluate=evaluate,
        path=str(ctx.python_root),
        therapy_data=ctx.therapy_data,
        therapy_model=ctx.therapy_model,
        hce_w1=ctx.hce_w1,
        hce_w2=ctx.hce_w2,
        hce_w12=ctx.hce_w12,
        hce_w_l12head=ctx.hce_w_l12head,
        hce_w_l3=ctx.hce_w_l3,
        hce_w_l4=ctx.hce_w_l4,
        n_splits=cv_k,
        stratify_target=stratify_target,
        patience=ctx.patience,
        max_epochs=ctx.max_epochs,
        loader_kwargs=_train_loader_kwargs(ctx.seed, ctx.train_batch_size),
        resume_from_checkpoints=ctx.resume_from_checkpoints,
        kfold_checkpoint_dir=str(ckpt_dir),
        hidden_dims=ctx.hidden_dims,
        val_selection_metric=ctx.val_selection_metric,
        cv_selection_metric=ctx.cv_selection_metric,
    )

    best_fold = lp["best_fold"]
    dest = ctx.cases_root / ctx.therapy_data / ctx.therapy_model / "best_mlp_gpu.pt"
    best_ckpt = sync_best_mlp_from_logo_fold(
        str(ctx.python_root),
        ctx.therapy_data,
        ctx.therapy_model,
        best_fold["checkpoint"],
        dest_path=str(dest),
    )
    print(f"  Best fold={best_fold['fold']}  checkpoint={best_ckpt}", flush=True)

    ctx.g.update(lp)
    ctx.g["LP"] = lp
    ctx.g["BEST_MLP_CHECKPOINT"] = best_ckpt
    ctx.g["model"] = lp["model"]
    ctx.g["hce_w1"] = ctx.hce_w1
    ctx.g["hce_w2"] = ctx.hce_w2
    ctx.g["hce_w12"] = ctx.hce_w12
    ctx.g["hce_w_l12head"] = ctx.hce_w_l12head
    ctx.g["hce_w_l3"] = ctx.hce_w_l3
    ctx.g["hce_w_l4"] = ctx.hce_w_l4


def step_he_validate(ctx: RunContext) -> None:
    if "LP" not in ctx.g:
        raise RuntimeError("Run training first (--steps train or all).")

    g = ctx.g
    cv_data = g["cv_data"]
    scaler = g["scaler"]
    model = g["model"]
    rf = ctx.result_fig

    if g.get("model") is not None:
        ensure_lp_extra_insample_preds(
            g["model"], scaler, cv_data, ctx.device, predict_all_label_heads, g
        )

    print("\n[HE validate] confusion matrices", flush=True)
    plot_he_confusion_matrices(plot_confusion_matrix, rf, cv_data, g)

    logo_summary = g.get("logo_summary") or g["LP"].get("logo_summary")
    insample_tiers, num_classes = build_insample_tier_metrics(g, cv_data)
    tier_extra = collect_hce_tier_metrics_extras(
        logo_summary=logo_summary,
        insample=insample_tiers,
        num_classes=num_classes,
    )
    mean_best_epoch = int(
        round(float(np.mean([f["best_epoch"] for f in logo_summary["folds"]])))
    )
    l2_ins = insample_tiers.get("l2") or {}
    l1_ins = insample_tiers.get("l1") or {}
    save_hce_validation_metrics(
        val_acc=l2_ins.get("accuracy", float("nan")),
        val_macro_f1=logo_summary["l2_macro_f1_mean"],
        val_weighted_f1=logo_summary["l2_weighted_f1_mean"],
        val_level1_acc=l1_ins.get("accuracy", float("nan")),
        val_level1_macro_f1=logo_summary["l1_macro_f1_mean"],
        val_level1_weighted_f1=logo_summary["l1_weighted_f1_mean"],
        class_names=g["class_names"],
        class_names_level1=g["class_names_level1"],
        best_epoch=mean_best_epoch,
        hce_lambda=None,
        hce_w1=g.get("hce_w1", 1.0),
        hce_w2=g.get("hce_w2", 1.0),
        hce_w12=g.get("hce_w12", 2.0),
        hce_w_l12head=g.get("hce_w_l12head", 1.0),
        hce_w_l3=g.get("hce_w_l3", 1.0),
        hce_w_l4=g.get("hce_w_l4", 1.0),
        therapy_data=f"{ctx.therapy_data}_kfold_CV",
        metrics_csv_path=rf("validation_internal_metrics"),
        extra_metrics=tier_extra,
    )

    print("[HE validate] per-class F1", flush=True)
    plot_per_class_f1(
        g["val_labels"],
        g["val_preds"],
        g["class_names"],
        model=model,
        device=ctx.device,
        train_dataset=g.get("train_dataset"),
        val_dataset=g.get("val_dataset"),
        input_dim=g["input_dim"],
        best_epoch=g.get("best_epoch"),
        test_acc=g["val_acc"],
        test_macro_f1=g["val_macro_f1"],
        test_weighted_f1=g["val_weighted_f1"],
        model_name="MLP-Level2",
        y_sort_by="f1_asc",
        save_path=rf("f1_perclass_level2"),
    )
    plot_per_class_f1(
        g["val_labels_level1"],
        g["val_preds_level1"],
        g["class_names_level1"],
        model=model,
        device=ctx.device,
        train_dataset=g.get("train_dataset"),
        val_dataset=g.get("val_dataset"),
        input_dim=g["input_dim"],
        best_epoch=g.get("best_epoch"),
        test_acc=g["val_level1_acc"],
        test_macro_f1=g["val_level1_macro_f1"],
        test_weighted_f1=g["val_level1_weighted_f1"],
        model_name="MLP-Level1",
        y_sort_by="f1_asc",
        save_path=rf("f1_perclass_level1"),
    )
    if g.get("val_preds_level1_head") is not None:
        plot_per_class_f1(
            g["val_labels_level1"],
            g["val_preds_level1_head"],
            g["class_names_level1"],
            model=model,
            device=ctx.device,
            train_dataset=g.get("train_dataset"),
            val_dataset=g.get("val_dataset"),
            input_dim=g["input_dim"],
            best_epoch=g.get("best_epoch"),
            test_acc=g["val_level1_acc"],
            test_macro_f1=g["val_level1_macro_f1"],
            test_weighted_f1=g["val_level1_weighted_f1"],
            model_name="MLP-Level1-head",
            y_sort_by="f1_asc",
            save_path=rf("f1_perclass_level1_L1head"),
        )
    plot_he_f1_extra_tiers(
        plot_per_class_f1,
        rf,
        model,
        ctx.device,
        g.get("train_dataset"),
        g.get("val_dataset"),
        g["input_dim"],
        g.get("best_epoch"),
        g,
    )

    print("[HE validate] spatial plots", flush=True)
    evaluate_and_plot_on_all_data(
        model=model,
        matched_features_path="",
        class_names=g["class_names"],
        evaluate=evaluate,
        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
        CellTypeDataset=CellTypeDataset,
        device=ctx.device,
        scaler=scaler,
        X=cv_data["X_f"],
        y=cv_data["y_f"],
        y_encoded=cv_data["y_encoded_f"],
        X_coords_matched=g.get("X_coords_plot"),
        celltype_pred_dir=rf("celltype_valid_level2"),
        celltype_true_dir=rf("celltype_true_level2"),
        spatial_plot_mode="pred_true_l2",
        spatial_color_scheme="xenium_ct",
        spatial_title_pred_l2=f"{ctx.therapy_data} predicted level2",
        spatial_title_true_l2=f"{ctx.therapy_data} ground truth level2",
    )
    plot_level1_spatial_distribution(
        matched_features_path=g["matched_features_path"],
        all_preds=g["val_preds"],
        class_names_level1=g["class_names_level1"],
        class_names=g["class_names"],
        y_encoded_f=g["y_encoded_f"],
        y_level1_encoded_f=g["y_level1_encoded_f"],
        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
        save_path_pred=rf("celltype_valid_level1"),
        save_path_true=rf("celltype_true_level1"),
        spatial_color_scheme="xenium_lineage",
        X_coords_matched=g.get("X_coords_plot"),
        y_level1_f=cv_data["y_level1_f"],
        spatial_title_pred_l1=f"{ctx.therapy_data} predicted level1",
        spatial_title_true_l1=f"{ctx.therapy_data} ground truth level1",
    )
    plot_he_spatial_extra_tiers(
        plot_tier_spatial_distribution,
        plot_celltype_spatial_distribution,
        rf,
        ctx.therapy_data,
        g.get("X_coords_plot"),
        cv_data,
        g,
    )


def _ensure_stardist_h5ad(ctx: RunContext) -> None:
    if ctx.matched_stardist_h5ad.is_file() and not ctx.force_rebuild_h5ad:
        print(f"  StarDist h5ad exists: {ctx.matched_stardist_h5ad}", flush=True)
        return
    if not ctx.build_stardist_h5ad:
        raise FileNotFoundError(
            f"StarDist h5ad not found: {ctx.matched_stardist_h5ad}. "
            "Run transer_embedding_label_h5ad.py first or pass --build-stardist-h5ad."
        )

    print("[StarDist h5ad] building matched features...", flush=True)
    stardist_raw = ctx.sample_dir / f"{ctx.therapy_data}_Float_prob0.01_nms_0.3.csv"
    xenium_cell = ctx.sample_dir / (
        f"{ctx.therapy_data}_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
    )
    stardist_csv = ctx.sample_dir / f"{ctx.therapy_data}_cells_matched_by_stardist.csv"
    emb_dir = ctx.sample_dir / ctx.therapy_model / "ImgEmbeddings_all_stardist/sc_pth_16_16"

    celltype_xenium = load_cell_pixcoords(
        str(xenium_cell), column_rename=ctx.column_rename, auto_rename=False
    )
    star_raw = pd.read_csv(stardist_raw)
    matched = match_celltype2stardist(
        celltype_xenium,
        star_raw,
        celltype_pixel_coords_cols=("X_pix_HE", "Y_pix_HE"),
        stardist_pixel_coords_cols=("centroid_x", "centroid_y"),
    )
    matched = matched.dropna(subset=["centroid_x", "centroid_y"])
    restore = {
        tgt: src for src, tgt in ctx.column_rename.items() if tgt in matched.columns
    }
    matched.rename(columns=restore).to_csv(stardist_csv, index=False)

    adata_star = match_hist2cell_h5ad(
        cell_coords_path=str(stardist_csv),
        hist_embedding_dir=emb_dir,
        matched_h5ad_path=str(ctx.matched_stardist_h5ad),
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=ctx.match_tolerance,
        pth_prefix=ctx.therapy_data,
        level1_name="celltype_level1",
        column_rename=ctx.column_rename,
        auto_rename=False,
        spatial_cols=("centroid_x", "centroid_y"),
        spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
        force_rebuild=ctx.force_rebuild_h5ad,
    )
    ctx.g["adata_star"] = adata_star
    print(f"  → {adata_star.n_obs:,} cells", flush=True)


def _load_stardist_arrays(ctx: RunContext) -> None:
    if not ctx.matched_stardist_h5ad.is_file():
        raise FileNotFoundError(ctx.matched_stardist_h5ad)
    if "adata_star" not in ctx.g:
        import anndata as ad

        ctx.g["adata_star"] = ad.read_h5ad(ctx.matched_stardist_h5ad)
    adata_star = ctx.g["adata_star"]
    ctx.g.update(
        {
            "matched_features_stardist_path": str(ctx.matched_stardist_h5ad),
            "X_star": adata_X_to_dense(adata_star.X),
            "y_star_level1": adata_star.obs["final_lineage"].to_numpy(),
            "y_star_level12": adata_star.obs["final_sublineage"].to_numpy(),
            "y_star_level3": adata_star.obs["CNiche"].to_numpy(),
            "y_star_level4": adata_star.obs["TNiche"].to_numpy(),
            "X_coords_star": adata_star.obsm["spatial"],
        }
    )


def step_stardist(ctx: RunContext, checkpoint_path: str | None = None) -> None:
    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        if ctx.matched_he_h5ad.is_file():
            step_he_h5ad(ctx)
            step_prepare_cv(ctx)
        else:
            raise RuntimeError(
                "Need trained scaler/class_names. Run --steps train first or ensure HE h5ad exists."
            )

    _ensure_stardist_h5ad(ctx)
    _load_stardist_arrays(ctx)

    g = ctx.g
    rf = ctx.result_fig
    class_names_star = g["class_names"]
    scaler = g["scaler"]
    cv_data = g["cv_data"]

    star_ckpt = resolve_model_checkpoint(ctx, checkpoint_path or g.get("BEST_MLP_CHECKPOINT"))
    print(f"\n[StarDist] load model from {star_ckpt}", flush=True)
    model_star = load_model_for_predict(
        str(ctx.python_root),
        ctx.therapy_data,
        ctx.therapy_model,
        parent_dir=True,
        checkpoint_path=star_ckpt,
        device=ctx.device,
    )
    g["model_star"] = model_star

    all_acc, all_macro_f1, all_weighted_f1, all_preds, all_labels = (
        evaluate_and_plot_on_all_data(
            model=model_star,
            matched_features_path=g["matched_features_stardist_path"],
            class_names=class_names_star,
            evaluate=evaluate,
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            CellTypeDataset=CellTypeDataset,
            device=ctx.device,
            scaler=scaler,
            prediction_only=True,
            X_coords_matched=g["X_coords_star"],
            y_level1_f=g["y_star_level1"],
            celltype_pred_dir=rf("stardist_pred_level2"),
            spatial_plot_mode="pred_true_l2",
            spatial_color_scheme="xenium_ct",
            spatial_title_pred_l2=f"{ctx.therapy_data} StarDist pred level2",
            spatial_title_true_l2=f"{ctx.therapy_data} ground truth level2",
        )
    )
    g.update(
        {
            "all_acc": all_acc,
            "all_macro_f1": all_macro_f1,
            "all_weighted_f1": all_weighted_f1,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }
    )

    plot_level1_spatial_distribution(
        matched_features_path=g["matched_features_stardist_path"],
        all_preds=all_preds,
        class_names_level1=g["class_names_level1"],
        class_names=class_names_star,
        y_encoded_f=g["y_encoded_f"],
        y_level1_encoded_f=g["y_level1_encoded_f"],
        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
        save_path_pred=rf("stardist_pred_level1"),
        spatial_color_scheme="xenium_lineage",
        X_coords_matched=g["X_coords_star"],
        y_level1_f=g["y_star_level1"],
        spatial_title_pred_l1=f"{ctx.therapy_data} StarDist pred level1",
        spatial_title_true_l1=f"{ctx.therapy_data} ground truth level1",
    )

    if hasattr(model_star, "level12_head"):
        star_heads = stardist_head_preds(
            model_star, scaler, g["X_star"], ctx.device, predict_all_label_heads
        )
        plot_stardist_spatial_extra(
            plot_tier_spatial_distribution,
            plot_celltype_spatial_distribution,
            rf,
            ctx.therapy_data,
            g["X_coords_star"],
            star_heads,
            cv_data,
            g,
        )
    else:
        print("  Skip L12/L3/L4 StarDist spatial: not five-head.", flush=True)

    print("[StarDist] per-class accuracy", flush=True)
    plot_per_class_accuracy(
        all_labels,
        all_preds,
        class_names_star,
        model=model_star,
        device=ctx.device,
        test_acc=all_acc,
        test_macro_f1=all_macro_f1,
        test_weighted_f1=all_weighted_f1,
        model_name="MLP-StarDist-Level2",
        y_sort_by="acc_asc",
        save_path=rf("acc_level2_stardist"),
    )
    plot_level1_accuracy_from_level2_predictions(
        matched_features_path=g["matched_features_stardist_path"],
        all_preds=all_preds,
        class_names=class_names_star,
        class_names_level1=g["class_names_level1"],
        y_encoded_f=g["y_encoded_f"],
        y_level1_encoded_f=g["y_level1_encoded_f"],
        plot_per_class_accuracy=plot_per_class_accuracy,
        model=model_star,
        device=ctx.device,
        model_name="MLP-StarDist-Level1",
        y_sort_by="acc_asc",
        save_path=rf("acc_level1_stardist"),
    )
    plot_level1_accuracy_from_level1_head(
        matched_features_path=g["matched_features_stardist_path"],
        model=model_star,
        device=ctx.device,
        scaler=scaler,
        class_names_level1=g["class_names_level1"],
        plot_per_class_accuracy=plot_per_class_accuracy,
        model_name="MLP-StarDist-Level1-head",
        y_sort_by="acc_asc",
        save_path=rf("acc_level1_stardist_L1head"),
    )
    if hasattr(model_star, "level12_head"):
        star_heads = stardist_head_preds(
            model_star, scaler, g["X_star"], ctx.device, predict_all_label_heads
        )
        plot_stardist_acc_extra(
            plot_per_class_accuracy, rf, model_star, ctx.device, star_heads, cv_data, g
        )

    print("[StarDist] confusion matrices", flush=True)
    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names_star,
        figsize=(10, 8),
        save_path=rf("conf_matrix_level2_stardist"),
    )
    if hasattr(model_star, "level12_head"):
        star_heads = stardist_head_preds(
            model_star, scaler, g["X_star"], ctx.device, predict_all_label_heads
        )
        plot_stardist_confusion_extra(plot_confusion_matrix, rf, star_heads, cv_data, g)

    print("[StarDist] per-class F1", flush=True)
    plot_per_class_f1(
        all_labels,
        all_preds,
        class_names_star,
        model=model_star,
        device=ctx.device,
        test_acc=all_acc,
        test_macro_f1=all_macro_f1,
        test_weighted_f1=all_weighted_f1,
        y_sort_by="f1_asc",
        model_name="MLP-StarDist-Level2",
        save_path=rf("f1_level2_stardist"),
    )
    plot_level1_f1_from_level2_predictions(
        matched_features_path=g["matched_features_stardist_path"],
        all_preds=all_preds,
        class_names=class_names_star,
        class_names_level1=g["class_names_level1"],
        y_encoded_f=g["y_encoded_f"],
        y_level1_encoded_f=g["y_level1_encoded_f"],
        plot_per_class_f1=plot_per_class_f1,
        model=model_star,
        device=ctx.device,
        y_sort_by="f1_asc",
        model_name="MLP-StarDist-Level1",
        save_path=rf("f1_level1_stardist"),
    )
    plot_level1_f1_from_level1_head(
        matched_features_path=g["matched_features_stardist_path"],
        model=model_star,
        device=ctx.device,
        scaler=scaler,
        class_names_level1=g["class_names_level1"],
        plot_per_class_f1=plot_per_class_f1,
        y_sort_by="f1_asc",
        model_name="MLP-StarDist-Level1-head",
        save_path=rf("f1_level1_stardist_L1head"),
    )
    if hasattr(model_star, "level12_head"):
        star_heads = stardist_head_preds(
            model_star, scaler, g["X_star"], ctx.device, predict_all_label_heads
        )
        plot_stardist_f1_extra(
            plot_per_class_f1, rf, model_star, ctx.device, star_heads, cv_data, g
        )

    print("[StarDist] ROC curves", flush=True)
    bundle = load_matched_features_bundle(g["matched_features_stardist_path"])
    x_star_scaled = scaler.transform(bundle["X"])
    probs_l2 = mlp_collect_softmax_probs(model_star, x_star_scaled, ctx.device)
    probs_heads = None
    if hasattr(model_star, "level12_head"):
        probs_heads = mlp_collect_five_head_softmax_probs(
            model_star, x_star_scaled, ctx.device
        )
    if "y" in bundle:
        y_roc = encode_labels_with_class_names(bundle["y"], class_names_star)
        m = min(len(probs_l2), len(y_roc))
        valid = (y_roc[:m] >= 0) & (y_roc[:m] < len(class_names_star))
        if np.any(valid):
            y_l1_roc = bundle["y_level1"][:m][valid] if "y_level1" in bundle else None
            x_roc = x_star_scaled[:m][valid]
            probs_roc = probs_l2[:m][valid]
            y_v = y_roc[:m][valid]
            plot_multiclass_roc_curves(
                y_v,
                probs_roc,
                class_names_star,
                figsize=(3.0, 3.0),
                max_curves=len(class_names_star),
                save_path=rf("roc_stardist_level2"),
                title="StarDist Level2 ROC",
                roc_color_scheme="xenium_ct",
            )
            plot_level1_roc_from_level2_scores(
                g["matched_features_stardist_path"],
                probs_roc,
                class_names_star,
                g["class_names_level1"],
                g["y_encoded_f"],
                g["y_level1_encoded_f"],
                figsize=(3.0, 3.0),
                save_path=rf("roc_stardist_level1"),
                title="StarDist Level1 ROC from L2 probs",
                roc_color_scheme="xenium_lineage",
                y_level1_f=y_l1_roc,
            )
            plot_level1_roc_from_level1_head(
                g["matched_features_stardist_path"],
                model_star,
                ctx.device,
                scaler,
                g["class_names_level1"],
                figsize=(3.0, 3.0),
                save_path=rf("roc_stardist_level1_L1head"),
                title="StarDist Level1 ROC from L1 head (softmax)",
                roc_color_scheme="xenium_lineage",
                X_f=x_roc,
                y_level1_f=y_l1_roc,
            )
            if hasattr(model_star, "level12_head") and probs_heads is not None:
                x_roc_len = len(x_roc)
                probs_heads_roc = {
                    k: v[:x_roc_len] for k, v in probs_heads.items()
                }
                plot_stardist_roc_extra(
                    plot_multiclass_roc_curves,
                    rf,
                    probs_heads_roc,
                    cv_data,
                    len(y_v),
                    g,
                )

    print("[StarDist] external metrics CSV", flush=True)
    logo_summary = g.get("logo_summary") or (g.get("LP") or {}).get("logo_summary")
    metrics_df, combined_df, tier_aucs = save_stardist_external_validation_metrics(
        all_labels=all_labels,
        all_preds=all_preds,
        matched_features_path=g["matched_features_stardist_path"],
        class_names_star=class_names_star,
        class_names_level1=g["class_names_level1"],
        y_encoded_f=g["y_encoded_f"],
        y_level1_encoded_f=g["y_level1_encoded_f"],
        therapy_data=ctx.therapy_data,
        metrics_csv_path=rf("validation_external_stardist_matched_metrics"),
        model_star=model_star,
        scaler=scaler,
        X_star=g["X_star"],
        device=ctx.device,
        predict_all_label_heads=predict_all_label_heads,
        cv_data=cv_data,
        g=g,
        logo_summary=logo_summary,
        hce_w1=g.get("hce_w1", 1.0),
        hce_w2=g.get("hce_w2", 1.0),
        hce_w12=g.get("hce_w12", 2.0),
        hce_w_l12head=g.get("hce_w_l12head", 1.0),
        hce_w_l3=g.get("hce_w_l3", 1.0),
        hce_w_l4=g.get("hce_w_l4", 1.0),
        probs_l2=probs_l2,
        head_probs=probs_heads,
    )
    print(
        "\nStarDist matched per-tier columns "
        "(insample_* = StarDist-matched cells, best-fold model):",
        flush=True,
    )
    tier_cols = [
        c
        for c in combined_df.columns
        if c.startswith(("insample_", "num_", "val_"))
        and ("auc" in c or not c.endswith("_std"))
    ]
    print(combined_df[tier_cols].tail(1).T, flush=True)
    print("\nMacro AUROC by tier:", flush=True)
    for tier, val in sorted(tier_aucs.items()):
        print(f"  {tier}: {val:.4f}", flush=True)

    metrics_csv = rf("validation_external_stardist_matched_metrics")
    auroc_csv = _auroc_csv_path_from_metrics(metrics_csv)
    print(f"\nLevel2 AUROC table: {auroc_csv}", flush=True)
    if Path(auroc_csv).is_file():
        print("[StarDist] ROC from AUROC CSV", flush=True)
        roc_info_l2_csv = plot_stardist_roc_from_auroc_csv(
            auroc_csv,
            class_names_star,
            save_path=rf("roc_stardist_level2_from_AUROC_csv"),
            title="StarDist Level2 ROC (from AUROC CSV)",
            roc_color_scheme="xenium_ct",
        )
        if roc_info_l2_csv:
            print(
                f"  Macro AUROC from CSV: {roc_info_l2_csv['macro_auc']:.4f}",
                flush=True,
            )
    else:
        print("  ⚠ AUROC CSV not found; skip ROC replay.", flush=True)


########################################################
# 2026.06.25 LLY: use Xenium_lung Complete_Cases 25 datasets for training and validation
########################################################
def _stardist_neighbor_index(ctx: PooledRunContext, model, X_coords) -> np.ndarray | None:
    """Per-sample kNN on StarDist spatial coords when the model uses spatial context."""
    if not ctx.use_spatial_context or not getattr(model, "use_spatial_context", False):
        return None
    return build_spatial_neighbor_index(np.asarray(X_coords), k_neighbors=ctx.spatial_k)


def step_pooled_prepare(ctx: PooledRunContext) -> None:
    print(f"\n[Pooled prepare] {len(ctx.samples)} datasets", flush=True)
    cv_data = prepare_pooled_cv_data_from_h5ads(ctx.cases_root, ctx.samples)
    scaler = StandardScaler()
    scaler.fit(cv_data["X_f"])
    use_five_head = all(
        k in cv_data for k in ("y_level12_encoded_f", "y_level3_encoded_f", "y_level4_encoded_f")
    )
    ctx.g.update(
        {
            "cv_data": cv_data,
            "scaler": scaler,
            "class_names": cv_data["class_names"],
            "class_names_level1": cv_data["class_names_level1"],
            "class_names_level12": cv_data.get("class_names_level12"),
            "class_names_level3": cv_data.get("class_names_level3"),
            "class_names_level4": cv_data.get("class_names_level4"),
            "USE_FIVE_HEAD": use_five_head,
            "input_dim": int(cv_data["X_f"].shape[1]),
        }
    )
    print(f"  input_dim={ctx.g['input_dim']}", flush=True)
    if ctx.use_spatial_context:
        if "X_coords_f" not in cv_data:
            raise ValueError("use_spatial_context=True requires X_coords_f in pooled cv_data.")
        nbr_idx = _build_spatial_neighbor_index_for_cv_data(cv_data, k_neighbors=ctx.spatial_k)
        ctx.g["spatial_neighbor_index"] = nbr_idx
        print(
            f"  spatial: k={ctx.spatial_k}, mode={ctx.spatial_mode!r}, "
            f"per-sample kNN index {nbr_idx.shape}",
            flush=True,
        )


def step_pooled_train(ctx: PooledRunContext) -> None:
    if "cv_data" not in ctx.g:
        step_pooled_prepare(ctx)
    cv_data = ctx.g["cv_data"]
    class_names = ctx.g["class_names"]
    ctx.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Pooled train] group CV k={ctx.cv_k} train_frac={ctx.train_group_frac} "
        f"tag={ctx.ablation_tag} val_sel={ctx.val_selection_metric} cv_sel={ctx.cv_selection_metric}"
        + (f" spatial_k={ctx.spatial_k}" if ctx.use_spatial_context else ""),
        flush=True,
    )
    lp = run_group_kfold_cv_with_oof_report(
        device=ctx.device,
        cv_data=cv_data,
        scaler=ctx.g["scaler"],
        class_names=class_names,
        evaluate=evaluate,
        hce_w1=ctx.hce_w1,
        hce_w2=ctx.hce_w2,
        hce_w12=ctx.hce_w12,
        hce_w_l12head=ctx.hce_w_l12head,
        hce_w_l3=ctx.hce_w_l3,
        hce_w_l4=ctx.hce_w_l4,
        n_splits=ctx.cv_k,
        train_group_frac=ctx.train_group_frac,
        patience=ctx.patience,
        max_epochs=ctx.max_epochs,
        loader_kwargs=_train_loader_kwargs(ctx.seed, ctx.train_batch_size),
        resume_from_checkpoints=ctx.resume_from_checkpoints,
        group_checkpoint_dir=str(ctx.ckpt_dir),
        hidden_dims=ctx.hidden_dims,
        random_state=ctx.seed,
        val_selection_metric=ctx.val_selection_metric,
        cv_selection_metric=ctx.cv_selection_metric,
        use_spatial_context=ctx.use_spatial_context,
        spatial_k=ctx.spatial_k,
        spatial_mode=ctx.spatial_mode,
    )

    best_ckpt = lp["best_fold"]["checkpoint"]
    dest = ctx.ckpt_dir / "best_mlp_gpu.pt"
    import shutil

    shutil.copy2(best_ckpt, dest)
    print(f"  Best fold={lp['best_fold']['fold']}  checkpoint={dest}", flush=True)

    ctx.g.update(lp)
    ctx.g["LP"] = lp
    ctx.g["BEST_MLP_CHECKPOINT"] = str(dest)
    ctx.g["model"] = lp["model"]


def step_pooled_he_validate(ctx: PooledRunContext) -> None:
    if "LP" not in ctx.g:
        raise RuntimeError("Run pooled training first (--steps train).")
    g = ctx.g
    rf = ctx.result_fig
    val_labels = g.get("val_labels")
    val_preds = g.get("val_preds")
    if val_labels is None or val_preds is None:
        raise RuntimeError("Missing OOF predictions from group CV.")

    macro_auc = plot_he_validate_level2_minimal(
        plot_confusion_matrix,
        plot_multiclass_roc_curves,
        rf,
        g["class_names"],
        val_labels,
        val_preds,
        val_probs_l2=g.get("val_probs_l2"),
        title_prefix="Cross-dataset OOF Level2",
    )
    save_pooled_internal_validation_metrics(
        rf("validation_internal_metrics"),
        macro_auc_l2=macro_auc,
        n_samples=len(ctx.samples),
        n_cells=int(g["cv_data"]["X_f"].shape[0]),
        n_folds=ctx.cv_k,
        train_group_frac=ctx.train_group_frac,
        best_fold=g["best_fold"],
        group_summary=g.get("group_summary") or g["logo_summary"],
    )


def _load_sample_stardist_arrays(ctx: PooledRunContext, sample: str) -> dict:
    import anndata as ad

    stardist_h5ad = ctx.cases_root / sample / f"{sample}_matched_features_stardist.h5ad"
    if not stardist_h5ad.is_file():
        raise FileNotFoundError(stardist_h5ad)
    adata_star = ad.read_h5ad(stardist_h5ad)
    return {
        "matched_features_stardist_path": str(stardist_h5ad),
        "X_star": adata_X_to_dense(adata_star.X),
        "y_star_level1": adata_star.obs["final_lineage"].to_numpy(),
        "X_coords_star": adata_star.obsm["spatial"],
        "y_star": adata_star.obs["final_CT"].to_numpy(),
    }


def step_pooled_stardist_one_sample(ctx: PooledRunContext, sample: str) -> None:
    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        raise RuntimeError("Run pooled train first.")
    g = ctx.g
    star = _load_sample_stardist_arrays(ctx, sample)
    rf = ctx.stardist_result_fig(sample)
    ckpt = g.get("BEST_MLP_CHECKPOINT")
    model_star = load_model_for_predict(
        str(ctx.python_root),
        sample,
        f"{sample}_project_all_UNI",
        parent_dir=True,
        checkpoint_path=ckpt,
        device=ctx.device,
        hidden_dims=ctx.hidden_dims,
    )
    class_names = g["class_names"]
    scaler = g["scaler"]
    cv_data = g["cv_data"]

    evaluate_and_plot_on_all_data(
        model=model_star,
        matched_features_path=star["matched_features_stardist_path"],
        class_names=class_names,
        evaluate=evaluate,
        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
        CellTypeDataset=CellTypeDataset,
        device=ctx.device,
        scaler=scaler,
        prediction_only=True,
        X_coords_matched=star["X_coords_star"],
        y_level1_f=star["y_star_level1"],
        celltype_pred_dir=rf("stardist_pred_level2"),
        spatial_plot_mode="pred_true_l2",
        spatial_color_scheme="xenium_ct",
        spatial_title_pred_l2=f"{sample} StarDist pred level2",
        spatial_title_true_l2=f"{sample} ground truth level2",
    )

    x_star_scaled = scaler.transform(star["X_star"])
    neighbor_index = _stardist_neighbor_index(ctx, model_star, star["X_coords_star"])
    probs_l2 = mlp_collect_softmax_probs(
        model_star, x_star_scaled, ctx.device, neighbor_index=neighbor_index
    )
    y_roc = encode_labels_with_class_names(star["y_star"], class_names)
    m = min(len(probs_l2), len(y_roc))
    valid = (y_roc[:m] >= 0) & (y_roc[:m] < len(class_names))
    macro_auc = float("nan")
    if np.any(valid):
        roc_info = plot_multiclass_roc_curves(
            y_roc[:m][valid],
            probs_l2[:m][valid],
            class_names,
            figsize=(3.0, 3.0),
            max_curves=len(class_names),
            save_path=rf("roc_stardist_level2"),
            title=f"{sample} StarDist Level2 ROC",
            roc_color_scheme="xenium_ct",
        )
        if roc_info is not None:
            macro_auc = float(roc_info.get("macro_auc", float("nan")))
    print(f"  {sample}: StarDist L2 macro AUROC = {macro_auc:.4f}", flush=True)

    all_labels = y_roc[:m]
    all_preds = np.argmax(probs_l2[:m], axis=1)
    save_stardist_external_validation_metrics(
        all_labels=all_labels,
        all_preds=all_preds,
        matched_features_path=star["matched_features_stardist_path"],
        class_names_star=class_names,
        class_names_level1=g["class_names_level1"],
        y_encoded_f=cv_data["y_encoded_f"],
        y_level1_encoded_f=cv_data["y_level1_encoded_f"],
        therapy_data=sample,
        metrics_csv_path=rf("validation_external_stardist_matched_metrics"),
        model_star=model_star,
        scaler=scaler,
        X_star=star["X_star"],
        device=ctx.device,
        predict_all_label_heads=predict_all_label_heads,
        cv_data=cv_data,
        g=g,
        logo_summary=g.get("logo_summary"),
        hce_w1=g.get("hce_w1", ctx.hce_w1),
        hce_w2=g.get("hce_w2", ctx.hce_w2),
        hce_w12=g.get("hce_w12", ctx.hce_w12),
        hce_w_l12head=g.get("hce_w_l12head", ctx.hce_w_l12head),
        hce_w_l3=g.get("hce_w_l3", ctx.hce_w_l3),
        hce_w_l4=g.get("hce_w_l4", ctx.hce_w_l4),
        probs_l2=probs_l2[:m],
        head_probs=None,
    )


def step_pooled_stardist(ctx: PooledRunContext) -> None:
    print(f"\n[Pooled StarDist] {len(ctx.samples)} datasets", flush=True)
    failures = []
    for sample in ctx.samples:
        try:
            step_pooled_stardist_one_sample(ctx, sample)
        except Exception as exc:
            print(f"  FAIL {sample}: {exc}", flush=True)
            failures.append((sample, str(exc)))
    if failures:
        print(f"\n  StarDist failures: {len(failures)}/{len(ctx.samples)}", flush=True)


########################################################
## 2026.06.27 LLY: Add the function to load the all_features_stardist.h5ad
########################################################
def _load_sample_stardist_all_arrays(ctx: PooledRunContext, sample: str) -> dict:
    import anndata as ad

    all_h5ad = ctx.cases_root / sample / f"{sample}_all_features_stardist.h5ad"
    if not all_h5ad.is_file():
        raise FileNotFoundError(
            f"{all_h5ad} not found. Run transer_embedding_label_h5ad.py --steps stardist_all_h5ad."
        )
    adata = ad.read_h5ad(all_h5ad)
    spatial = adata.obsm.get("spatial")
    if spatial is None:
        spatial = adata.obsm.get("spatial_HE")
    if spatial is None:
        raise KeyError(f"{all_h5ad} has no obsm['spatial'] or obsm['spatial_HE'].")
    return {
        "all_features_stardist_path": str(all_h5ad),
        "adata": adata,
        "X_all": adata_X_to_dense(adata.X),
        "X_coords_all": np.asarray(spatial),
    }


def step_pooled_stardist_all_one_sample(ctx: PooledRunContext, sample: str) -> Path:
    """
    Predict all StarDist nuclei (no GT AUROC) and save labeled h5ad under
    ``Data/{save_result}/stardist/{sample}/{sample}_all_features_stardist_label.h5ad``.
    """
    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        raise RuntimeError("Run pooled prepare/train first (--steps train or stardist_all).")
    g = ctx.g
    star = _load_sample_stardist_all_arrays(ctx, sample)
    out_path = stardist_all_label_h5ad_path(ctx.data_root, sample, ctx.save_result)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = g.get("BEST_MLP_CHECKPOINT")
    model_star = load_model_for_predict(
        str(ctx.python_root),
        sample,
        f"{sample}_project_all_UNI",
        parent_dir=True,
        checkpoint_path=ckpt,
        device=ctx.device,
        hidden_dims=ctx.hidden_dims,
    )
    scaler = g["scaler"]
    x_scaled = scaler.transform(star["X_all"])
    neighbor_index = _stardist_neighbor_index(ctx, model_star, star["X_coords_all"])
    head_probs = mlp_collect_five_head_softmax_probs(
        model_star,
        x_scaled,
        ctx.device,
        neighbor_index=neighbor_index,
    )
    if not head_probs:
        raise RuntimeError(f"No softmax probabilities returned for {sample}.")

    adata = star["adata"].copy()
    attach_five_head_probs_to_adata_obs(
        adata,
        head_probs,
        g=g,
        cv_data=g.get("cv_data"),
    )
    adata.write_h5ad(out_path)
    n_prob_cols = sum(
        1 for c in adata.obs.columns if any(c.startswith(f"{h}_prob_") for h in head_probs)
    )
    print(
        f"  {sample}: saved {out_path.name} "
        f"({adata.n_obs:,} cells; {n_prob_cols} prob columns; heads={sorted(head_probs.keys())})",
        flush=True,
    )
    return out_path


def step_pooled_stardist_all(ctx: PooledRunContext) -> None:
    print(f"\n[Pooled StarDist all nuclei] {len(ctx.samples)} datasets", flush=True)
    failures = []
    for sample in ctx.samples:
        try:
            step_pooled_stardist_all_one_sample(ctx, sample)
        except Exception as exc:
            print(f"  FAIL {sample}: {exc}", flush=True)
            failures.append((sample, str(exc)))
    if failures:
        print(f"\n  StarDist-all failures: {len(failures)}/{len(ctx.samples)}", flush=True)

########################################################
## 2026.06.27 LLY: Add the function to ensure the pooled inference ready
########################################################
def _ensure_pooled_inference_ready(ctx: PooledRunContext, *, require_train_if_missing: bool = True) -> None:
    """
    Prepare scaler/class names and resolve ``BEST_MLP_CHECKPOINT`` for StarDist inference.

    Loads ``{ckpt_dir}/best_mlp_gpu.pt`` when present; optionally runs full training if missing.
    """
    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        step_pooled_prepare(ctx)
    if "BEST_MLP_CHECKPOINT" not in ctx.g:
        ckpt = ctx.ckpt_dir / "best_mlp_gpu.pt"
        if ckpt.is_file():
            ctx.g["BEST_MLP_CHECKPOINT"] = str(ckpt)
            print(f"  Using existing checkpoint: {ckpt}", flush=True)
        elif require_train_if_missing:
            step_pooled_train(ctx)
        else:
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}. Run --steps train first or set --ablation-tag correctly."
            )
########################################################

def process_pooled(ctx: PooledRunContext, steps: set[str]) -> tuple[bool, str | None]:
    print(f"\n{'=' * 60}\nPooled cross-dataset CV ({len(ctx.samples)} samples)\n{'=' * 60}", flush=True)
    print(f"  Result dir: {ctx.result_dir}", flush=True)
    try:
        if "he_h5ad" in steps:
            step_pooled_prepare(ctx)
        if "train" in steps or "he_validate" in steps or "stardist" in steps or "stardist_all" in steps:
            if "cv_data" not in ctx.g and "he_h5ad" not in steps:
                step_pooled_prepare(ctx)
        if "train" in steps:
            step_pooled_train(ctx)
        if "he_validate" in steps:
            if "LP" not in ctx.g:
                step_pooled_train(ctx)
            step_pooled_he_validate(ctx)
        if "stardist" in steps:
            _ensure_pooled_inference_ready(ctx)
            step_pooled_stardist(ctx)
        if "stardist_all" in steps:
            _ensure_pooled_inference_ready(ctx)
            step_pooled_stardist_all(ctx)
        print("  OK: pooled cross-dataset pipeline", flush=True)
        return True, None
    except Exception as exc:
        print(f"  FAIL pooled: {exc}", flush=True)
        return False, str(exc)
########################################################

def process_sample(ctx: RunContext, steps: set[str], checkpoint_path: str | None = None) -> tuple[bool, str | None]:
    print(f"\n{'=' * 60}\nSample: {ctx.therapy_data}\n{'=' * 60}", flush=True)
    if not ctx.sample_dir.is_dir():
        msg = f"sample directory not found: {ctx.sample_dir}"
        print(f"  SKIP: {msg}", flush=True)
        return False, msg
    try:
        if "he_h5ad" in steps:
            step_he_h5ad(ctx)
            step_prepare_cv(ctx)
        if "train" in steps:
            step_train(ctx)
        if "he_validate" in steps:
            if "LP" not in ctx.g:
                step_train(ctx)
            step_he_validate(ctx)
        if "stardist" in steps:
            step_stardist(ctx, checkpoint_path=checkpoint_path)
        print(f"  OK: {ctx.therapy_data}", flush=True)
        return True, None
    except Exception as exc:
        print(f"  FAIL: {ctx.therapy_data}: {exc}", flush=True)
        return False, str(exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Xenium lung UNI-label train/validate (HE + StarDist).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--python-root", type=Path, default=DEFAULT_PYTHON_ROOT)
    p.add_argument("--cases-root", type=Path, default=None)
    p.add_argument(
        "--cases-set",
        choices=("complete", "incomplete"),
        default="complete",
        help="Which cases folder when --cases-root is omitted.",
    )
    p.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Single sample ID (default: all samples under cases-root).",
    )
    p.add_argument(
        "--model-suffix",
        type=str,
        default="_project_all_UNI",
        help="Appended to sample ID for therapy_model dir name.",
    )
    p.add_argument(
        "--save-result",
        type=str,
        default=DEFAULT_SAVE_RESULT,
        help=(
            f"Per-sample result subfolder when --mode per-sample (default {DEFAULT_SAVE_RESULT!r} "
            f"→ {DEFAULT_PER_SAMPLE_SAVE_RESULT!r}). For cross-dataset, prefer --pooled-save-result."
        ),
    )
    p.add_argument(
        "--pooled-save-result",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            f"Cross-dataset output subfolder under Data/ (default: {DEFAULT_POOLED_SAVE_RESULT!r}). "
            "Example: --pooled-save-result result_spatial → Data/result_spatial/."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("per-sample", "cross-dataset"),
        default="per-sample",
        help="per-sample: train/validate each dataset separately; "
        "cross-dataset: pool all Complete_Cases h5ads, dataset-level 5-fold CV, "
        f"outputs under Data/{{pooled_save_result}} (default {DEFAULT_POOLED_SAVE_RESULT!r}).",
    )
    p.add_argument(
        "--train-group-frac",
        type=float,
        default=0.7,
        help="Cross-dataset mode: fraction of datasets for training per fold "
        "(ceil(n×frac) train; default 0.7 → 18 train / 7 test for 25 samples).",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=("he_h5ad", "train", "he_validate", "stardist", "stardist_all", "all"),
        default=["all"],
        help=(
            "Pipeline steps. 'all' = he_h5ad+train+he_validate+stardist (matched). "
            "stardist_all = predict every StarDist nucleus → "
            "{sample}_all_features_stardist_label.h5ad (cross-dataset only)."
        ),
    )
    p.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="UNI embedding dimension (default: infer from matched h5ad, usually 1024).",
    )
    p.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[1024, 512, 256],
        help="MLP hidden layer sizes (output heads sized from label encoders).",
    )
    p.add_argument("--cv-k", type=int, default=5)
    p.add_argument(
        "--stratify-target",
        choices=("level2", "level1", "joint"),
        default="joint",
    )
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help=f"Training DataLoader batch size on GPU (default: {DEFAULT_TRAIN_BATCH_SIZE}). "
        "Val batch size is 2× this value.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match-tolerance", type=float, default=1.0)
    p.add_argument("--ablation-tag", type=str, default=DEFAULT_ABLATION_TAG)
    p.add_argument("--hce-w1", type=float, default=1.0)
    p.add_argument("--hce-w2", type=float, default=2.0)
    p.add_argument("--hce-w12", type=float, default=1.0)
    p.add_argument("--hce-w-l12head", type=float, default=1.0)
    p.add_argument("--hce-w-l3", type=float, default=1.0)
    p.add_argument("--hce-w-l4", type=float, default=1.0)
    p.add_argument(
        "--val-selection-metric",
        choices=("four_term_sum", "l2_macro_priority", "five_tier_auc_sum"),
        default=DEFAULT_VAL_SELECTION_METRIC,
        help=(
            "Per-epoch checkpoint selection during training. "
            "five_tier_auc_sum = sum of head-level macro AUROC (L2,L1,L12,L3,L4)."
        ),
    )
    p.add_argument(
        "--cv-selection-metric",
        choices=("four_term_sum", "l2_macro_priority", "five_tier_auc_sum"),
        default=DEFAULT_CV_SELECTION_METRIC,
        help="Metric to pick the best CV fold after all folds complete.",
    )
    p.add_argument(
        "--use-spatial-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse kNN neighbor UNI embeddings before MLP. Cross-dataset: kNN built "
            "separately within each sample's spatial_HE coordinates."
        ),
    )
    p.add_argument("--spatial-k", type=int, default=DEFAULT_SPATIAL_K)
    p.add_argument(
        "--spatial-mode",
        choices=("mean", "attention"),
        default=DEFAULT_SPATIAL_MODE,
        help="Neighbor aggregation before MLP: mean (default) or attention.",
    )
    p.add_argument(
        "--resume-from-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing k-fold fold checkpoints when present.",
    )
    p.add_argument(
        "--force-rebuild-h5ad",
        action="store_true",
        help="Rebuild HE / StarDist h5ad even when cache exists.",
    )
    p.add_argument(
        "--build-stardist-h5ad",
        action="store_true",
        help="Build StarDist matched h5ad if missing (else require transer_embedding_label_h5ad.py).",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit model checkpoint for StarDist / inference (default: auto-detect).",
    )
    p.add_argument(
        "--no-auto-cv-k",
        action="store_true",
        help="Do not auto-reduce --cv-k or coarsen --stratify-target for rare strata.",
    )
    p.add_argument("--cuda-device", type=str, default="0")
    p.add_argument("--allow-cpu-train", action="store_true")
    return p.parse_args(argv)


def resolve_save_result(
    mode: str,
    save_result: str,
    pooled_save_result: str | None = None,
) -> str:
    """
    Resolve output subfolder name under Data/ (cross-dataset) or per-sample project dir.

    cross-dataset priority: ``--pooled-save-result`` > explicit ``--save-result`` > ``result_all``
    per-sample priority: explicit ``--save-result`` > ``result``
    """
    if mode == "cross-dataset":
        if pooled_save_result is not None:
            return pooled_save_result
        if save_result != DEFAULT_SAVE_RESULT:
            return save_result
        return DEFAULT_POOLED_SAVE_RESULT
    if save_result != DEFAULT_SAVE_RESULT:
        return save_result
    return DEFAULT_PER_SAMPLE_SAVE_RESULT


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_cuda(args.cuda_device)
    device = setup_device(args.allow_cpu_train)
    setup_seed(args.seed)

    cases_root = resolve_cases_root(args)
    steps = resolve_steps(args.steps)
    data_root = cases_root.parent
    save_result = resolve_save_result(args.mode, args.save_result, args.pooled_save_result)

    if "stardist_all" in steps and args.mode != "cross-dataset":
        print("ERROR: --steps stardist_all requires --mode cross-dataset.", flush=True)
        return 2

    if args.mode == "cross-dataset":
        samples = discover_h5ad_case_samples(cases_root, args.sample)
        print(f"Mode:       cross-dataset (pooled)", flush=True)
        print(f"Data root:  {data_root}", flush=True)
        print(f"Cases root: {cases_root}", flush=True)
        print(f"Samples:    {len(samples)} (with matched_features.h5ad)", flush=True)
        print(f"Steps:      {', '.join(sorted(steps))}", flush=True)
        print(f"Device:     {device}", flush=True)
        print(f"Train batch size: {args.train_batch_size}", flush=True)
        print(f"Results:    {data_root / save_result}", flush=True)
        if args.use_spatial_context:
            print(
                f"Spatial:    k={args.spatial_k}, mode={args.spatial_mode!r} (per-sample kNN)",
                flush=True,
            )
        if "stardist_all" in steps:
            print(
                "StarDist-all: Complete_Cases/{sample}/{sample}_all_features_stardist.h5ad "
                f"→ {save_result}/stardist/{{sample}}/{{sample}}_all_features_stardist_label.h5ad",
                flush=True,
            )

        ctx = PooledRunContext(
            data_root=data_root,
            cases_root=cases_root,
            python_root=args.python_root.expanduser().resolve(),
            samples=samples,
            device=device,
            seed=args.seed,
            save_result=save_result,
            input_dim=args.input_dim,
            hidden_dims=tuple(args.hidden_dims),
            cv_k=args.cv_k,
            train_group_frac=args.train_group_frac,
            patience=args.patience,
            max_epochs=args.max_epochs,
            train_batch_size=args.train_batch_size,
            resume_from_checkpoints=args.resume_from_checkpoints,
            ablation_tag=args.ablation_tag,
            hce_w1=args.hce_w1,
            hce_w2=args.hce_w2,
            hce_w12=args.hce_w12,
            hce_w_l12head=args.hce_w_l12head,
            hce_w_l3=args.hce_w_l3,
            hce_w_l4=args.hce_w_l4,
            val_selection_metric=args.val_selection_metric,
            cv_selection_metric=args.cv_selection_metric,
            use_spatial_context=args.use_spatial_context,
            spatial_k=args.spatial_k,
            spatial_mode=args.spatial_mode,
        )
        success, err = process_pooled(ctx, steps)
        if success:
            print("\nDone: pooled cross-dataset pipeline succeeded.", flush=True)
            return 0
        print(f"\nDone: pooled pipeline failed: {err}", flush=True)
        return 1

    samples = discover_case_samples(cases_root, args.sample)
    column_rename = dict(XENIUM_CELL_COORD_COLUMN_RENAME_FULL)

    print(f"Cases root: {cases_root}", flush=True)
    print(f"Samples:    {len(samples)}", flush=True)
    print(f"Steps:      {', '.join(sorted(steps))}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Train batch size: {args.train_batch_size}", flush=True)
    print(f"Results:    .../{{sample}}{args.model_suffix}/{save_result}/", flush=True)

    ok = skipped = 0
    failures: list[tuple[str, str]] = []
    for sample in samples:
        ctx = RunContext(
            sample=sample,
            cases_root=cases_root,
            python_root=args.python_root.expanduser().resolve(),
            therapy_data=sample,
            therapy_model=f"{sample}{args.model_suffix}",
            save_result=save_result,
            device=device,
            seed=args.seed,
            match_tolerance=args.match_tolerance,
            column_rename=column_rename,
            force_rebuild_h5ad=args.force_rebuild_h5ad,
            input_dim=args.input_dim,
            hidden_dims=tuple(args.hidden_dims),
            cv_k=args.cv_k,
            stratify_target=args.stratify_target,
            patience=args.patience,
            max_epochs=args.max_epochs,
            train_batch_size=args.train_batch_size,
            resume_from_checkpoints=args.resume_from_checkpoints,
            ablation_tag=args.ablation_tag,
            hce_w1=args.hce_w1,
            hce_w2=args.hce_w2,
            hce_w12=args.hce_w12,
            hce_w_l12head=args.hce_w_l12head,
            hce_w_l3=args.hce_w_l3,
            hce_w_l4=args.hce_w_l4,
            val_selection_metric=args.val_selection_metric,
            cv_selection_metric=args.cv_selection_metric,
            build_stardist_h5ad=args.build_stardist_h5ad,
            auto_cv_k=not args.no_auto_cv_k,
        )
        success, err = process_sample(ctx, steps, checkpoint_path=args.checkpoint_path)
        if success:
            ok += 1
        else:
            skipped += 1
            if err:
                failures.append((sample, err))

    print(f"\nDone: {ok} succeeded, {skipped} failed/skipped, {len(samples)} total.", flush=True)
    if failures:
        print("\nFailed/skipped samples:", flush=True)
        for sample, err in failures:
            print(f"  - {sample}: {err}", flush=True)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
