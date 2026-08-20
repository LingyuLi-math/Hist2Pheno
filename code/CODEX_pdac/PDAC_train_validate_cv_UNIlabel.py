## 2026.08.20 LLY: CODEX PDAC CLI twin of CODEX_hcc/HCC_train_validate_cv_UNIlabel.py
## Paths: data/CODEX/HCC/Michael_data_transfer/s1167/{ACQUISITION_ID}/project_all_UNI/
## Samples: 278 Pancreas TMA cores with cell-type CSV (list_aligned_annotated_regions)
## Incomplete_Cases: 195 PDAC cores without cell-type CSV → stardist_Incomplete_Cases/
## Hierarchy: three-head (L2/L12/L1); selection metric three_tier_auc_sum
## Spatial: build_spatial_neighbor_index defaults match lung/HCC (k=8, mode=mean)

## cd /home/lingyu/ssd2/Python/Hist2Pheno
## conda activate SeededNTM

## Preprocess CSVs (once per annotated ACQUISITION_ID):
# conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py

## One sample, full pipeline (HE h5ad + train + validate + StarDist):
# python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \
#   --sample Charvill-94_c001_v001_r001_reg001

## All 278 annotated cores (per-sample):
# python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py

## Cross-dataset pool of 278 h5ads + spatial kNN:
# python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \
#   --mode cross-dataset \
#   --use-spatial-context --spatial-k 8 --spatial-mode mean \
#   --pooled-save-result result_all_spatial \
#   --ablation-tag D_emph_L2_spatial_bs4096

## StarDist-all h5ad (annotated):
# conda run --no-capture-output -n SeededNTM python -u \
#   code/CODEX_pdac/transer_embedding_label_h5ad.py \
#   --sample Charvill-94_c001_v001_r001_reg001 --steps stardist_all_h5ad

## StarDist-all h5ad (Incomplete_Cases):
# conda run --no-capture-output -n SeededNTM python -u \
#   code/CODEX_pdac/transer_embedding_label_h5ad.py \
#   --incomplete --steps stardist_all_h5ad



#!/usr/bin/env python3
"""
CODEX PDAC: stratified K-fold HCE training + HE internal validation + StarDist external validation.

Assumes UNI embeddings exist under
``s1167/{ACQUISITION_ID}/project_all_UNI/ImgEmbeddings_all/``. GT / StarDist
cell CSVs come from ``match_codex_cells_with_pixel.py``; StarDist centroids from
``data/CODEX/HCC/StarDist_Segment_pdac/pdac_result/{ACQUISITION_ID}/``.

By default processes all **278** annotated Pancreas TMA cores and enables spatial
neighbor fusion (``build_spatial_neighbor_index``, k=8, mode=mean).

Terminal usage (from repo root ``Hist2Pheno``):

  python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \\
    --sample Charvill-94_c001_v001_r001_reg001

  python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py

  python -u code/CODEX_pdac/PDAC_train_validate_cv_UNIlabel.py \\
    --mode cross-dataset --use-spatial-context --spatial-k 8 --spatial-mode mean \\
    --pooled-save-result result_all_spatial

CSV outputs under ``s1167/{ACQUISITION_ID}/project_all_UNI/result/``.
Incomplete_Cases StarDist-all labels go to
``s1167/{save_result}/stardist_Incomplete_Cases/{ACQUISITION_ID}/``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
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
_CODEX_PDAC_DIR = _REPO_ROOT / "code" / "CODEX_pdac"
for _p in (_PKG_DIR, _CODEX_PDAC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import base  # noqa: E402
from base import (  # noqa: E402
    HCC_H5AD_OBS_COLUMNS,
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
from match_codex_cells_with_pixel import (  # noqa: E402
    HCC_COLUMN_RENAME,
    cells_matched_stardist_path,
    cells_with_pixel_path,
    list_aligned_annotated_regions,
    list_incomplete_pdac_regions,
    stardist_csv_path,
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
    plot_level1_roc_from_level2_scores,
    plot_level1_spatial_distribution,
    plot_multiclass_roc_curves,
    plot_per_class_accuracy,
    plot_per_class_f1,
    plot_tier_spatial_distribution,
)
from uni_label_cv_helpers import (  # noqa: E402
    build_insample_tier_metrics,
    save_stardist_external_validation_metrics,
    plot_stardist_roc_from_auroc_csv,
    _auroc_csv_path_from_metrics,
    ensure_lp_extra_insample_preds,
    make_pooled_result_fig,
    make_pooled_stardist_result_fig,
    make_result_fig,
    attach_five_head_probs_to_adata_obs,
    discover_stardist_all_h5ad_samples,
    stardist_all_label_h5ad_path,
    stardist_incomplete_all_label_h5ad_path,
    plot_he_confusion_matrices,
    plot_he_f1_extra_tiers,
    plot_he_spatial_extra_tiers,
    plot_he_validate_level2_minimal,
    save_pooled_internal_validation_metrics,
    plot_stardist_roc_extra,
    plot_stardist_spatial_extra,
    save_stardist_external_auroc_tier_csv,
    stardist_tier_auroc_csv_path,
    stardist_head_preds,
    macro_auc_ovr,
)

DEFAULT_S1167_ROOT = (
    _REPO_ROOT / "data/CODEX/HCC/Michael_data_transfer/s1167"
)
DEFAULT_CASES_ROOT = DEFAULT_S1167_ROOT
DEFAULT_DATA_ROOT = DEFAULT_S1167_ROOT  # pooled outputs under s1167/{save_result}/
DEFAULT_STARDIST_ROOT = _REPO_ROOT / "data/CODEX/HCC/StarDist_Segment_pdac/pdac_result"

## 2026.08.20, StarDist-all for 278 annotated PDAC cores + 195 Incomplete_Cases
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"
STARDIST_INCOMPLETE_RESULT_SUBDIR = "stardist_Incomplete_Cases"
INCOMPLETE_CASES_DIRNAME = "Incomplete_Cases"

DEFAULT_PYTHON_ROOT = Path("/home/lingyu/ssd2/Python/")
DEFAULT_ABLATION_TAG = "D_emph_L2"
DEFAULT_VAL_SELECTION_METRIC = "three_tier_auc_sum"
DEFAULT_CV_SELECTION_METRIC = "three_tier_auc_sum"
DEFAULT_SAVE_RESULT = "result"

# Canonical organ selector for shared plotting APIs.
PAN_ORGAN = "codex_pdac"
DEFAULT_PER_SAMPLE_SAVE_RESULT = "result"
DEFAULT_POOLED_SAVE_RESULT = "result_all"
DEFAULT_SPATIAL_K = 8
DEFAULT_SPATIAL_MODE = "mean"
DEFAULT_USE_SPATIAL_CONTEXT = True
DEFAULT_TRAIN_BATCH_SIZE = 4096
DEFAULT_THERAPY_MODEL = "project_all_UNI"


def _train_loader_kwargs(seed: int, train_batch_size: int) -> dict:
    return {
        "seed": seed,
        "train_balance_sampler": False,
        "num_workers_cuda": 0,
        "batch_size_cuda": train_batch_size,
    }



def load_he_acq_map(cases_root: Path | None = None) -> dict[str, str]:
    """ACQUISITION_ID → ACQUISITION_ID (PDAC has no separate MATCHED_HE)."""
    from s1167_img_cell_mapping import load_s1167_metadata

    del cases_root
    df = load_s1167_metadata(cohort="PDAC", annotated_only=False, with_he_only=True)
    return {str(aid): str(aid) for aid in df["ACQUISITION_ID"].astype(str).str.strip()}


def discover_pdac_samples(
    cases_root: Path,
    sample: str | None = None,
    *,
    require_h5ad: bool = False,
    require_cells_csv: bool = True,
) -> list[str]:
    """
    Return ACQUISITION_ID keys for the 278 annotated PDAC cores.

    When ``sample`` is set, return that one key (still validated against the acq dir).
    """
    cases_root = Path(cases_root)
    if sample is not None:
        sample_dir = cases_root / sample
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample dir not found: {sample_dir}")
        if require_h5ad and not (sample_dir / f"{sample}_matched_features.h5ad").is_file():
            raise FileNotFoundError(sample_dir / f"{sample}_matched_features.h5ad")
        return [sample]

    annotated = list_aligned_annotated_regions()
    keys = [str(k) for k in annotated["MATCHED_HE"].tolist()]
    out = []
    for he in keys:
        sample_dir = cases_root / he
        if not sample_dir.is_dir():
            print(f"  skip {he}: missing HE dir", flush=True)
            continue
        if require_cells_csv and not cells_with_pixel_path(sample_dir, he).is_file():
            print(f"  skip {he}: missing cells_with_pixel.csv (run match_codex_cells_with_pixel.py)", flush=True)
            continue
        if require_h5ad and not (sample_dir / f"{he}_matched_features.h5ad").is_file():
            print(f"  skip {he}: missing matched_features.h5ad", flush=True)
            continue
        out.append(he)
    if not out:
        raise FileNotFoundError(
            f"No annotated PDAC samples under {cases_root}. "
            "Run match_codex_cells_with_pixel.py and/or --steps he_h5ad first."
        )
    return out


def prepare_pooled_cv_data_from_h5ads(
    cases_root: Path,
    samples: list[str],
    min_l2_samples: int = 2,
):
    """Pool HE matched h5ads for PDAC three-head (no CNiche/TNiche)."""
    import anndata as ad
    from base import prepare_data_leave_one_group_out

    if not samples:
        raise ValueError("samples list is empty")

    sample_to_group = {s: i for i, s in enumerate(samples)}
    X_parts, y_parts, y_l1_parts, y_l12_parts = [], [], [], []
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
        coord_parts.append(np.asarray(adata.obsm["spatial_HE"], dtype=np.float64))
        group_parts.append(np.full(n_obs, sample_to_group[sample], dtype=np.int64))
        print(f"  {sample}: {n_obs:,} cells", flush=True)

    cv_data = prepare_data_leave_one_group_out(
        np.vstack(X_parts),
        np.concatenate(y_parts),
        np.concatenate(y_l1_parts),
        groups=np.concatenate(group_parts),
        min_l2_samples=min_l2_samples,
        y_level12=np.concatenate(y_l12_parts),
        y_level3=None,
        y_level4=None,
        X_coords=np.vstack(coord_parts),
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
    acq_id: str
    stardist_root: Path = DEFAULT_STARDIST_ROOT
    val_selection_metric: str = DEFAULT_VAL_SELECTION_METRIC
    cv_selection_metric: str = DEFAULT_CV_SELECTION_METRIC
    auto_cv_k: bool = True
    use_spatial_context: bool = DEFAULT_USE_SPATIAL_CONTEXT
    spatial_k: int = DEFAULT_SPATIAL_K
    spatial_mode: str = DEFAULT_SPATIAL_MODE
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
    use_spatial_context: bool = DEFAULT_USE_SPATIAL_CONTEXT
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
    return DEFAULT_CASES_ROOT.resolve()


def resolve_data_root(args: argparse.Namespace, cases_root: Path) -> Path:
    """Pooled result parent (s1167/). Override with --data-root."""
    if getattr(args, "data_root", None) is not None:
        return args.data_root.expanduser().resolve()
    # HE/ → s4769/
    if cases_root.name == "HE":
        return cases_root.parent.resolve()
    return cases_root.resolve()


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
    cell_coords = cells_with_pixel_path(ctx.sample_dir, ctx.therapy_data)
    if not cell_coords.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed GT table: {cell_coords}\n"
            "Run: conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py "
            f"--he-key {ctx.therapy_data}"
        )
    emb_dir = ctx.sample_dir / ctx.therapy_model / "ImgEmbeddings_all/sc_pth_16_16"
    pth_prefix = f"sc_{ctx.acq_id}"
    adata = match_hist2cell_h5ad(
        cell_coords_path=str(cell_coords),
        hist_embedding_dir=emb_dir,
        matched_h5ad_path=str(ctx.matched_he_h5ad),
        coord_cols=("X_pix_HE", "Y_pix_HE"),
        tolerance=ctx.match_tolerance,
        pth_prefix=pth_prefix,
        level1_name="celltype_level1",
        column_rename=ctx.column_rename,
        auto_rename=False,
        force_rebuild=ctx.force_rebuild_h5ad,
        obs_columns=HCC_H5AD_OBS_COLUMNS,
        cell_id_col="cell_id",
    )
    ctx.g["adata"] = adata
    ctx.g["matched_features_path"] = str(ctx.matched_he_h5ad)
    print(f"  → {adata.n_obs:,} cells × {adata.n_vars} features", flush=True)


def step_prepare_cv(ctx: RunContext) -> None:
    adata = ctx.g["adata"]
    cv_data = prepare_data_from_matched_h5ad(
        adata,
        groups=np.arange(adata.n_obs, dtype=np.int64),
        require_niche_heads=False,
    )
    use_five_head = False
    use_three_head = "y_level12_encoded_f" in cv_data
    if use_three_head:
        print(
            "  Three-head training (HCC): L2=final_CT, L12=final_sublineage, "
            "L1=final_lineage",
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
            "USE_THREE_HEAD": use_three_head,
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
    if ctx.use_spatial_context:
        if "X_coords_f" not in cv_data:
            raise ValueError("use_spatial_context=True requires X_coords_f in cv_data.")
        nbr_idx = _build_spatial_neighbor_index_for_cv_data(cv_data, k_neighbors=ctx.spatial_k)
        ctx.g["spatial_neighbor_index"] = nbr_idx
        print(
            f"  spatial: k={ctx.spatial_k}, mode={ctx.spatial_mode!r}, "
            f"kNN index {nbr_idx.shape}",
            flush=True,
        )


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
    print(
        f"  val_sel={ctx.val_selection_metric} cv_sel={ctx.cv_selection_metric}"
        + (
            f" spatial_k={ctx.spatial_k} mode={ctx.spatial_mode!r}"
            if ctx.use_spatial_context
            else ""
        ),
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
        use_spatial_context=ctx.use_spatial_context,
        spatial_k=ctx.spatial_k,
        spatial_mode=ctx.spatial_mode,
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
        pan_organ=PAN_ORGAN,
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
        pan_organ=PAN_ORGAN,
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
        pan_organ=PAN_ORGAN,
    )


def _ensure_stardist_h5ad(ctx: RunContext) -> None:
    if ctx.matched_stardist_h5ad.is_file() and not ctx.force_rebuild_h5ad:
        print(f"  StarDist h5ad exists: {ctx.matched_stardist_h5ad}", flush=True)
        return
    if not ctx.build_stardist_h5ad:
        raise FileNotFoundError(
            f"StarDist h5ad not found: {ctx.matched_stardist_h5ad}. "
            "Pass --build-stardist-h5ad (uses prebuilt "
            f"{ctx.therapy_data}_cells_matched_by_stardist.csv)."
        )

    print("[StarDist h5ad] building matched features...", flush=True)
    stardist_csv = cells_matched_stardist_path(ctx.sample_dir, ctx.therapy_data)
    if not stardist_csv.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed StarDist-matched table: {stardist_csv}\n"
            "Run: conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py "
            f"--he-key {ctx.therapy_data}"
        )
    stardist_raw = stardist_csv_path(ctx.stardist_root, ctx.therapy_data)
    if not stardist_raw.is_file():
        print(f"  ⚠ StarDist_Segment CSV missing (optional check): {stardist_raw}", flush=True)

    emb_dir = ctx.sample_dir / ctx.therapy_model / "ImgEmbeddings_all_stardist/sc_pth_16_16"
    pth_prefix = f"sc_{ctx.therapy_data}"  # StarDist UNI files use HE key

    adata_star = match_hist2cell_h5ad(
        cell_coords_path=str(stardist_csv),
        hist_embedding_dir=emb_dir,
        matched_h5ad_path=str(ctx.matched_stardist_h5ad),
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=ctx.match_tolerance,
        pth_prefix=pth_prefix,
        level1_name="celltype_level1",
        column_rename=ctx.column_rename,
        auto_rename=False,
        spatial_cols=("centroid_x", "centroid_y"),
        spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
        force_rebuild=ctx.force_rebuild_h5ad,
        obs_columns=HCC_H5AD_OBS_COLUMNS,
        cell_id_col="cell_id",
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
    spatial_key = "spatial" if "spatial" in adata_star.obsm else "spatial_HE"
    ctx.g.update(
        {
            "matched_features_stardist_path": str(ctx.matched_stardist_h5ad),
            "X_star": adata_X_to_dense(adata_star.X),
            "y_star_level1": adata_star.obs["final_lineage"].to_numpy(),
            "y_star_level12": adata_star.obs["final_sublineage"].to_numpy(),
            "y_star_level3": None,
            "y_star_level4": None,
            "X_coords_star": adata_star.obsm[spatial_key],
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
            pan_organ=PAN_ORGAN,
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
        pan_organ=PAN_ORGAN,
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
            pan_organ=PAN_ORGAN,
        )
    else:
        print("  Skip L12/L3/L4 StarDist spatial: not five-head.", flush=True)

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
                pan_organ=PAN_ORGAN,
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
                pan_organ=PAN_ORGAN,
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
                    pan_organ=PAN_ORGAN,
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
            pan_organ=PAN_ORGAN,
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
    use_five_head = False
    use_three_head = "y_level12_encoded_f" in cv_data
    if use_three_head:
        print(
            "  Three-head pooled training (HCC): L2=final_CT, L12=final_sublineage, "
            "L1=final_lineage",
            flush=True,
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
            "USE_THREE_HEAD": use_three_head,
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
        pan_organ=PAN_ORGAN,
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
    spatial_key = "spatial" if "spatial" in adata_star.obsm else "spatial_HE"
    out = {
        "matched_features_stardist_path": str(stardist_h5ad),
        "X_star": adata_X_to_dense(adata_star.X),
        "y_star_level1": adata_star.obs["final_lineage"].to_numpy(),
        "X_coords_star": adata_star.obsm[spatial_key],
        "y_star": adata_star.obs["final_CT"].to_numpy(),
    }
    for obs_col, key in (
        ("final_sublineage", "y_star_level12"),
        ("CNiche", "y_star_level3"),
        ("TNiche", "y_star_level4"),
    ):
        if obs_col in adata_star.obs.columns:
            out[key] = adata_star.obs[obs_col].to_numpy()
    return out

POOLED_STARDIST_LEVELS: tuple[str, ...] = ("l2", "l1", "l12")  # HCC three-head


def _normalize_pooled_stardist_levels(levels: list[str] | tuple[str, ...] | None) -> frozenset[str]:
    if levels is None:
        return frozenset(POOLED_STARDIST_LEVELS)
    out = {str(x).lower() for x in levels}
    unknown = out - set(POOLED_STARDIST_LEVELS)
    if unknown:
        raise ValueError(
            f"Unknown StarDist levels {sorted(unknown)}; choose from {POOLED_STARDIST_LEVELS}"
        )
    if not out:
        raise ValueError("At least one StarDist level is required.")
    return frozenset(out)


def _pooled_stardist_g_star(g: dict, star: dict) -> dict:
    """Merge per-sample StarDist GT labels into globals for L12/L3/L4 tier plots."""
    g_star = dict(g)
    for key in ("y_star_level12", "y_star_level3", "y_star_level4"):
        if key in star:
            g_star[key] = star[key]
    return g_star


def step_pooled_stardist_one_sample(
    ctx: PooledRunContext,
    sample: str,
    *,
    levels: list[str] | tuple[str, ...] | None = None,
    save_metrics: bool | None = None,
) -> None:
    levels_set = _normalize_pooled_stardist_levels(levels)
    if save_metrics is None:
        save_metrics = "l2" in levels_set

    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        raise RuntimeError("Run pooled train first.")
    g = ctx.g
    star = _load_sample_stardist_arrays(ctx, sample)
    rf = ctx.stardist_result_fig(sample)
    ckpt = g.get("BEST_MLP_CHECKPOINT")
    model_star = load_model_for_predict(
        str(ctx.python_root),
        sample,
        DEFAULT_THERAPY_MODEL,
        parent_dir=True,
        checkpoint_path=ckpt,
        device=ctx.device,
        hidden_dims=ctx.hidden_dims,
    )
    class_names = g["class_names"]
    scaler = g["scaler"]
    cv_data = g["cv_data"]
    g_star = _pooled_stardist_g_star(g, star)

    run_l2 = "l2" in levels_set
    run_l1 = "l1" in levels_set
    extra_tiers = tuple(sorted(levels_set & {"l12", "l3", "l4"}))

    all_preds: np.ndarray | None = None
    all_labels: np.ndarray | None = None
    probs_l2: np.ndarray | None = None
    head_probs = None
    macro_auc = float("nan")
    m = 0
    valid = np.array([], dtype=bool)

    x_star_scaled = scaler.transform(star["X_star"])
    neighbor_index = _stardist_neighbor_index(ctx, model_star, star["X_coords_star"])
    y_roc = encode_labels_with_class_names(star["y_star"], class_names)

    if run_l2 or run_l1 or extra_tiers:
        probs_l2 = mlp_collect_softmax_probs(
            model_star, x_star_scaled, ctx.device, neighbor_index=neighbor_index
        )
        m = min(len(probs_l2), len(y_roc))
        valid = (y_roc[:m] >= 0) & (y_roc[:m] < len(class_names))

    if run_l2:
        _, _, _, all_preds, all_labels = evaluate_and_plot_on_all_data(
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
            pan_organ=PAN_ORGAN,
            spatial_title_pred_l2=f"{sample} StarDist pred level2",
            spatial_title_true_l2=f"{sample} ground truth level2",
            neighbor_index=neighbor_index,
        )
        if np.any(valid):
            y_v = y_roc[:m][valid]
            probs_roc = probs_l2[:m][valid]
            roc_info = plot_multiclass_roc_curves(
                y_v,
                probs_roc,
                class_names,
                figsize=(3.0, 3.0),
                max_curves=len(class_names),
                save_path=rf("roc_stardist_level2"),
                title=f"{sample} StarDist Level2 ROC",
                pan_organ=PAN_ORGAN,
            )
            if roc_info is not None:
                macro_auc = float(roc_info.get("macro_auc", float("nan")))
        print(f"  {sample}: StarDist L2 macro AUROC = {macro_auc:.4f}", flush=True)
    elif run_l1 and probs_l2 is not None:
        all_preds = np.argmax(probs_l2[:m], axis=1)
        all_labels = y_roc[:m]

    if run_l1:
        if all_preds is None and probs_l2 is not None:
            all_preds = np.argmax(probs_l2[:m], axis=1)
            all_labels = y_roc[:m]
        plot_level1_spatial_distribution(
            matched_features_path=star["matched_features_stardist_path"],
            all_preds=all_preds,
            class_names_level1=g["class_names_level1"],
            class_names=class_names,
            y_encoded_f=cv_data["y_encoded_f"],
            y_level1_encoded_f=cv_data["y_level1_encoded_f"],
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            save_path_pred=rf("stardist_pred_level1"),
            pan_organ=PAN_ORGAN,
            X_coords_matched=star["X_coords_star"],
            y_level1_f=star["y_star_level1"],
            spatial_title_pred_l1=f"{sample} StarDist pred level1",
            spatial_title_true_l1=f"{sample} ground truth level1",
        )
        if np.any(valid):
            y_v = y_roc[:m][valid]
            probs_roc = probs_l2[:m][valid]
            bundle = load_matched_features_bundle(star["matched_features_stardist_path"])
            y_l1_roc = bundle["y_level1"][:m][valid] if "y_level1" in bundle else None
            plot_level1_roc_from_level2_scores(
                star["matched_features_stardist_path"],
                probs_roc,
                class_names,
                g["class_names_level1"],
                cv_data["y_encoded_f"],
                cv_data["y_level1_encoded_f"],
                figsize=(3.0, 3.0),
                save_path=rf("roc_stardist_level1"),
                title=f"{sample} StarDist Level1 ROC from L2 probs",
                pan_organ=PAN_ORGAN,
                y_level1_f=y_l1_roc,
            )
        if probs_l2 is not None:
            try:
                save_stardist_external_auroc_tier_csv(
                    "l1",
                    auroc_csv_path=stardist_tier_auroc_csv_path(rf, "l1"),
                    matched_features_path=star["matched_features_stardist_path"],
                    class_names_star=class_names,
                    class_names_level1=g["class_names_level1"],
                    y_encoded_f=cv_data["y_encoded_f"],
                    y_level1_encoded_f=cv_data["y_level1_encoded_f"],
                    probs_l2=probs_l2[:m],
                )
            except Exception as exc:
                print(f"  ⚠ Skip L1 AUROC CSV: {exc}", flush=True)

    if extra_tiers:
        if not hasattr(model_star, "level12_head"):
            print("  Skip L12/L3/L4 StarDist: model has no level12_head.", flush=True)
        else:
            head_probs = mlp_collect_five_head_softmax_probs(
                model_star,
                x_star_scaled[:m],
                ctx.device,
                neighbor_index=neighbor_index,
            )
            star_heads = stardist_head_preds(
                model_star,
                scaler,
                star["X_star"],
                ctx.device,
                predict_all_label_heads,
                neighbor_index=neighbor_index,
            )
            plot_stardist_spatial_extra(
                plot_tier_spatial_distribution,
                plot_celltype_spatial_distribution,
                rf,
                sample,
                star["X_coords_star"],
                star_heads,
                cv_data,
                g_star,
                tiers=extra_tiers,
                pan_organ=PAN_ORGAN,
            )
            if np.any(valid):
                y_v = y_roc[:m][valid]
                probs_heads_roc = {k: v[: len(y_v)] for k, v in head_probs.items()}
                plot_stardist_roc_extra(
                    plot_multiclass_roc_curves,
                    rf,
                    probs_heads_roc,
                    cv_data,
                    len(y_v),
                    g_star,
                    tiers=extra_tiers,
                    pan_organ=PAN_ORGAN,
                )
            for tier in extra_tiers:
                try:
                    save_stardist_external_auroc_tier_csv(
                        tier,
                        auroc_csv_path=stardist_tier_auroc_csv_path(rf, tier),
                        matched_features_path=star["matched_features_stardist_path"],
                        head_probs=head_probs,
                        cv_data=cv_data,
                        g=g_star,
                        m=m,
                    )
                except Exception as exc:
                    print(f"  ⚠ Skip {tier.upper()} AUROC CSV: {exc}", flush=True)

    if save_metrics and probs_l2 is not None and all_preds is not None and all_labels is not None:
        save_stardist_external_validation_metrics(
            all_labels=all_labels[:m],
            all_preds=np.asarray(all_preds[:m], dtype=np.int64),
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
            g=g_star,
            logo_summary=g.get("logo_summary"),
            hce_w1=g.get("hce_w1", ctx.hce_w1),
            hce_w2=g.get("hce_w2", ctx.hce_w2),
            hce_w12=g.get("hce_w12", ctx.hce_w12),
            hce_w_l12head=g.get("hce_w_l12head", ctx.hce_w_l12head),
            hce_w_l3=g.get("hce_w_l3", ctx.hce_w_l3),
            hce_w_l4=g.get("hce_w_l4", ctx.hce_w_l4),
            probs_l2=probs_l2[:m],
            head_probs=head_probs,
        )


def step_pooled_stardist(
    ctx: PooledRunContext,
    *,
    levels: list[str] | tuple[str, ...] | None = None,
    samples: list[str] | tuple[str, ...] | None = None,
    save_metrics: bool | None = None,
) -> None:
    levels_set = _normalize_pooled_stardist_levels(levels)
    sample_list = list(samples) if samples is not None else list(ctx.samples)
    print(
        f"\n[Pooled StarDist] levels={sorted(levels_set)} | {len(sample_list)} datasets",
        flush=True,
    )
    failures = []
    for sample in sample_list:
        try:
            step_pooled_stardist_one_sample(
                ctx,
                sample,
                levels=tuple(sorted(levels_set)),
                save_metrics=save_metrics,
            )
        except Exception as exc:
            print(f"  FAIL {sample}: {exc}", flush=True)
            failures.append((sample, str(exc)))
    if failures:
        print(f"\n  StarDist failures: {len(failures)}/{len(sample_list)}", flush=True)


########################################################
## 2026.06.27 LLY: Add the function to load the all_features_stardist.h5ad
########################################################
def _write_h5ad_atomic(adata, out_path: Path) -> None:
    """Write h5ad via temp file + rename to avoid truncated/corrupt outputs on lock errors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".h5ad.tmp", dir=out_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_sample_stardist_all_arrays(
    ctx: PooledRunContext,
    sample: str,
    *,
    cases_root: Path | None = None,
) -> dict:
    import anndata as ad

    root = cases_root or ctx.cases_root
    all_h5ad = root / sample / f"{sample}_all_features_stardist.h5ad"
    if not all_h5ad.is_file():
        raise FileNotFoundError(
            f"{all_h5ad} not found. Run "
            "code/CODEX_pdac/transer_embedding_label_h5ad.py "
            f"--sample {sample} --steps stardist_all_h5ad."
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


def step_pooled_stardist_all_one_sample(
    ctx: PooledRunContext,
    sample: str,
    *,
    cases_root: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Predict all StarDist nuclei (no GT AUROC) and save labeled h5ad under
    ``Data/{save_result}/stardist/{sample}/{sample}_all_features_stardist_label.h5ad``
    unless *out_path* is set (e.g. Incomplete_Cases discovery outputs).
    """
    if "scaler" not in ctx.g or "class_names" not in ctx.g:
        raise RuntimeError("Run pooled prepare/train first (--steps train or stardist_all).")
    g = ctx.g
    star = _load_sample_stardist_all_arrays(ctx, sample, cases_root=cases_root)
    out_path = out_path or stardist_all_label_h5ad_path(
        ctx.data_root, sample, ctx.save_result
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = g.get("BEST_MLP_CHECKPOINT")
    model_star = load_model_for_predict(
        str(ctx.python_root),
        sample,
        DEFAULT_THERAPY_MODEL,
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
    _write_h5ad_atomic(adata, out_path)
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
## 2026.08.20 LLY: StarDist all-nuclei inference for 195 Incomplete_Cases
########################################################
def discover_pdac_incomplete_samples(
    cases_root: Path | None = None,
    stardist_root: Path | None = None,
    *,
    require_h5ad: bool = False,
) -> list[str]:
    """ACQUISITION_IDs of Pancreas TMA cores without cell-type CSV (195)."""
    cases_root = Path(cases_root or DEFAULT_CASES_ROOT)
    stardist_root = Path(stardist_root or DEFAULT_STARDIST_ROOT)
    incomplete = list_incomplete_pdac_regions()
    out = []
    for aid in incomplete["ACQUISITION_ID"].astype(str).str.strip():
        if not stardist_csv_path(stardist_root, aid).is_file():
            continue
        h5ad = cases_root / aid / f"{aid}{STARDIST_ALL_H5AD_SUFFIX}"
        if require_h5ad and not h5ad.is_file():
            print(f"  skip {aid}: missing {h5ad.name}", flush=True)
            continue
        out.append(aid)
    return out


def step_pooled_stardist_all_pdac_incomplete(
    ctx: PooledRunContext,
    samples: list[str] | None = None,
    *,
    cases_root: Path | None = None,
    stardist_root: Path | None = None,
) -> None:
    """
    Predict all StarDist nuclei for Incomplete_Cases (no CODEX cell-type CSV).

    Input: ``s1167/{sample}/{sample}_all_features_stardist.h5ad``
    Output: ``s1167/{save_result}/stardist_Incomplete_Cases/{sample}/..._label.h5ad``
    """
    cases_root = Path(cases_root or ctx.cases_root)
    if samples is None:
        samples = discover_pdac_incomplete_samples(
            cases_root, stardist_root, require_h5ad=True,
        )
    if not samples:
        print(
            f"\n[Pooled StarDist all nuclei — Incomplete_Cases] no samples "
            f"(need {STARDIST_ALL_H5AD_SUFFIX} under {cases_root})",
            flush=True,
        )
        return

    print(
        f"\n[Pooled StarDist all nuclei — Incomplete_Cases] {len(samples)} datasets",
        flush=True,
    )
    failures = []
    for sample in samples:
        try:
            out_path = stardist_incomplete_all_label_h5ad_path(
                ctx.data_root, sample, ctx.save_result
            )
            step_pooled_stardist_all_one_sample(
                ctx,
                sample,
                cases_root=cases_root,
                out_path=out_path,
            )
        except Exception as exc:
            print(f"  FAIL {sample}: {exc}", flush=True)
            failures.append((sample, str(exc)))
    if failures:
        print(
            f"\n  StarDist-all Incomplete_Cases failures: {len(failures)}/{len(samples)}",
            flush=True,
        )


def step_pooled_stardist_all_incomplete(
    ctx: PooledRunContext,
    samples: list[str] | None = None,
    *,
    cases_root: Path | None = None,
) -> None:
    """Lung-named wrapper: PDAC Incomplete_Cases stay under ``s1167/{acq}/``."""
    step_pooled_stardist_all_pdac_incomplete(
        ctx, samples, cases_root=cases_root
    )

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
        description="CODEX PDAC UNI-label train/validate (278 annotated cores; HE + StarDist).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--python-root", type=Path, default=DEFAULT_PYTHON_ROOT)
    p.add_argument(
        "--cases-root",
        type=Path,
        default=None,
        help=f"s1167 root with ACQUISITION_ID dirs (default: {DEFAULT_CASES_ROOT}).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Pooled result parent (default: {DEFAULT_DATA_ROOT}).",
    )
    p.add_argument(
        "--stardist-root",
        type=Path,
        default=DEFAULT_STARDIST_ROOT,
        help="Prepared StarDist CSV root (StarDist_Segment).",
    )
    p.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Single ACQUISITION_ID (default: all 278 annotated PDAC cores).",
    )
    p.add_argument(
        "--therapy-model",
        type=str,
        default=DEFAULT_THERAPY_MODEL,
        help="Model dir under each HE sample (default: project_all_UNI).",
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
            f"Cross-dataset output subfolder under s1167/ (default: {DEFAULT_POOLED_SAVE_RESULT!r}). "
            "Example: --pooled-save-result result_spatial → s1167/result_spatial/."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("per-sample", "cross-dataset"),
        default="per-sample",
        help="per-sample: train/validate each ACQUISITION_ID separately; "
        "cross-dataset: pool PDAC h5ads, dataset-level CV, "
        f"outputs under s1167/{{pooled_save_result}} (default {DEFAULT_POOLED_SAVE_RESULT!r}).",
    )
    p.add_argument(
        "--train-group-frac",
        type=float,
        default=0.7,
        help="Cross-dataset mode: fraction of datasets for training per fold "
        "(ceil(n×frac) train; default 0.7 → ~25 train / ~11 test for 36 samples).",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=("he_h5ad", "train", "he_validate", "stardist", "stardist_all", "all"),
        default=["all"],
        help=(
            "Pipeline steps. 'all' = he_h5ad+train+he_validate+stardist (matched). "
            "stardist_all = predict every StarDist nucleus → "
            "{sample}_all_features_stardist_label.h5ad (cross-dataset only; optional)."
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
        choices=("four_term_sum", "l2_macro_priority", "three_tier_auc_sum", "five_tier_auc_sum"),
        default=DEFAULT_VAL_SELECTION_METRIC,
        help=(
            "Per-epoch checkpoint selection during training. "
            "three_tier_auc_sum = sum of L2+L1+L12 macro AUROC (HCC default)."
        ),
    )
    p.add_argument(
        "--cv-selection-metric",
        choices=("four_term_sum", "l2_macro_priority", "three_tier_auc_sum", "five_tier_auc_sum"),
        default=DEFAULT_CV_SELECTION_METRIC,
        help="Metric to pick the best CV fold after all folds complete.",
    )
    p.add_argument(
        "--use-spatial-context",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SPATIAL_CONTEXT,
        help=(
            "Fuse kNN neighbor UNI embeddings before MLP via build_spatial_neighbor_index "
            f"(default True; lung defaults k={DEFAULT_SPATIAL_K}, mode={DEFAULT_SPATIAL_MODE!r}). "
            "Use --no-use-spatial-context to disable."
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
        help="Build StarDist matched h5ad if missing from cells_matched_by_stardist.csv.",
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
    Resolve output subfolder name under s4769/ (cross-dataset) or per-sample project dir.

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
    data_root = resolve_data_root(args, cases_root)
    save_result = resolve_save_result(args.mode, args.save_result, args.pooled_save_result)
    acq_map = load_he_acq_map(cases_root)
    stardist_root = Path(args.stardist_root).expanduser().resolve()
    therapy_model = args.therapy_model

    if "stardist_all" in steps and args.mode != "cross-dataset":
        print("ERROR: --steps stardist_all requires --mode cross-dataset.", flush=True)
        return 2

    if args.mode == "cross-dataset":
        samples = discover_pdac_samples(
            cases_root,
            args.sample,
            require_h5ad=False,
            require_cells_csv=True,
        )
        ## 2026.08.12 LLy add
        # Ensure matched HE h5ads exist before pooling (build if missing / requested).
        column_rename = dict(HCC_COLUMN_RENAME)
        for sample in samples:
            if sample not in acq_map:
                raise KeyError(f"Unknown PDAC ACQUISITION_ID={sample!r}")
            h5ad = cases_root / sample / f"{sample}_matched_features.h5ad"
            need_build = (
                "he_h5ad" in steps
                or args.force_rebuild_h5ad
                or not h5ad.is_file()
            )
            if not need_build:
                continue
            ctx_pre = RunContext(
                sample=sample,
                cases_root=cases_root,
                python_root=args.python_root.expanduser().resolve(),
                therapy_data=sample,
                therapy_model=therapy_model,
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
                build_stardist_h5ad=args.build_stardist_h5ad,
                acq_id=acq_map[sample],
                stardist_root=stardist_root,
                val_selection_metric=args.val_selection_metric,
                cv_selection_metric=args.cv_selection_metric,
                auto_cv_k=not args.no_auto_cv_k,
                use_spatial_context=args.use_spatial_context,
                spatial_k=args.spatial_k,
                spatial_mode=args.spatial_mode,
            )
            step_he_h5ad(ctx_pre)
        samples = discover_pdac_samples(cases_root, args.sample, require_h5ad=True)

        print(f"Mode:       cross-dataset (pooled HCC)", flush=True)
        print(f"Data root:  {data_root}", flush=True)
        print(f"Cases root: {cases_root}", flush=True)
        print(f"Samples:    {len(samples)} (with matched_features.h5ad)", flush=True)
        print(f"Steps:      {', '.join(sorted(steps))}", flush=True)
        print(f"Device:     {device}", flush=True)
        print(f"Train batch size: {args.train_batch_size}", flush=True)
        print(f"Results:    {data_root / save_result}", flush=True)
        if args.use_spatial_context:
            print(
                f"Spatial:    k={args.spatial_k}, mode={args.spatial_mode!r} "
                "(build_spatial_neighbor_index, per-sample kNN)",
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
        success, err = process_pooled(ctx, steps - {"he_h5ad"})
        if success:
            print("\nDone: pooled cross-dataset pipeline succeeded.", flush=True)
            return 0
        print(f"\nDone: pooled pipeline failed: {err}", flush=True)
        return 1

    samples = discover_pdac_samples(cases_root, args.sample, require_h5ad=False)
    column_rename = dict(HCC_COLUMN_RENAME)

    print(f"Cases root: {cases_root}", flush=True)
    print(f"Samples:    {len(samples)} (ALIGNED annotated HCC)", flush=True)
    print(f"Steps:      {', '.join(sorted(steps))}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Train batch size: {args.train_batch_size}", flush=True)
    print(f"Therapy model: {therapy_model}", flush=True)
    print(f"Results:    .../{{sample}}/{therapy_model}/{save_result}/", flush=True)
    if args.use_spatial_context:
        print(
            f"Spatial:    k={args.spatial_k}, mode={args.spatial_mode!r} "
            "(build_spatial_neighbor_index)",
            flush=True,
        )

    ok = skipped = 0
    failures: list[tuple[str, str]] = []
    for sample in samples:
        if sample not in acq_map:
            failures.append((sample, f"Unknown PDAC ACQUISITION_ID={sample!r}"))
            skipped += 1
            continue
        ctx = RunContext(
            sample=sample,
            cases_root=cases_root,
            python_root=args.python_root.expanduser().resolve(),
            therapy_data=sample,
            therapy_model=therapy_model,
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
            acq_id=acq_map[sample],
            stardist_root=stardist_root,
            auto_cv_k=not args.no_auto_cv_k,
            use_spatial_context=args.use_spatial_context,
            spatial_k=args.spatial_k,
            spatial_mode=args.spatial_mode,
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
