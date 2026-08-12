## 2026.04.16 LLY add --train_balance_sampler 
##                to use WeightedRandomSampler in CV training loaders to upweight rare level2 classes.
##                add --cv_selection_metric to select the best fold based on the l2_macro_priority metric.


"""
CLI: Train + validate hierarchical cell type classifier (level2 + level1).

Aligned with ``NCRT_train_validate_*_cv.ipynb``:
- Default ``--cv_mode logo``: Leave-One-Group-Out by ``TumorID`` (or spatial tiles if degenerate),
  using ``matched_features_{parent}_logo.npz`` and ``run_logo_cv_with_insample_report``.
- ``--cv_mode stratified_kfold``: label-stratified K-fold CV with configurable ``--cv_k``
  and ``--cv_stratify_target`` (level2 / level1 / joint).
- ``--cv_mode random_split``: legacy stratified train/test split (notebook-style 70/30).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Make project code importable from any cwd.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PARENT = os.path.dirname(CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if CODE_PARENT not in sys.path:
    sys.path.insert(0, CODE_PARENT)
PKG_DIR = os.path.join(CODE_PARENT, "Hist2Pheno_pkg")
if os.path.isdir(PKG_DIR) and PKG_DIR not in sys.path:
    # Prefer package-style module location when present.
    sys.path.insert(0, PKG_DIR)

try:
    from base import (  # type: ignore  # noqa: E402
        CellTypeDataset,
        evaluate,
        evaluate_and_plot_on_all_data,
        export_all_data_prediction_confidence,
        loader_train_test,
        match_hist2cell_matrix,
        prepare_data_leave_one_group_out,
        save_hce_validation_metrics,
        spatial_tile_groups_for_logo,
        split_train_test,
    )
    from model import (  # type: ignore  # noqa: E402
        load_model_for_predict,
        mode_validation_from_split,
        run_logo_cv_with_insample_report,
        run_stratified_kfold_cv_with_insample_report,
        train_and_save_model_from_split,
    )
    from plot import (  # type: ignore  # noqa: E402
        plot_celltype_spatial_distribution,
        plot_confusion_matrix,
        plot_level1_spatial_distribution,
        plot_level1_accuracy_from_level2_predictions,
        plot_per_class_accuracy,
        plot_per_class_f1,
    )
except ModuleNotFoundError:
    # Fallback to package imports when running without sys.path injection.
    from Hist2Pheno_pkg.base import (  # type: ignore  # noqa: E402
        CellTypeDataset,
        evaluate,
        evaluate_and_plot_on_all_data,
        export_all_data_prediction_confidence,
        loader_train_test,
        match_hist2cell_matrix,
        prepare_data_leave_one_group_out,
        save_hce_validation_metrics,
        spatial_tile_groups_for_logo,
        split_train_test,
    )
    from Hist2Pheno_pkg.model import (  # type: ignore  # noqa: E402
        load_model_for_predict,
        mode_validation_from_split,
        run_logo_cv_with_insample_report,
        run_stratified_kfold_cv_with_insample_report,
        train_and_save_model_from_split,
    )
    from Hist2Pheno_pkg.plot import (  # type: ignore  # noqa: E402
        plot_celltype_spatial_distribution,
        plot_confusion_matrix,
        plot_level1_spatial_distribution,
        plot_level1_accuracy_from_level2_predictions,
        plot_per_class_accuracy,
        plot_per_class_f1,
    )


class TeeStream:
    """Mirror writes to terminal and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


def plot_per_class_accuracy(
    test_labels,
    test_preds,
    class_names,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    test_acc=None,
    test_macro_f1=None,
    test_weighted_f1=None,
    model_name="MLP",
    save_path=None,
):
    """Per-class accuracy plot (class-wise recall) with save support."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    test_labels = np.asarray(test_labels).astype(np.int64, copy=False)
    test_preds = np.asarray(test_preds).astype(np.int64, copy=False)
    max_idx = len(class_names) - 1
    valid_mask = (
        (test_labels >= 0)
        & (test_labels <= max_idx)
        & (test_preds >= 0)
        & (test_preds <= max_idx)
    )
    if not np.all(valid_mask):
        n_drop = int((~valid_mask).sum())
        bad_vals = np.unique(
            np.concatenate([test_labels[~valid_mask], test_preds[~valid_mask]])
        )[:10]
        print(
            f"  ⚠ Dropping {n_drop} samples with out-of-range class indices "
            f"(valid range 0..{max_idx}). Sample bad values: {bad_vals}"
        )
        test_labels = test_labels[valid_mask]
        test_preds = test_preds[valid_mask]
    if test_labels.size == 0:
        print("  ⚠ No valid samples for per-class accuracy plot.")
        return pd.DataFrame(columns=["Precision", "Accuracy", "F1-Score", "Support"])

    unique_labels = np.unique(np.concatenate([test_labels, test_preds]))
    actual_class_names = [class_names[i] for i in unique_labels]
    per_class_metrics = precision_recall_fscore_support(
        test_labels,
        test_preds,
        labels=unique_labels,
        zero_division=0,
    )
    metrics_df = pd.DataFrame(
        {
            "Precision": per_class_metrics[0],
            "Accuracy": per_class_metrics[1],
            "F1-Score": per_class_metrics[2],
            "Support": per_class_metrics[3],
        },
        index=actual_class_names,
    )

    metrics_df_sorted = metrics_df.sort_values("Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(metrics_df_sorted))
    colors = plt.cm.viridis(
        metrics_df_sorted["Accuracy"] / max(metrics_df_sorted["Accuracy"].max(), 1e-8)
    )
    ax.barh(x_pos, metrics_df_sorted["Accuracy"], color=colors)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(metrics_df_sorted.index, fontsize=9)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title(f"Per-Class Accuracy Performance ({model_name})", fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for i, (_, row) in enumerate(metrics_df_sorted.iterrows()):
        ax.text(row["Accuracy"] + 0.01, i, f"{row['Accuracy']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
    return metrics_df


def plot_level1_accuracy_from_level2_predictions(
    matched_features_path,
    all_preds,
    class_names,
    class_names_level1,
    y_encoded_f,
    y_level1_encoded_f,
    model=None,
    device=None,
    train_dataset=None,
    val_dataset=None,
    input_dim=None,
    best_epoch=None,
    model_name="MLP-AllData-Level1",
    save_path=None,
):
    """Compute level1 metrics from level2 predictions and save per-class accuracy plot."""
    loaded_data = np.load(matched_features_path)
    if "y_level1" not in loaded_data:
        print(f"y_level1 not found in {matched_features_path}; skip level1 accuracy plot.")
        return None

    y_true_l1_raw = loaded_data["y_level1"]

    num_l2_classes = len(class_names)
    child_to_parent = np.full(num_l2_classes, -1, dtype=np.int64)
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

    preds_l2 = np.asarray(all_preds).astype(np.int64)
    valid_pred_mask = (preds_l2 >= 0) & (preds_l2 < num_l2_classes)
    if not np.all(valid_pred_mask):
        n_bad = int((~valid_pred_mask).sum())
        bad_vals = np.unique(preds_l2[~valid_pred_mask])[:10]
        print(
            f"  ⚠ Found {n_bad} out-of-range level2 predictions for level1 mapping; "
            f"dropping them. Sample bad indices: {bad_vals}"
        )
    preds_l2 = preds_l2[valid_pred_mask]
    preds_l1 = child_to_parent[preds_l2]

    if y_true_l1_raw.dtype.kind in ["U", "S", "O"]:
        name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
        y_true_l1 = np.array([name_to_idx.get(str(v), -1) for v in y_true_l1_raw], dtype=np.int64)
    else:
        y_true_l1 = np.asarray(y_true_l1_raw).astype(np.int64)

    # Keep alignment with filtered predictions.
    y_true_l1 = y_true_l1[: len(valid_pred_mask)]
    y_true_l1 = y_true_l1[valid_pred_mask]
    n = min(len(y_true_l1), len(preds_l1))
    y_true_l1 = y_true_l1[:n]
    preds_l1 = preds_l1[:n]
    valid_mask = y_true_l1 >= 0
    if not np.any(valid_mask):
        print("No valid aligned level1 labels; skip level1 accuracy plot.")
        return None

    y_true_l1_valid = y_true_l1[valid_mask]
    preds_l1_valid = preds_l1[valid_mask]

    from sklearn.metrics import accuracy_score, f1_score

    acc_l1 = accuracy_score(y_true_l1_valid, preds_l1_valid)
    macro_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="macro", zero_division=0)
    weighted_f1_l1 = f1_score(y_true_l1_valid, preds_l1_valid, average="weighted", zero_division=0)
    print(
        "All-data level1 metrics from level2 preds: "
        f"acc={acc_l1:.4f}, macro_f1={macro_f1_l1:.4f}, weighted_f1={weighted_f1_l1:.4f}"
    )

    return plot_per_class_accuracy(
        y_true_l1_valid,
        preds_l1_valid,
        class_names_level1,
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        best_epoch=best_epoch,
        test_acc=acc_l1,
        test_macro_f1=macro_f1_l1,
        test_weighted_f1=weighted_f1_l1,
        model_name=model_name,
        save_path=save_path,
    )


def configure_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for deterministic CUDA GEMM behavior on some operators.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def pick_device(allow_cpu_train: bool, cuda_device: int = 0) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if cuda_device < 0 or cuda_device >= n:
            raise RuntimeError(
                f"Invalid --cuda_device {cuda_device}: need 0 <= index < {n} "
                f"({n} CUDA device(s) visible)."
            )
        torch.cuda.set_device(cuda_device)
        device = torch.device(f"cuda:{cuda_device}")
        print(f"Using CUDA device {cuda_device}: {torch.cuda.get_device_name(cuda_device)}")
        return device

    if allow_cpu_train:
        print("CUDA unavailable, fallback to CPU (--allow_cpu_train).")
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA unavailable. Use a CUDA-enabled PyTorch build, "
        "or pass --allow_cpu_train to run on CPU."
    )


def run(args: argparse.Namespace) -> None:
    device = pick_device(args.allow_cpu_train, cuda_device=args.cuda_device)
    configure_reproducibility(args.seed)

    if args.embedding_pth_cap is not None:
        os.environ["ESCCAI_EMBEDDING_PTH_CAP"] = str(args.embedding_pth_cap)

    # Align with NCRT_train_validate.ipynb: pth_prefix=f"sc_{therapy_data}" by default.
    pth_prefix = args.pth_prefix if args.pth_prefix is not None else f"sc_{args.therapy_data}"

    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data" / "CODEX" / "ESCC"

    therapy_model = args.therapy_model or f"{args.therapy_data}_project_{args.parent_value}_{args.method}"
    model_data_dir = data_root / args.therapy_data / therapy_model
    model_data_dir.mkdir(parents=True, exist_ok=True)

    # Save run artifacts directly under:
    #   data/<therapy_data>/<therapy_model>/<save_result>/<run_prefix><timestamp>/
    run_id = args.run_name or datetime.now().strftime("%Y%m%d%H%M%S%f")
    save_root_dir = model_data_dir / args.save_result
    result_dir = save_root_dir / f"{args.run_prefix}{run_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "run.log"

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(orig_stdout, log_f)
    sys.stderr = TeeStream(orig_stderr, log_f)

    try:
        print(f"Log file: {log_path}")
        # Inputs for matching embeddings/labels.
        cell_coords_path = Path(args.cell_coords_path) if args.cell_coords_path else (
            data_root / f"{args.transfer_data}" / f"{args.therapy_data}_CellPixCoords_{args.parent_value}.csv"
        )
        hist_embedding_dir = Path(args.hist_embedding_dir) if args.hist_embedding_dir else (
            model_data_dir / f"ImgEmbeddings_{args.parent_value}" / args.embedding_subdir
        )
        use_logo = args.cv_mode == "logo"
        use_stratified_kfold = args.cv_mode == "stratified_kfold"
        use_cv_fullmatrix = use_logo or use_stratified_kfold
        if use_cv_fullmatrix and args.checkpoint_exists:
            raise ValueError(
                "Cross-validation mode uses per-fold checkpoints (hce_logo_fold_*.pt or hce_kfold_fold_*.pt). "
                "Use --logo_resume auto|true|false, or --cv_mode random_split with --checkpoint_exists."
            )
        if args.matched_features_path:
            matched_features_path = Path(args.matched_features_path)
        elif use_cv_fullmatrix:
            matched_features_path = model_data_dir / f"matched_features_{args.parent_value}_logo.npz"
        else:
            matched_features_path = model_data_dir / f"matched_features_{args.parent_value}.npz"
        best_model_path = Path(args.best_model_path) if args.best_model_path else (
            result_dir / "best_mlp_gpu.pt"
        )
        metrics_csv_path = Path(args.metrics_csv_path) if args.metrics_csv_path else (
            result_dir / f"hce_validation_metrics_{args.parent_value}.csv"
        )

        python_root = args.python_root.replace("\\", "/")
        if not python_root.endswith("/"):
            python_root = python_root + "/"

        cv_data = None
        result = None
        loaders = None
        LP = None

        print("=" * 72)
        print("Step 1/6: Match embeddings with cell labels")
        print(f"run_output_dir: {result_dir}")
        print(f"cell_coords_path: {cell_coords_path}")
        print(f"hist_embedding_dir: {hist_embedding_dir}")
        print(f"matched_features_path: {matched_features_path}")
        print(f"cv_mode: {args.cv_mode}")

        if use_logo:
            X, y, y_level1, y_level0, y_level01, y_level12, X_coords_matched, groups = match_hist2cell_matrix(
                cell_coords_path=cell_coords_path,
                hist_embedding_dir=hist_embedding_dir,
                matched_features_path=matched_features_path,
                coord_cols=(args.coord_x_col, args.coord_y_col),
                tolerance=args.match_tolerance,
                pth_prefix=pth_prefix,
                force_full_embedding_scan=args.force_full_embedding_scan,
                embedding_load_workers=args.embedding_load_workers,
                selective_embedding_strategy=args.selective_embedding_strategy,
                group_col=args.logo_group_col,
                return_groups=True,
            )
            if X_coords_matched is not None:
                print(f"Matched coordinates shape: {X_coords_matched.shape}")
            groups = np.asarray(groups)
            n_unique_grp = len(np.unique(groups))
            if n_unique_grp < 2:
                if X_coords_matched is None:
                    raise RuntimeError(
                        "LOGO needs ≥2 groups or HE coordinates for spatial tiling; "
                        f"{args.logo_group_col!r} is degenerate and coordinates are missing."
                    )
                print(
                    f"  ⚠ Only {n_unique_grp} unique group(s) in {args.logo_group_col!r}; "
                    f"using spatial_tile_groups_for_logo(nx={args.logo_spatial_nx}, "
                    f"ny={args.logo_spatial_ny})."
                )
                groups = spatial_tile_groups_for_logo(
                    X_coords_matched[:, 0],
                    X_coords_matched[:, 1],
                    nx=args.logo_spatial_nx,
                    ny=args.logo_spatial_ny,
                )
            n_final = len(np.unique(groups))
            if n_final < 2:
                raise RuntimeError("Fewer than 2 LOGO groups after spatial fallback.")
            print(f"LOGO: {n_final} unique groups, {len(groups)} cells")

            print("=" * 72)
            print("Step 2/6: Leave-one-group-out label prep (filtered full matrix)")
            cv_data = prepare_data_leave_one_group_out(X, y, y_level1, groups)
            class_names = cv_data["class_names"]
            class_names_level1 = cv_data["class_names_level1"]
            scaler = StandardScaler()
            scaler.fit(cv_data["X_f"])

            logo_ckpt_dir = (
                Path(args.logo_checkpoint_dir).resolve()
                if args.logo_checkpoint_dir
                else result_dir.resolve()
            )
            _logo_resume = _parse_logo_resume(args.logo_resume)
            _ckpt_glob = sorted(logo_ckpt_dir.glob("hce_logo_fold_*.pt"))
            print("=" * 72)
            print("Step 3-5/6: LOGO CV, best-fold checkpoint, in-sample L2/L1 report")
            print(f"LOGO fold checkpoint directory: {logo_ckpt_dir}")
            if not args.logo_checkpoint_dir:
                print(
                    "  (default: this run's result folder under save_result; "
                    "same LOGO resume: reuse --run_name <id> or --logo_checkpoint_dir <path>)"
                )
            print(f"LOGO existing fold checkpoints in that dir: {len(_ckpt_glob)} file(s)")
            print(
                "LOGO resume: "
                f"--logo_resume={args.logo_resume!r} -> resume_from_checkpoints={_logo_resume!r} "
                "(None=auto; True=load .pt when present; False=retrain every fold — expect Epoch 1/50)"
            )
            print(
                "CV options: "
                f"cv_selection_metric={args.cv_selection_metric!r}, "
                f"train_balance_sampler={args.train_balance_sampler}, "
                f"train_balance_sampler_mode={args.train_balance_sampler_mode!r}, "
                f"val_selection_metric={args.val_selection_metric!r}, "
                f"l2_label_smoothing={args.l2_label_smoothing}, "
                f"l1_label_smoothing={args.l1_label_smoothing}, "
                f"class_weight_mode={args.class_weight_mode!r}, "
                f"class_balanced_beta={args.class_balanced_beta}, "
                f"l2_focal_gamma={args.l2_focal_gamma}"
            )
            LP = run_logo_cv_with_insample_report(
                device,
                cv_data,
                scaler,
                class_names,
                evaluate,
                python_root,
                args.therapy_data,
                therapy_model,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                patience=args.patience,
                max_epochs=args.max_epochs,
                loader_kwargs={
                    "seed": args.seed,
                    "train_balance_sampler": args.train_balance_sampler,
                    "train_balance_sampler_mode": args.train_balance_sampler_mode,
                },
                resume_from_checkpoints=_logo_resume,
                logo_checkpoint_dir=str(logo_ckpt_dir),
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                cv_selection_metric=args.cv_selection_metric,
                l2_label_smoothing=args.l2_label_smoothing,
                l1_label_smoothing=args.l1_label_smoothing,
                val_selection_metric=args.val_selection_metric,
                class_weight_mode=args.class_weight_mode,
                class_balanced_beta=args.class_balanced_beta,
                l2_focal_gamma=args.l2_focal_gamma,
            )
            model = LP["model"]
            input_dim = LP["input_dim"]
            val_acc = LP["val_acc"]
            val_macro_f1 = LP["val_macro_f1"]
            val_weighted_f1 = LP["val_weighted_f1"]
            val_preds = LP["val_preds"]
            val_labels = LP["val_labels"]
            val_level1_acc = LP["val_level1_acc"]
            val_level1_macro_f1 = LP["val_level1_macro_f1"]
            val_level1_weighted_f1 = LP["val_level1_weighted_f1"]
            val_preds_level1 = LP["val_preds_level1"]
            val_labels_level1 = LP["val_labels_level1"]
            best_epoch = LP["best_epoch"]
            best_weighted_f1 = float(LP["best_weighted_f1"])
            best_macro_f1 = float(LP["best_macro_f1"])
            ckpt_src = Path(LP["best_fold"]["checkpoint"])
            if ckpt_src.is_file():
                shutil.copy2(ckpt_src, best_model_path)
                print(f"Copied best LOGO fold checkpoint to {best_model_path}")
            print(
                f"LOGO best fold (held-out val): epoch={best_epoch}, "
                f"macro_f1={best_macro_f1:.4f}, weighted_f1={best_weighted_f1:.4f}"
            )
        elif use_stratified_kfold:
            X, y, y_level1, y_level0, y_level01, y_level12, X_coords_matched = match_hist2cell_matrix(
                cell_coords_path=cell_coords_path,
                hist_embedding_dir=hist_embedding_dir,
                matched_features_path=matched_features_path,
                coord_cols=(args.coord_x_col, args.coord_y_col),
                tolerance=args.match_tolerance,
                pth_prefix=pth_prefix,
                force_full_embedding_scan=args.force_full_embedding_scan,
                embedding_load_workers=args.embedding_load_workers,
                selective_embedding_strategy=args.selective_embedding_strategy,
            )
            if X_coords_matched is not None:
                print(f"Matched coordinates shape: {X_coords_matched.shape}")

            print("=" * 72)
            print("Step 2/6: K-fold label prep (filtered full matrix)")
            # Reuse same filtering/encoding path as LOGO; groups are placeholders here.
            cv_data = prepare_data_leave_one_group_out(
                X, y, y_level1, groups=np.arange(X.shape[0], dtype=np.int64)
            )
            class_names = cv_data["class_names"]
            class_names_level1 = cv_data["class_names_level1"]
            scaler = StandardScaler()
            scaler.fit(cv_data["X_f"])

            kfold_ckpt_dir = (
                Path(args.logo_checkpoint_dir).resolve()
                if args.logo_checkpoint_dir
                else result_dir.resolve()
            )
            _kfold_resume = _parse_logo_resume(args.logo_resume)
            _ckpt_glob = sorted(kfold_ckpt_dir.glob("hce_kfold_fold_*.pt"))
            print("=" * 72)
            print("Step 3-5/6: Stratified K-fold CV, best-fold checkpoint, in-sample L2/L1 report")
            print(f"K-fold checkpoint directory: {kfold_ckpt_dir}")
            print(f"K-fold existing fold checkpoints in that dir: {len(_ckpt_glob)} file(s)")
            print(
                "K-fold resume: "
                f"--logo_resume={args.logo_resume!r} -> resume_from_checkpoints={_kfold_resume!r} "
                "(None=auto; True=load .pt when present; False=retrain every fold)"
            )
            print(
                "CV options: "
                f"cv_selection_metric={args.cv_selection_metric!r}, "
                f"train_balance_sampler={args.train_balance_sampler}, "
                f"train_balance_sampler_mode={args.train_balance_sampler_mode!r}, "
                f"val_selection_metric={args.val_selection_metric!r}, "
                f"l2_label_smoothing={args.l2_label_smoothing}, "
                f"l1_label_smoothing={args.l1_label_smoothing}, "
                f"class_weight_mode={args.class_weight_mode!r}, "
                f"class_balanced_beta={args.class_balanced_beta}, "
                f"l2_focal_gamma={args.l2_focal_gamma}"
            )
            LP = run_stratified_kfold_cv_with_insample_report(
                device,
                cv_data,
                scaler,
                class_names,
                evaluate,
                python_root,
                args.therapy_data,
                therapy_model,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                n_splits=args.cv_k,
                stratify_target=args.cv_stratify_target,
                patience=args.patience,
                max_epochs=args.max_epochs,
                loader_kwargs={
                    "seed": args.seed,
                    "train_balance_sampler": args.train_balance_sampler,
                    "train_balance_sampler_mode": args.train_balance_sampler_mode,
                },
                resume_from_checkpoints=_kfold_resume,
                kfold_checkpoint_dir=str(kfold_ckpt_dir),
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                cv_selection_metric=args.cv_selection_metric,
                l2_label_smoothing=args.l2_label_smoothing,
                l1_label_smoothing=args.l1_label_smoothing,
                val_selection_metric=args.val_selection_metric,
                class_weight_mode=args.class_weight_mode,
                class_balanced_beta=args.class_balanced_beta,
                l2_focal_gamma=args.l2_focal_gamma,
            )
            model = LP["model"]
            input_dim = LP["input_dim"]
            val_acc = LP["val_acc"]
            val_macro_f1 = LP["val_macro_f1"]
            val_weighted_f1 = LP["val_weighted_f1"]
            val_preds = LP["val_preds"]
            val_labels = LP["val_labels"]
            val_level1_acc = LP["val_level1_acc"]
            val_level1_macro_f1 = LP["val_level1_macro_f1"]
            val_level1_weighted_f1 = LP["val_level1_weighted_f1"]
            val_preds_level1 = LP["val_preds_level1"]
            val_labels_level1 = LP["val_labels_level1"]
            best_epoch = LP["best_epoch"]
            best_weighted_f1 = float(LP["best_weighted_f1"])
            best_macro_f1 = float(LP["best_macro_f1"])
            ckpt_src = Path(LP["best_fold"]["checkpoint"])
            if ckpt_src.is_file():
                shutil.copy2(ckpt_src, best_model_path)
                print(f"Copied best K-fold checkpoint to {best_model_path}")
            print(
                f"K-fold best fold (held-out val): epoch={best_epoch}, "
                f"macro_f1={best_macro_f1:.4f}, weighted_f1={best_weighted_f1:.4f}"
            )
        else:
            X, y, y_level1, y_level0, y_level01, y_level12, X_coords_matched = match_hist2cell_matrix(
                cell_coords_path=cell_coords_path,
                hist_embedding_dir=hist_embedding_dir,
                matched_features_path=matched_features_path,
                coord_cols=(args.coord_x_col, args.coord_y_col),
                tolerance=args.match_tolerance,
                pth_prefix=pth_prefix,
                force_full_embedding_scan=args.force_full_embedding_scan,
                embedding_load_workers=args.embedding_load_workers,
                selective_embedding_strategy=args.selective_embedding_strategy,
            )
            if X_coords_matched is not None:
                print(f"Matched coordinates shape: {X_coords_matched.shape}")

            print("=" * 72)
            print("Step 2/6: Split train/validation")
            result = split_train_test(
                X=X,
                y=y,
                y_level1=y_level1,
                test_size=args.test_size,
                random_state=args.seed,
            )
            class_names = result["class_names"]
            class_names_level1 = result["class_names_level1"]
            scaler = result["scaler"]

            print("=" * 72)
            print("Step 3/6: Build loaders")
            loaders = loader_train_test(
                result=result,
                batch_size_cuda=args.batch_size_cuda,
                batch_size_cpu=args.batch_size_cpu,
                num_workers_cuda=args.num_workers_cuda,
                num_workers_cpu=args.num_workers_cpu,
                pin_memory_cuda=not args.disable_pin_memory_cuda,
                pin_memory_cpu=args.pin_memory_cpu,
                seed=args.seed,
                train_balance_sampler=args.train_balance_sampler,
                train_balance_sampler_mode=args.train_balance_sampler_mode,
            )

            print("=" * 72)
            print("Step 4/6: Train model with HCE")
            best_weighted_f1, best_macro_f1, best_epoch = train_and_save_model_from_split(
                device=device,
                loaders=loaders,
                result=result,
                evaluate=evaluate,
                save_bestmodel_path=str(best_model_path),
                class_names=class_names,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                patience=args.patience,
                max_epochs=args.max_epochs,
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                l2_label_smoothing=args.l2_label_smoothing,
                l1_label_smoothing=args.l1_label_smoothing,
                val_selection_metric=args.val_selection_metric,
                class_weight_mode=args.class_weight_mode,
                class_balanced_beta=args.class_balanced_beta,
                l2_focal_gamma=args.l2_focal_gamma,
                checkpoint_exists=args.checkpoint_exists,
            )
            print(
                f"Training done. best_epoch={best_epoch}, "
                f"best_weighted_f1={best_weighted_f1:.4f}, best_macro_f1={best_macro_f1:.4f}"
            )

            print("=" * 72)
            print("Step 5/6: Validate best checkpoint (level2 + level1)")
            (
                model,
                input_dim,
                val_acc,
                val_macro_f1,
                val_weighted_f1,
                val_preds,
                val_labels,
                val_level1_acc,
                val_level1_macro_f1,
                val_level1_weighted_f1,
                val_preds_level1,
                val_labels_level1,
            ) = mode_validation_from_split(
                device=device,
                path=python_root,
                therapy_data=args.therapy_data,
                loaders=loaders,
                result=result,
                class_names=class_names,
                class_names_level1=class_names_level1,
                therapy_model=therapy_model,
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
            )

        if use_cv_fullmatrix:
            y_enc_f = cv_data["y_encoded_f"]
            y_l1_enc_f = cv_data["y_level1_encoded_f"]
        else:
            y_enc_f = result["y_encoded_f"]
            y_l1_enc_f = result["y_level1_encoded_f"]

        if use_logo:
            plot_tag = f"{args.parent_value}_logo"
        elif use_stratified_kfold:
            plot_tag = f"{args.parent_value}_kfold"
        else:
            plot_tag = args.parent_value
        train_ds = loaders["train_dataset"] if loaders else None
        val_ds = loaders["val_dataset"] if loaders else None

        #######################################
        # Internal validation metrics
        #######################################
        ## Confusion matrices (level2 + level1).
        plot_confusion_matrix(
            val_labels,
            val_preds,
            class_names=class_names,
            normalize=not args.confusion_matrix_counts,
            hierarchical=args.hierarchical_confusion_matrix,
            save_path=str(result_dir / f"confusion_{plot_tag}_level2.png"),
        )
        plot_confusion_matrix(
            val_labels_level1,
            val_preds_level1,
            class_names=class_names_level1,
            normalize=not args.confusion_matrix_counts,
            hierarchical=args.hierarchical_confusion_matrix,
            save_path=str(result_dir / f"confusion_{plot_tag}_level1.png"),
        )

        ## Per-class F1 (level2 + level1).
        metrics_df_level2 = plot_per_class_f1(
            test_labels=val_labels,
            test_preds=val_preds,
            class_names=class_names,
            model=model,
            device=device,
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            best_epoch=best_epoch,
            test_acc=val_acc,
            test_macro_f1=val_macro_f1,
            test_weighted_f1=val_weighted_f1,
            model_name="MLP-HCE Level2",
            save_path=str(result_dir / f"f1_{plot_tag}_level2.png"),
        )
        metrics_df_level1 = plot_per_class_f1(
            test_labels=val_labels_level1,
            test_preds=val_preds_level1,
            class_names=class_names_level1,
            model=None,
            device=device,
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            best_epoch=best_epoch,
            test_acc=val_level1_acc,
            test_macro_f1=val_level1_macro_f1,
            test_weighted_f1=val_level1_weighted_f1,
            model_name="MLP-HCE Level1",
            save_path=str(result_dir / f"f1_{plot_tag}_level1.png"),
        )
        metrics_df_level2.to_csv(result_dir / f"f1_{plot_tag}_level2.csv")
        metrics_df_level1.to_csv(result_dir / f"f1_{plot_tag}_level1.csv")

        ## Per-class Accuracy (level2 + level1).
        metrics_acc_level2 = plot_per_class_accuracy(
            test_labels=val_labels,
            test_preds=val_preds,
            class_names=class_names,
            model=model,
            device=device,
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            best_epoch=best_epoch,
            test_acc=val_acc,
            test_macro_f1=val_macro_f1,
            test_weighted_f1=val_weighted_f1,
            model_name="MLP-HCE Level2",
            save_path=str(result_dir / f"acc_{plot_tag}_level2.png"),
        )
        metrics_acc_level1 = plot_per_class_accuracy(
            test_labels=val_labels_level1,
            test_preds=val_preds_level1,
            class_names=class_names_level1,
            model=model,
            device=device,
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            best_epoch=best_epoch,
            test_acc=val_level1_acc,
            test_macro_f1=val_level1_macro_f1,
            test_weighted_f1=val_level1_weighted_f1,
            model_name="MLP-HCE Level1",
            save_path=str(result_dir / f"acc_{plot_tag}_level1.png"),
        )
        metrics_acc_level2.to_csv(result_dir / f"acc_{plot_tag}_level2.csv")
        metrics_acc_level1.to_csv(result_dir / f"acc_{plot_tag}_level1.csv")

        # Save one-row run summary (LOGO: fold-mean F1s, NaN accs — notebook-aligned).
        if use_logo:
            logo_summary = LP["logo_summary"]
            mean_best_epoch = int(
                round(float(np.mean([f["best_epoch"] for f in logo_summary["folds"]])))
            )
            save_hce_validation_metrics(
                val_acc=float("nan"),
                val_macro_f1=logo_summary["l2_macro_f1_mean"],
                val_weighted_f1=logo_summary["l2_weighted_f1_mean"],
                val_level1_acc=float("nan"),
                val_level1_macro_f1=logo_summary["l1_macro_f1_mean"],
                val_level1_weighted_f1=logo_summary["l1_weighted_f1_mean"],
                class_names=class_names,
                class_names_level1=class_names_level1,
                best_epoch=mean_best_epoch,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                therapy_data=f"{args.therapy_data}_{args.parent_value}_LOGO",
                metrics_csv_path=str(metrics_csv_path),
            )
        elif use_stratified_kfold:
            kfold_summary = LP["logo_summary"]
            mean_best_epoch = int(
                round(float(np.mean([f["best_epoch"] for f in kfold_summary["folds"]])))
            )
            save_hce_validation_metrics(
                val_acc=float("nan"),
                val_macro_f1=kfold_summary["l2_macro_f1_mean"],
                val_weighted_f1=kfold_summary["l2_weighted_f1_mean"],
                val_level1_acc=float("nan"),
                val_level1_macro_f1=kfold_summary["l1_macro_f1_mean"],
                val_level1_weighted_f1=kfold_summary["l1_weighted_f1_mean"],
                class_names=class_names,
                class_names_level1=class_names_level1,
                best_epoch=mean_best_epoch,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                therapy_data=f"{args.therapy_data}_{args.parent_value}_KFOLD{kfold_summary.get('n_splits', args.cv_k)}",
                metrics_csv_path=str(metrics_csv_path),
            )
        else:
            save_hce_validation_metrics(
                val_acc=val_acc,
                val_macro_f1=val_macro_f1,
                val_weighted_f1=val_weighted_f1,
                val_level1_acc=val_level1_acc,
                val_level1_macro_f1=val_level1_macro_f1,
                val_level1_weighted_f1=val_level1_weighted_f1,
                class_names=class_names,
                class_names_level1=class_names_level1,
                best_epoch=best_epoch,
                hce_w1=args.hce_w1,
                hce_w2=args.hce_w2,
                hce_w12=args.hce_w12,
                therapy_data=args.therapy_data,
                metrics_csv_path=str(metrics_csv_path),
            )

        print("=" * 72)
        print("Step 6/6: Optional all-data evaluation + spatial distribution")
        if args.run_all_data_eval:
            spatial_tag = plot_tag
            level2_pred_save_path = args.save_path_level2_pred or str(
                result_dir / f"celltype_valid_{spatial_tag}_level2_hce.jpg"
            )
            level2_true_save_path = args.save_path_level2_true or str(
                result_dir / f"celltype_true_{spatial_tag}_level2_hce.jpg"
            )
            level1_pred_save_path = args.save_path_level1_pred or str(
                result_dir / f"celltype_valid_{spatial_tag}_level1_hce.jpg"
            )
            level1_true_save_path = args.save_path_level1_true or str(
                result_dir / f"celltype_true_{spatial_tag}_level1_hce.jpg"
            )
            conf_csv_path = result_dir / f"pred_confidence_{spatial_tag}.csv"
            conf_l2_hist_path = result_dir / f"pred_confidence_{spatial_tag}_level2_hist.png"
            conf_l1_hist_path = result_dir / f"pred_confidence_{spatial_tag}_level1_hist.png"

            if use_cv_fullmatrix:
                X_coords_plot = None
                if X_coords_matched is not None:
                    X_coords_plot = X_coords_matched[cv_data["mask"]]
                all_acc, all_macro_f1, all_weighted_f1, all_preds, all_labels = (
                    evaluate_and_plot_on_all_data(
                        model=model,
                        matched_features_path="",
                        class_names=class_names,
                        evaluate=evaluate,
                        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                        CellTypeDataset=CellTypeDataset,
                        device=device,
                        scaler=scaler,
                        val_loader=None,
                        X=cv_data["X_f"],
                        y=None,
                        y_encoded=cv_data["y_encoded_f"],
                        X_train=None,
                        X_train_scaled=None,
                        X_test_scaled=None,
                        y_train_encoded=None,
                        y_test_encoded=None,
                        X_coords_matched=X_coords_plot,
                        celltype_pred_dir=level2_pred_save_path,
                        celltype_true_dir=level2_true_save_path,
                        prediction_only=args.prediction_only,
                        spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                        spatial_point_size=args.spatial_point_size,
                    )
                )
                logo_l1_npz = result_dir / f"_logo_level1_aux_{args.parent_value}.npz"
                np.savez(
                    logo_l1_npz,
                    y_level1=np.asarray(cv_data["y_level1_f"]),
                )
                if X_coords_plot is not None:
                    np.savez(
                        result_dir / f"_logo_level1_spatial_{args.parent_value}.npz",
                        X_coords=np.asarray(X_coords_plot),
                        y_level1=np.asarray(cv_data["y_level1_f"]),
                    )
                    plot_level1_spatial_distribution(
                        matched_features_path=str(
                            result_dir / f"_logo_level1_spatial_{args.parent_value}.npz"
                        ),
                        all_preds=all_preds,
                        class_names_level1=class_names_level1,
                        class_names=class_names,
                        y_encoded_f=y_enc_f,
                        y_level1_encoded_f=y_l1_enc_f,
                        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                        save_path_pred=level1_pred_save_path,
                        save_path_true=level1_true_save_path,
                        spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                        spatial_point_size=args.spatial_point_size,
                    )
                else:
                    print(
                        "  ⚠ LOGO all-data: no coordinates after mask; skip level1 spatial maps."
                    )
                metrics_acc_all_level1 = plot_level1_accuracy_from_level2_predictions(
                    matched_features_path=str(logo_l1_npz),
                    all_preds=all_preds,
                    class_names=class_names,
                    class_names_level1=class_names_level1,
                    y_encoded_f=y_enc_f,
                    y_level1_encoded_f=y_l1_enc_f,
                    model=model,
                    device=device,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    input_dim=input_dim,
                    best_epoch=best_epoch,
                    model_name="MLP-HCE AllData Level1",
                    save_path=str(result_dir / f"acc_{spatial_tag}_alldata_level1.png"),
                )
                export_all_data_prediction_confidence(
                    model=model,
                    X=cv_data["X_f"],
                    device=device,
                    class_names=class_names,
                    class_names_level1=class_names_level1,
                    y_encoded_f=y_enc_f,
                    y_level1_encoded_f=y_l1_enc_f,
                    scaler=scaler,
                    X_coords=X_coords_plot,
                    y_true_l2=cv_data["y_encoded_f"],
                    y_true_l1=cv_data["y_level1_encoded_f"],
                    csv_path=str(conf_csv_path),
                    level2_hist_path=str(conf_l2_hist_path),
                    level1_hist_path=str(conf_l1_hist_path),
                )
            else:
                all_acc, all_macro_f1, all_weighted_f1, all_preds, all_labels = (
                    evaluate_and_plot_on_all_data(
                        model=model,
                        matched_features_path=str(matched_features_path),
                        class_names=class_names,
                        evaluate=evaluate,
                        plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                        CellTypeDataset=CellTypeDataset,
                        device=device,
                        scaler=scaler,
                        val_loader=loaders["val_loader_eval"],
                        X=X,
                        y=y,
                        y_encoded=result["y_encoded_f"],
                        X_train=result["X_train"],
                        X_train_scaled=result["X_train_scaled"],
                        X_test_scaled=result["X_test_scaled"],
                        y_train_encoded=result["y_train_encoded"],
                        y_test_encoded=result["y_test_encoded"],
                        X_coords_matched=X_coords_matched,
                        celltype_pred_dir=level2_pred_save_path,
                        celltype_true_dir=level2_true_save_path,
                        prediction_only=args.prediction_only,
                        spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                        spatial_point_size=args.spatial_point_size,
                    )
                )
                plot_level1_spatial_distribution(
                    matched_features_path=str(matched_features_path),
                    all_preds=all_preds,
                    class_names_level1=class_names_level1,
                    class_names=class_names,
                    y_encoded_f=y_enc_f,
                    y_level1_encoded_f=y_l1_enc_f,
                    plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                    save_path_pred=level1_pred_save_path,
                    save_path_true=level1_true_save_path,
                    spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                    spatial_point_size=args.spatial_point_size,
                )
                metrics_acc_all_level1 = plot_level1_accuracy_from_level2_predictions(
                    matched_features_path=str(matched_features_path),
                    all_preds=all_preds,
                    class_names=class_names,
                    class_names_level1=class_names_level1,
                    y_encoded_f=y_enc_f,
                    y_level1_encoded_f=y_l1_enc_f,
                    model=model,
                    device=device,
                    train_dataset=loaders["train_dataset"],
                    val_dataset=loaders["val_dataset"],
                    input_dim=input_dim,
                    best_epoch=best_epoch,
                    model_name="MLP-HCE AllData Level1",
                    save_path=str(result_dir / f"acc_{args.parent_value}_stardist_level1.png"),
                )
                export_all_data_prediction_confidence(
                    model=model,
                    X=X,
                    device=device,
                    class_names=class_names,
                    class_names_level1=class_names_level1,
                    y_encoded_f=y_enc_f,
                    y_level1_encoded_f=y_l1_enc_f,
                    scaler=scaler,
                    X_coords=X_coords_matched,
                    y_true_l2=y_enc_f if len(y_enc_f) == len(X) else None,
                    y_true_l1=y_l1_enc_f if len(y_l1_enc_f) == len(X) else None,
                    csv_path=str(conf_csv_path),
                    level2_hist_path=str(conf_l2_hist_path),
                    level1_hist_path=str(conf_l1_hist_path),
                )
            if metrics_acc_all_level1 is not None:
                alldata_l1_csv = (
                    result_dir / f"acc_{spatial_tag}_alldata_level1.csv"
                    if use_logo
                    else result_dir / f"acc_{args.parent_value}_stardist_level1.csv"
                )
                metrics_acc_all_level1.to_csv(alldata_l1_csv)
            print(
                "All-data metrics: "
                f"acc={all_acc}, macro_f1={all_macro_f1}, weighted_f1={all_weighted_f1}; "
                f"n_preds={len(all_preds)}, n_labels={len(all_labels) if all_labels is not None else 0}"
            )
        else:
            print("Skipped all-data evaluation (use --run_all_data_eval to enable).")

        print("=" * 72)
        print("Step 7/7: Optional StarDist prediction (model from project_all)")
        if args.run_stardist_eval:
            # Keep notebook behavior: model_star is loaded from project_all by default.
            stardist_model_therapy_model = (
                args.stardist_model_therapy_model
                or therapy_model
            )
            model_star = load_model_for_predict(
                args.python_root,
                args.therapy_data,
                stardist_model_therapy_model,
                device=device,
                parent_dir=False,
                run_log_path=str(log_path),
            )

            # StarDist data paths (default to parent-specific project path).
            stardist_data_therapy_model = (
                args.stardist_data_therapy_model
                or f"{args.therapy_data}_project_{args.parent_value_stardist}_{args.method}"
            )
            stardist_model_data_dir = data_root / args.therapy_data / stardist_data_therapy_model
            celltype_pixel_stardist_path = (
                Path(args.stardist_cell_coords_path)
                if args.stardist_cell_coords_path
                else data_root
                / args.transfer_data
                / f"{args.therapy_data}_CellPixCoords_{args.parent_value_stardist}_StarDist.csv"
            )
            embedding_star_dir = (
                Path(args.stardist_embedding_dir)
                if args.stardist_embedding_dir
                else stardist_model_data_dir
                / f"ImgEmbeddings_{args.parent_value_stardist}_stardist"
                / args.embedding_subdir
            )
            matched_features_star_path = (
                Path(args.stardist_matched_features_path)
                if args.stardist_matched_features_path
                else stardist_model_data_dir / f"matched_features_{args.parent_value_stardist}_stardist.npz"
            )
            print(f"stardist_model_therapy_model: {stardist_model_therapy_model}")
            print(f"stardist_data_therapy_model: {stardist_data_therapy_model}")
            print(f"stardist_cell_coords_path: {celltype_pixel_stardist_path}")
            print(f"stardist_embedding_dir: {embedding_star_dir}")
            print(f"stardist_matched_features_path: {matched_features_star_path}")

            X_star, y_star, y_star_level1, y_star_level0, y_star_level01, y_star_level12, X_coords_star = match_hist2cell_matrix(
                cell_coords_path=celltype_pixel_stardist_path,
                hist_embedding_dir=embedding_star_dir,
                matched_features_path=matched_features_star_path,
                coord_cols=("centroid_x", "centroid_y"),
                tolerance=args.match_tolerance,
                pth_prefix=pth_prefix,
                force_full_embedding_scan=args.force_full_embedding_scan,
                embedding_load_workers=args.embedding_load_workers,
                selective_embedding_strategy=args.selective_embedding_strategy,
            )
            print(
                "StarDist matched shapes: "
                f"X={getattr(X_star, 'shape', None)}, y={getattr(y_star, 'shape', None)}, "
                f"y_level1={getattr(y_star_level1, 'shape', None)}, "
                f"coords={getattr(X_coords_star, 'shape', None)}"
            )

            # Build StarDist class names from model output dimension first.
            # This avoids truncating predictions when training split class_names has fewer classes.
            num_classes_star = int(model_star.output_layer.weight.shape[0])
            if class_names is None:
                class_names_star = [str(i) for i in range(num_classes_star)]
            else:
                class_names_star = list(class_names)
                if len(class_names_star) < num_classes_star:
                    n_missing = num_classes_star - len(class_names_star)
                    print(
                        f"  ⚠ class_names has {len(class_names_star)} classes but model_star outputs "
                        f"{num_classes_star}. Appending {n_missing} placeholder names."
                    )
                    class_names_star.extend(
                        [f"Unknown_{i}" for i in range(len(class_names_star), num_classes_star)]
                    )
                elif len(class_names_star) > num_classes_star:
                    print(
                        f"  ⚠ class_names has {len(class_names_star)} classes but model_star outputs "
                        f"{num_classes_star}. Truncating class_names_star to output dimension."
                    )
                    class_names_star = class_names_star[:num_classes_star]
            star_level2_pred_path = args.stardist_save_path_level2_pred or str(
                result_dir / f"celltype_pred_{args.parent_value_stardist}_level2_hce.jpg"
            )
            star_level2_true_path = args.stardist_save_path_level2_true or str(
                result_dir / f"celltype_true_{args.parent_value_stardist}_level2_hce.jpg"
            )
            star_level1_pred_path = args.stardist_save_path_level1_pred or str(
                result_dir / f"celltype_pred_{args.parent_value_stardist}_level1_hce.jpg"
            )
            star_level1_true_path = args.stardist_save_path_level1_true or str(
                result_dir / f"celltype_true_{args.parent_value_stardist}_level1_hce.jpg"
            )

            all_acc_star, all_macro_f1_star, all_weighted_f1_star, all_preds_star, all_labels_star = evaluate_and_plot_on_all_data(
                model=model_star,
                matched_features_path=str(matched_features_star_path),
                class_names=class_names_star,
                evaluate=evaluate,
                plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                CellTypeDataset=CellTypeDataset,
                device=device,
                scaler=scaler,
                val_loader=None,
                prediction_only=True,
                X_coords_matched=X_coords_star,
                celltype_pred_dir=star_level2_pred_path,
                celltype_true_dir=star_level2_true_path,
                spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                spatial_point_size=args.spatial_point_size,
            )

            plot_level1_spatial_distribution(
                matched_features_path=str(matched_features_star_path),
                all_preds=all_preds_star,
                class_names_level1=class_names_level1,
                class_names=class_names,
                y_encoded_f=y_enc_f,
                y_level1_encoded_f=y_l1_enc_f,
                plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
                save_path_pred=star_level1_pred_path,
                save_path_true=star_level1_true_path,
                spatial_figsize=(args.spatial_fig_w, args.spatial_fig_h),
                spatial_point_size=args.spatial_point_size,
            )

            if all_labels_star is not None and all_preds_star is not None:
                metrics_acc_star = plot_per_class_accuracy(
                    all_labels_star,
                    all_preds_star,
                    class_names_star,
                    model=model_star,
                    device=device,
                    train_dataset=None,
                    val_dataset=None,
                    input_dim=None,
                    best_epoch=None,
                    test_acc=all_acc_star,
                    test_macro_f1=all_macro_f1_star,
                    test_weighted_f1=all_weighted_f1_star,
                    model_name="MLP-StarDist-Level2",
                    save_path=str(
                        result_dir / f"acc_{args.parent_value}_level2_stardist_hce.png"
                    ),
                )
                metrics_acc_star.to_csv(
                    result_dir / f"acc_metrics_{args.parent_value}_level2_stardist_hce.csv"
                )

            metrics_acc_star_l1 = plot_level1_accuracy_from_level2_predictions(
                matched_features_path=str(matched_features_star_path),
                all_preds=all_preds_star,
                class_names=class_names,
                class_names_level1=class_names_level1,
                y_encoded_f=y_enc_f,
                y_level1_encoded_f=y_l1_enc_f,
                model=model_star,
                device=device,
                model_name="MLP-StarDist-Level1",
                save_path=str(
                    result_dir / f"acc_{args.parent_value}_level1_stardist_hce.png"
                ),
            )
            if metrics_acc_star_l1 is not None:
                metrics_acc_star_l1.to_csv(
                    result_dir / f"acc_metrics_{args.parent_value}_level1_stardist_hce.csv"
                )

            # StarDist confusion matrices (level2 + level1), saved in the same result_dir as acc bar plots.
            if all_labels_star is not None and all_preds_star is not None:
                plot_confusion_matrix(
                    all_labels_star,
                    all_preds_star,
                    class_names_star,
                    save_path=str(
                        result_dir / f"conf_matrix_{args.parent_value}_level2_stardist_hce.pdf"
                    ),
                )

            try:
                _loaded_star = np.load(matched_features_star_path, allow_pickle=True)
                if "y_level1" in _loaded_star and all_preds_star is not None:
                    _y_true_l1_raw = _loaded_star["y_level1"]
                    _child_to_parent = np.full(len(class_names_star), -1, dtype=np.int64)
                    for _l2, _l1 in zip(y_enc_f, y_l1_enc_f):
                        _l2_i, _l1_i = int(_l2), int(_l1)
                        if 0 <= _l2_i < len(_child_to_parent) and _child_to_parent[_l2_i] == -1:
                            _child_to_parent[_l2_i] = _l1_i

                    _preds_l1 = _child_to_parent[np.asarray(all_preds_star, dtype=np.int64)]

                    if (
                        getattr(_y_true_l1_raw, "dtype", None) is not None
                        and _y_true_l1_raw.dtype.kind in ["U", "S", "O"]
                    ):
                        _name_to_idx = {str(c): i for i, c in enumerate(class_names_level1)}
                        _y_true_l1 = np.array(
                            [_name_to_idx.get(str(v), -1) for v in _y_true_l1_raw],
                            dtype=np.int64,
                        )
                    else:
                        _y_true_l1 = np.asarray(_y_true_l1_raw, dtype=np.int64)

                    _n = min(len(_y_true_l1), len(_preds_l1))
                    _y_true_l1 = _y_true_l1[:_n]
                    _preds_l1 = _preds_l1[:_n]
                    _valid = _y_true_l1 >= 0

                    if np.any(_valid):
                        plot_confusion_matrix(
                            _y_true_l1[_valid],
                            _preds_l1[_valid],
                            class_names_level1,
                            save_path=str(
                                result_dir
                                / f"conf_matrix_{args.parent_value}_level1_stardist_hce.pdf"
                            ),
                        )
                    else:
                        print("Skip level1 StarDist confusion matrix: no valid y_level1 labels.")
                else:
                    print(
                        f"Skip level1 StarDist confusion matrix: y_level1 not found in {matched_features_star_path}"
                    )
            except Exception as e:
                print(f"Skip StarDist confusion matrix generation due to error: {e}")
        else:
            print("Skipped StarDist evaluation (use --run_stardist_eval to enable).")

        print("=" * 72)
        print("Completed.")
        model_copy_path = result_dir / "best_mlp_gpu.pt"
        if best_model_path.exists():
            if best_model_path.resolve() != model_copy_path.resolve():
                shutil.copy2(best_model_path, model_copy_path)
            print(f"Best model (training path): {best_model_path}")
            print(f"Best model (copied): {model_copy_path}")
        else:
            print(f"Best model not found at: {best_model_path}")
        print(f"Metrics CSV: {metrics_csv_path}")
        print(f"Result directory: {result_dir}")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_f.close()


def _cli_bool(value) -> bool:
    """Parse true/false for --checkpoint_exists (avoids argparse type=bool pitfalls)."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean (e.g. true/false), got {value!r}")


def _parse_logo_resume(value: str | None):
    """auto -> None (detect fold checkpoints); true/false -> explicit per-fold resume flag."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == "auto":
        return None
    return _cli_bool(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/validate MLP model from terminal.")

    ## Path and dataset settings.
    parser.add_argument("--python_root", type=str, default="/home/lingyu/ssd2/Python/", help="Root path",)
    parser.add_argument("--project_root", type=str, default="/home/lingyu/ssd2/Python/Hist2Pheno", 
        help="Hist2Pheno repo root; ESCC data resolved as <project_root>/data/CODEX/ESCC.",)
    parser.add_argument("--therapy_data", type=str, default="NCRT")
    parser.add_argument("--parent_value", type=str, default="tumor1")
    parser.add_argument("--transfer_data", type=str, default="he_cell_coords")
    parser.add_argument("--method", type=str, default="HIPT")
    parser.add_argument("--therapy_model", type=str, default=None, 
        help="Defaults to '{therapy_data}_project_{parent_value}_{method}'.",)
    parser.add_argument("--save_result", type=str, default="result", 
        help="Result subdirectory under data/<therapy_data>/<therapy_model>/.",)
    parser.add_argument("--run_prefix", type=str, default="train_validate_", 
        help="Prefix for run folder under data/<therapy_data>/<therapy_model>/<save_result>/.",)
    parser.add_argument("--run_name", type=str, default=None, 
        help="Optional run folder name. Default: timestamp YYYYmmddHHMMSSffffff.",)

    ## Optional explicit paths (auto-derived if omitted).
    parser.add_argument("--cell_coords_path", type=str, default=None)
    parser.add_argument("--hist_embedding_dir", type=str, default=None)
    parser.add_argument(
        "--matched_features_path",
        type=str,
        default=None,
        help="Default: matched_features_{parent}_logo.npz (cv_mode=logo/stratified_kfold) or matched_features_{parent}.npz.",
    )
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--metrics_csv_path", type=str, default=None)
    parser.add_argument("--embedding_subdir", type=str, default="sc_pth_16_16")

    ## Matching / split / loader.
    parser.add_argument("--coord_x_col", type=str, default="X_pix_HE")
    parser.add_argument("--coord_y_col", type=str, default="Y_pix_HE")
    parser.add_argument("--match_tolerance", type=float, default=1.0)
    parser.add_argument("--pth_prefix", type=str, default=None,
        help="Embedding filename prefix, e.g. sc_NCRT -> sc_NCRT_x_y.pth. Default: sc_{therapy_data}.",)
    parser.add_argument("--embedding_pth_cap", type=int, default=None, metavar="N",
        help="Sets ESCCAI_EMBEDDING_PTH_CAP before loading: directories with >=N.",)
    parser.add_argument("--force_full_embedding_scan", action="store_true",
        help="Always glob and load all *.pth (skip selective path; slow if millions of files).",
    )
    parser.add_argument("--embedding_load_workers", type=int, default=0, 
        help= "Parallel torch.load threads for full scan (0 = auto, ~4-16). Use 1 for single-threaded.", )
    parser.add_argument("--selective_embedding_strategy", type=str, choices=("batch", "per_cell"), default="batch",
        help="batch: vectorized unique patches + parallel load; per_cell: legacy row loop (very slow).",
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        choices=("logo", "stratified_kfold", "random_split"),
        default="stratified_kfold",
        help="logo: Leave-One-Group-Out; stratified_kfold: label-stratified K-fold; random_split: stratified 70/30.",
    )
    parser.add_argument(
        "--cv_k",
        type=int,
        default=5,
        help="Number of folds for --cv_mode stratified_kfold.",
    )
    parser.add_argument(
        "--cv_stratify_target",
        type=str,
        choices=("level2", "level1", "joint"),
        default="joint",
        help="Stratification target for stratified_kfold: level2, level1, or joint(level1|level2).",
    )
    parser.add_argument(
        "--cv_selection_metric",
        type=str,
        choices=("four_term_sum", "l2_macro_priority"),
        default="four_term_sum",
        help="Best-fold selection metric for LOGO/K-fold CV.",
    )
    parser.add_argument(
        "--train_balance_sampler",
        action="store_true",
        help="Use WeightedRandomSampler in CV training loaders to upweight rare level2 classes.",
    )
    parser.add_argument(
        "--train_balance_sampler_mode",
        type=str,
        choices=("global_l2", "hierarchical_l1_l2"),
        default="global_l2",
        help="global_l2: inverse level2 frequency. hierarchical_l1_l2: first balance level1, then level2 within each level1.",
    )
    parser.add_argument(
        "--val_selection_metric",
        type=str,
        choices=("four_term_sum", "l2_macro_priority"),
        default="four_term_sum",
        help="Epoch-level validation metric used for early stopping / best checkpoint during training.",
    )
    parser.add_argument(
        "--logo_resume",
        type=str,
        default="true",
        metavar="{auto,true,false}",
        help="CV fold resume: true (default)=load existing fold .pt when present (no Epoch loop); missing folds still train. "
        "auto=like true only if any such .pt exists in fold dir, else train all. "
        "false=always retrain every fold (ignore existing .pt).",
    )
    parser.add_argument(
        "--logo_checkpoint_dir",
        type=str,
        default=None,
        help="LOGO: directory for hce_logo_fold_*.pt. Default: this run's result dir "
        "(.../<therapy_model>/<save_result>/<run_prefix><run_name_or_timestamp>/).",
    )
    parser.add_argument(
        "--logo_group_col",
        type=str,
        default="TumorID",
        help="CSV column for LOGO groups (stored in *_logo.npz).",
    )
    parser.add_argument(
        "--logo_spatial_nx",
        type=int,
        default=12,
        help="When TumorID (or logo_group_col) is constant: HE X-axis bins for spatial_tile_groups_for_logo (default 12 → 144 folds with ny=12).",
    )
    parser.add_argument(
        "--logo_spatial_ny",
        type=int,
        default=12,
        help="When group column is constant: HE Y-axis bins (use e.g. 2 and nx=2 for 2×2 LOGO).",
    )
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size_cuda", type=int, default=1024)
    parser.add_argument("--batch_size_cpu", type=int, default=256)
    parser.add_argument("--num_workers_cuda", type=int, default=4)
    parser.add_argument("--num_workers_cpu", type=int, default=0)
    parser.add_argument("--disable_pin_memory_cuda", action="store_true")
    parser.add_argument("--pin_memory_cpu", action="store_true")
    parser.add_argument("--allow_cpu_train", action="store_true")
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        metavar="INDEX",
        help="CUDA device index when GPU is available (e.g. 2 for the third GPU). Ignored on CPU.",
    )

    ## Train hyperparameters.
    parser.add_argument("--hce_w1", type=float, default=1.0, help="Weight for CE(L1 head).")
    parser.add_argument("--hce_w2", type=float, default=1.0, help="Weight for CE(L2 head).")
    parser.add_argument("--hce_w12", type=float, default=1.0, help="Weight for NLL(L1 from L2 agg).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        choices=("none", "balanced", "class_balanced"),
        default="balanced",
        help="Class weighting mode for CE losses on level2 and level1.",
    )
    parser.add_argument(
        "--class_balanced_beta",
        type=float,
        default=0.9999,
        help="Beta used by class-balanced weighting when --class_weight_mode class_balanced.",
    )
    parser.add_argument(
        "--l2_focal_gamma",
        type=float,
        default=0.0,
        help="Focal gamma for level2 loss only (0.0 disables focal weighting).",
    )
    parser.add_argument(
        "--l2_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for level2 CE loss (0.0 disables).",
    )
    parser.add_argument(
        "--l1_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for level1 CE loss (0.0 disables).",
    )
    parser.add_argument(
        "--checkpoint_exists",
        type=_cli_bool,
        default=False,
        metavar="{true,false}",
        help="false (default): train and save to best_mlp_gpu.pt. true: skip training and load that file only.",
    )
    ## Plot/eval behavior.
    parser.add_argument("--confusion_matrix_counts", action="store_true", 
        help="Default is row-normalized. Set to True to plot confusion matrices with raw counts.",)
    parser.add_argument("--hierarchical_confusion_matrix", action="store_true")
    parser.add_argument("--run_all_data_eval", action="store_true")
    parser.add_argument("--parent_value_stardist", type=str, default="tumor1")
    parser.add_argument("--run_stardist_eval", action="store_true",
        help="Run StarDist prediction/evaluation pipeline (model from project_all by default).",)
    parser.add_argument("--stardist_model_therapy_model", type=str, default=None, 
        help="Therapy model used to load model_star (default: {therapy_data}_project_all).",)
    parser.add_argument("--stardist_data_therapy_model", type=str, default=None, 
        help="Therapy model used for StarDist embeddings/matched_features (default: {therapy_data}_project_{parent_value}).",)
    parser.add_argument("--stardist_cell_coords_path", type=str, default=None)
    parser.add_argument("--stardist_embedding_dir", type=str, default=None)
    parser.add_argument("--stardist_matched_features_path", type=str, default=None)
    parser.add_argument("--stardist_save_path_level2_pred", type=str, default=None)
    parser.add_argument("--stardist_save_path_level2_true", type=str, default=None)
    parser.add_argument("--stardist_save_path_level1_pred", type=str, default=None)
    parser.add_argument("--stardist_save_path_level1_true", type=str, default=None)
    parser.add_argument("--save_path_level2_pred", type=str, default=None)
    parser.add_argument("--save_path_level2_true", type=str, default=None)
    parser.add_argument("--save_path_level1_pred", type=str, default=None)
    parser.add_argument("--save_path_level1_true", type=str, default=None)
    parser.add_argument("--prediction_only", action="store_true",
        help="If set with --run_all_data_eval, skip all-data metric eval and only predict/plot.",)
    parser.add_argument("--spatial_fig_w",
        type=float, default=12.0, help="Spatial plot figure width (inches) for all-data evaluation maps.",)
    parser.add_argument("--spatial_fig_h", type=float, default=10.0, 
        help="Spatial plot figure height (inches) for all-data evaluation maps.",)
    parser.add_argument("--spatial_point_size", type=float, default=0.6, 
        help="Spatial plot point size (scatter marker size) for all-data evaluation maps.",)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

## model_train_validate_cv.py — LOGO CV (default) or legacy random_split
## conda activate SeededNTM
## cd /home/lingyu/ssd2/Python/Hist2Pheno
##
## Defaults: cv_mode=logo, matched_features_{parent}_logo.npz, group column TumorID
## Fold weights: hce_logo_fold_*.pt 默认写在本次 run 目录下：
##   data/<therapy>/<therapy_model>/<save_result>/<run_prefix><时间戳或--run_name>/
## （与 run.log、混淆矩阵等同目录）。续跑同一目录请用相同 --run_name 或 --logo_checkpoint_dir。
## --logo_resume: true (default) = 若该 fold 的 .pt 已存在则只 load；缺文件的 fold 仍会训练。
##   false = 每一折重训。auto = 目录里已有任意 hce_logo_fold_*.pt 则按 true 处理。
## --logo_checkpoint_dir 可显式指定 fold .pt 目录（覆盖默认 result 子文件夹）。勿与 --checkpoint_exists 同用（LOGO）。
## TumorID 只有 1 类时会用空间分箱代替 LOGO 组：默认 --logo_spatial_nx 12 --logo_spatial_ny 12 → 12×12=144 折。
## 若要 2×2 共 4 折：加 --logo_spatial_nx 2 --logo_spatial_ny 2（与 notebook 里 LOGO_SPATIAL_NX/Y 一致）。

## LOGO — NCRT tumor1（--logo_resume true 与默认 true 行为相同）
# 是否还会出现 Train/Epoch：取决于本次 result 目录下是否已有 hce_logo_fold_0.pt …（与折数一致）。
# 已全部存在 → 每折只 load，无训练；部分存在 → 仅有文件的折不训，缺的折仍会训；全无 → 整次 LOGO 都会训。
# 不设 --python_root / --project_root 时用 argparse 默认值（与早期示例一致）；需改路径时再显式传入。
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --cv_mode logo \
#   --logo_spatial_nx 2 \
#   --logo_spatial_ny 2 \
#   --logo_resume true \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1

## LOGO — 强制每一折重新训练（会覆盖/重写 hce_logo_fold_*.pt）
# python code/model_train_validate_cv.py \
#   --project_root /path/to/esccAI \
#   --parent_value tumor1 \
#   --cv_mode logo \
#   --logo_resume false

## LOGO — NCRT all ROIs (~28 tumors; matched_features_all_logo.npz, ImgEmbeddings_all, CellPixCoords_all.csv)
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value all \
#   --cuda_device 2 \
#   --cv_mode logo \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1 \
#   --spatial_fig_w 64 \
#   --spatial_fig_h 48 \
#   --spatial_point_size 0.5

## Legacy stratified train/val (70/30), matched_features_{parent}.npz — same script, different mode
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cv_mode random_split \
#   --checkpoint_exists false \
#   --hidden_dims 256 256 256 \
#   --save_result result \
#   --run_all_data_eval


# 示例（k=5，按 level2 分层）：
# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cv_mode stratified_kfold \
#   --cv_k 5 \
#   --cv_stratify_target level2 \
#   --logo_resume true

# 如果你要更强的层级比例约束（L1+L2同时尽量均衡）：
# --cv_stratify_target joint


# python code/model_train_validate_cv.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --logo_resume true \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1



####################################################################
# 2026.04.15 UNI-stratified_5fold CV-
####################################################################
## tumor1
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --logo_resume false \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1

## using level2 label to split data into 5 folds
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --cv_stratify_target level2 \
#   --logo_resume false \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1

## ALL
# 1 run tumor1 prediction
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value all \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --logo_resume false \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1 

## 2 run all prediction - --logo_resume true (use existing fold .pt)
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value all \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --logo_resume true \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist all \
#   --spatial_fig_w 64 \
#   --spatial_fig_h 48 \
#   --spatial_point_size 0.5


####################################################################
# 2026.04.16 UNI-stratified_5fold CV-balance_sampler & l2_macro_priority
####################################################################
## tumor1
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --train_balance_sampler \
#   --train_balance_sampler_mode hierarchical_l1_l2 \
#   --cv_selection_metric l2_macro_priority \
#   --val_selection_metric l2_macro_priority \
#   --class_weight_mode class_balanced \
#   --class_balanced_beta 0.9999 \
#   --l2_focal_gamma 1.5 \
#   --l2_label_smoothing 0.05 \
#   --l1_label_smoothing 0.02 \
#   --logo_resume false \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1

## tumor1
## 验证长尾改进
# python code/model_train_validate_cv_method.py \
#   --method UNI \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --cuda_device 2 \
#   --hce_w2 2.0 \
#   --cv_mode stratified_kfold \
#   --cv_stratify_target level2 \
#   --train_balance_sampler \
#   --train_balance_sampler_mode hierarchical_l1_l2 \
#   --cv_selection_metric l2_macro_priority \
#   --val_selection_metric l2_macro_priority \
#   --class_weight_mode class_balanced \
#   --class_balanced_beta 0.9999 \
#   --l2_focal_gamma 1.5 \
#   --l2_label_smoothing 0.05 \
#   --l1_label_smoothing 0.02 \
#   --logo_resume false \
#   --save_result result \
#   --run_all_data_eval \
#   --run_stardist_eval \
#   --parent_value_stardist tumor1