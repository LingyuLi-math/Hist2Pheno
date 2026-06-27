######################################################
# 2026.03.23 For model training and validation: LLY Adjust model to use level1 and level2
# 2026.06.26 For each dataset, add spatial context for model training and validation
######################################################
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from base import (
    DualHeadImprovedMLPClassifier,
    FiveHeadImprovedMLPClassifier,
    ImprovedMLPClassifier,
    build_spatial_neighbor_index,
    build_spatial_neighbor_index_by_group,
    evaluate,
    forward_heads_l2_l1,
    gather_neighbor_embeddings,
    result_has_extra_head_labels,
)


def _checkpoint_is_spatial(state_dict):
    return "spatial_fusion.proj.weight" in state_dict


def _spatial_ctx_from_result(result):
    if not result.get("use_spatial_context"):
        return None
    return {
        "X_all": torch.as_tensor(result["X_all_scaled"], dtype=torch.float32),
        "neighbor_index": torch.as_tensor(result["spatial_neighbor_index"], dtype=torch.long),
    }


########################################################
# 2026.06.26 For each dataset, add spatial context for model training and validation
########################################################
def _build_spatial_neighbor_index_for_cv_data(cv_data, k_neighbors=8):
    """kNN over ``X_coords_f``; per-sample groups when ``groups_f`` is present."""
    if "X_coords_f" not in cv_data:
        raise ValueError(
            "Spatial context requires X_coords_f in cv_data "
            "(pass X_coords=... to prepare_data_leave_one_group_out)."
        )
    coords = cv_data["X_coords_f"]
    if "groups_f" in cv_data:
        return build_spatial_neighbor_index_by_group(
            coords, cv_data["groups_f"], k_neighbors=k_neighbors
        )
    return build_spatial_neighbor_index(coords, k_neighbors=k_neighbors)
########################################################

def _unpack_spatial_batch(batch, use_extra_heads):
    """Return (x, y_l2, y_l1, extras..., global_idx|None)."""
    global_idx = None
    if use_extra_heads:
        if len(batch) == 7:
            x, y_l2, y_l1, y_l12, y_l3, y_l4, global_idx = batch
            return x, y_l2, y_l1, y_l12, y_l3, y_l4, global_idx
        x, y_l2, y_l1, y_l12, y_l3, y_l4 = batch
        return x, y_l2, y_l1, y_l12, y_l3, y_l4, None
    if len(batch) == 4:
        x, y_l2, y_l1, global_idx = batch
        return x, y_l2, y_l1, None, None, None, global_idx
    x, y_l2, y_l1 = batch
    return x, y_l2, y_l1, None, None, None, None


def _neighbor_x_from_batch(spatial_ctx, global_idx, device):
    if spatial_ctx is None or global_idx is None:
        return None
    X_all = spatial_ctx["X_all"].to(device, non_blocking=True)
    neighbor_index = spatial_ctx["neighbor_index"].to(device, non_blocking=True)
    global_idx = global_idx.to(device, non_blocking=True)
    return gather_neighbor_embeddings(X_all, neighbor_index, global_idx, device=device)


def _redirect_stdlib_tempdir():
    """
    PyTorch AdamW can lazy-import torch._dynamo -> distributed, which uses
    tempfile under the default dir (often /tmp). If /tmp hits disk quota, set
    NCRT_TMPDIR or use ~/ssd2/tmp and force the stdlib tempfile module to use it.
    """
    import tempfile

    tdir = os.environ.get("NCRT_TMPDIR", os.path.join(os.path.expanduser("~"), "ssd2", "tmp"))
    try:
        os.makedirs(tdir, exist_ok=True)
    except OSError:
        return
    os.environ["TMPDIR"] = tdir
    os.environ["TEMP"] = tdir
    os.environ["TMP"] = tdir
    tempfile.tempdir = tdir


## Helper utilities shared by training/validation modes.
def infer_input_dim_and_num_classes(
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    class_names=None,
):
    """Infer model input dimension and class count."""
    input_dim = X_train_scaled.shape[1]
    if class_names is not None:
        num_classes = len(class_names)
    else:
        # Use full label space to avoid out-of-bounds targets.
        num_classes = int(np.max(np.concatenate([y_train_encoded, y_test_encoded]))) + 1
    return input_dim, num_classes


def build_mlp_classifier(
    input_dim,
    num_classes,
    device,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    num_level1_classes=None,
    use_dual_head=True,
    use_five_head=False,
    num_level12_classes=None,
    num_level3_classes=None,
    num_level4_classes=None,
    use_spatial_context=False,
    spatial_mode="mean",
    spatial_k=8,
):
    """
    Create the MLP on ``device``.

    Training uses ``UnifiedHCELoss`` / ``UnifiedFiveHeadLoss``:

    - ``use_five_head=True``: five heads (L2, L1, L12, CNiche, TNiche) for Xenium.
    - ``use_dual_head=True`` (default when not five-head): L2 + L1 (NCRT / esccAI).
    - ``use_dual_head=False``: legacy single-head checkpoints only.
    - ``use_spatial_context=True``: fuse kNN neighbor embeddings before the MLP backbone.
      Default ``False`` preserves the original per-cell model and checkpoints.
    """
    spatial_kw = dict(
        use_spatial_context=use_spatial_context,
        spatial_mode=spatial_mode,
        spatial_k=spatial_k,
    )
    if use_five_head:
        if num_level1_classes is None or num_level12_classes is None:
            raise ValueError(
                "num_level1_classes and num_level12_classes are required when use_five_head=True"
            )
        if num_level3_classes is None or num_level4_classes is None:
            raise ValueError(
                "num_level3_classes and num_level4_classes are required when use_five_head=True"
            )
        return FiveHeadImprovedMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_level1_classes=num_level1_classes,
            num_level12_classes=num_level12_classes,
            num_level3_classes=num_level3_classes,
            num_level4_classes=num_level4_classes,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            **spatial_kw,
        ).to(device)
    if use_dual_head:
        if num_level1_classes is None:
            raise ValueError("num_level1_classes is required when use_dual_head=True")
        return DualHeadImprovedMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            num_level1_classes=num_level1_classes,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            **spatial_kw,
        ).to(device)
    return ImprovedMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=list(hidden_dims),
        dropout=dropout,
        **spatial_kw,
    ).to(device)


def _append_logo_fold_extra_tiers(result_fold, cv_data, tr_idx, va_idx):
    """Copy optional level12 / level3 / level4 encodings into a per-fold result dict."""
    for short in ("level12", "level3", "level4"):
        key_full = f"y_{short}_encoded_f"
        if key_full not in cv_data:
            continue
        enc = cv_data[key_full]
        result_fold[f"y_train_{short}_encoded"] = enc[tr_idx]
        result_fold[f"y_test_{short}_encoded"] = enc[va_idx]
        result_fold[key_full] = enc
    return result_fold


def _build_l2_to_l1_map(num_classes, y_enc_full, y_l1_full, device):
    """Return (child_to_parent, map_l2_to_l1) for hierarchical L1 aggregation from L2 softmax."""
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_enc_full, y_l1_full):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
        elif child_to_parent[int(l2)] != int(l1):
            raise ValueError(f"Inconsistent hierarchy at L2={l2}")
    num_l1 = int(np.max(y_l1_full)) + 1
    map_l2_to_l1 = torch.zeros(num_classes, num_l1, dtype=torch.float32, device=device)
    for k, p in enumerate(child_to_parent):
        if p >= 0:
            map_l2_to_l1[k, int(p)] = 1.0
    return map_l2_to_l1


def _macro_auc_ovr_evaluable(y_true, y_score, n_classes):
    """Multiclass macro AUROC: mean of evaluable one-vs-rest class AUCs (softmax scores)."""
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


def _five_tier_auc_sum_from_head_aucs(metrics, *, use_extra_heads=None):
    """Sum macro AUROC from each prediction head (excludes L2→L1 aggregated AUC)."""
    if use_extra_heads is False:
        keys = ("l2", "l1_head")
    else:
        keys = ("l2", "l1_head", "l12", "l3", "l4")
    aucs = [float(metrics.get(f"{k}_macro_auc", float("nan"))) for k in keys]
    finite = [a for a in aucs if np.isfinite(a)]
    return float(sum(finite)) if finite else float("nan")


def _validation_selection_score(metrics, metric, *, use_extra_heads=False):
    """
    Validation / CV fold selection score from a metrics dict.

    ``five_tier_auc_sum``: sum of head-level macro AUROC (L2, L1 head, L12, L3, L4).
    Does **not** include L1 derived from L2 aggregation.
    """
    if metric == "four_term_sum":
        return (
            float(metrics["l2_macro_f1"])
            + float(metrics["l2_weighted_f1"])
            + float(metrics["l1_macro_f1"])
            + float(metrics["l1_weighted_f1"])
        )
    if metric == "l2_macro_priority":
        return (
            2.0 * float(metrics["l2_macro_f1"])
            + float(metrics["l2_weighted_f1"])
            + float(metrics["l1_macro_f1"])
            + float(metrics["l1_weighted_f1"])
        )
    if metric == "five_tier_auc_sum":
        if "five_tier_auc_sum" in metrics and np.isfinite(metrics["five_tier_auc_sum"]):
            return float(metrics["five_tier_auc_sum"])
        return _five_tier_auc_sum_from_head_aucs(metrics, use_extra_heads=use_extra_heads)
    raise ValueError(
        "val_selection_metric / cv_selection_metric must be one of: "
        "four_term_sum, l2_macro_priority, five_tier_auc_sum"
    )


def _append_prob_bucket(buckets, tier, y_true, logits, device):
    buckets.setdefault(tier, ([], []))
    buckets[tier][0].append(np.asarray(y_true, dtype=np.int64))
    buckets[tier][1].append(torch.softmax(logits.float(), dim=1).cpu().numpy())


def _eval_head_macro_aucs_on_loader(
    model,
    loader,
    device,
    spatial_ctx=None,
    *,
    num_classes,
    num_level1_classes,
    num_level12_classes=None,
    num_level3_classes=None,
    num_level4_classes=None,
):
    """Macro AUROC per prediction head on a validation loader (softmax scores)."""
    five_head = hasattr(model, "level12_head")
    has_l1_head = hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))
    prob_buckets = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if five_head:
                if spatial_ctx is not None and len(batch) == 7:
                    x, _, _, yt_l12, yt_l3, yt_l4, global_idx = batch
                    yt_l2 = batch[1].cpu().numpy()
                    yt_l1 = batch[2].cpu().numpy()
                    yt_l12 = yt_l12.cpu().numpy()
                    yt_l3 = yt_l3.cpu().numpy()
                    yt_l4 = yt_l4.cpu().numpy()
                else:
                    x = batch[0]
                    yt_l2 = batch[1].cpu().numpy()
                    yt_l1 = batch[2].cpu().numpy()
                    yt_l12 = batch[3].cpu().numpy()
                    yt_l3 = batch[4].cpu().numpy()
                    yt_l4 = batch[5].cpu().numpy()
                    global_idx = None
            elif spatial_ctx is not None and len(batch) == 4:
                x, _, _, global_idx = batch
                yt_l2 = batch[1].cpu().numpy()
                yt_l1 = batch[2].cpu().numpy()
            else:
                x = batch[0]
                yt_l2 = batch[1].cpu().numpy()
                yt_l1 = batch[2].cpu().numpy()
                global_idx = None

            x = x.to(device, non_blocking=True)
            neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)

            if five_head:
                logits_l2, logits_l1, logits_l12, logits_l3, logits_l4 = model.forward_heads(
                    x, neighbor_x=neighbor_x
                )
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)
                _append_prob_bucket(prob_buckets, "l1_head", yt_l1, logits_l1, device)
                _append_prob_bucket(prob_buckets, "l12", yt_l12, logits_l12, device)
                _append_prob_bucket(prob_buckets, "l3", yt_l3, logits_l3, device)
                _append_prob_bucket(prob_buckets, "l4", yt_l4, logits_l4, device)
            elif has_l1_head:
                logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)
                _append_prob_bucket(prob_buckets, "l1_head", yt_l1, logits_l1, device)
            else:
                logits_l2 = model(x, neighbor_x=neighbor_x)
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)

    tier_nc = {"l2": num_classes, "l1_head": num_level1_classes}
    if five_head:
        tier_nc.update(
            {
                "l12": num_level12_classes,
                "l3": num_level3_classes,
                "l4": num_level4_classes,
            }
        )

    out = {}
    for tier, n_cls in tier_nc.items():
        if tier not in prob_buckets or n_cls is None:
            continue
        y_true = np.concatenate(prob_buckets[tier][0])
        probs = np.concatenate(prob_buckets[tier][1], axis=0)
        out[f"{tier}_macro_auc"] = _macro_auc_ovr_evaluable(y_true, probs, int(n_cls))

    out["five_tier_auc_sum"] = _five_tier_auc_sum_from_head_aucs(
        out, use_extra_heads=five_head
    )
    return out


def _eval_fold_tier_metrics(
    model, val_loader, device, num_classes, y_enc_full, y_l1_full, spatial_ctx=None, *, return_oof=False
):
    """
    Validation metrics per label tier on one CV fold (best checkpoint already loaded).

    Returns flat keys: ``l2_accuracy``, ``l2_macro_f1``, ``l1_macro_f1`` (L2→L1 agg),
    ``l1_head_*`` when a L1 head exists, ``l12_*`` / ``l3_*`` / ``l4_*`` for five-head,
    plus ``{tier}_macro_auc`` and ``five_tier_auc_sum`` (head-level AUROC only).
    """
    from sklearn.metrics import accuracy_score, f1_score

    map_l2_to_l1 = _build_l2_to_l1_map(num_classes, y_enc_full, y_l1_full, device)
    five_head = hasattr(model, "level12_head")
    has_l1_head = hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))

    buckets = {
        "l2": ([], []),
        "l1": ([], []),
        "l1_head": ([], []),
        "l12": ([], []),
        "l3": ([], []),
        "l4": ([], []),
    }
    prob_buckets = {}
    num_level1_classes = int(np.max(y_l1_full)) + 1
    num_level12_classes = num_level3_classes = num_level4_classes = None

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if five_head:
                if spatial_ctx is not None and len(batch) == 7:
                    x, _, _, yt_l12, yt_l3, yt_l4, global_idx = batch
                    yt_l2 = batch[1].cpu().numpy()
                    yt_l1 = batch[2].cpu().numpy()
                    yt_l12 = yt_l12.cpu().numpy()
                    yt_l3 = yt_l3.cpu().numpy()
                    yt_l4 = yt_l4.cpu().numpy()
                else:
                    x = batch[0]
                    yt_l2 = batch[1].cpu().numpy()
                    yt_l1 = batch[2].cpu().numpy()
                    yt_l12 = batch[3].cpu().numpy()
                    yt_l3 = batch[4].cpu().numpy()
                    yt_l4 = batch[5].cpu().numpy()
                    global_idx = None
            elif spatial_ctx is not None and len(batch) == 4:
                x, _, _, global_idx = batch
                yt_l2 = batch[1].cpu().numpy()
                yt_l1 = batch[2].cpu().numpy()
            else:
                x = batch[0]
                yt_l2 = batch[1].cpu().numpy()
                yt_l1 = batch[2].cpu().numpy()
                global_idx = None

            x = x.to(device, non_blocking=True)
            neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)
            buckets["l2"][0].append(yt_l2)
            buckets["l1"][0].append(yt_l1)

            if five_head:
                logits_l2, logits_l1, logits_l12, logits_l3, logits_l4 = model.forward_heads(
                    x, neighbor_x=neighbor_x
                )
                logits_l2 = logits_l2.float()
                logits_l1 = logits_l1.float()
                if num_level12_classes is None:
                    num_level12_classes = int(logits_l12.shape[1])
                    num_level3_classes = int(logits_l3.shape[1])
                    num_level4_classes = int(logits_l4.shape[1])
                buckets["l2"][1].append(torch.argmax(logits_l2, dim=1).cpu().numpy())
                buckets["l1_head"][0].append(yt_l1)
                buckets["l1_head"][1].append(torch.argmax(logits_l1, dim=1).cpu().numpy())
                buckets["l12"][0].append(yt_l12)
                buckets["l12"][1].append(torch.argmax(logits_l12, dim=1).cpu().numpy())
                buckets["l3"][0].append(yt_l3)
                buckets["l3"][1].append(torch.argmax(logits_l3, dim=1).cpu().numpy())
                buckets["l4"][0].append(yt_l4)
                buckets["l4"][1].append(torch.argmax(logits_l4, dim=1).cpu().numpy())
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)
                _append_prob_bucket(prob_buckets, "l1_head", yt_l1, logits_l1, device)
                _append_prob_bucket(prob_buckets, "l12", yt_l12, logits_l12, device)
                _append_prob_bucket(prob_buckets, "l3", yt_l3, logits_l3, device)
                _append_prob_bucket(prob_buckets, "l4", yt_l4, logits_l4, device)
            elif has_l1_head:
                logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                logits_l2 = logits_l2.float()
                logits_l1 = logits_l1.float()
                buckets["l2"][1].append(torch.argmax(logits_l2, dim=1).cpu().numpy())
                buckets["l1_head"][0].append(yt_l1)
                buckets["l1_head"][1].append(torch.argmax(logits_l1, dim=1).cpu().numpy())
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)
                _append_prob_bucket(prob_buckets, "l1_head", yt_l1, logits_l1, device)
            else:
                logits_l2 = model(x, neighbor_x=neighbor_x).float()
                buckets["l2"][1].append(torch.argmax(logits_l2, dim=1).cpu().numpy())
                _append_prob_bucket(prob_buckets, "l2", yt_l2, logits_l2, device)

            pr_l2 = torch.softmax(logits_l2, dim=1)
            pr_l1 = torch.matmul(pr_l2, map_l2_to_l1)
            buckets["l1"][1].append(torch.argmax(pr_l1, dim=1).cpu().numpy())

    out = {}
    for tier, (yl, yp) in buckets.items():
        if not yl or not yp:
            continue
        y_true = np.concatenate(yl)
        y_pred = np.concatenate(yp)
        out[f"{tier}_accuracy"] = float(accuracy_score(y_true, y_pred))
        out[f"{tier}_macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        out[f"{tier}_weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

    tier_nc = {"l2": num_classes, "l1_head": num_level1_classes}
    if five_head:
        tier_nc.update(
            {
                "l12": num_level12_classes,
                "l3": num_level3_classes,
                "l4": num_level4_classes,
            }
        )
    for tier, n_cls in tier_nc.items():
        if tier not in prob_buckets or n_cls is None:
            continue
        y_true = np.concatenate(prob_buckets[tier][0])
        probs = np.concatenate(prob_buckets[tier][1], axis=0)
        out[f"{tier}_macro_auc"] = _macro_auc_ovr_evaluable(y_true, probs, int(n_cls))
    out["five_tier_auc_sum"] = _five_tier_auc_sum_from_head_aucs(
        out, use_extra_heads=five_head
    )
    if return_oof and "l2" in prob_buckets and buckets["l2"][0]:
        out["_oof_labels_l2"] = np.concatenate(buckets["l2"][0]).astype(np.int64)
        out["_oof_preds_l2"] = np.concatenate(buckets["l2"][1]).astype(np.int64)
        out["_oof_probs_l2"] = np.concatenate(prob_buckets["l2"][1], axis=0).astype(np.float64)
    return out


def _split_oof_from_tier_metrics(tier_m):
    """Pop optional OOF arrays stored by ``_eval_fold_tier_metrics(..., return_oof=True)``."""
    labels = tier_m.pop("_oof_labels_l2", None)
    preds = tier_m.pop("_oof_preds_l2", None)
    probs = tier_m.pop("_oof_probs_l2", None)
    return labels, preds, probs, tier_m


def _append_tier_means_to_cv_summary(summary, fold_rows):
    """Add ``{tier}_{metric}_mean/std`` keys present in ``fold_rows``."""
    if not fold_rows:
        return summary

    def _mean_std(key):
        vals = np.array([r[key] for r in fold_rows if key in r], dtype=np.float64)
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    sample = fold_rows[0]
    for k in sorted(sample.keys()):
        if k in ("fold", "held_group", "n_train", "n_val", "checkpoint", "best_epoch"):
            continue
        if not (
            k.endswith("_accuracy")
            or k.endswith("_macro_f1")
            or k.endswith("_weighted_f1")
            or k.endswith("_macro_auc")
            or k == "five_tier_auc_sum"
        ):
            continue
        mean_v, std_v = _mean_std(k)
        summary[f"{k}_mean"] = mean_v
        summary[f"{k}_std"] = std_v
    return summary


def _load_fold_model_for_eval(
    save_path,
    device,
    result_fold,
    class_names,
    hidden_dims,
    dropout,
):
    """Reload fold checkpoint and return ``(model, num_classes)``."""
    input_dim, num_classes = infer_input_dim_and_num_classes(
        result_fold["X_train_scaled"],
        result_fold["y_train_encoded"],
        result_fold["y_test_encoded"],
        class_names=class_names,
    )
    num_l1 = int(np.max(result_fold["y_level1_encoded_f"])) + 1
    sd_fold = torch.load(save_path, map_location=device)
    model, _ = build_mlp_classifier_for_state_dict(
        sd_fold,
        input_dim=input_dim,
        num_classes=num_classes,
        num_level1_classes=num_l1,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(_remap_legacy_five_head_state_dict(sd_fold))
    return model, num_classes


def _checkpoint_is_five_head(state_dict):
    """True if ``state_dict`` was saved from ``FiveHeadImprovedMLPClassifier``."""
    return (
        "level12_head.weight" in state_dict
        or "level3_head.weight" in state_dict
        or "level0_head.weight" in state_dict
    )


def _remap_legacy_five_head_state_dict(state_dict):
    """Map ``level0_head`` / ``level01_head`` checkpoint keys to ``level3_head`` / ``level4_head``."""
    if "level3_head.weight" in state_dict:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        out[k.replace("level0_head", "level3_head").replace("level01_head", "level4_head")] = v
    return out


def _five_head_class_counts_from_state_dict(state_dict):
    """Return (num_l12, num_l3, num_l4) from a five-head checkpoint."""
    sd = _remap_legacy_five_head_state_dict(state_dict)
    return (
        int(sd["level12_head.weight"].shape[0]),
        int(sd["level3_head.weight"].shape[0]),
        int(sd["level4_head.weight"].shape[0]),
    )


def build_mlp_classifier_for_state_dict(
    state_dict,
    *,
    input_dim,
    num_classes,
    num_level1_classes,
    device,
    hidden_dims,
    dropout=0.2,
):
    """Instantiate Dual- or Five-head MLP to match an on-disk ``state_dict``."""
    use_five = _checkpoint_is_five_head(state_dict)
    use_spatial = _checkpoint_is_spatial(state_dict)
    kw = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        hidden_dims=list(hidden_dims),
        dropout=dropout,
        num_level1_classes=num_level1_classes,
        use_dual_head=not use_five,
        use_five_head=use_five,
        use_spatial_context=use_spatial,
    )
    if use_five:
        n12, n3, n4 = _five_head_class_counts_from_state_dict(state_dict)
        kw.update(
            num_level12_classes=n12,
            num_level3_classes=n3,
            num_level4_classes=n4,
        )
    return build_mlp_classifier(**kw), use_five


def get_best_checkpoint_path(path, therapy_data, therapy_model=None):
    """Return the checkpoint path saved during training."""
    if therapy_model is None:
        # Default naming used across notebooks.
        therapy_model = f"{therapy_data}_project_tumor1"
    return (
        f"{path}Collaborate/esccAI/data/{therapy_data}/"
        f"{therapy_model}/best_mlp_gpu.pt"
    )


def get_select4_best_checkpoint_path(path, therapy_data, therapy_model=None):
    """Return ``best_mlp_gpu.pt`` under Xenium Complete_Cases_Select4 (same tree as kfold ckpts)."""
    if therapy_model is None:
        therapy_model = f"{therapy_data}_project_tumor1"
    return (
        f"{path}Collaborate/esccAI/data/Xemiun/weiqin/SpatialPF-NGenetics/"
        f"Spatial-PF-Processed/Data/Complete_Cases_Select4/"
        f"{therapy_data}/{therapy_model}/best_mlp_gpu.pt"
    )


def get_run_dir_checkpoint_path(path, therapy_data, therapy_model=None, run_log_path=None):
    """Return ``best_mlp_gpu.pt`` under a run timestamp directory (same dir as ``run.log``)."""
    if therapy_model is None:
        therapy_model = f"{therapy_data}_project_tumor1"

    if run_log_path is not None:
        run_log = Path(run_log_path).expanduser().resolve()
        return str(run_log.parent / "best_mlp_gpu.pt")

    therapy_root = (
        Path(path).expanduser().resolve()
        / "Collaborate"
        / "esccAI"
        / "data"
        / therapy_data
        / therapy_model
    )
    run_logs = sorted(
        therapy_root.glob("**/run.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_logs:
        raise FileNotFoundError(
            "No run.log found under "
            f"{therapy_root}. Pass run_log_path explicitly or use parent_dir=True."
        )
    return str(run_logs[0].parent / "best_mlp_gpu.pt")


def sync_best_mlp_from_logo_fold(
    path,
    therapy_data,
    therapy_model,
    fold_checkpoint_path,
    dest_path=None,
):
    """
    Copy a LOGO/HCE fold checkpoint (``hce_logo_fold_*.pt`` / ``hce_kfold_fold_*.pt``)
    to ``best_mlp_gpu.pt``.

    Default destination: ``.../data/{therapy_data}/{therapy_model}/best_mlp_gpu.pt``.
    For Xenium Select4 notebooks, pass ``dest_path=get_select4_best_checkpoint_path(...)``
    so the copy lands next to ``ablation_kfold_ckpts`` (same tree as fold writes).

    Use after ablation or ``run_logo_cv_with_insample_report`` when
    ``LP['best_fold']['checkpoint']`` points to the weights you want for
    ``load_model_for_predict`` without passing ``checkpoint_path``.

    Returns:
        str: Absolute path of the written ``best_mlp_gpu.pt``.
    """
    import shutil

    src = os.path.normpath(os.path.abspath(os.path.expanduser(fold_checkpoint_path)))
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Fold checkpoint not found: {src}")

    dest = dest_path or get_best_checkpoint_path(path, therapy_data, therapy_model)
    dest = os.path.normpath(os.path.abspath(os.path.expanduser(dest)))
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"sync_best_mlp_from_logo_fold: wrote {dest!r} ← {src!r}")
    return dest


## 训练中途变成 nan
def train_and_save_model(
    device,
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    y_train_level1_encoded,
    y_encoded_f,
    y_level1_encoded_f,
    train_loader,
    val_loader,
    evaluate,
    save_bestmodel_path,
    class_names=None,
    val_loader_eval=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    use_extra_heads=False,
    y_level12_encoded_f=None,
    y_level3_encoded_f=None,
    y_level4_encoded_f=None,
    patience=10,
    max_epochs=50,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    use_spatial_context=False,
    spatial_mode="mean",
    spatial_k=8,
    spatial_neighbor_index=None,
    X_all_scaled=None,
):
    """
    Train dual-head or five-head MLP with unified HCE loss.

    **Dual-head (NCRT / esccAI, default):**

    ``total = w1*CE(L1) + w2*CE(L2) + w12*NLL(L1_from_L2_agg)``

    **Five-head (Xenium, ``use_extra_heads=True``):** adds direct CE on L12, CNiche (L0), TNiche (L01):

    ``+ w_l12head*CE(L12) + w_l3*CE(L3) + w_l4*CE(L4)`` (no L2→L12 or L12→L1 logit-chain losses).

    The best checkpoint maximizes validation score controlled by ``val_selection_metric``:
    - ``five_tier_auc_sum`` (recommended for five-head): sum of head-level macro AUROC
      (L2, L1 head, L12, L3, L4); excludes L2→L1 aggregated AUC.
    - ``four_term_sum``: ``l2_macro + l2_weighted + l1_macro + l1_weighted`` (L1 from L2 agg)
    - ``l2_macro_priority``: ``2*l2_macro + l2_weighted + l1_macro + l1_weighted``

    Returns (at that checkpoint) ``(l2_weighted_f1, l2_macro_f1, best_epoch)``.
    """
    import time
    import torch.nn as nn
    import torch.nn.functional as F

    class UnifiedHCELoss(nn.Module):
        """
        Three- or six-term hierarchical objective (weights are independent scalars).

        Core (always): CE(L1 head), CE(L2 head), NLL(L1 from L2 aggregation).

        Optional (``use_extra_heads``): direct CE on L12, L0 (CNiche), L01 (TNiche) heads only.
        """

        def __init__(
            self,
            child_to_parent,
            num_level1_classes,
            class_weights_level2=None,
            class_weights_level1=None,
            w1=1.0,
            w2=1.0,
            w12=1.0,
            w_l12head=1.0,
            w_l3=1.0,
            w_l4=1.0,
            use_extra_heads=False,
            l2_label_smoothing=0.0,
            l1_label_smoothing=0.0,
            l2_focal_gamma=0.0,
        ):
            super().__init__()
            self.w1 = float(w1)
            self.w2 = float(w2)
            self.w12 = float(w12)
            self.w_l12head = float(w_l12head)
            self.w_l3 = float(w_l3)
            self.w_l4 = float(w_l4)
            self.use_extra_heads = bool(use_extra_heads)
            self.l2_label_smoothing = float(l2_label_smoothing)
            self.l1_label_smoothing = float(l1_label_smoothing)
            self.l2_focal_gamma = float(l2_focal_gamma)
            if class_weights_level2 is not None:
                self.register_buffer("class_weights_level2", class_weights_level2)
            else:
                self.class_weights_level2 = None
            if class_weights_level1 is not None:
                self.register_buffer("class_weights_level1", class_weights_level1)
            else:
                self.class_weights_level1 = None

            map_l2_to_l1 = torch.zeros(len(child_to_parent), num_level1_classes, dtype=torch.float32)
            for k, p in enumerate(child_to_parent):
                map_l2_to_l1[k, int(p)] = 1.0
            self.register_buffer("map_l2_to_l1", map_l2_to_l1)

        def forward(
            self,
            logits_l2,
            logits_l1_head,
            target_l2,
            target_l1,
            logits_l12=None,
            logits_l3=None,
            logits_l4=None,
            target_l12=None,
            target_l3=None,
            target_l4=None,
        ):
            logits_fp32 = torch.nan_to_num(logits_l2.float(), nan=0.0, posinf=30.0, neginf=-30.0)

            ce_l2_raw = F.cross_entropy(
                logits_fp32,
                target_l2,
                weight=self.class_weights_level2,
                label_smoothing=self.l2_label_smoothing,
                reduction="none",
            )
            if self.l2_focal_gamma > 0.0:
                probs_l2_detached = torch.softmax(logits_fp32, dim=1)
                pt = probs_l2_detached.gather(1, target_l2.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
                focal_factor = torch.pow(1.0 - pt, self.l2_focal_gamma)
                ce_l2 = (focal_factor * ce_l2_raw).mean()
            else:
                ce_l2 = ce_l2_raw.mean()
            ce_l1_std = F.cross_entropy(
                logits_l1_head.float(),
                target_l1,
                weight=self.class_weights_level1,
                label_smoothing=self.l1_label_smoothing,
            )

            probs_l2 = torch.softmax(logits_fp32, dim=1)
            probs_l1_agg = torch.matmul(probs_l2, self.map_l2_to_l1)
            log_probs_agg = torch.log(probs_l1_agg.clamp_min(1e-6))
            ce_l1_agg = F.nll_loss(log_probs_agg, target_l1)

            total_loss = self.w1 * ce_l1_std + self.w2 * ce_l2 + self.w12 * ce_l1_agg
            ce_l12 = ce_l3 = ce_l4 = None
            if self.use_extra_heads:
                ce_l12 = F.cross_entropy(logits_l12.float(), target_l12)
                ce_l3 = F.cross_entropy(logits_l3.float(), target_l3)
                ce_l4 = F.cross_entropy(logits_l4.float(), target_l4)
                total_loss = (
                    total_loss
                    + self.w_l12head * ce_l12
                    + self.w_l3 * ce_l3
                    + self.w_l4 * ce_l4
                )
            return (
                total_loss,
                ce_l2.detach(),
                ce_l1_std.detach(),
                ce_l1_agg.detach(),
                None if ce_l12 is None else ce_l12.detach(),
                None if ce_l3 is None else ce_l3.detach(),
                None if ce_l4 is None else ce_l4.detach(),
            )

    # Create the parent directory safely (dirname() can be empty).
    save_dir = os.path.dirname(save_bestmodel_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use AMP only on CUDA.
    amp_scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    # Infer input dimension and class count.
    input_dim, num_classes = infer_input_dim_and_num_classes(
        X_train_scaled,
        y_train_encoded,
        y_test_encoded,
        class_names=class_names,
    )
    # Use full filtered label space so val folds never see out-of-range L1 indices.
    num_level1_classes = int(np.max(y_level1_encoded_f)) + 1
    num_level12_classes = num_level3_classes = num_level4_classes = None
    if use_extra_heads:
        if y_level12_encoded_f is None or y_level3_encoded_f is None or y_level4_encoded_f is None:
            raise ValueError(
                "use_extra_heads=True requires y_level12_encoded_f, y_level3_encoded_f, "
                "y_level4_encoded_f in train_and_save_model"
            )
        num_level12_classes = int(np.max(y_level12_encoded_f)) + 1
        num_level3_classes = int(np.max(y_level3_encoded_f)) + 1
        num_level4_classes = int(np.max(y_level4_encoded_f)) + 1

    print("Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of level2 classes: {num_classes}")
    print(f"  Number of level1 classes: {num_level1_classes}")
    if use_extra_heads:
        print(f"  Number of level12 classes: {num_level12_classes}")
        print(f"  Number of level3 (CNiche) classes: {num_level3_classes}")
        print(f"  Number of level4 (TNiche) classes: {num_level4_classes}")
    if use_spatial_context:
        print(f"  Spatial context: enabled (k={spatial_k}, mode={spatial_mode!r})")
    else:
        print("  Spatial context: disabled (original per-cell MLP)")

    # Build model before creating optimizer/scheduler.
    model = build_mlp_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_level1_classes=num_level1_classes,
        use_dual_head=not use_extra_heads,
        use_five_head=use_extra_heads,
        num_level12_classes=num_level12_classes,
        num_level3_classes=num_level3_classes,
        num_level4_classes=num_level4_classes,
        use_spatial_context=use_spatial_context,
        spatial_mode=spatial_mode,
        spatial_k=spatial_k,
    )

    spatial_ctx = None
    if use_spatial_context:
        if spatial_neighbor_index is None or X_all_scaled is None:
            raise ValueError(
                "use_spatial_context=True requires spatial_neighbor_index and X_all_scaled"
            )
        spatial_ctx = {
            "X_all": torch.as_tensor(X_all_scaled, dtype=torch.float32),
            "neighbor_index": torch.as_tensor(spatial_neighbor_index, dtype=torch.long),
        }

    print(f"Model initialized on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    _redirect_stdlib_tempdir()

    # Optimizer + scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    def _class_balanced_weights(labels, n_classes, beta):
        labels = np.asarray(labels, dtype=np.int64)
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        weights = np.ones(n_classes, dtype=np.float64)
        nonzero = counts > 0
        effective_num = 1.0 - np.power(beta, counts[nonzero])
        weights_nonzero = (1.0 - beta) / np.clip(effective_num, 1e-12, None)
        weights[nonzero] = weights_nonzero
        # Normalize around 1.0 so loss scale stays comparable across runs.
        weights *= (np.sum(nonzero) / np.clip(weights[nonzero].sum(), 1e-12, None))
        return weights

    # Class weights for imbalanced level2 classes.
    unique_classes = np.unique(y_train_encoded)
    from sklearn.utils.class_weight import compute_class_weight

    if class_weight_mode == "balanced":
        cw = compute_class_weight("balanced", classes=unique_classes, y=y_train_encoded)
        full_weights = np.ones(num_classes, dtype=np.float64)
        full_weights[unique_classes.astype(int)] = cw
    elif class_weight_mode == "class_balanced":
        full_weights = _class_balanced_weights(
            y_train_encoded, n_classes=num_classes, beta=float(class_balanced_beta)
        )
    elif class_weight_mode == "none":
        full_weights = np.ones(num_classes, dtype=np.float64)
    else:
        raise ValueError(
            "class_weight_mode must be one of: none, balanced, class_balanced"
        )
    class_weights_tensor = torch.as_tensor(full_weights, dtype=torch.float32, device=device)

    unique_l1 = np.unique(y_train_level1_encoded)
    if class_weight_mode == "balanced":
        cw_l1 = compute_class_weight("balanced", classes=unique_l1, y=y_train_level1_encoded)
        full_weights_l1 = np.ones(num_level1_classes, dtype=np.float64)
        full_weights_l1[unique_l1.astype(int)] = cw_l1
    elif class_weight_mode == "class_balanced":
        full_weights_l1 = _class_balanced_weights(
            y_train_level1_encoded, n_classes=num_level1_classes, beta=float(class_balanced_beta)
        )
    elif class_weight_mode == "none":
        full_weights_l1 = np.ones(num_level1_classes, dtype=np.float64)
    else:
        raise ValueError(
            "class_weight_mode must be one of: none, balanced, class_balanced"
        )
    class_weights_l1_tensor = torch.as_tensor(full_weights_l1, dtype=torch.float32, device=device)

    # Build level2 -> level1 mapping from the filtered dataset.
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
        elif child_to_parent[int(l2)] != int(l1):
            raise ValueError(f"Inconsistent hierarchy: level2 class {l2} maps to multiple level1 classes")
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes: {missing}")

    criterion = UnifiedHCELoss(
        child_to_parent=child_to_parent,
        num_level1_classes=num_level1_classes,
        class_weights_level2=class_weights_tensor,
        class_weights_level1=class_weights_l1_tensor,
        w1=hce_w1,
        w2=hce_w2,
        w12=hce_w12,
        w_l12head=hce_w_l12head,
        w_l3=hce_w_l3,
        w_l4=hce_w_l4,
        use_extra_heads=use_extra_heads,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        l2_focal_gamma=l2_focal_gamma,
    ).to(device)

    loss_msg = (
        f"{hce_w1:.3f}*CE(L1) + {hce_w2:.3f}*CE(L2) + {hce_w12:.3f}*NLL(L1_from_L2_agg)"
    )
    if use_extra_heads:
        loss_msg += (
            f" + {hce_w_l12head:.3f}*CE(L12) + {hce_w_l3:.3f}*CE(CNiche) + {hce_w_l4:.3f}*CE(TNiche)"
        )
    print(f"Unified HCE: total_loss = {loss_msg}")
    print(
        "Loss options: "
        f"l2_label_smoothing={l2_label_smoothing:.4f}, "
        f"l1_label_smoothing={l1_label_smoothing:.4f}, "
        f"class_weight_mode={class_weight_mode}, "
        f"class_balanced_beta={class_balanced_beta:.6f}, "
        f"l2_focal_gamma={l2_focal_gamma:.4f}"
    )
    print(f"Validation selection metric: {val_selection_metric}")

    from sklearn.metrics import accuracy_score, f1_score

    def evaluate_level1_from_level2_logits(model, loader, device, child_to_parent_arr, num_l1):
        model.eval()
        all_preds_l1, all_labels_l1 = [], []

        map_l2_to_l1 = torch.zeros(len(child_to_parent_arr), num_l1, dtype=torch.float32, device=device)
        for k, p in enumerate(child_to_parent_arr):
            map_l2_to_l1[k, int(p)] = 1.0

        with torch.no_grad():
            for batch in loader:
                if spatial_ctx is not None and len(batch) >= 4:
                    x, _, y_l1, global_idx = batch[0], batch[1], batch[2], batch[-1]
                else:
                    x, _, y_l1 = batch[0], batch[1], batch[2]
                    global_idx = None
                x = x.to(device, non_blocking=True)
                y_l1 = y_l1.to(device, non_blocking=True)
                neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)

                logits_l2 = model(x, neighbor_x=neighbor_x).float()
                probs_l2 = torch.softmax(logits_l2, dim=1)
                probs_l1 = torch.matmul(probs_l2, map_l2_to_l1)
                preds_l1 = torch.argmax(probs_l1, dim=1)

                all_preds_l1.append(preds_l1.cpu().numpy())
                all_labels_l1.append(y_l1.cpu().numpy())

        preds = np.concatenate(all_preds_l1)
        labels = np.concatenate(all_labels_l1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return acc, macro_f1, weighted_f1, preds, labels

    def evaluate_extra_heads(model, loader, device, head_idx):
        """Macro/weighted F1 for L12 (3), L0 (4), or L01 (5) head on validation loader."""
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in loader:
                global_idx = batch[-1] if spatial_ctx is not None and len(batch) >= 5 else None
                x = batch[0].to(device, non_blocking=True)
                y = batch[head_idx].to(device, non_blocking=True)
                neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)
                heads = model.forward_heads(x, neighbor_x=neighbor_x)
                logits = heads[head_idx - 1]
                preds_all.append(torch.argmax(logits, dim=1).cpu().numpy())
                labels_all.append(y.cpu().numpy())
        preds = np.concatenate(preds_all)
        labels = np.concatenate(labels_all)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return acc, macro_f1, weighted_f1

    # Train with hierarchical targets. Best checkpoint: max validation selection score.
    best_weighted_f1 = 0.0
    best_macro_f1 = 0.0
    best_acc = 0.0
    best_l1_macro_f1 = 0.0
    best_l1_weighted_f1 = 0.0
    best_l1_acc = 0.0
    best_score = -1.0
    best_five_tier_auc_sum = float("nan")
    best_epoch = 0
    counter = 0

    eval_loader = val_loader_eval if val_loader_eval is not None else val_loader

    print("Starting training with HCE...")
    print("=" * 60)

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_ce_l2 = 0.0
        total_ce_l1_std = 0.0
        total_ce_l1_agg = 0.0
        total_ce_l12 = 0.0
        total_ce_l3 = 0.0
        total_ce_l4 = 0.0
        num_batches = 0

        for batch in train_loader:
            x, y_l2, y_l1, y_l12, y_l3, y_l4, global_idx = _unpack_spatial_batch(
                batch, use_extra_heads
            )
            if not use_extra_heads:
                y_l12 = y_l3 = y_l4 = None
            x = x.to(device, non_blocking=True)
            y_l2 = y_l2.to(device, non_blocking=True)
            y_l1 = y_l1.to(device, non_blocking=True)
            if use_extra_heads:
                y_l12 = y_l12.to(device, non_blocking=True)
                y_l3 = y_l3.to(device, non_blocking=True)
                y_l4 = y_l4.to(device, non_blocking=True)
            neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)

            optimizer.zero_grad()

            if amp_scaler is not None:
                with torch.amp.autocast("cuda"):
                    if use_extra_heads:
                        logits_l2, logits_l1_head, logits_l12, logits_l3, logits_l4 = (
                            model.forward_heads(x, neighbor_x=neighbor_x)
                        )
                        loss, ce_l2, ce_l1_std, ce_l1_agg, ce_l12, ce_l3, ce_l4 = criterion(
                            logits_l2,
                            logits_l1_head,
                            y_l2,
                            y_l1,
                            logits_l12=logits_l12,
                            logits_l3=logits_l3,
                            logits_l4=logits_l4,
                            target_l12=y_l12,
                            target_l3=y_l3,
                            target_l4=y_l4,
                        )
                    else:
                        logits_l2, logits_l1_head = model.forward_heads(
                            x, neighbor_x=neighbor_x
                        )
                        loss, ce_l2, ce_l1_std, ce_l1_agg, _, _, _ = criterion(
                            logits_l2, logits_l1_head, y_l2, y_l1
                        )
                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss encountered (AMP), skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                if use_extra_heads:
                    logits_l2, logits_l1_head, logits_l12, logits_l3, logits_l4 = (
                        model.forward_heads(x, neighbor_x=neighbor_x)
                    )
                    loss, ce_l2, ce_l1_std, ce_l1_agg, ce_l12, ce_l3, ce_l4 = criterion(
                        logits_l2,
                        logits_l1_head,
                        y_l2,
                        y_l1,
                        logits_l12=logits_l12,
                        logits_l3=logits_l3,
                        logits_l4=logits_l4,
                        target_l12=y_l12,
                        target_l3=y_l3,
                        target_l4=y_l4,
                    )
                else:
                    logits_l2, logits_l1_head = model.forward_heads(
                        x, neighbor_x=neighbor_x
                    )
                    loss, ce_l2, ce_l1_std, ce_l1_agg, _, _, _ = criterion(
                        logits_l2, logits_l1_head, y_l2, y_l1
                    )
                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss encountered, skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())
            total_ce_l2 += float(ce_l2.item())
            total_ce_l1_std += float(ce_l1_std.item())
            total_ce_l1_agg += float(ce_l1_agg.item())
            if use_extra_heads:
                total_ce_l12 += float(ce_l12.item())
                total_ce_l3 += float(ce_l3.item())
                total_ce_l4 += float(ce_l4.item())
            num_batches += 1

        # Validation: F1 passes only needed for F1-based selection metrics.
        need_f1_val = val_selection_metric in ("four_term_sum", "l2_macro_priority")
        if need_f1_val:
            val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate(
                model, eval_loader, device, amp_scaler, spatial_ctx=spatial_ctx
            )
            val_l1_acc, val_l1_macro_f1, val_l1_weighted_f1, _, _ = evaluate_level1_from_level2_logits(
                model, val_loader, device, child_to_parent, num_level1_classes
            )
        else:
            val_acc = val_macro_f1 = val_weighted_f1 = float("nan")
            val_l1_acc = val_l1_macro_f1 = val_l1_weighted_f1 = float("nan")

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce_l2 = total_ce_l2 / max(num_batches, 1)
        avg_ce_l1_std = total_ce_l1_std / max(num_batches, 1)
        avg_ce_l1_agg = total_ce_l1_agg / max(num_batches, 1)
        avg_ce_l12 = total_ce_l12 / max(num_batches, 1)
        avg_ce_l3 = total_ce_l3 / max(num_batches, 1)
        avg_ce_l4 = total_ce_l4 / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{max_epochs} ({epoch_time:.2f}s, LR: {current_lr:.6f})")
        print(f"  Train unified HCE: {avg_loss:.4f}")
        loss_parts = [
            f"CE(L1_head): {avg_ce_l1_std:.4f}",
            f"CE(L2_head): {avg_ce_l2:.4f}",
            f"NLL(L1_agg): {avg_ce_l1_agg:.4f}",
        ]
        if use_extra_heads:
            loss_parts.extend(
                [
                    f"CE(L12): {avg_ce_l12:.4f}",
                    f"CE(L3): {avg_ce_l3:.4f}",
                    f"CE(L4): {avg_ce_l4:.4f}",
                ]
            )
        print(f"  {', '.join(loss_parts)}")

        tier_aucs = _eval_head_macro_aucs_on_loader(
            model,
            val_loader,
            device,
            spatial_ctx=spatial_ctx,
            num_classes=num_classes,
            num_level1_classes=num_level1_classes,
            num_level12_classes=num_level12_classes,
            num_level3_classes=num_level3_classes,
            num_level4_classes=num_level4_classes,
        )
        if tier_aucs:
            if use_extra_heads:
                print(
                    "  Val head macro-AUROC: "
                    f"L1={tier_aucs.get('l1_head_macro_auc', float('nan')):.4f}  "
                    f"L12={tier_aucs.get('l12_macro_auc', float('nan')):.4f}  "
                    f"L2={tier_aucs.get('l2_macro_auc', float('nan')):.4f}  "
                    f"L3={tier_aucs.get('l3_macro_auc', float('nan')):.4f}  "
                    f"L4={tier_aucs.get('l4_macro_auc', float('nan')):.4f}"
                )
            else:
                print(
                    "  Val head macro-AUROC: "
                    f"L1={tier_aucs.get('l1_head_macro_auc', float('nan')):.4f}  "
                    f"L2={tier_aucs.get('l2_macro_auc', float('nan')):.4f}"
                )
            print(f"  Val five-tier AUC sum: {tier_aucs.get('five_tier_auc_sum', float('nan')):.4f}")

        val_metrics = {
            "l2_macro_f1": val_macro_f1,
            "l2_weighted_f1": val_weighted_f1,
            "l1_macro_f1": val_l1_macro_f1,
            "l1_weighted_f1": val_l1_weighted_f1,
            **tier_aucs,
        }
        val_selection_score = _validation_selection_score(
            val_metrics,
            val_selection_metric,
            use_extra_heads=use_extra_heads,
        )
        if not np.isfinite(val_selection_score):
            val_selection_score = -1.0
        if val_selection_score > best_score:
            best_score = val_selection_score
            best_epoch = epoch + 1

            best_macro_f1 = val_macro_f1
            best_weighted_f1 = val_weighted_f1
            best_acc = val_acc
            best_l1_macro_f1 = val_l1_macro_f1
            best_l1_weighted_f1 = val_l1_weighted_f1
            best_l1_acc = val_l1_acc
            best_five_tier_auc_sum = float(tier_aucs.get("five_tier_auc_sum", float("nan")))
            torch.save(model.state_dict(), save_bestmodel_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    print("\nTraining completed!")
    if val_selection_metric == "five_tier_auc_sum" and np.isfinite(best_five_tier_auc_sum):
        print(
            f"Best validation ({val_selection_metric}): {best_score:.4f} at epoch {best_epoch} "
            f"with five-tier AUC sum={best_five_tier_auc_sum:.4f}"
        )
    else:
        print(
            f"Best validation ({val_selection_metric}): {best_score:.4f} at epoch {best_epoch}"
        )

    return best_weighted_f1, best_macro_f1, best_epoch


def train_and_save_model_from_split(
    device,
    loaders,
    result,
    evaluate,
    save_bestmodel_path,
    class_names=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    checkpoint_exists=False,
):
    """
    Convenience wrapper: train directly from split/result + loaders dict.

    Parameters
    ----------
    checkpoint_exists : bool
        If False (default), run full training and save weights to
        ``save_bestmodel_path`` via ``train_and_save_model``.
        If True, **do not train**: load weights from ``save_bestmodel_path`` only
        (file must already exist), evaluate on the validation loader, and return
        metrics. ``best_epoch`` in the return value is 0 (unknown from plain state_dict).

    When training, best checkpoint uses ``val_selection_metric`` (default
    ``five_tier_auc_sum``: sum of head-level macro AUROC on validation).
    Returns ``(weighted_f1, macro_f1, best_epoch)`` at that checkpoint.
    """
    required_loader_keys = ["train_loader", "val_loader"]
    for k in required_loader_keys:
        if k not in loaders:
            raise KeyError(f"Missing loaders['{k}']")

    required_result_keys = [
        "X_train_scaled",
        "y_train_encoded",
        "y_test_encoded",
        "y_train_level1_encoded",
        "y_encoded_f",
        "y_level1_encoded_f",
    ]
    for k in required_result_keys:
        if k not in result:
            raise KeyError(f"Missing result['{k}']")

    use_extra_heads = result_has_extra_head_labels(result)

    if checkpoint_exists:
        if not os.path.isfile(save_bestmodel_path):
            raise FileNotFoundError(
                f"checkpoint_exists=True requires an existing file: {save_bestmodel_path}"
            )
        print(f"checkpoint_exists=True: loading checkpoint (no training): {save_bestmodel_path}")
        input_dim, num_classes = infer_input_dim_and_num_classes(
            result["X_train_scaled"],
            result["y_train_encoded"],
            result["y_test_encoded"],
            class_names=class_names,
        )
        num_l1 = int(np.max(result["y_level1_encoded_f"])) + 1
        state_dict = torch.load(save_bestmodel_path, map_location=device)
        model, _ = build_mlp_classifier_for_state_dict(
            state_dict,
            input_dim=input_dim,
            num_classes=num_classes,
            num_level1_classes=num_l1,
            device=device,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        model.load_state_dict(_remap_legacy_five_head_state_dict(state_dict))
        eval_loader = loaders.get("val_loader_eval", loaders["val_loader"])
        spatial_ctx = _spatial_ctx_from_result(result)
        acc, macro_f1, weighted_f1, _, _ = evaluate(
            model, eval_loader, device, spatial_ctx=spatial_ctx
        )
        print(
            "Loaded existing checkpoint metrics: "
            f"accuracy={acc:.4f}, weighted_f1={weighted_f1:.4f}, macro_f1={macro_f1:.4f}"
        )
        # Epoch unknown from plain state_dict checkpoint.
        return weighted_f1, macro_f1, 0

    train_kw = dict(
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        use_extra_heads=use_extra_heads,
    )
    if use_extra_heads:
        train_kw.update(
            y_level12_encoded_f=result["y_level12_encoded_f"],
            y_level3_encoded_f=result["y_level3_encoded_f"],
            y_level4_encoded_f=result["y_level4_encoded_f"],
        )
    if result.get("use_spatial_context"):
        train_kw.update(
            use_spatial_context=True,
            spatial_mode=result.get("spatial_mode", "mean"),
            spatial_k=int(result.get("spatial_k", 8)),
            spatial_neighbor_index=result["spatial_neighbor_index"],
            X_all_scaled=result["X_all_scaled"],
        )

    return train_and_save_model(
        device=device,
        X_train_scaled=result["X_train_scaled"],
        y_train_encoded=result["y_train_encoded"],
        y_test_encoded=result["y_test_encoded"],
        y_train_level1_encoded=result["y_train_level1_encoded"],
        y_encoded_f=result["y_encoded_f"],
        y_level1_encoded_f=result["y_level1_encoded_f"],
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        evaluate=evaluate,
        save_bestmodel_path=save_bestmodel_path,
        class_names=class_names,
        val_loader_eval=loaders.get("val_loader_eval", None),
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        patience=patience,
        **train_kw,
        max_epochs=max_epochs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
    )


def train_hce_leave_one_group_out_cv(
    device,
    cv_data,
    evaluate,
    class_names=None,
    save_bestmodel_path_pattern=None,
    min_train_l2_classes=2,
    min_train_samples=1,
    loader_kwargs=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    resume_from_checkpoints=False,
):
    """
    Repeated train/val with ``LeaveOneGroupOut``: each fold holds out one group, fits
    ``StandardScaler`` on train only, then trains the HCE MLP like ``train_and_save_model_from_split``.

    This mirrors sklearn's ``cross_val_predict(..., cv=LeaveOneGroupOut(), groups=groups)``
    idea for a PyTorch model (explicit fold loop instead of ``cross_val_predict``).

    Parameters
    ----------
    cv_data : dict
        Output of ``base.prepare_data_leave_one_group_out`` (``X_f``, ``y_encoded_f``,
        ``y_level1_encoded_f``, ``groups_f``, etc.).
    save_bestmodel_path_pattern : str or None
        ``str.format(fold_idx)`` path template, e.g.
        ``"/tmp/hce_logo_fold_{}.pt"``. If None, uses ``NCRT_TMPDIR`` or ``~/ssd2/tmp``.
    resume_from_checkpoints : bool
        If True, for each fold when ``save_path`` already exists, skip training and only
        load weights and compute validation metrics (same as ``checkpoint_exists=True`` in
        ``train_and_save_model_from_split``). Missing files are still trained.

    Returns
    -------
    dict
        Per-fold rows and ``mean``/``std`` for L2/L1 weighted and macro F1 at best checkpoint.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    from base import loader_train_test

    loader_kwargs = loader_kwargs or {}

    X_f = cv_data["X_f"]
    y_enc = cv_data["y_encoded_f"]
    y_l1_enc = cv_data["y_level1_encoded_f"]
    y_enc_full = cv_data["y_encoded_f"]
    y_l1_full = cv_data["y_level1_encoded_f"]
    groups_f = cv_data["groups_f"]
    y_f = cv_data["y_f"]
    y_level1_f = cv_data["y_level1_f"]

    _uniq_g = np.unique(np.asarray(groups_f))
    if len(_uniq_g) < 2:
        raise ValueError(
            "LeaveOneGroupOut requires at least 2 unique groups; "
            f"got {len(_uniq_g)}: {_uniq_g[:8]!r}. "
            "If your CSV marks every row with the same TumorID (e.g. NCRT_tumor1), use a column "
            "with multiple slides/ROIs/patients, or build groups with "
            "``base.spatial_tile_groups_for_logo(X_pix_HE, Y_pix_HE, nx, ny)`` (see tumor1_cv notebook)."
        )

    if save_bestmodel_path_pattern is None:
        tdir = os.environ.get("NCRT_TMPDIR", os.path.join(os.path.expanduser("~"), "ssd2", "tmp"))
        os.makedirs(tdir, exist_ok=True)
        save_bestmodel_path_pattern = os.path.join(tdir, "hce_logo_fold_{}.pt")

    logo = LeaveOneGroupOut()
    fold_rows = []
    skipped = []

    for fold_idx, (tr_idx, va_idx) in enumerate(logo.split(X_f, y_enc, groups_f)):
        held = np.unique(np.asarray(groups_f)[va_idx])
        held_repr = held[0] if len(held) == 1 else tuple(held.tolist())

        if len(tr_idx) < min_train_samples or len(va_idx) < 1:
            skipped.append((fold_idx, held_repr, "empty train or val"))
            continue
        n_l2_tr = len(np.unique(y_enc[tr_idx]))
        if n_l2_tr < min_train_l2_classes:
            skipped.append(
                (fold_idx, held_repr, f"train has only {n_l2_tr} L2 classes (need {min_train_l2_classes})")
            )
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_f[tr_idx])
        X_va = scaler.transform(X_f[va_idx])

        result_fold = {
            "X_train_scaled": X_tr,
            "X_test_scaled": X_va,
            "y_train_encoded": y_enc[tr_idx],
            "y_test_encoded": y_enc[va_idx],
            "y_train_level1_encoded": y_l1_enc[tr_idx],
            "y_test_level1_encoded": y_l1_enc[va_idx],
            "y_train": y_f[tr_idx],
            "y_test": y_f[va_idx],
            "y_train_level1": y_level1_f[tr_idx],
            "y_test_level1": y_level1_f[va_idx],
            "y_encoded_f": y_enc_full,
            "y_level1_encoded_f": y_l1_full,
        }
        _append_logo_fold_extra_tiers(result_fold, cv_data, tr_idx, va_idx)

        loaders = loader_train_test(result_fold, **loader_kwargs)
        save_path = os.path.normpath(os.path.abspath(save_bestmodel_path_pattern.format(fold_idx)))
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        print(f"\n{'=' * 60}\nLOGO fold {fold_idx}: held-out group = {held_repr!r}\n{'=' * 60}")
        ckpt_exists = os.path.isfile(save_path)
        use_resume = bool(resume_from_checkpoints and ckpt_exists)
        if use_resume:
            print(f"  Resume: checkpoint exists, skip training → {save_path}")
        else:
            if not resume_from_checkpoints:
                print(
                    "  Training: resume_from_checkpoints=False "
                    "(e.g. CLI --logo_resume false) — full HCE training for this fold."
                )
            else:
                print(
                    f"  Training: no file at {save_path!r} — "
                    "train this fold; after checkpoints exist, re-run to load-only for that fold."
                )

        w_f1, macro_f1, best_ep = train_and_save_model_from_split(
            device=device,
            loaders=loaders,
            result=result_fold,
            evaluate=evaluate,
            save_bestmodel_path=save_path,
            class_names=class_names,
            hce_w1=hce_w1,
            hce_w2=hce_w2,
            hce_w12=hce_w12,
            hce_w_l12head=hce_w_l12head,
            hce_w_l3=hce_w_l3,
            hce_w_l4=hce_w_l4,
            patience=patience,
            max_epochs=max_epochs,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            l2_label_smoothing=l2_label_smoothing,
            l1_label_smoothing=l1_label_smoothing,
            val_selection_metric=val_selection_metric,
            class_weight_mode=class_weight_mode,
            class_balanced_beta=class_balanced_beta,
            l2_focal_gamma=l2_focal_gamma,
            checkpoint_exists=use_resume,
        )

        model, num_classes = _load_fold_model_for_eval(
            save_path, device, result_fold, class_names, hidden_dims, dropout
        )
        tier_m = _eval_fold_tier_metrics(
            model,
            loaders["val_loader"],
            device,
            num_classes,
            y_enc_full,
            y_l1_full,
            spatial_ctx=_spatial_ctx_from_result(result_fold),
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "held_group": held_repr,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "l2_weighted_f1": float(w_f1),
                "l2_macro_f1": float(macro_f1),
                "best_epoch": int(best_ep),
                "checkpoint": save_path,
                **tier_m,
            }
        )

    def _mean_std(key):
        vals = np.array([r[key] for r in fold_rows], dtype=np.float64)
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "folds": fold_rows,
        "skipped": skipped,
        "n_folds_ran": len(fold_rows),
        "l2_weighted_f1_mean": _mean_std("l2_weighted_f1")[0],
        "l2_weighted_f1_std": _mean_std("l2_weighted_f1")[1],
        "l2_macro_f1_mean": _mean_std("l2_macro_f1")[0],
        "l2_macro_f1_std": _mean_std("l2_macro_f1")[1],
        "l1_weighted_f1_mean": _mean_std("l1_weighted_f1")[0],
        "l1_weighted_f1_std": _mean_std("l1_weighted_f1")[1],
        "l1_macro_f1_mean": _mean_std("l1_macro_f1")[0],
        "l1_macro_f1_std": _mean_std("l1_macro_f1")[1],
    }
    _append_tier_means_to_cv_summary(summary, fold_rows)

    print("\n" + "=" * 60)
    print("Leave-One-Group-Out summary (best checkpoint per fold)")
    print(
        f"  L2 weighted-F1: {summary['l2_weighted_f1_mean']:.4f} ± {summary['l2_weighted_f1_std']:.4f}\n"
        f"  L2 macro-F1:    {summary['l2_macro_f1_mean']:.4f} ± {summary['l2_macro_f1_std']:.4f}\n"
        f"  L1 weighted-F1: {summary['l1_weighted_f1_mean']:.4f} ± {summary['l1_weighted_f1_std']:.4f}\n"
        f"  L1 macro-F1:    {summary['l1_macro_f1_mean']:.4f} ± {summary['l1_macro_f1_std']:.4f}"
    )
    if skipped:
        print(f"  Skipped {len(skipped)} fold(s): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    print("=" * 60)

    return summary


########################################################
# 2026.06.25 LLY: group kfold cv for Xenium_lung Complete_Cases 25 datasets for training and validation
# 2026.06.26 For each dataset, supports use_spatial_context / spatial_k / spatial_mode for spatial context.
########################################################
def train_hce_group_kfold_cv(
    device,
    cv_data,
    evaluate,
    class_names=None,
    save_bestmodel_path_pattern=None,
    n_splits=5,
    train_group_frac=0.7,
    min_train_l2_classes=2,
    min_train_samples=1,
    loader_kwargs=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    resume_from_checkpoints=False,
    random_state=42,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """
    Dataset-level CV with ``GroupShuffleSplit``: each fold holds out ~``(1-train_group_frac)``
    of groups (e.g. samples) for validation; scaler is fit on train groups only.

    With 25 samples and ``train_group_frac=0.7``, each fold trains on 18 datasets and
    validates on 7 (ceil(25×0.7) train size).

    ``use_spatial_context``: fuse kNN neighbor UNI embeddings before the MLP.
    When ``groups_f`` is present, kNN is built **per sample** (not across pooled coords).
    """
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    from base import loader_train_test

    loader_kwargs = loader_kwargs or {}

    if use_spatial_context and "X_coords_f" not in cv_data:
        raise ValueError(
            "use_spatial_context=True requires X_coords_f in cv_data. "
            "Pooled h5ad prep must pass spatial_HE coordinates."
        )

    neighbor_index_full = None
    if use_spatial_context:
        neighbor_index_full = _build_spatial_neighbor_index_for_cv_data(
            cv_data, k_neighbors=spatial_k
        )
        group_note = "per-sample kNN" if "groups_f" in cv_data else "global kNN"
        print(
            f"Spatial context: k={spatial_k}, mode={spatial_mode!r}, {group_note}, "
            f"neighbor_index shape={neighbor_index_full.shape}",
            flush=True,
        )

    X_f = cv_data["X_f"]
    y_enc = cv_data["y_encoded_f"]
    y_l1_enc = cv_data["y_level1_encoded_f"]
    y_enc_full = cv_data["y_encoded_f"]
    y_l1_full = cv_data["y_level1_encoded_f"]
    y_f = cv_data["y_f"]
    y_level1_f = cv_data["y_level1_f"]
    groups_f = np.asarray(cv_data["groups_f"])

    uniq_g = np.unique(groups_f)
    n_groups = len(uniq_g)
    if n_groups < 2:
        raise ValueError(
            f"Group CV requires at least 2 unique groups; got {n_groups}: {uniq_g[:8]!r}"
        )

    n_train_groups = int(np.ceil(n_groups * float(train_group_frac)))
    n_train_groups = min(max(n_train_groups, 1), n_groups - 1)
    test_frac = (n_groups - n_train_groups) / float(n_groups)

    if save_bestmodel_path_pattern is None:
        tdir = os.environ.get("NCRT_TMPDIR", os.path.join(os.path.expanduser("~"), "ssd2", "tmp"))
        os.makedirs(tdir, exist_ok=True)
        save_bestmodel_path_pattern = os.path.join(tdir, "hce_group_fold_{}.pt")

    gss = GroupShuffleSplit(
        n_splits=int(n_splits),
        test_size=test_frac,
        random_state=int(random_state),
    )
    fold_rows = []
    skipped = []
    oof_preds_parts = []
    oof_labels_parts = []
    oof_probs_parts = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gss.split(X_f, y_enc, groups_f)):
        held = np.unique(groups_f[va_idx])
        held_repr = tuple(int(x) for x in held.tolist())

        if len(tr_idx) < min_train_samples or len(va_idx) < 1:
            skipped.append((fold_idx, held_repr, "empty train or val"))
            continue
        n_l2_tr = len(np.unique(y_enc[tr_idx]))
        if n_l2_tr < min_train_l2_classes:
            skipped.append(
                (fold_idx, held_repr, f"train has only {n_l2_tr} L2 classes (need {min_train_l2_classes})")
            )
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_f[tr_idx])
        X_va = scaler.transform(X_f[va_idx])

        result_fold = {
            "X_train_scaled": X_tr,
            "X_test_scaled": X_va,
            "y_train_encoded": y_enc[tr_idx],
            "y_test_encoded": y_enc[va_idx],
            "y_train_level1_encoded": y_l1_enc[tr_idx],
            "y_test_level1_encoded": y_l1_enc[va_idx],
            "y_train": y_f[tr_idx],
            "y_test": y_f[va_idx],
            "y_train_level1": y_level1_f[tr_idx],
            "y_test_level1": y_level1_f[va_idx],
            "y_encoded_f": y_enc_full,
            "y_level1_encoded_f": y_l1_full,
        }
        if use_spatial_context:
            result_fold.update(
                {
                    "use_spatial_context": True,
                    "spatial_mode": spatial_mode,
                    "spatial_k": spatial_k,
                    "spatial_neighbor_index": neighbor_index_full,
                    "X_all_scaled": scaler.transform(X_f),
                    "train_global_indices": tr_idx,
                    "val_global_indices": va_idx,
                }
            )
        _append_logo_fold_extra_tiers(result_fold, cv_data, tr_idx, va_idx)

        loaders = loader_train_test(result_fold, **loader_kwargs)
        save_path = os.path.normpath(os.path.abspath(save_bestmodel_path_pattern.format(fold_idx)))
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        print(
            f"\n{'=' * 60}\nGroup CV fold {fold_idx}: "
            f"train_groups={n_groups - len(held)}/{n_groups}, "
            f"held_out={len(held)} group(s) {held_repr!r}\n{'=' * 60}"
        )
        ckpt_exists = os.path.isfile(save_path)
        use_resume = bool(resume_from_checkpoints and ckpt_exists)
        if use_resume:
            print(f"  Resume: checkpoint exists, skip training → {save_path}")
        else:
            print("  Training: full HCE training for this fold.")

        w_f1, macro_f1, best_ep = train_and_save_model_from_split(
            device=device,
            loaders=loaders,
            result=result_fold,
            evaluate=evaluate,
            save_bestmodel_path=save_path,
            class_names=class_names,
            hce_w1=hce_w1,
            hce_w2=hce_w2,
            hce_w12=hce_w12,
            hce_w_l12head=hce_w_l12head,
            hce_w_l3=hce_w_l3,
            hce_w_l4=hce_w_l4,
            patience=patience,
            max_epochs=max_epochs,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            l2_label_smoothing=l2_label_smoothing,
            l1_label_smoothing=l1_label_smoothing,
            val_selection_metric=val_selection_metric,
            class_weight_mode=class_weight_mode,
            class_balanced_beta=class_balanced_beta,
            l2_focal_gamma=l2_focal_gamma,
            checkpoint_exists=use_resume,
        )

        model, num_classes = _load_fold_model_for_eval(
            save_path, device, result_fold, class_names, hidden_dims, dropout
        )
        tier_m = _eval_fold_tier_metrics(
            model,
            loaders["val_loader"],
            device,
            num_classes,
            y_enc_full,
            y_l1_full,
            spatial_ctx=_spatial_ctx_from_result(result_fold),
            return_oof=True,
        )
        val_labels, val_preds, val_probs, tier_m = _split_oof_from_tier_metrics(tier_m)
        if val_labels is None or val_preds is None or val_probs is None:
            raise RuntimeError("_eval_fold_tier_metrics failed to produce OOF outputs.")
        oof_preds_parts.append(np.asarray(val_preds, dtype=np.int64))
        oof_labels_parts.append(np.asarray(val_labels, dtype=np.int64))
        oof_probs_parts.append(np.asarray(val_probs, dtype=np.float64))

        fold_rows.append(
            {
                "fold": fold_idx,
                "held_groups": held_repr,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "n_train_groups": int(n_groups - len(held)),
                "n_val_groups": int(len(held)),
                "l2_weighted_f1": float(w_f1),
                "l2_macro_f1": float(macro_f1),
                "best_epoch": int(best_ep),
                "checkpoint": save_path,
                "val_indices": va_idx,
                **tier_m,
            }
        )

    def _mean_std(key):
        vals = np.array([r[key] for r in fold_rows], dtype=np.float64)
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "folds": fold_rows,
        "skipped": skipped,
        "n_folds_ran": len(fold_rows),
        "l2_weighted_f1_mean": _mean_std("l2_weighted_f1")[0],
        "l2_weighted_f1_std": _mean_std("l2_weighted_f1")[1],
        "l2_macro_f1_mean": _mean_std("l2_macro_f1")[0],
        "l2_macro_f1_std": _mean_std("l2_macro_f1")[1],
        "l1_weighted_f1_mean": _mean_std("l1_weighted_f1")[0],
        "l1_weighted_f1_std": _mean_std("l1_weighted_f1")[1],
        "l1_macro_f1_mean": _mean_std("l1_macro_f1")[0],
        "l1_macro_f1_std": _mean_std("l1_macro_f1")[1],
        "cv_mode": "group_kfold",
        "n_splits": int(n_splits),
        "train_group_frac": float(train_group_frac),
        "n_train_groups_per_fold": int(n_train_groups),
        "n_val_groups_per_fold": int(n_groups - n_train_groups),
    }
    _append_tier_means_to_cv_summary(summary, fold_rows)

    if oof_preds_parts:
        summary["oof_val_preds"] = np.concatenate(oof_preds_parts)
        summary["oof_val_labels"] = np.concatenate(oof_labels_parts)
        summary["oof_val_probs"] = np.concatenate(oof_probs_parts, axis=0)

    print("\n" + "=" * 60)
    print(
        f"GroupShuffleSplit summary (n_splits={n_splits}, "
        f"train_group_frac={train_group_frac}, "
        f"~{n_train_groups} train / {n_groups - n_train_groups} val datasets per fold)"
    )
    print(
        f"  L2 weighted-F1: {summary['l2_weighted_f1_mean']:.4f} ± {summary['l2_weighted_f1_std']:.4f}\n"
        f"  L2 macro-F1:    {summary['l2_macro_f1_mean']:.4f} ± {summary['l2_macro_f1_std']:.4f}\n"
        f"  L1 weighted-F1: {summary['l1_weighted_f1_mean']:.4f} ± {summary['l1_weighted_f1_std']:.4f}\n"
        f"  L1 macro-F1:    {summary['l1_macro_f1_mean']:.4f} ± {summary['l1_macro_f1_std']:.4f}"
    )
    if skipped:
        print(f"  Skipped {len(skipped)} fold(s): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    print("=" * 60)

    return summary
########################################################


def train_hce_stratified_kfold_cv(
    device,
    cv_data,
    evaluate,
    class_names=None,
    save_bestmodel_path_pattern=None,
    n_splits=5,
    stratify_target="level2",
    min_train_l2_classes=2,
    min_train_samples=1,
    loader_kwargs=None,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    resume_from_checkpoints=False,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """
    Repeated train/val with ``StratifiedKFold`` on filtered full matrix.

    ``stratify_target`` controls how folds preserve label proportions:
    - ``"level2"``: stratify by level2 labels (recommended default).
    - ``"level1"``: stratify by level1 labels.
    - ``"joint"``: stratify by joint key ``"{l1}|{l2}"`` for tighter hierarchy balance.

    ``use_spatial_context`` (default False): fuse kNN neighbor UNI embeddings before the MLP.
    Requires ``cv_data['X_coords_f']`` from ``prepare_data_leave_one_group_out(..., X_coords=...)``.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from base import loader_train_test

    loader_kwargs = loader_kwargs or {}

    if use_spatial_context and "X_coords_f" not in cv_data:
        raise ValueError(
            "use_spatial_context=True requires X_coords_f in cv_data. "
            "Pass X_coords=... to prepare_data_leave_one_group_out()."
        )
    neighbor_index_full = None
    if use_spatial_context:
        neighbor_index_full = _build_spatial_neighbor_index_for_cv_data(
            cv_data, k_neighbors=spatial_k
        )
        group_note = "per-sample kNN" if "groups_f" in cv_data else "global kNN"
        print(
            f"Spatial context: k={spatial_k}, mode={spatial_mode!r}, {group_note}, "
            f"neighbor_index shape={neighbor_index_full.shape}"
        )

    X_f = cv_data["X_f"]
    y_enc = cv_data["y_encoded_f"]
    y_l1_enc = cv_data["y_level1_encoded_f"]
    y_enc_full = cv_data["y_encoded_f"]
    y_l1_full = cv_data["y_level1_encoded_f"]
    y_f = cv_data["y_f"]
    y_level1_f = cv_data["y_level1_f"]

    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >=2, got {n_splits}")

    if stratify_target == "level2":
        y_strat = np.asarray(y_enc)
    elif stratify_target == "level1":
        y_strat = np.asarray(y_l1_enc)
    elif stratify_target == "joint":
        y_strat = np.asarray([f"{int(a)}|{int(b)}" for a, b in zip(y_l1_enc, y_enc)], dtype=object)
    else:
        raise ValueError("stratify_target must be one of: level2, level1, joint")

    unique_keys, key_counts = np.unique(y_strat, return_counts=True)
    min_count = int(np.min(key_counts)) if key_counts.size else 0
    if min_count < n_splits:
        raise ValueError(
            f"Cannot run StratifiedKFold(n_splits={n_splits}): minimum class/group count in "
            f"stratification target is {min_count}. Reduce --cv_k or use coarser stratify target."
        )

    if save_bestmodel_path_pattern is None:
        tdir = os.environ.get("NCRT_TMPDIR", os.path.join(os.path.expanduser("~"), "ssd2", "tmp"))
        os.makedirs(tdir, exist_ok=True)
        save_bestmodel_path_pattern = os.path.join(tdir, "hce_kfold_fold_{}.pt")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rows = []
    skipped = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_f, y_strat)):
        if len(tr_idx) < min_train_samples or len(va_idx) < 1:
            skipped.append((fold_idx, "empty train or val"))
            continue
        n_l2_tr = len(np.unique(y_enc[tr_idx]))
        if n_l2_tr < min_train_l2_classes:
            skipped.append(
                (fold_idx, f"train has only {n_l2_tr} L2 classes (need {min_train_l2_classes})")
            )
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_f[tr_idx])
        X_va = scaler.transform(X_f[va_idx])

        result_fold = {
            "X_train_scaled": X_tr,
            "X_test_scaled": X_va,
            "y_train_encoded": y_enc[tr_idx],
            "y_test_encoded": y_enc[va_idx],
            "y_train_level1_encoded": y_l1_enc[tr_idx],
            "y_test_level1_encoded": y_l1_enc[va_idx],
            "y_train": y_f[tr_idx],
            "y_test": y_f[va_idx],
            "y_train_level1": y_level1_f[tr_idx],
            "y_test_level1": y_level1_f[va_idx],
            "y_encoded_f": y_enc_full,
            "y_level1_encoded_f": y_l1_full,
        }
        if use_spatial_context:
            result_fold.update(
                {
                    "use_spatial_context": True,
                    "spatial_mode": spatial_mode,
                    "spatial_k": spatial_k,
                    "spatial_neighbor_index": neighbor_index_full,
                    "X_all_scaled": scaler.transform(X_f),
                    "train_global_indices": tr_idx,
                    "val_global_indices": va_idx,
                }
            )
        _append_logo_fold_extra_tiers(result_fold, cv_data, tr_idx, va_idx)

        loaders = loader_train_test(result_fold, **loader_kwargs)
        save_path = os.path.normpath(os.path.abspath(save_bestmodel_path_pattern.format(fold_idx)))
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        print(f"\n{'=' * 60}\nKFold fold {fold_idx}: stratify={stratify_target!r}\n{'=' * 60}")
        ckpt_exists = os.path.isfile(save_path)
        use_resume = bool(resume_from_checkpoints and ckpt_exists)
        if use_resume:
            print(f"  Resume: checkpoint exists, skip training -> {save_path}")
        else:
            if not resume_from_checkpoints:
                print("  Training: resume_from_checkpoints=False -> full HCE training for this fold.")
            else:
                print(
                    f"  Training: no file at {save_path!r} -> train this fold; re-run to load-only later."
                )

        w_f1, macro_f1, best_ep = train_and_save_model_from_split(
            device=device,
            loaders=loaders,
            result=result_fold,
            evaluate=evaluate,
            save_bestmodel_path=save_path,
            class_names=class_names,
            hce_w1=hce_w1,
            hce_w2=hce_w2,
            hce_w12=hce_w12,
            hce_w_l12head=hce_w_l12head,
            hce_w_l3=hce_w_l3,
            hce_w_l4=hce_w_l4,
            patience=patience,
            max_epochs=max_epochs,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            l2_label_smoothing=l2_label_smoothing,
            l1_label_smoothing=l1_label_smoothing,
            val_selection_metric=val_selection_metric,
            class_weight_mode=class_weight_mode,
            class_balanced_beta=class_balanced_beta,
            l2_focal_gamma=l2_focal_gamma,
            checkpoint_exists=use_resume,
        )

        model, num_classes = _load_fold_model_for_eval(
            save_path, device, result_fold, class_names, hidden_dims, dropout
        )
        tier_m = _eval_fold_tier_metrics(
            model,
            loaders["val_loader"],
            device,
            num_classes,
            y_enc_full,
            y_l1_full,
            spatial_ctx=_spatial_ctx_from_result(result_fold),
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "l2_weighted_f1": float(w_f1),
                "l2_macro_f1": float(macro_f1),
                "best_epoch": int(best_ep),
                "checkpoint": save_path,
                **tier_m,
            }
        )

    def _mean_std(key):
        vals = np.array([r[key] for r in fold_rows], dtype=np.float64)
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "folds": fold_rows,
        "skipped": skipped,
        "n_folds_ran": len(fold_rows),
        "l2_weighted_f1_mean": _mean_std("l2_weighted_f1")[0],
        "l2_weighted_f1_std": _mean_std("l2_weighted_f1")[1],
        "l2_macro_f1_mean": _mean_std("l2_macro_f1")[0],
        "l2_macro_f1_std": _mean_std("l2_macro_f1")[1],
        "l1_weighted_f1_mean": _mean_std("l1_weighted_f1")[0],
        "l1_weighted_f1_std": _mean_std("l1_weighted_f1")[1],
        "l1_macro_f1_mean": _mean_std("l1_macro_f1")[0],
        "l1_macro_f1_std": _mean_std("l1_macro_f1")[1],
        "cv_mode": "stratified_kfold",
        "stratify_target": stratify_target,
        "n_splits": int(n_splits),
    }
    _append_tier_means_to_cv_summary(summary, fold_rows)

    print("\n" + "=" * 60)
    print(f"StratifiedKFold summary (n_splits={n_splits}, stratify={stratify_target})")
    print(
        f"  L2 weighted-F1: {summary['l2_weighted_f1_mean']:.4f} ± {summary['l2_weighted_f1_std']:.4f}\n"
        f"  L2 macro-F1:    {summary['l2_macro_f1_mean']:.4f} ± {summary['l2_macro_f1_std']:.4f}\n"
        f"  L1 weighted-F1: {summary['l1_weighted_f1_mean']:.4f} ± {summary['l1_weighted_f1_std']:.4f}\n"
        f"  L1 macro-F1:    {summary['l1_macro_f1_mean']:.4f} ± {summary['l1_macro_f1_std']:.4f}"
    )
    if skipped:
        print(f"  Skipped {len(skipped)} fold(s): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    print("=" * 60)

    return summary


def run_stratified_kfold_cv_and_load_best_model(
    device,
    cv_data,
    evaluate,
    class_names,
    path,
    therapy_data,
    therapy_model,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    n_splits=5,
    stratify_target="level2",
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    kfold_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """Run StratifiedKFold HCE CV then load the best fold by 4-term selection score."""
    from pathlib import Path

    if kfold_checkpoint_dir is not None:
        kfold_dir = Path(kfold_checkpoint_dir).expanduser().resolve()
        kfold_ckpt_pattern = str(kfold_dir / "hce_kfold_fold_{}.pt")
    else:
        kfold_ckpt_pattern = (
            f"{path}Collaborate/esccAI/data/{therapy_data}/{therapy_model}/hce_kfold_fold_{{}}.pt"
        )
        kfold_dir = Path(f"{path}Collaborate/esccAI/data/{therapy_data}/{therapy_model}")
    if resume_from_checkpoints is None:
        resume_from_checkpoints = any(kfold_dir.glob("hce_kfold_fold_*.pt"))

    loader_kwargs = loader_kwargs or {"seed": 42, "train_balance_sampler": False}

    kfold_summary = train_hce_stratified_kfold_cv(
        device=device,
        cv_data=cv_data,
        evaluate=evaluate,
        class_names=class_names,
        save_bestmodel_path_pattern=kfold_ckpt_pattern,
        n_splits=n_splits,
        stratify_target=stratify_target,
        loader_kwargs=loader_kwargs,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        patience=patience,
        max_epochs=max_epochs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
        resume_from_checkpoints=resume_from_checkpoints,
        use_spatial_context=use_spatial_context,
        spatial_k=spatial_k,
        spatial_mode=spatial_mode,
    )

    best_fold = max(
        kfold_summary["folds"],
        key=lambda r: _validation_selection_score(r, cv_selection_metric),
    )
    best_fold_score = float(_validation_selection_score(best_fold, cv_selection_metric))
    print(
        "Best fold selection: "
        f"metric={cv_selection_metric}, "
        f"fold={best_fold.get('fold')}, "
        f"score={best_fold_score:.6f}"
    )
    input_dim = int(cv_data["X_f"].shape[1])
    num_classes = len(class_names)
    ckpt_path = best_fold["checkpoint"]
    state_dict = torch.load(ckpt_path, map_location=device)
    model, use_five = build_mlp_classifier_for_state_dict(
        state_dict,
        input_dim=input_dim,
        num_classes=num_classes,
        num_level1_classes=int(np.max(cv_data["y_level1_encoded_f"])) + 1,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(_remap_legacy_five_head_state_dict(state_dict))
    model.eval()

    spatial_neighbor_index = None
    if use_spatial_context:
        spatial_neighbor_index = build_spatial_neighbor_index(cv_data["X_coords_f"], k_neighbors=spatial_k)

    return {
        "logo_summary": kfold_summary,  # keep key name for downstream compatibility
        "model": model,
        "best_fold": best_fold,
        "best_weighted_f1": float(best_fold["l2_weighted_f1"]),
        "best_macro_f1": float(best_fold["l2_macro_f1"]),
        "best_epoch": int(best_fold["best_epoch"]),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "logo_ckpt_pattern": kfold_ckpt_pattern,
        "hce_w1": float(hce_w1),
        "hce_w2": float(hce_w2),
        "hce_w12": float(hce_w12),
        "hce_w_l12head": float(hce_w_l12head),
        "hce_w_l3": float(hce_w_l3),
        "hce_w_l4": float(hce_w_l4),
        "use_five_head": use_five,
        "use_spatial_context": bool(use_spatial_context),
        "spatial_k": int(spatial_k),
        "spatial_mode": spatial_mode,
        "spatial_neighbor_index": spatial_neighbor_index,
    }


def run_stratified_kfold_cv_with_insample_report(
    device,
    cv_data,
    scaler,
    class_names,
    evaluate,
    path,
    therapy_data,
    therapy_model,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    n_splits=5,
    stratify_target="level2",
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    kfold_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    insample_batch_size=1024,
    print_level2_classification_report=True,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """Convenience wrapper: stratified K-fold CV + load best fold + in-sample report."""
    ctx = run_stratified_kfold_cv_and_load_best_model(
        device,
        cv_data,
        evaluate,
        class_names,
        path,
        therapy_data,
        therapy_model,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        n_splits=n_splits,
        stratify_target=stratify_target,
        patience=patience,
        max_epochs=max_epochs,
        loader_kwargs=loader_kwargs,
        resume_from_checkpoints=resume_from_checkpoints,
        kfold_checkpoint_dir=kfold_checkpoint_dir,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        cv_selection_metric=cv_selection_metric,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
        use_spatial_context=use_spatial_context,
        spatial_k=spatial_k,
        spatial_mode=spatial_mode,
    )
    ins = report_logo_cv_means_and_insample_eval(
        device,
        cv_data,
        scaler,
        class_names,
        ctx["model"],
        evaluate,
        ctx["logo_summary"],
        batch_size=insample_batch_size,
        print_level2_classification_report=print_level2_classification_report,
        neighbor_index=ctx.get("spatial_neighbor_index"),
    )
    merged = {**ctx, **ins}
    return merged

########################################################
# 2026.06.25 LLY: group kfold cv for Xenium_lung Complete_Cases 25 datasets for training and validation
########################################################
def run_group_kfold_cv_and_load_best_model(
    device,
    cv_data,
    evaluate,
    class_names,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    n_splits=5,
    train_group_frac=0.7,
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    group_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    random_state=42,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """Run dataset-level GroupShuffleSplit CV, then load the best fold by 4-term score."""
    from pathlib import Path

    if group_checkpoint_dir is not None:
        group_dir = Path(group_checkpoint_dir).expanduser().resolve()
        group_ckpt_pattern = str(group_dir / "hce_group_fold_{}.pt")
    else:
        tdir = os.environ.get("NCRT_TMPDIR", os.path.join(os.path.expanduser("~"), "ssd2", "tmp"))
        group_dir = Path(tdir)
        group_ckpt_pattern = str(group_dir / "hce_group_fold_{}.pt")

    if resume_from_checkpoints is None:
        resume_from_checkpoints = any(group_dir.glob("hce_group_fold_*.pt"))

    loader_kwargs = loader_kwargs or {"seed": 42, "train_balance_sampler": False}

    group_summary = train_hce_group_kfold_cv(
        device=device,
        cv_data=cv_data,
        evaluate=evaluate,
        class_names=class_names,
        save_bestmodel_path_pattern=group_ckpt_pattern,
        n_splits=n_splits,
        train_group_frac=train_group_frac,
        loader_kwargs=loader_kwargs,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        patience=patience,
        max_epochs=max_epochs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
        resume_from_checkpoints=resume_from_checkpoints,
        random_state=random_state,
        use_spatial_context=use_spatial_context,
        spatial_k=spatial_k,
        spatial_mode=spatial_mode,
    )

    best_fold = max(
        group_summary["folds"],
        key=lambda r: _validation_selection_score(r, cv_selection_metric),
    )
    best_fold_score = float(_validation_selection_score(best_fold, cv_selection_metric))
    print(
        "Best fold selection: "
        f"metric={cv_selection_metric}, "
        f"fold={best_fold.get('fold')}, "
        f"score={best_fold_score:.6f}"
    )
    input_dim = int(cv_data["X_f"].shape[1])
    num_classes = len(class_names)
    ckpt_path = best_fold["checkpoint"]
    state_dict = torch.load(ckpt_path, map_location=device)
    model, use_five = build_mlp_classifier_for_state_dict(
        state_dict,
        input_dim=input_dim,
        num_classes=num_classes,
        num_level1_classes=int(np.max(cv_data["y_level1_encoded_f"])) + 1,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(_remap_legacy_five_head_state_dict(state_dict))
    model.eval()

    spatial_neighbor_index = None
    if use_spatial_context:
        spatial_neighbor_index = _build_spatial_neighbor_index_for_cv_data(
            cv_data, k_neighbors=spatial_k
        )

    return {
        "logo_summary": group_summary,
        "group_summary": group_summary,
        "model": model,
        "best_fold": best_fold,
        "best_weighted_f1": float(best_fold["l2_weighted_f1"]),
        "best_macro_f1": float(best_fold["l2_macro_f1"]),
        "best_epoch": int(best_fold["best_epoch"]),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "group_ckpt_pattern": group_ckpt_pattern,
        "hce_w1": float(hce_w1),
        "hce_w2": float(hce_w2),
        "hce_w12": float(hce_w12),
        "hce_w_l12head": float(hce_w_l12head),
        "hce_w_l3": float(hce_w_l3),
        "hce_w_l4": float(hce_w_l4),
        "use_five_head": use_five,
        "use_spatial_context": bool(use_spatial_context),
        "spatial_k": int(spatial_k),
        "spatial_mode": spatial_mode,
        "spatial_neighbor_index": spatial_neighbor_index,
    }


def run_group_kfold_cv_with_oof_report(
    device,
    cv_data,
    scaler,
    class_names,
    evaluate,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    n_splits=5,
    train_group_frac=0.7,
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    group_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    random_state=42,
    use_spatial_context=False,
    spatial_k=8,
    spatial_mode="mean",
):
    """Dataset-level group CV + load best fold + out-of-fold (held-out dataset) predictions."""
    ctx = run_group_kfold_cv_and_load_best_model(
        device,
        cv_data,
        evaluate,
        class_names,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        n_splits=n_splits,
        train_group_frac=train_group_frac,
        patience=patience,
        max_epochs=max_epochs,
        loader_kwargs=loader_kwargs,
        resume_from_checkpoints=resume_from_checkpoints,
        group_checkpoint_dir=group_checkpoint_dir,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        cv_selection_metric=cv_selection_metric,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
        random_state=random_state,
        use_spatial_context=use_spatial_context,
        spatial_k=spatial_k,
        spatial_mode=spatial_mode,
    )
    gs = ctx["group_summary"]
    oof = {
        "val_preds": gs.get("oof_val_preds"),
        "val_labels": gs.get("oof_val_labels"),
        "val_probs_l2": gs.get("oof_val_probs"),
        "val_labels_level1": None,
        "val_preds_level1": None,
    }
    if oof["val_labels"] is not None:
        child_to_parent = np.full(len(class_names), -1, dtype=np.int64)
        for l2, l1 in zip(cv_data["y_encoded_f"], cv_data["y_level1_encoded_f"]):
            if child_to_parent[int(l2)] == -1:
                child_to_parent[int(l2)] = int(l1)
        oof["val_labels_level1"] = np.array(
            [child_to_parent[int(p)] for p in oof["val_preds"]], dtype=np.int64
        )
        oof["val_preds_level1"] = np.array(
            [child_to_parent[int(p)] for p in oof["val_preds"]], dtype=np.int64
        )
    merged = {**ctx, **oof}
    return merged
########################################################

def run_logo_cv_and_load_best_model(
    device,
    cv_data,
    evaluate,
    class_names,
    path,
    therapy_data,
    therapy_model,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    logo_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
):
    """
    Run Leave-One-Group-Out HCE training (or resume per-fold if checkpoints exist), then load
    the single best fold (by 4-term score:
    ``l2_macro_f1 + l2_weighted_f1 + l1_macro_f1 + l1_weighted_f1``) for downstream plots.

    Parameters
    ----------
    path : str
        Notebook-style base, e.g. ``'/path/to/Python/'``; used only when ``logo_checkpoint_dir``
        is None to build
        ``{path}Collaborate/esccAI/data/{therapy_data}/{therapy_model}/hce_logo_fold_{{k}}.pt``.
    logo_checkpoint_dir : str, pathlib.Path, or None
        If set, fold checkpoints are read/written as ``{logo_checkpoint_dir}/hce_logo_fold_{k}.pt``
        (same folder as ``matched_features_*.npz`` when using ``project_root/data/...`` layouts).
        Recommended for CLI runs so paths match ``--project_root``. If None, uses ``path`` layout
        above (notebook default).
    resume_from_checkpoints : bool or None
        If None, auto-enable when any ``hce_logo_fold_*.pt`` exists in the fold checkpoint dir.

    Returns
    -------
    dict
        Keys: ``logo_summary``, ``model``, ``best_fold``, ``best_weighted_f1``, ``best_macro_f1``,
        ``best_epoch``, ``input_dim``, ``num_classes``, ``logo_ckpt_pattern``.
    """
    from pathlib import Path

    if logo_checkpoint_dir is not None:
        logo_dir = Path(logo_checkpoint_dir).expanduser().resolve()
        logo_ckpt_pattern = str(logo_dir / "hce_logo_fold_{}.pt")
    else:
        logo_ckpt_pattern = (
            f"{path}Collaborate/esccAI/data/{therapy_data}/{therapy_model}/hce_logo_fold_{{}}.pt"
        )
        logo_dir = Path(f"{path}Collaborate/esccAI/data/{therapy_data}/{therapy_model}")
    if resume_from_checkpoints is None:
        resume_from_checkpoints = any(logo_dir.glob("hce_logo_fold_*.pt"))

    loader_kwargs = loader_kwargs or {"seed": 42, "train_balance_sampler": False}

    logo_summary = train_hce_leave_one_group_out_cv(
        device=device,
        cv_data=cv_data,
        evaluate=evaluate,
        class_names=class_names,
        save_bestmodel_path_pattern=logo_ckpt_pattern,
        loader_kwargs=loader_kwargs,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        patience=patience,
        max_epochs=max_epochs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
        resume_from_checkpoints=resume_from_checkpoints,
    )

    best_fold = max(
        logo_summary["folds"],
        key=lambda r: _validation_selection_score(r, cv_selection_metric),
    )
    best_fold_score = float(_validation_selection_score(best_fold, cv_selection_metric))
    print(
        "Best fold selection: "
        f"metric={cv_selection_metric}, "
        f"fold={best_fold.get('fold')}, "
        f"score={best_fold_score:.6f}"
    )
    input_dim = int(cv_data["X_f"].shape[1])
    num_classes = len(class_names)
    ckpt_path = best_fold["checkpoint"]
    state_dict = torch.load(ckpt_path, map_location=device)
    model, use_five = build_mlp_classifier_for_state_dict(
        state_dict,
        input_dim=input_dim,
        num_classes=num_classes,
        num_level1_classes=int(np.max(cv_data["y_level1_encoded_f"])) + 1,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(_remap_legacy_five_head_state_dict(state_dict))
    model.eval()

    return {
        "logo_summary": logo_summary,
        "model": model,
        "best_fold": best_fold,
        "best_weighted_f1": float(best_fold["l2_weighted_f1"]),
        "best_macro_f1": float(best_fold["l2_macro_f1"]),
        "best_epoch": int(best_fold["best_epoch"]),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "logo_ckpt_pattern": logo_ckpt_pattern,
        "hce_w1": float(hce_w1),
        "hce_w2": float(hce_w2),
        "hce_w12": float(hce_w12),
        "hce_w_l12head": float(hce_w_l12head),
        "hce_w_l3": float(hce_w_l3),
        "hce_w_l4": float(hce_w_l4),
        "use_five_head": use_five,
    }


def report_logo_cv_means_and_insample_eval(
    device,
    cv_data,
    scaler,
    class_names,
    model,
    evaluate,
    logo_summary,
    *,
    batch_size=1024,
    print_level2_classification_report=True,
    neighbor_index=None,
):
    """
    Print LOGO aggregate (mean ± std), then evaluate the given ``model`` on **all** filtered
    cells (in-sample; not OOF). Computes L2 metrics via ``evaluate`` and L1 via hierarchical softmax.

    Returns
    -------
    dict
        L2/L1 metrics and prediction arrays for confusion-matrix / per-class plots.
        ``val_preds_level1`` is argmax of L2-softmax aggregated to L1 (hierarchy matrix).
        ``val_preds_level1_head`` is argmax of the dedicated L1 logits head when the model
        implements ``forward_heads`` (else ``None``).
        Five-head models also return ``val_preds_level12`` / ``val_preds_level3`` /
        ``val_preds_level4`` and matching ``val_labels_*`` when ``cv_data`` contains those tiers.
    """
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    print(
        "LOGO (mean ± std over folds): L2 macro F1 "
        f"{logo_summary['l2_macro_f1_mean']:.4f} ± {logo_summary['l2_macro_f1_std']:.4f}, "
        f"L2 weighted F1 {logo_summary['l2_weighted_f1_mean']:.4f} ± "
        f"{logo_summary['l2_weighted_f1_std']:.4f}"
    )
    #######################################################################################
    ## 2026.05.12 LLY check val_preds, val_labels
    X_all_s = scaler.transform(cv_data["X_f"])
    pin = torch.cuda.is_available()
    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None
    spatial_ctx = None
    if use_spatial:
        spatial_ctx = {
            "X_all": torch.as_tensor(X_all_s, dtype=torch.float32),
            "neighbor_index": torch.as_tensor(neighbor_index, dtype=torch.long),
        }
        all_ds = TensorDataset(
            torch.as_tensor(X_all_s, dtype=torch.float32),
            torch.as_tensor(cv_data["y_encoded_f"], dtype=torch.long),
            torch.as_tensor(cv_data["y_level1_encoded_f"], dtype=torch.long),
            torch.arange(len(cv_data["y_encoded_f"]), dtype=torch.long),
        )
    else:
        all_ds = TensorDataset(
            torch.as_tensor(X_all_s, dtype=torch.float32),
            torch.as_tensor(cv_data["y_encoded_f"], dtype=torch.long),
            torch.as_tensor(cv_data["y_level1_encoded_f"], dtype=torch.long),
        )
    all_loader = DataLoader(all_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    val_acc, val_macro_f1, val_weighted_f1, val_preds, val_labels = evaluate(
        model, all_loader, device, spatial_ctx=spatial_ctx
    )
    #######################################################################################

    num_classes = len(class_names)
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(cv_data["y_encoded_f"], cv_data["y_level1_encoded_f"]):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
    num_level1_classes = int(np.max(cv_data["y_level1_encoded_f"])) + 1
    map_l2_to_l1 = torch.zeros(num_classes, num_level1_classes, dtype=torch.float32, device=device)
    for k, p in enumerate(child_to_parent):
        if p >= 0:
            map_l2_to_l1[k, int(p)] = 1.0

    #######################################################################################
    ## 2026.05.12 LLY check val_preds_level1 from level1_head [NO], or map_l2_to_l1 [YES]?
    model.eval()
    pl, yl, pl_head = [], [], []
    has_l1_head = hasattr(model, "forward_heads") and callable(getattr(model, "forward_heads"))
    with torch.no_grad():
        for batch in all_loader:
            if use_spatial:
                x, _, y_l1, global_idx = batch
            else:
                x, _, y_l1 = batch
                global_idx = None
            x = x.to(device, non_blocking=True)
            y_l1 = y_l1.to(device, non_blocking=True)
            neighbor_x = _neighbor_x_from_batch(spatial_ctx, global_idx, device)
            if has_l1_head:
                logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                logits_l2 = logits_l2.float()
                logits_l1 = logits_l1.float()
                pl_head.append(torch.argmax(logits_l1, dim=1).cpu().numpy())
            else:
                logits_l2 = model(x, neighbor_x=neighbor_x).float()
            probs_l2 = torch.softmax(logits_l2, dim=1)
            probs_l1 = torch.matmul(probs_l2, map_l2_to_l1)
            pl.append(torch.argmax(probs_l1, dim=1).cpu().numpy())
            yl.append(y_l1.cpu().numpy())
    #######################################################################################
    val_preds_level1 = np.concatenate(pl)
    val_labels_level1 = np.concatenate(yl)
    val_preds_level1_head = np.concatenate(pl_head) if has_l1_head else None
    val_level1_acc = accuracy_score(val_labels_level1, val_preds_level1)
    val_level1_macro_f1 = f1_score(
        val_labels_level1, val_preds_level1, average="macro", zero_division=0
    )
    val_level1_weighted_f1 = f1_score(
        val_labels_level1, val_preds_level1, average="weighted", zero_division=0
    )

    print(
        "\nIn-sample (all cells, best-fold model): "
        f"L2 acc={val_acc:.4f}, macro F1={val_macro_f1:.4f}, weighted F1={val_weighted_f1:.4f}"
    )

    if print_level2_classification_report:
        unique_labels = np.unique(np.concatenate([val_labels, val_preds]))
        names_l2 = [class_names[i] for i in unique_labels]
        print("\nClassification report L2 (in-sample):\n")
        print(
            classification_report(
                val_labels,
                val_preds,
                labels=unique_labels,
                target_names=names_l2,
                zero_division=0,
            )
        )

    out = {
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "val_level1_acc": val_level1_acc,
        "val_level1_macro_f1": val_level1_macro_f1,
        "val_level1_weighted_f1": val_level1_weighted_f1,
        "val_preds_level1": val_preds_level1,
        "val_preds_level1_head": val_preds_level1_head,
        "val_labels_level1": val_labels_level1,
    }

    if hasattr(model, "level12_head"):
        head_preds = predict_all_label_heads(
            model, X_all_s, device, batch_size=batch_size, neighbor_index=neighbor_index
        )
        if val_preds_level1_head is None and "preds_l1" in head_preds:
            out["val_preds_level1_head"] = head_preds["preds_l1"]
        for short, pred_key in (
            ("level12", "preds_l12"),
            ("level3", "preds_l3"),
            ("level4", "preds_l4"),
        ):
            y_key = f"y_{short}_encoded_f"
            if y_key in cv_data and pred_key in head_preds:
                out[f"val_preds_{short}"] = head_preds[pred_key]
                out[f"val_labels_{short}"] = cv_data[y_key]

    return out


def run_logo_cv_with_insample_report(
    device,
    cv_data,
    scaler,
    class_names,
    evaluate,
    path,
    therapy_data,
    therapy_model,
    *,
    hce_w1=1.0,
    hce_w2=1.0,
    hce_w12=1.0,
    hce_w_l12head=1.0,
    hce_w_l3=1.0,
    hce_w_l4=1.0,
    patience=10,
    max_epochs=50,
    loader_kwargs=None,
    resume_from_checkpoints=None,
    logo_checkpoint_dir=None,
    hidden_dims=(768, 384, 192),
    dropout=0.2,
    lr=1e-3,
    weight_decay=5e-5,
    cv_selection_metric="five_tier_auc_sum",
    l2_label_smoothing=0.0,
    l1_label_smoothing=0.0,
    val_selection_metric="five_tier_auc_sum",
    class_weight_mode="balanced",
    class_balanced_beta=0.9999,
    l2_focal_gamma=0.0,
    insample_batch_size=1024,
    print_level2_classification_report=True,
):
    """
    Convenience wrapper: ``run_logo_cv_and_load_best_model`` then
    ``report_logo_cv_means_and_insample_eval``.

    ``resume_from_checkpoints`` is forwarded to ``run_logo_cv_and_load_best_model`` /
    ``train_hce_leave_one_group_out_cv``: per fold, if ``hce_logo_fold_{k}.pt`` already exists,
    that fold **does not train** (load weights + validation metrics only). When every fold file
    is present, the full LOGO loop runs without training. Use ``False`` to force retrain all folds.

    ``logo_checkpoint_dir`` is passed through to ``run_logo_cv_and_load_best_model`` (optional
    absolute directory for fold ``.pt`` files).

    Returns
    -------
    dict
        Merges the return dicts of both steps (no key overlap): model, logo_summary, best_fold,
        input_dim, num_classes, val_acc, val_preds, …
    """
    ctx = run_logo_cv_and_load_best_model(
        device,
        cv_data,
        evaluate,
        class_names,
        path,
        therapy_data,
        therapy_model,
        hce_w1=hce_w1,
        hce_w2=hce_w2,
        hce_w12=hce_w12,
        hce_w_l12head=hce_w_l12head,
        hce_w_l3=hce_w_l3,
        hce_w_l4=hce_w_l4,
        patience=patience,
        max_epochs=max_epochs,
        loader_kwargs=loader_kwargs,
        resume_from_checkpoints=resume_from_checkpoints,
        logo_checkpoint_dir=logo_checkpoint_dir,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        cv_selection_metric=cv_selection_metric,
        l2_label_smoothing=l2_label_smoothing,
        l1_label_smoothing=l1_label_smoothing,
        val_selection_metric=val_selection_metric,
        class_weight_mode=class_weight_mode,
        class_balanced_beta=class_balanced_beta,
        l2_focal_gamma=l2_focal_gamma,
    )
    ins = report_logo_cv_means_and_insample_eval(
        device,
        cv_data,
        scaler,
        class_names,
        ctx["model"],
        evaluate,
        ctx["logo_summary"],
        batch_size=insample_batch_size,
        print_level2_classification_report=print_level2_classification_report,
    )
    merged = {**ctx, **ins}
    return merged


######################################################
# 2026.03.23 For model validation: LLY Adjust model to use level1 and level2
######################################################
# Override mode_validation: report both level2 and level1 metrics.
def mode_validation(
    val_loader,
    device,
    path,
    therapy_data,
    X_train_scaled,
    y_train_encoded,
    y_test_encoded,
    y_encoded_f=None,
    y_level1_encoded_f=None,
    class_names=None,
    class_names_level1=None,
    val_loader_eval=None,
    therapy_model=None,
    hidden_dims=(1024, 512, 256),
    dropout=0.2,
):
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    input_dim, num_classes = infer_input_dim_and_num_classes(
        X_train_scaled,
        y_train_encoded,
        y_test_encoded,
        class_names=class_names,
    )

    if y_encoded_f is None or y_level1_encoded_f is None:
        raise ValueError("y_encoded_f and y_level1_encoded_f are required.")

    checkpoint_path = get_best_checkpoint_path(path, therapy_data, therapy_model=therapy_model)
    state_dict = torch.load(checkpoint_path, map_location=device)

    num_l1 = int(np.max(y_level1_encoded_f)) + 1
    if "level1_head.weight" in state_dict:
        model = build_mlp_classifier(
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_level1_classes=num_l1,
            use_dual_head=True,
        )
    else:
        model = build_mlp_classifier(
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_level1_classes=None,
            use_dual_head=False,
        )
    model.load_state_dict(state_dict)

    # Level2 validation
    eval_loader = val_loader_eval if val_loader_eval is not None else val_loader
    val_acc, val_macro_f1, val_weighted_f1, val_preds, val_labels = evaluate(model, eval_loader, device)

    print("\nValidation Performance (Level2, best checkpoint):")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Macro F1-score: {val_macro_f1:.4f}")
    print(f"  Weighted F1-score: {val_weighted_f1:.4f}")

    unique_labels = np.unique(np.concatenate([val_labels, val_preds]))
    actual_class_names = [class_names[i] for i in unique_labels] if class_names is not None else [str(i) for i in unique_labels]
    print("\nDetailed Classification Report (Level2):")
    print(classification_report(val_labels, val_preds, labels=unique_labels, target_names=actual_class_names, zero_division=0))

    # Build level2->level1 mapping
    child_to_parent = np.full(num_classes, -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        if child_to_parent[int(l2)] == -1:
            child_to_parent[int(l2)] = int(l1)
    if np.any(child_to_parent < 0):
        missing = np.where(child_to_parent < 0)[0]
        raise ValueError(f"Missing level1 mapping for level2 classes in validation: {missing}")

    num_level1_classes = int(np.max(child_to_parent)) + 1
    map_l2_to_l1 = torch.zeros(num_classes, num_level1_classes, dtype=torch.float32, device=device)
    for k, p in enumerate(child_to_parent):
        map_l2_to_l1[k, int(p)] = 1.0

    # Level1 validation by aggregating level2 probabilities
    model.eval()
    val_preds_level1_list, val_labels_level1_list = [], []
    with torch.no_grad():
        for x, _, y_l1 in val_loader:
            x = x.to(device, non_blocking=True)
            y_l1 = y_l1.to(device, non_blocking=True)

            logits_l2 = model(x).float()
            probs_l2 = torch.softmax(logits_l2, dim=1)
            probs_l1 = torch.matmul(probs_l2, map_l2_to_l1)
            preds_l1 = torch.argmax(probs_l1, dim=1)

            val_preds_level1_list.append(preds_l1.cpu().numpy())
            val_labels_level1_list.append(y_l1.cpu().numpy())

    val_preds_level1 = np.concatenate(val_preds_level1_list)
    val_labels_level1 = np.concatenate(val_labels_level1_list)

    val_level1_acc = accuracy_score(val_labels_level1, val_preds_level1)
    val_level1_macro_f1 = f1_score(val_labels_level1, val_preds_level1, average="macro", zero_division=0)
    val_level1_weighted_f1 = f1_score(val_labels_level1, val_preds_level1, average="weighted", zero_division=0)

    print("\nValidation Performance (Level1):")
    print(f"  Accuracy: {val_level1_acc:.4f}")
    print(f"  Macro F1-score: {val_level1_macro_f1:.4f}")
    print(f"  Weighted F1-score: {val_level1_weighted_f1:.4f}")

    unique_labels_l1 = np.unique(np.concatenate([val_labels_level1, val_preds_level1]))
    actual_class_names_l1 = [class_names_level1[i] for i in unique_labels_l1] if class_names_level1 is not None else [str(i) for i in unique_labels_l1]
    print("\nDetailed Classification Report (Level1):")
    print(classification_report(val_labels_level1, val_preds_level1, labels=unique_labels_l1, target_names=actual_class_names_l1, zero_division=0))

    return (
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
    )


def mode_validation_from_split(
    device,
    path,
    therapy_data,
    loaders,
    result,
    class_names=None,
    class_names_level1=None,
    therapy_model=None,
    hidden_dims=(1024, 512, 256),
    dropout=0.2,
):
    """
    Convenience wrapper: validate directly from split/result + loaders dict.
    """
    if "val_loader" not in loaders:
        raise KeyError("Missing loaders['val_loader']")

    required_result_keys = [
        "X_train_scaled",
        "y_train_encoded",
        "y_test_encoded",
        "y_encoded_f",
        "y_level1_encoded_f",
    ]
    for k in required_result_keys:
        if k not in result:
            raise KeyError(f"Missing result['{k}']")

    return mode_validation(
        val_loader=loaders["val_loader"],
        device=device,
        path=path,
        therapy_data=therapy_data,
        therapy_model=therapy_model,
        X_train_scaled=result["X_train_scaled"],
        y_train_encoded=result["y_train_encoded"],
        y_test_encoded=result["y_test_encoded"],
        y_encoded_f=result["y_encoded_f"],
        y_level1_encoded_f=result["y_level1_encoded_f"],
        class_names=class_names,
        class_names_level1=class_names_level1,
        val_loader_eval=loaders.get("val_loader_eval", None),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


######################################################
# 2026.03.23 LLY Load best model for StarDist prediction
######################################################
def infer_hidden_dims_improved_mlp(state_dict):
    """
    Reconstruct ``hidden_dims`` for ``ImprovedMLPClassifier`` from ``state_dict``
    (``input_layer.0`` and each ``hidden_layers.*.0`` Linear out_features chain).
    """
    in_key = "input_layer.0.weight"
    if in_key not in state_dict:
        raise KeyError(f"Expected {in_key!r} in state_dict (ImprovedMLPClassifier).")
    dims = [int(state_dict[in_key].shape[0])]
    k = 0
    while True:
        wkey = f"hidden_layers.{k}.0.weight"
        if wkey not in state_dict:
            break
        dims.append(int(state_dict[wkey].shape[0]))
        k += 1
    out_in = int(state_dict["output_layer.weight"].shape[1])
    if dims[-1] != out_in:
        raise ValueError(
            f"Inferred hidden widths end with {dims[-1]} but output_layer expects "
            f"in_features={out_in}."
        )
    return dims


def predict_all_label_heads(model, X, device, batch_size=4096, neighbor_index=None):
    """
    Batch inference for all supervised heads.

    Returns dict with keys ``preds_l2``, ``preds_l1`` (L1 head), and when the checkpoint is
    five-head: ``preds_l12``, ``preds_l3`` (CNiche), ``preds_l4`` (TNiche).

    When the model was trained with ``use_spatial_context=True``, pass ``neighbor_index``
    (``(N, K)`` from ``build_spatial_neighbor_index``) so neighbor embeddings are fused at inference.
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    n = X_t.shape[0]
    use_spatial = bool(getattr(model, "use_spatial_context", False)) and neighbor_index is not None
    nbr_idx_t = (
        torch.as_tensor(neighbor_index, dtype=torch.long) if use_spatial else None
    )
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks = {k: [] for k in ("l2", "l1", "l12", "l3", "l4")}
    five_head = hasattr(model, "level12_head")
    row_offset = 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            neighbor_x = None
            if use_spatial:
                b = x.shape[0]
                global_idx = torch.arange(row_offset, row_offset + b, dtype=torch.long, device=device)
                neighbor_x = gather_neighbor_embeddings(
                    X_t.to(device), nbr_idx_t.to(device), global_idx, device=device
                )
                row_offset += b
            if five_head:
                logits = model.forward_heads(x, neighbor_x=neighbor_x)
                names = ("l2", "l1", "l12", "l3", "l4")
                for name, lg in zip(names, logits):
                    chunks[name].append(torch.argmax(lg, dim=1).cpu().numpy())
            else:
                logits_l2, logits_l1 = forward_heads_l2_l1(model, x, neighbor_x=neighbor_x)
                chunks["l2"].append(torch.argmax(logits_l2, dim=1).cpu().numpy())
                chunks["l1"].append(torch.argmax(logits_l1, dim=1).cpu().numpy())

    out = {
        "preds_l2": np.concatenate(chunks["l2"]),
        "preds_l1": np.concatenate(chunks["l1"]),
    }
    if five_head:
        out["preds_l12"] = np.concatenate(chunks["l12"])
        out["preds_l3"] = np.concatenate(chunks["l3"])
        out["preds_l4"] = np.concatenate(chunks["l4"])
    return out


def load_model_for_predict(
    path,
    therapy_data,
    therapy_model,
    model_class=ImprovedMLPClassifier,
    device=None,
    hidden_dims=None,
    dropout=0.2,
    checkpoint_path=None,
    parent_dir=False,
    run_log_path=None,
):
    """
    Load the best model for StarDist prediction.

    Args:
        path (str): Base directory.
        therapy_data (str): Dataset identifier.
        therapy_model (str): Model name or identifier.
        model_class (nn.Module): Model class to instantiate.
        device (torch.device, optional): Device to load the model onto. If None, automatically selected.
        hidden_dims (sequence of int, optional): MLP hidden widths. If None, inferred from checkpoint
            so architecture matches training (e.g. ``[256, 256, 256]`` vs ``[1024, 512, 256]``).
        dropout (float): Dropout for ``model_class`` construction (default 0.2, match training).
        checkpoint_path (str, optional): Explicit ``.pt`` path. If provided, this file is loaded
            directly; otherwise ``get_best_checkpoint_path(...)`` (``best_mlp_gpu.pt``). This default
            file is synchronized from the selected best LOGO fold in
            ``run_logo_cv_and_load_best_model``.
        parent_dir (bool): Checkpoint resolution mode when ``checkpoint_path`` is not provided.
            ``True`` loads from ``.../data/{therapy_data}/{therapy_model}/best_mlp_gpu.pt``.
            ``False`` (default) loads from the run timestamp folder (same directory as ``run.log``).
        run_log_path (str, optional): Explicit ``run.log`` path used when ``parent_dir=False``.
            If omitted, the most recently modified ``run.log`` under the therapy model directory is used.

    Returns:
        model_star (nn.Module): Loaded model.
    """
    print("Loading best model for StarDist prediction...")

    if checkpoint_path is None:
        if parent_dir:
            checkpoint_path = get_best_checkpoint_path(path, therapy_data, therapy_model)
        else:
            checkpoint_path = get_run_dir_checkpoint_path(
                path, therapy_data, therapy_model, run_log_path=run_log_path
            )
    print(f"  Checkpoint file: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Expected the selected best LOGO fold to be synced to best_mlp_gpu.pt.\n"
            "Run run_logo_cv_and_load_best_model / run_logo_cv_with_insample_report first, or run\n"
            "  sync_best_mlp_from_logo_fold(path, therapy_data, therapy_model, LP['best_fold']['checkpoint'])\n"
            "or pass checkpoint_path explicitly."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(checkpoint_path, map_location=device)

    input_dim = int(state_dict["input_layer.0.weight"].shape[1])
    num_classes = int(state_dict["output_layer.weight"].shape[0])
    if hidden_dims is None:
        hidden_dims_use = infer_hidden_dims_improved_mlp(state_dict)
        print(f"  Inferred hidden_dims from checkpoint: {hidden_dims_use}")
    else:
        hidden_dims_use = list(hidden_dims)
        print(f"  Using provided hidden_dims: {hidden_dims_use}")

    if "level1_head.weight" in state_dict or _checkpoint_is_five_head(state_dict):
        num_l1 = int(state_dict["level1_head.weight"].shape[0])
        model_star, _ = build_mlp_classifier_for_state_dict(
            state_dict,
            input_dim=input_dim,
            num_classes=num_classes,
            num_level1_classes=num_l1,
            device=device,
            hidden_dims=hidden_dims_use,
            dropout=dropout,
        )
    else:
        model_star = model_class(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims_use,
            dropout=dropout,
        ).to(device)

    model_star.load_state_dict(_remap_legacy_five_head_state_dict(state_dict))

    print("✓ StarDist model loaded")
    return model_star