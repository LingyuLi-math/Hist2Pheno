"""Shared helpers for NCRT level012 notebooks (insample + StarDist multi-tier plots).

Internal tier keys match five-head training (``level3`` = level0, ``level4`` = level01).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# (short, head_pred_key, labels_glob, preds_glob, names_glob, y_f_key in cv_data)
EXTRA_TIER_SPECS = (
    ("level12", "preds_l12", "val_labels_level12", "val_preds_level12", "class_names_level12", "y_level12_f"),
    ("level3", "preds_l3", "val_labels_level3", "val_preds_level3", "class_names_level3", "y_level3_f"),
    ("level4", "preds_l4", "val_labels_level4", "val_preds_level4", "class_names_level4", "y_level4_f"),
)

STARDIST_EXTRA_SPECS = (
    ("Level12 sublineage", "preds_l12", "class_names_level12", "y_star_level12", "le_level12"),
    ("Level0 compartment", "preds_l3", "class_names_level3", "y_star_level3", "le_level3"),
    ("Level01 bucket", "preds_l4", "class_names_level4", "y_star_level4", "le_level4"),
)

# Display title, pred key, names key, y_f key, fig pred, fig true, celltype_col for NCRT colors
HE_SPATIAL_TIER_ROWS = (
    ("Level12 sublineage", "val_preds_level12", "class_names_level12", "y_level12_f", "celltype_valid_level12", "celltype_true_level12", "celltype_level12"),
    ("Level0 compartment", "val_preds_level3", "class_names_level3", "y_level3_f", "celltype_valid_level0", "celltype_true_level0", "celltype_level0"),
    ("Level01 bucket", "val_preds_level4", "class_names_level4", "y_level4_f", "celltype_valid_level01", "celltype_true_level01", "celltype_level01"),
)


def tier_class_names(names_key, cv_data=None, g=None):
    g = g or {}
    names = g.get(names_key)
    if names is None and cv_data is not None:
        names = cv_data.get(names_key)
    return names


def ensure_lp_extra_insample_preds(model, scaler, cv_data, device, predict_all_label_heads, g=None):
    """Populate globals val_preds/val_labels for L12/L0/L01 (+ L1 head) from best-fold model."""
    g = g if g is not None else {}
    if g.get("val_preds_level12") is not None:
        return
    if not hasattr(model, "level12_head"):
        return
    X_ins = scaler.transform(cv_data["X_f"])
    head_preds = predict_all_label_heads(model, X_ins, device)
    for short, pk in (("level12", "preds_l12"), ("level3", "preds_l3"), ("level4", "preds_l4")):
        yk = f"y_{short}_encoded_f"
        if yk in cv_data and pk in head_preds:
            g[f"val_preds_{short}"] = head_preds[pk]
            g[f"val_labels_{short}"] = cv_data[yk]
    if g.get("val_preds_level1_head") is None and "preds_l1" in head_preds:
        g["val_preds_level1_head"] = head_preds["preds_l1"]


def stardist_head_preds(model_star, scaler, X_star, device, predict_all_label_heads):
    if not hasattr(model_star, "level12_head"):
        return None
    return predict_all_label_heads(model_star, scaler.transform(X_star), device)


_STARDIST_Y_KEYS = ("y_star_level12", "y_star_level3", "y_star_level4")
NCRT_LEVEL0_LABELS = frozenset({"Immune", "Stromal", "Tumor", "Epithelial"})
NCRT_LEVEL01_LABELS = frozenset({"B", "T", "Stromal", "Myeloid", "Tumor", "Epithelial", "Immune"})


def _ncrt_tier_tag_from_title(title: str) -> str:
    """Return ``Level12`` / ``Level01`` / ``Level0``; check Level01 before Level0 (substring trap)."""
    if "Level12" in title:
        return "Level12"
    if "Level01" in title:
        return "Level01"
    if "Level0" in title:
        return "Level0"
    raise ValueError(f"Unknown NCRT tier in title: {title!r}")


def _unique_label_set(y_arr):
    if y_arr is None:
        return set()
    return {str(x).strip() for x in np.asarray(y_arr).astype(str) if str(x).strip()}


def _is_level0_label_set(label_set):
    return bool(label_set) and label_set.issubset(NCRT_LEVEL0_LABELS)


def _is_level01_label_set(label_set):
    return bool(label_set) and label_set.issubset(NCRT_LEVEL01_LABELS) and bool(
        label_set & {"B", "T", "Myeloid"}
    )


def encode_star_labels(y_raw, le):
    return encode_star_tier_labels(y_raw, le.classes_, le=le)


def encode_star_tier_labels(y_raw, class_names, le=None):
    """Encode string tier labels in ``class_names`` order; drop unknown with a warning."""
    names = [str(c) for c in class_names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    if le is not None:
        name_to_idx = {str(c): int(i) for i, c in enumerate(le.classes_)}
    encoded = []
    bad = 0
    for v in np.asarray(y_raw).astype(str):
        idx = name_to_idx.get(v.strip(), -1)
        if idx < 0:
            bad += 1
            idx = 0
        encoded.append(idx)
    if bad:
        print(f"  ⚠ {bad} StarDist labels not in training class list {names!r}; mapped to index 0")
    return np.asarray(encoded, dtype=np.int64)


def metrics_triplet(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_he_confusion_matrices(plot_confusion_matrix, result_fig, cv_data=None, g=None):
    """In-sample confusion matrices for L2/L1/L12/Level0/Level01 (+ L1 head if available)."""
    g = g or {}
    rows = [
        ("L2 celltype", "val_labels", "val_preds", "class_names", "conf_matrix_level2", (10, 8)),
        ("L1 lineage (L2→L1 agg)", "val_labels_level1", "val_preds_level1", "class_names_level1", "conf_matrix_level1", (4, 4)),
        ("L1 lineage (L1 head)", "val_labels_level1", "val_preds_level1_head", "class_names_level1", "conf_matrix_level1_L1head", (4, 4)),
        ("Level12 sublineage", "val_labels_level12", "val_preds_level12", "class_names_level12", "conf_matrix_level12", (6, 5)),
        ("Level0 compartment", "val_labels_level3", "val_preds_level3", "class_names_level3", "conf_matrix_level0", (4, 4)),
        ("Level01 bucket", "val_labels_level4", "val_preds_level4", "class_names_level4", "conf_matrix_level01", (5, 4)),
    ]
    for title, lk, pk, nk, fig_key, figsize in rows:
        y_true, y_pred = g.get(lk), g.get(pk)
        if y_pred is None:
            print(f"Skip {title}: no {pk} (re-run ablation with five-head labels, or model has no extra heads)")
            continue
        names = tier_class_names(nk, cv_data, g)
        if names is None:
            print(f"Skip {title}: missing {nk}")
            continue
        print(f"\n{title}")
        plot_confusion_matrix(y_true, y_pred, names, figsize=figsize, save_path=result_fig(fig_key))
    if g.get("val_preds_level1_head") is not None:
        print("L1 agg vs L1 head agreement:", (g["val_preds_level1"] == g["val_preds_level1_head"]).mean())


def plot_he_spatial_extra_tiers(
    plot_tier_spatial_distribution,
    plot_celltype_spatial_distribution,
    result_fig,
    therapy_data,
    X_coords_plot,
    cv_data,
    g=None,
):
    g = g or {}
    for title, pk, nk, yk, fig_pred, fig_true, ct_col in HE_SPATIAL_TIER_ROWS:
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
            spatial_color_scheme="ncrt",
            celltype_col=ct_col,
            title_pred=f"{therapy_data} predicted {title}",
            title_true=f"{therapy_data} ground truth {title}",
            fig_size=(10, 8),
            show=True,
            X_coords_matched=X_coords_plot,
            y_tier_f=y_f,
        )


def plot_he_f1_extra_tiers(plot_per_class_f1, result_fig, model, device, train_dataset, val_dataset, input_dim, best_epoch, g=None):
    g = g or {}
    rows = (
        ("Level12 sublineage", "val_labels_level12", "val_preds_level12", "class_names_level12", "f1_perclass_level12", "MLP-Level12"),
        ("Level0 compartment", "val_labels_level3", "val_preds_level3", "class_names_level3", "f1_perclass_level0", "MLP-Level0"),
        ("Level01 bucket", "val_labels_level4", "val_preds_level4", "class_names_level4", "f1_perclass_level01", "MLP-Level01"),
    )
    for title, lk, pk, nk, fig_key, mname in rows:
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



def stardist_tier_labels(head_preds, cv_data, g):
    g = g or {}
    for (title, pk, nk, _yk, lek), y_star_key in zip(STARDIST_EXTRA_SPECS, _STARDIST_Y_KEYS):
        if pk not in head_preds:
            print(f"Skip {title}: missing {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        le = cv_data.get(lek) if cv_data is not None else None
        y_raw = g.get(y_star_key)
        if names is None or le is None or y_raw is None:
            print(f"Skip {title}: need {nk}, {lek}, {y_star_key}")
            continue
        u = _unique_label_set(y_raw)
        tag = _ncrt_tier_tag_from_title(title)
        if tag == "Level01" and _is_level0_label_set(u) and not _is_level01_label_set(u):
            print(
                f"  ⚠ {y_star_key} looks like Level0 labels {sorted(u)!r}; "
                "re-run ensure_stardist_hierarchy_labels() before plotting."
            )
            continue
        if tag == "Level0" and _is_level01_label_set(u) and not _is_level0_label_set(u):
            print(
                f"  ⚠ {y_star_key} looks like Level01 labels {sorted(u)!r}; "
                "re-run ensure_stardist_hierarchy_labels() before plotting."
            )
            continue
        print(f"  Ground-truth tier check ({title}): {sorted(u)!r}")
        y_true = encode_star_tier_labels(y_raw, names, le=le)
        y_pred = np.asarray(head_preds[pk], dtype=np.int64)
        n = min(len(y_true), len(y_pred))
        yield title, y_true[:n], y_pred[:n], names


def plot_stardist_confusion_extra(plot_confusion_matrix, result_fig, head_preds, cv_data, g=None):
    g = g or {}
    key_map = {
        "Level12": "conf_matrix_level12_stardist",
        "Level0": "conf_matrix_level0_stardist",
        "Level01": "conf_matrix_level01_stardist",
    }
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g):
        tag = _ncrt_tier_tag_from_title(title)
        figsize = (6, 5) if tag == "Level12" else (5, 4)
        fig_key = key_map[tag]
        print(f"\n{title} (StarDist) confusion matrix")
        plot_confusion_matrix(y_true, y_pred, names, figsize=figsize, save_path=result_fig(fig_key))
        print(f"Saved: {result_fig(fig_key)}")


def plot_stardist_f1_extra(plot_per_class_f1, result_fig, model_star, device, head_preds, cv_data, g=None):
    g = g or {}
    key_map = {"Level12": "f1_level12_stardist", "Level0": "f1_level0_stardist", "Level01": "f1_level01_stardist"}
    name_map = {"Level12": "MLP-StarDist-Level12", "Level0": "MLP-StarDist-Level0", "Level01": "MLP-StarDist-Level01"}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g):
        tag = _ncrt_tier_tag_from_title(title)
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


def plot_stardist_acc_extra(plot_per_class_accuracy, result_fig, model_star, device, head_preds, cv_data, g=None):
    g = g or {}
    key_map = {"Level12": "acc_level12_stardist", "Level0": "acc_level0_stardist", "Level01": "acc_level01_stardist"}
    name_map = {"Level12": "MLP-StarDist-Level12", "Level0": "MLP-StarDist-Level0", "Level01": "MLP-StarDist-Level01"}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g):
        tag = _ncrt_tier_tag_from_title(title)
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
):
    g = g or {}
    fig_map = {
        "Level12": ("stardist_pred_level12", "stardist_true_level12", "celltype_level12"),
        "Level0": ("stardist_pred_level0", "stardist_true_level0", "celltype_level0"),
        "Level01": ("stardist_pred_level01", "stardist_true_level01", "celltype_level01"),
    }
    y_key_map = {"Level12": "y_star_level12", "Level0": "y_star_level3", "Level01": "y_star_level4"}
    for title, y_true, y_pred, names in stardist_tier_labels(head_preds, cv_data, g):
        tag = _ncrt_tier_tag_from_title(title)
        y_key = y_key_map[tag]
        fp, ft, ct_col = fig_map[tag]
        print(f"\n{title} (StarDist)")
        plot_tier_spatial_distribution(
            pred_encoded=y_pred,
            class_names=names,
            plot_celltype_spatial_distribution=plot_celltype_spatial_distribution,
            save_path_pred=result_fig(fp),
            save_path_true=result_fig(ft),
            spatial_color_scheme="ncrt",
            celltype_col=ct_col,
            title_pred=f"{therapy_data} StarDist pred {title}",
            title_true=f"{therapy_data} ground truth {title}",
            fig_size=(10, 8),
            show=True,
            X_coords_matched=X_coords_star,
            y_tier_f=g.get(y_key),
        )
        print(f"Saved: {result_fig(fp)}")


def plot_stardist_roc_extra(plot_multiclass_roc_curves, result_fig, head_probs, cv_data, m, g=None):
    g = g or {}
    roc_specs = (
        ("Level12 sublineage", "l12", "class_names_level12", "y_star_level12", "le_level12", "roc_stardist_level12", "StarDist Level12 ROC", "celltype_level12"),
        ("Level0 compartment", "l3", "class_names_level3", "y_star_level3", "le_level3", "roc_stardist_level0", "StarDist Level0 ROC", "celltype_level0"),
        ("Level01 bucket", "l4", "class_names_level4", "y_star_level4", "le_level4", "roc_stardist_level01", "StarDist Level01 ROC", "celltype_level01"),
    )
    for title, pk, nk, yk, lek, fig_key, roc_title, color_tier in roc_specs:
        if pk not in head_probs:
            print(f"Skip {title} ROC: missing probs {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        le = cv_data.get(lek) if cv_data is not None else None
        y_raw = g.get(yk)
        if names is None or le is None or y_raw is None:
            print(f"Skip {title} ROC: need {nk}, {lek}, {yk}")
            continue
        u = _unique_label_set(y_raw)
        tag = _ncrt_tier_tag_from_title(title)
        if tag == "Level01" and _is_level0_label_set(u) and not _is_level01_label_set(u):
            print(
                f"Skip {title} ROC: {yk} has Level0 labels {sorted(u)!r}. "
                "Call ensure_stardist_hierarchy_labels() after StarDist match."
            )
            continue
        if tag == "Level0" and _is_level01_label_set(u) and not _is_level0_label_set(u):
            print(
                f"Skip {title} ROC: {yk} has Level01 labels {sorted(u)!r}. "
                "Call ensure_stardist_hierarchy_labels() after StarDist match."
            )
            continue
        y_true = encode_star_tier_labels(y_raw, names, le=le)
        scores = head_probs[pk]
        mm = min(len(y_true), len(scores), m)
        print(f"\n{title} (StarDist) ROC — ground truth labels: {sorted(u)!r}")
        plot_multiclass_roc_curves(
            y_true[:mm],
            scores[:mm],
            names,
            figsize=(3.0, 3.0),
            max_curves=len(names),
            save_path=result_fig(fig_key),
            title=roc_title,
            roc_color_scheme="ncrt",
            ncrt_color_tier=color_tier,
        )
        print(f"Saved: {result_fig(fig_key)}")


def plot_he_roc_extra(plot_multiclass_roc_curves, result_fig, head_probs, cv_data=None, g=None):
    """In-sample ROC for Level12 / Level0 / Level01 heads (HE matched_features)."""
    g = g or {}
    cv_data = cv_data or {}
    roc_specs = (
        ("Level12 sublineage", "l12", "class_names_level12", "val_labels_level12", "roc_level12", "HE Level12 ROC", "celltype_level12"),
        ("Level0 compartment", "l3", "class_names_level3", "val_labels_level3", "roc_level0", "HE Level0 ROC", "celltype_level0"),
        ("Level01 bucket", "l4", "class_names_level4", "val_labels_level4", "roc_level01", "HE Level01 ROC", "celltype_level01"),
    )
    for title, pk, nk, yk, fig_key, roc_title, color_tier in roc_specs:
        if pk not in head_probs:
            print(f"Skip {title} ROC: missing probs {pk}")
            continue
        names = tier_class_names(nk, cv_data, g)
        y_true = g.get(yk)
        if names is None or y_true is None:
            print(f"Skip {title} ROC: need {nk}, {yk}")
            continue
        scores = head_probs[pk]
        mm = min(len(y_true), len(scores))
        print(f"\n{title} (HE in-sample) ROC")
        plot_multiclass_roc_curves(
            y_true[:mm],
            scores[:mm],
            names,
            figsize=(3.0, 3.0),
            max_curves=len(names),
            save_path=result_fig(fig_key),
            title=roc_title,
            roc_color_scheme="ncrt",
            ncrt_color_tier=color_tier,
        )
        print(f"Saved: {result_fig(fig_key)}")


def build_insample_tier_metrics(g=None, cv_data=None):
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


def alias_ncrt_hierarchy_labels(g=None, cv_data=None):
    """Map NCRT npz keys (level0/level01) to five-head keys (level3/level4)."""
    g = g if g is not None else {}
    cd = cv_data or {}
    if g.get("y_level0") is not None and g.get("y_level3") is None:
        g["y_level3"] = g["y_level0"]
    if g.get("y_level01") is not None and g.get("y_level4") is None:
        g["y_level4"] = g["y_level01"]
    if g.get("y_star_level0") is not None and g.get("y_star_level3") is None:
        g["y_star_level3"] = g["y_star_level0"]
    if g.get("y_star_level01") is not None and g.get("y_star_level4") is None:
        g["y_star_level4"] = g["y_star_level01"]
    for src, dst in (
        ("le_level12", "le_level12"),
        ("le_level3", "le_level3"),
        ("le_level4", "le_level4"),
    ):
        if g.get(dst) is None and cd.get(src) is not None:
            g[dst] = cd[src]


def ensure_stardist_hierarchy_labels(g=None, cv_data=None, meta_path=None, force=False):
    """Repair StarDist y_star_level12/3/4 when missing or wrong tier.

    Prefer labels from ``match_hist2cell_matrix`` (CSV columns celltype_level12/0/01).
    Falls back to deriving from ``y_star`` via codex meta when cache lacks hierarchy.
    """
    g = g if g is not None else {}
    alias_ncrt_hierarchy_labels(g, cv_data)
    if g.get("y_star") is None:
        return
    from base import load_codex_meta_hierarchy_df

    meta = load_codex_meta_hierarchy_df(meta_path)
    y_star = np.asarray(g["y_star"]).astype(str)
    tier_map = (
        ("y_star_level12", "celltype_level12", None),
        ("y_star_level3", "celltype_level0", _is_level0_label_set),
        ("y_star_level4", "celltype_level01", _is_level01_label_set),
    )

    def _derive(meta_col):
        lookup = {
            str(ct).strip(): str(lab).strip()
            for ct, lab in zip(meta["celltype"], meta[meta_col])
        }
        mapped = []
        miss = 0
        for ct in y_star:
            lab = lookup.get(ct.strip())
            if lab is None:
                miss += 1
                mapped.append("")
            else:
                mapped.append(lab)
        return np.asarray(mapped, dtype=str), miss

    for yk, meta_col, validator in tier_map:
        current = g.get(yk)
        labels = _unique_label_set(current)
        needs = force or current is None or not labels
        if not needs and validator is not None:
            needs = not validator(labels)
            if needs:
                print(
                    f"  ⚠ {yk} has unexpected labels {sorted(labels)!r} for {meta_col}; "
                    "re-deriving from y_star / codex meta"
                )
        if not needs:
            continue
        mapped, miss = _derive(meta_col)
        g[yk] = mapped
        if miss:
            print(
                f"  ⚠ {yk}: {miss}/{len(y_star)} celltypes missing from codex meta "
                f"(derived from y_star / {meta_col})"
            )
        else:
            print(
                f"  ✓ {yk}: derived from y_star via codex meta ({len(y_star)} cells) "
                f"→ {sorted(_unique_label_set(mapped))!r}"
            )


def _tumor_slug(group_id):
    s = str(group_id).strip()
    for prefix in ("NCRT_", "ncrt_"):
        if s.startswith(prefix):
            return s[len(prefix) :].lower()
    return s.lower().replace(" ", "_").replace("/", "_")


def _child_to_parent_map(y_encoded_f, y_level1_encoded_f, num_l2):
    child_to_parent = np.full(int(num_l2), -1, dtype=np.int64)
    for l2, l1 in zip(y_encoded_f, y_level1_encoded_f):
        l2_i, l1_i = int(l2), int(l1)
        if child_to_parent[l2_i] == -1:
            child_to_parent[l2_i] = l1_i
        elif child_to_parent[l2_i] != l1_i:
            raise ValueError(
                f"Inconsistent L2→L1 map for class {l2_i}: "
                f"{child_to_parent[l2_i]} vs {l1_i}"
            )
    return child_to_parent


def plot_per_tumor_logo_roc(
    model,
    scaler,
    device,
    cv_data,
    result_fig,
    plot_multiclass_roc_curves,
    mlp_collect_softmax_probs,
    mlp_collect_five_head_softmax_probs,
    min_cells=50,
    g=None,
):
    """Per-``TumorID`` multiclass AUROC on LOGO-filtered HE cells (all label tiers)."""
    from plot import _aggregate_l2_probs_to_l1

    groups_f = cv_data.get("groups_f")
    if groups_f is None:
        print("Skip per-tumor ROC: cv_data has no groups_f")
        return

    g = g or {}
    class_names = cv_data["class_names"]
    class_names_level1 = cv_data["class_names_level1"]
    X_scaled = scaler.transform(cv_data["X_f"])
    probs_l2_all = mlp_collect_softmax_probs(model, X_scaled, device)
    has_five = hasattr(model, "level12_head")
    head_probs_all = (
        mlp_collect_five_head_softmax_probs(model, X_scaled, device) if has_five else {}
    )
    child_to_parent = _child_to_parent_map(
        cv_data["y_encoded_f"],
        cv_data["y_level1_encoded_f"],
        len(class_names),
    )

    extra_tiers = (
        ("Level12 sublineage", "l12", "class_names_level12", "y_level12_encoded_f", "level12", "celltype_level12"),
        ("Level0 compartment", "l3", "class_names_level3", "y_level3_encoded_f", "level0", "celltype_level0"),
        ("Level01 bucket", "l4", "class_names_level4", "y_level4_encoded_f", "level01", "celltype_level01"),
    )

    unique_groups = sorted({str(x) for x in groups_f})
    print(f"Per-tumor AUROC: {len(unique_groups)} LOGO groups")

    for gid in unique_groups:
        idx = np.array([str(x) == gid for x in groups_f], dtype=bool)
        n = int(idx.sum())
        if n < min_cells:
            print(f"  Skip {gid}: {n} cells (< {min_cells})")
            continue
        slug = _tumor_slug(gid)
        print(f"\n{'=' * 60}\n{gid} ({n} cells)\n{'=' * 60}")

        y_l2 = cv_data["y_encoded_f"][idx]
        plot_multiclass_roc_curves(
            y_l2,
            probs_l2_all[idx],
            class_names,
            figsize=(3.0, 3.0),
            max_curves=len(class_names),
            save_path=result_fig(f"roc_logo_{slug}_level2"),
            title=f"{gid} Level2 ROC",
            roc_color_scheme="ncrt",
        )

        probs_l1 = _aggregate_l2_probs_to_l1(
            probs_l2_all[idx], child_to_parent, len(class_names_level1)
        )
        plot_multiclass_roc_curves(
            cv_data["y_level1_encoded_f"][idx],
            probs_l1,
            class_names_level1,
            figsize=(3.0, 3.0),
            max_curves=len(class_names_level1),
            save_path=result_fig(f"roc_logo_{slug}_level1"),
            title=f"{gid} Level1 ROC (L2 agg)",
            roc_color_scheme="ncrt",
            ncrt_color_tier="celltype_level1",
        )

        if has_five and "l1" in head_probs_all:
            plot_multiclass_roc_curves(
                cv_data["y_level1_encoded_f"][idx],
                head_probs_all["l1"][idx],
                class_names_level1,
                figsize=(3.0, 3.0),
                max_curves=len(class_names_level1),
                save_path=result_fig(f"roc_logo_{slug}_level1_L1head"),
                title=f"{gid} Level1 ROC (L1 head)",
                roc_color_scheme="ncrt",
                ncrt_color_tier="celltype_level1",
            )

        if not has_five:
            continue
        for title, pk, nk, yk, tier_slug, color_tier in extra_tiers:
            names = tier_class_names(nk, cv_data, g)
            y_enc = cv_data.get(yk)
            if names is None or y_enc is None or pk not in head_probs_all:
                print(f"  Skip {gid} {title}: missing labels or head probs")
                continue
            plot_multiclass_roc_curves(
                y_enc[idx],
                head_probs_all[pk][idx],
                names,
                figsize=(3.0, 3.0),
                max_curves=len(names),
                save_path=result_fig(f"roc_logo_{slug}_{tier_slug}"),
                title=f"{gid} {title} ROC",
                roc_color_scheme="ncrt",
                ncrt_color_tier=color_tier,
            )
        print(f"  Saved ROC figures for {gid} → roc_logo_{slug}_*")
