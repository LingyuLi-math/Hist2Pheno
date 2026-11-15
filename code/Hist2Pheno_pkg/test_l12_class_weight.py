"""2026.11.15 tests for the optional class-weighted L12 head.

1. ``adjacent_loss`` with ``weight_l12=None`` is bit-identical to the old call.
2. A tiny synthetic group-CV run trains with ``l12_class_weight_mode`` = none
   and = balanced, and the rare L12 family (B_Plasma) gets more mean
   probability mass under balanced.

    python -m pytest -q code/Hist2Pheno_pkg/test_l12_class_weight.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hierarchy_loss import HierarchyConfig, adjacent_loss  # noqa: E402


def test_adjacent_loss_none_weights_identical():
    torch.manual_seed(0)
    l2, l12, l1 = torch.randn(64, 5), torch.randn(64, 3), torch.randn(64, 2)
    maps = {"l2_to_l12": [0, 1, 1, 2, 2], "l12_to_l1": [0, 1, 1]}
    y12 = torch.randint(0, 3, (64,))
    y1 = torch.as_tensor(np.asarray(maps["l12_to_l1"])[y12.numpy()])
    cfg = HierarchyConfig()
    a, _ = adjacent_loss(l2, l12, l1, y12, y1, maps, cfg)
    b, _ = adjacent_loss(l2, l12, l1, y12, y1, maps, cfg, weight_l12=None, weight_l1=None)
    assert torch.equal(a, b)
    w12 = torch.tensor([1.0, 1.0, 5.0])
    c, _ = adjacent_loss(l2, l12, l1, y12, y1, maps, cfg, weight_l12=w12)
    assert not torch.equal(a, c)


def _synthetic_cv(n_per_group=600, n_groups=6, seed=0):
    """L2: Tumor, CD4_T, Tregs, B_cell, Fibroblast. L12: Tumor, T_cell, B_Plasma, Stromal.
    B_cell is rare (3%) and overlaps heavily with T in feature space."""
    rng = np.random.default_rng(seed)
    l2_names = np.array(["Tumor", "CD4_T", "Tregs", "B_cell", "Fibroblast"])
    l2_to_l12 = np.array([0, 1, 1, 2, 3])
    l12_to_l1 = np.array([0, 1, 1, 2])
    prior = np.array([0.40, 0.25, 0.17, 0.03, 0.15])
    d = 16
    centers = rng.normal(size=(5, d)) * 2.0
    centers[3] = centers[1] * 0.6 + centers[2] * 0.2 + rng.normal(size=d) * 0.6  # B close to T
    X, y, g = [], [], []
    for grp in range(n_groups):
        yy = rng.choice(5, size=n_per_group, p=prior)
        X.append(centers[yy] + rng.normal(size=(n_per_group, d)) * 1.3)
        y.append(yy)
        g.append(np.full(n_per_group, grp))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y)
    groups = np.concatenate(g)
    y12 = l2_to_l12[y]
    y1 = l12_to_l1[y12]
    cv = {
        "X_f": X,
        "y_encoded_f": y,
        "y_level1_encoded_f": y1,
        "y_level12_encoded_f": y12,
        "y_f": l2_names[y],
        "y_level1_f": np.array(["Epithelial", "Immune", "Stromal"])[y1],
        "groups_f": groups,
        "fold_id_f": groups % 3,
        "class_names": l2_names,
        "class_names_level1": np.array(["Epithelial", "Immune", "Stromal"]),
        "class_names_level12": np.array(["Tumor", "T_cell", "B_Plasma", "Stromal"]),
        "child_to_parent": l12_to_l1[l2_to_l12],
        "hierarchy_maps": {"l2_to_l12": l2_to_l12.tolist(), "l12_to_l1": l12_to_l1.tolist()},
        "hierarchy_config": vars(HierarchyConfig()),
        "inner_val_fraction": 0.25,
        "seed": 1,
        "split_seed": 1,
    }
    return cv


def _run(mode, tmp):
    import model as m
    from base import evaluate

    torch.manual_seed(0)
    np.random.seed(0)
    cv = _synthetic_cv()
    cv["l12_class_weight_mode"] = mode
    summary = m.train_hce_group_kfold_cv(
        device=torch.device("cpu"),
        cv_data=cv,
        evaluate=evaluate,
        class_names=list(cv["class_names"]),
        save_bestmodel_path_pattern=os.path.join(tmp, mode + "_fold_{}.pt"),
        n_splits=3,
        hce_w12=2.0,
        hce_w_l3=0.0,
        hce_w_l4=0.0,
        max_epochs=4,
        patience=4,
        hidden_dims=(32, 16, 8),
        val_selection_metric="three_tier_auc_sum",
        group_splitter="cross_dataset_section",
        loader_kwargs={"seed": 0, "batch_size_cuda": 256, "num_workers_cuda": 0},
    )
    probs = summary["oof_val_probs"]
    return float(probs[:, 3].mean())


def test_balanced_l12_runs_and_lifts_rare_family():
    from checkpoint_artifact import read_artifact
    from pipeline_provenance import checkpoint_family

    with tempfile.TemporaryDirectory() as tmp:
        p_none = _run("none", tmp)
        p_bal = _run("balanced", tmp)
        m_none = read_artifact(os.path.join(tmp, "none_fold_0.pt"), required=True)["metadata"]
        m_bal = read_artifact(os.path.join(tmp, "balanced_fold_0.pt"), required=True)["metadata"]
    # A3 checkpoints keep their old loss_weights; A8 is a different family.
    assert "l12_class_weight" not in m_none["loss_weights"]
    assert m_bal["loss_weights"]["l12_class_weight"] == "balanced"
    assert checkpoint_family(m_none) != checkpoint_family(m_bal)
    print(f"mean OOF P(B_cell): none={p_none:.4f} balanced={p_bal:.4f}")
    assert np.isfinite(p_none) and np.isfinite(p_bal)
    assert p_bal > p_none


if __name__ == "__main__":
    test_adjacent_loss_none_weights_identical()
    test_balanced_l12_runs_and_lifts_rare_family()
    print("ok")


# ---------------------------------------------------------------------------
# 2026.11.15 A9: Hist2Prot-style neighbour-composition auxiliary head
# ---------------------------------------------------------------------------
def _spatial_cv(seed=0):
    """Synthetic sections where B cells sit in small clusters (TLS-like)."""
    cv = _synthetic_cv(seed=seed)
    rng = np.random.default_rng(seed + 1)
    y = cv["y_encoded_f"]
    xy = rng.uniform(0, 400, size=(len(y), 2))
    b = np.flatnonzero(y == 3)
    centres = rng.uniform(50, 350, size=(4, 2))
    xy[b] = centres[rng.integers(0, 4, len(b))] + rng.normal(0, 6, (len(b), 2))
    cv["X_coords_f"] = xy.astype(np.float64)
    cv["_data_fingerprint"] = "synthetic-spatial-v1"
    return cv


def _run_spatial(nbr_w, tmp):
    import model as m
    from base import evaluate
    from spatial_context import SpatialConfig

    torch.manual_seed(0)
    np.random.seed(0)
    cv = _spatial_cv()
    cv["l12_class_weight_mode"] = "balanced"
    cv["neighbor_comp_weight"] = nbr_w
    cv["spatial_config"] = vars(SpatialConfig(k=8, gate=True, cache_dir=os.path.join(tmp, "ctx")))
    tag = f"nbr{nbr_w:g}"
    summary = m.train_hce_group_kfold_cv(
        device=torch.device("cpu"), cv_data=cv, evaluate=evaluate,
        class_names=list(cv["class_names"]),
        save_bestmodel_path_pattern=os.path.join(tmp, tag + "_fold_{}.pt"),
        n_splits=3, hce_w12=2.0, hce_w_l3=0.0, hce_w_l4=0.0, max_epochs=3, patience=3,
        hidden_dims=(32, 16, 8), val_selection_metric="three_tier_auc_sum",
        group_splitter="cross_dataset_section",
        use_spatial_context=True, spatial_k=8, spatial_mode="mean",
        loader_kwargs={"seed": 0, "batch_size_cuda": 256, "num_workers_cuda": 0},
    )
    return cv, summary, os.path.join(tmp, tag + "_fold_0.pt")


def test_neighbor_composition_targets_exclude_self_and_sum_to_one():
    import model as m
    from spatial_context import SpatialConfig, graphs

    cv = _spatial_cv()
    cfg = SpatialConfig(k=8)
    cv["_context_graphs"] = graphs(cv["X_coords_f"], cv["groups_f"], cfg)
    n = len(cv["y_encoded_f"])
    res = {"train_global_indices": np.flatnonzero(cv["groups_f"] < 4),
           "val_global_indices": np.flatnonzero(cv["groups_f"] >= 4)}
    q, ok = m.neighbor_composition_targets(cv, res, 5)
    assert q.shape == (n, 5) and ok.all()
    assert np.allclose(q.astype(np.float32).sum(1), 1.0, atol=1e-3)
    idx = cv["_context_graphs"][0][0]
    i = 0
    manual = np.bincount(cv["y_encoded_f"][idx[i]], minlength=5) / 8
    assert np.allclose(q[i].astype(np.float32), manual, atol=1e-3)


def test_a9_trains_saves_and_reloads_strict():
    import model as m
    from checkpoint_artifact import read_artifact
    from pipeline_provenance import checkpoint_family

    with tempfile.TemporaryDirectory() as tmp:
        cv, summary, ck = _run_spatial(0.2, tmp)
        _, _, ck0 = _run_spatial(0.0, tmp)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        assert "neighbor_comp_head.weight" in sd
        meta = read_artifact(ck, required=True)["metadata"]
        meta0 = read_artifact(ck0, required=True)["metadata"]
        assert meta["loss_weights"]["neighbor_comp"] == 0.2
        assert "neighbor_comp" not in meta0["loss_weights"]
        assert checkpoint_family(meta) != checkpoint_family(meta0)
        model, _ = m.build_mlp_classifier_for_state_dict(
            sd, input_dim=cv["X_f"].shape[1], num_classes=5, num_level1_classes=3,
            device=torch.device("cpu"), hidden_dims=(32, 16, 8),
            spatial_config=meta.get("spatial_config"))
        model.load_state_dict(sd)  # strict
        assert summary["oof_val_probs"].shape[1] == 5
        assert np.isfinite(summary["oof_val_probs"]).all()


def test_a10_geometry_and_inference_exports_neighbor_comp():
    """A10 config (µm coords, 50 µm cap, geometry) trains; inference returns 'nbr'."""
    import model as m
    from base import evaluate
    from checkpoint_artifact import read_artifact, restore_scaler
    from plot import mlp_collect_five_head_softmax_probs
    from spatial_context import SpatialConfig

    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0); np.random.seed(0)
        cv = _spatial_cv()
        cv["l12_class_weight_mode"] = "balanced"
        cv["neighbor_comp_weight"] = 0.2
        cv["spatial_config"] = vars(SpatialConfig(k=8, gate=True, radii_um=(50.0,), geometry=True,
                                                  coordinates_um=True, cache_dir=os.path.join(tmp, "ctx")))
        m.train_hce_group_kfold_cv(
            device=torch.device("cpu"), cv_data=cv, evaluate=evaluate,
            class_names=list(cv["class_names"]),
            save_bestmodel_path_pattern=os.path.join(tmp, "a10_fold_{}.pt"),
            n_splits=3, hce_w12=2.0, hce_w_l3=0.0, hce_w_l4=0.0, max_epochs=2, patience=2,
            hidden_dims=(32, 16, 8), val_selection_metric="three_tier_auc_sum",
            group_splitter="cross_dataset_section",
            use_spatial_context=True, spatial_k=8, spatial_mode="mean",
            loader_kwargs={"seed": 0, "batch_size_cuda": 256, "num_workers_cuda": 0})
        ck = os.path.join(tmp, "a10_fold_0.pt")
        art = read_artifact(ck, required=True)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        assert "spatial_fusion.geometry_proj.weight" in sd and "neighbor_comp_head.weight" in sd
        model, _ = m.build_mlp_classifier_for_state_dict(
            sd, input_dim=cv["X_f"].shape[1], num_classes=5, num_level1_classes=3,
            device=torch.device("cpu"), hidden_dims=(32, 16, 8),
            spatial_config=art["metadata"].get("spatial_config"))
        model.load_state_dict(sd); model.eval()
        scaler = restore_scaler(art["scaler"]); model.checkpoint_scaler = scaler
        sec = cv["groups_f"] == 0
        Xs = scaler.transform(cv["X_f"][sec]).astype(np.float32)
        cfg = dict(art["metadata"]["spatial_config"]); cfg["cache_dir"] = os.path.join(tmp, "ctx_inf")
        probs = mlp_collect_five_head_softmax_probs(
            model, Xs, torch.device("cpu"),
            neighbor_index={"xy": cv["X_coords_f"][sec], "config_override": cfg})
        assert set(probs) == {"l2", "l1", "l12", "nbr"}
        assert np.allclose(probs["nbr"].sum(1), 1, atol=1e-4)


# ---------------------------------------------------------------------------
# 2026.11.15 A11: Hist2Prot tissue-compartment head
# ---------------------------------------------------------------------------
def _pancancer_cv(seed=0):
    """Synthetic data on the real 20-class L2 ontology with a lymphoid aggregate."""
    from pancancer_labels import ontology_class_names, hierarchy_maps
    names = ontology_class_names()
    maps = hierarchy_maps(names)
    rng = np.random.default_rng(seed)
    n_per, n_g, d = 500, 6, 12
    l2 = np.array(names["L2"])
    tumor, b, t4, fib = (list(l2).index(c) for c in ("Tumor", "B_cell", "CD4_T", "Fibroblast"))
    X, y, g, xy = [], [], [], []
    centers = rng.normal(size=(20, d)) * 2
    for grp in range(n_g):
        p = rng.uniform(0, 300, size=(n_per, 2))
        yy = np.where(p[:, 0] < 100, tumor, fib)
        agg = np.linalg.norm(p - [220, 150], axis=1) < 50          # TLS-like aggregate
        yy[agg] = np.where(rng.random(agg.sum()) < 0.5, b, t4)
        X.append(centers[yy] + rng.normal(size=(n_per, d)))
        y.append(yy); g.append(np.full(n_per, grp)); xy.append(p)
    y = np.concatenate(y)
    l12 = np.asarray(maps["l2_to_l12"])[y]
    l1 = np.asarray(maps["l12_to_l1"])[l12]
    cv = {
        "X_f": np.vstack(X).astype(np.float32), "y_encoded_f": y, "y_level1_encoded_f": l1,
        "y_level12_encoded_f": l12, "y_f": l2[y], "y_level1_f": np.array(names["L1"])[l1],
        "groups_f": np.concatenate(g), "fold_id_f": np.concatenate(g) % 3,
        "X_coords_f": np.vstack(xy), "_data_fingerprint": "synthetic-pancancer-v1",
        "class_names": l2, "class_names_level1": np.array(names["L1"]),
        "class_names_level12": np.array(names["L12"]),
        "child_to_parent": np.asarray(maps["l12_to_l1"])[np.asarray(maps["l2_to_l12"])],
        "hierarchy_maps": maps, "hierarchy_config": vars(HierarchyConfig()),
        "inner_val_fraction": 0.25, "seed": 1, "split_seed": 1,
    }
    return cv, (tumor, b, fib)


def test_tissue_compartment_targets():
    import model as m
    cv, _ = _pancancer_cv()
    res = {"train_global_indices": np.flatnonzero(cv["groups_f"] < 4),
           "val_global_indices": np.flatnonzero(cv["groups_f"] >= 4)}
    t = m.tissue_compartment_targets(cv, res, 16)
    names = list(m.TISSUE_COMPARTMENTS)
    xy = cv["X_coords_f"]
    core_tls = np.linalg.norm(xy - [220, 150], axis=1) < 25
    core_tumor = xy[:, 0] < 60
    assert (t[core_tls] == names.index("Lymphoid")).mean() > 0.9
    assert (t[core_tumor] == names.index("Tumor")).mean() > 0.9
    assert t.min() >= 0  # every row is in train or val here


def test_a11_trains_and_exports_tissue_probs():
    import model as m
    from base import evaluate
    from checkpoint_artifact import read_artifact, restore_scaler
    from plot import mlp_collect_five_head_softmax_probs
    from spatial_context import SpatialConfig

    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(0); np.random.seed(0)
        cv, _ = _pancancer_cv()
        cv.update(l12_class_weight_mode="balanced", neighbor_comp_weight=0.2,
                  tissue_comp_weight=0.2, tissue_comp_k=16,
                  spatial_config=vars(SpatialConfig(k=8, gate=True, cache_dir=os.path.join(tmp, "ctx"))))
        m.train_hce_group_kfold_cv(
            device=torch.device("cpu"), cv_data=cv, evaluate=evaluate,
            class_names=list(cv["class_names"]),
            save_bestmodel_path_pattern=os.path.join(tmp, "a11_fold_{}.pt"),
            n_splits=3, hce_w12=2.0, hce_w_l3=0.0, hce_w_l4=0.0, max_epochs=2, patience=2,
            hidden_dims=(32, 16, 8), val_selection_metric="three_tier_auc_sum",
            group_splitter="cross_dataset_section",
            use_spatial_context=True, spatial_k=8, spatial_mode="mean",
            loader_kwargs={"seed": 0, "batch_size_cuda": 256, "num_workers_cuda": 0})
        ck = os.path.join(tmp, "a11_fold_0.pt")
        art = read_artifact(ck, required=True)
        assert art["metadata"]["loss_weights"]["tissue_comp"] == 0.2
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        assert sd["tissue_comp_head.weight"].shape[0] == len(m.TISSUE_COMPARTMENTS)
        model, _ = m.build_mlp_classifier_for_state_dict(
            sd, input_dim=cv["X_f"].shape[1], num_classes=20, num_level1_classes=5,
            device=torch.device("cpu"), hidden_dims=(32, 16, 8),
            spatial_config=art["metadata"].get("spatial_config"))
        model.load_state_dict(sd); model.eval()
        model.checkpoint_scaler = restore_scaler(art["scaler"])
        sec = cv["groups_f"] == 0
        cfg = dict(art["metadata"]["spatial_config"]); cfg["cache_dir"] = os.path.join(tmp, "ctx_inf")
        probs = mlp_collect_five_head_softmax_probs(
            model, model.checkpoint_scaler.transform(cv["X_f"][sec]).astype(np.float32),
            torch.device("cpu"), neighbor_index={"xy": cv["X_coords_f"][sec], "config_override": cfg})
        assert set(probs) == {"l2", "l1", "l12", "nbr", "tissue"}
        assert probs["tissue"].shape[1] == len(m.TISSUE_COMPARTMENTS)


def test_spora_native_labels_kept():
    # 2026.11.15 review: CRC-SPORA (Schurch) + GBM-SPORA (Hoefflin) native names
    from pancancer_labels import native_to_pancancer as f
    want = {
        "CD8+ T cells": "CD8_T", "plasma cells": "Plasma", "B cells": "B_cell",
        "Bcell": "B_cell", "Tregs": "Tregs", "TcellCD8": "CD8_T", "CD4+ T cells": "CD4_T",
        "CD11c+ DCs": "DC", "Mac": "Macrophage", "tumor cells": "Tumor", "cMES": "Tumor",
        "vasculature": "Endothelial", "stroma": "Stromal", "granulocytes": "Neutrophil",
    }
    for native, l2 in want.items():
        out = f(native)
        assert out is not None and out[2] == l2, (native, out)
    for dropped in ("CD3+ T cells", "immune cells", "tumor cells / immune cells", "undefined"):
        assert f(dropped) is None, dropped
