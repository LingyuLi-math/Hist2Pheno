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
