# 2026.11.17, post-hoc decision layer (posthoc.py section 5) and merged helpers
import json
import numpy as np
import pandas as pd
import pytest

import posthoc as ph
from pancancer_labels import hierarchy_maps, ontology_class_names


def _fake_oof(tmp_path, n=40_000, seed=0):
    """Imbalanced truth; a 'balanced-trained' model that over-predicts rare classes."""
    rng = np.random.default_rng(seed)
    cls = ontology_class_names()
    maps = hierarchy_maps(cls)
    k2 = len(cls["L2"])
    prior = rng.dirichlet(np.full(k2, 0.35)); prior = np.sort(prior)[::-1]
    y = rng.choice(k2, size=n, p=prior)
    logit = rng.normal(0, 1.0, (n, k2))
    logit[np.arange(n), y] += 2.2                       # informative -> good AUROC
    logit -= np.log(prior)[None, :] * 0.8               # balanced training: rare classes inflated
    p2 = ph.softmax_rows(logit)
    m12 = np.asarray(maps["l2_to_l12"]); m1 = np.asarray(maps["l12_to_l1"])
    def agg(p, m, k):
        out = np.zeros((len(p), k)); np.add.at(out.T, m, p.T); return out
    p12 = ph.softmax_rows(np.log(agg(p2, m12, len(cls["L12"])) + 1e-6) + rng.normal(0, .3, (n, len(cls["L12"]))))
    p1 = ph.softmax_rows(np.log(agg(p12, m1, len(cls["L1"])) + 1e-6))
    folds = rng.integers(0, 5, n)
    np.savez_compressed(tmp_path / "oof_predictions.npz", row_indices=np.arange(n),
                        l2_probs=p2.astype(np.float32), l12_probs=p12.astype(np.float32),
                        l1_head_probs=p1.astype(np.float32), l2_labels=y, l12_labels=m12[y], l1_head_labels=m1[m12[y]],
                        l2_class_names=np.asarray(cls["L2"]), l12_class_names=np.asarray(cls["L12"]),
                        l1_head_class_names=np.asarray(cls["L1"]), fold_ids=folds)
    pd.DataFrame({"row_index": np.arange(n), "section_id": [f"D{i % 3}::s{i % 50}" for i in range(n)],
                  "fold": folds}).to_csv(tmp_path / "oof_rows.csv", index=False)
    return cls


def test_fit_decision_improves_labels_cross_fitted(tmp_path):
    _fake_oof(tmp_path)
    rep = ph.fit_decision(tmp_path, max_fit=20_000)
    a = rep[(rep.cohort == "ALL") & (rep["head"] == "L2")].set_index("method")
    base, dec = a.loc["argmax per head (current)"], a.loc["decision full (cross-fitted)"]
    assert dec.macro_f1 > base.macro_f1 + 0.01
    assert dec.accuracy > base.accuracy + 0.01
    cons = rep[rep["head"] == "hierarchy_consistent_fraction"].set_index("method").accuracy
    assert cons["decision full (cross-fitted)"] == 1.0
    for name in ("decision_rule.json", "decision_rule_l2_only.json", "decision_report.csv", "decision_per_class_f1.csv"):
        assert (tmp_path / name).is_file()
    rule = ph.DecisionRule.load(tmp_path / "decision_rule_l2_only.json")
    assert rule.variant == "l2_only" and rule.w12 == 0 and len(rule.bias) == 20


def test_objective_accuracy_vs_f1(tmp_path):
    _fake_oof(tmp_path, n=20_000, seed=1)
    z = np.load(tmp_path / "oof_predictions.npz")
    s, y = np.log(z["l2_probs"]), z["l2_labels"]
    b_acc, v_acc = ph.fit_class_bias(s, y, objective="accuracy")
    b_f1, v_f1 = ph.fit_class_bias(s, y, objective="macro_f1")
    acc0 = ph.label_metrics(y, s.argmax(1), 20)
    assert v_acc >= acc0["accuracy"] and v_f1 >= acc0["macro_f1"]


def test_decide_npz_and_abstain(tmp_path):
    cls = _fake_oof(tmp_path, n=10_000, seed=2)
    ph.fit_decision(tmp_path, max_fit=5_000)
    z = np.load(tmp_path / "oof_predictions.npz")
    npz = tmp_path / "S_l2_probs.npz"
    np.savez_compressed(npz, probs=z["l2_probs"][:500], class_names=np.asarray(cls["L2"], dtype=object),
                        obs_names=np.arange(500).astype(str))
    df = ph.decide_npz(npz, tmp_path / "decision_rule_l2_only.json")
    assert len(df) == 500 and (tmp_path / "S_l2_decision.csv").is_file()
    rule = ph.DecisionRule.load(tmp_path / "decision_rule_l2_only.json")
    m12 = np.asarray(rule.l2_to_l12)
    l2 = np.asarray([rule.classes_l2.index(v) for v in df.L2])
    assert (np.asarray([rule.classes_l12.index(v) for v in df.L12]) == m12[l2]).all()
    rule.min_confidence = 0.99
    out = rule.apply(z["l2_probs"][:500])
    assert (out["l2"] == -1).any() and "Unassigned" in set(rule.names(out["l2"]))
    full = ph.DecisionRule.load(tmp_path / "decision_rule.json")
    if full.w12 or full.w1:                      # a three-head rule cannot run on L2-only npz
        with pytest.raises(ValueError):
            ph.decide_npz(npz, tmp_path / "decision_rule.json")


def test_temperature_and_smoothing():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 5000)
    s = rng.normal(0, 1, (5000, 4)); s[np.arange(5000), y] += 1.0
    t = ph.fit_temperature(s * 3.0, y)
    assert 2.0 < t < 4.5                     # sharpened scores need T ~ 3
    p = ph.softmax_rows(s)
    xy = rng.random((5000, 2))
    q = ph.smooth_probs_knn(p, xy, k=5, alpha=0.5)
    assert np.allclose(q.sum(1), 1, atol=1e-5) and not np.allclose(q, p)
    assert np.array_equal(ph.smooth_probs_knn(p, xy, alpha=0), p.astype(np.float32))


def test_merged_sections_still_work(tmp_path):
    rng = np.random.default_rng(0)
    p = ph.softmax_rows(rng.normal(0, 1, (300, 5)))
    q, pi, it = ph.em_prior_shift(p, np.full(5, .2))
    assert q.shape == p.shape and abs(pi.sum() - 1) < 1e-9
    assert ph.lins_ccc([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    cls = _fake_oof(tmp_path, n=3000)
    met, pcl = ph.evaluate(tmp_path)              # no checkpoint needed any more
    assert {"accuracy", "balanced_accuracy", "macro_f1_all_classes"} <= set(met.columns)


def test_compare_runs_paired(tmp_path):
    for arm, seed in (("spatial_k8", 42), ("spatial_k8", 43), ("l12_balanced", 42)):
        d = tmp_path / f"result_uni2_ablation_{arm}_new1066axisfix_paired_v2_seed{seed}"
        d.mkdir()
        _fake_oof(d, n=6000, seed=seed + (7 if arm == "l12_balanced" else 0))
        ph.fit_decision(d, max_fit=3000)
        assert (d / "decision_crossfit_predictions.npz").is_file()
    tab = ph.compare_runs(tmp_path, "result_uni2_ablation_*_new1066axisfix_paired_v2_seed*", "spatial_k8",
                          tmp_path / "s.csv", data_tag="new1066axisfix")
    t = tab.set_index("arm")
    assert list(t.index) == ["spatial_k8", "l12_balanced"]
    assert t.loc["spatial_k8", "n_seeds"] == 2 and t.loc["l12_balanced", "n_seeds"] == 1
    assert t.loc["l12_balanced", "L2_macro_f1_decision_folds_better"].endswith("/5")
    assert 0.5 < t.loc["spatial_k8", "L2_macro_auroc_mean"] <= 1
    assert (tmp_path / "s_per_fold.csv").is_file()
    assert ph._arm_of("result_uni2_ablation_l12_links_1_stratsec_new1066axisfix_paired_v2_seed43",
                      "new1066axisfix") == ("l12_links_1", 43)


def test_label_modes_external(tmp_path):
    # 2026.11.17, label_modes: four modes, prior cached, class order checked
    cls = _fake_oof(tmp_path, n=20_000)
    ph.fit_decision(tmp_path, max_fit=10_000)
    rng = np.random.default_rng(3)
    k2 = len(cls["L2"])
    p = ph.softmax_rows(rng.normal(0, 1, (500, k2)))
    xy = rng.uniform(0, 100, (500, 2))
    labs = ph.label_modes(p, cls["L2"], tmp_path, xy=xy)
    assert set(labs) == set(ph.LABEL_MODES)
    assert all(len(v) == 500 and set(v) <= set(cls["L2"]) for v in labs.values())
    assert (labs["argmax"] == np.asarray(cls["L2"], dtype=object)[p.argmax(1)]).all()
    assert (tmp_path / "oof_l2_mean_prior.json").is_file()
    # no xy -> no smooth; reversed class order -> error
    assert "smooth" not in ph.label_modes(p, cls["L2"], tmp_path, modes=("smooth",))
    import pytest
    with pytest.raises(ValueError):
        ph.label_modes(p, cls["L2"][::-1], tmp_path, modes=("decision",))
