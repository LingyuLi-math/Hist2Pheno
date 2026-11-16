# 2026.11.16, L2-balanced section folds
import numpy as np
import pytest

from pancancer_cv import (assign_section_folds, assign_stratified_section_folds,
                          fold_balance_score, fold_balance_table, section_label_counts)


def _toy(seed=0, n_ds=5, per_ds=40, n_cls=12):
    rng = np.random.default_rng(seed)
    sids, ds, rows = [], [], []
    for d in range(n_ds):
        mix = rng.dirichlet(np.full(n_cls, 0.3))
        for j in range(per_ds):
            p = rng.dirichlet(mix * 8 + 0.05)
            rows.append(rng.multinomial(int(rng.integers(200, 5000)), p))
            sids.append(f"D{d}::s{j}")
            ds.append(f"D{d}")
    return sids, ds, np.asarray(rows)


def test_better_balance_than_round_robin_and_cohorts_kept():
    sids, ds, C = _toy()
    rr = assign_section_folds(sids, ds, n_splits=5, seed=42)
    st = assign_stratified_section_folds(sids, ds, C, n_splits=5, seed=42)
    s_rr = fold_balance_score(C, [rr[s] for s in sids])
    s_st = fold_balance_score(C, [st[s] for s in sids])
    assert s_st < 0.6 * s_rr, (s_rr, s_st)
    for d in set(ds):                                   # each cohort spread evenly
        f = np.bincount([st[s] for s, x in zip(sids, ds) if x == d], minlength=5)
        assert f.max() - f.min() <= 1
    assert set(st.values()) == set(range(5))


def test_deterministic_and_existing_frozen():
    sids, ds, C = _toy(1)
    a = assign_stratified_section_folds(sids, ds, C, n_splits=5, seed=7)
    b = assign_stratified_section_folds(sids, ds, C, n_splits=5, seed=7)
    assert a == b
    old = {s: a[s] for s in sids[:150]}
    c = assign_stratified_section_folds(sids, ds, C, n_splits=5, seed=7, existing=old)
    assert all(c[s] == old[s] for s in old)


def test_units_never_split():
    sids, ds, C = _toy(2)
    units = [s.rsplit("s", 1)[0] + str(int(s.rsplit("s", 1)[1]) // 3) for s in sids]  # 3 sections / patient
    st = assign_stratified_section_folds(sids, ds, C, n_splits=5, seed=0, units=units)
    for u in set(units):
        assert len({st[s] for s, x in zip(sids, units) if x == u}) == 1


def test_counts_and_table():
    g = np.array([0, 0, 1, 1, 1, 2]); y = np.array([0, 1, 1, 1, 2, 0])
    C = section_label_counts(g, y, 3, 3)
    assert C.tolist() == [[1, 1, 0], [0, 2, 1], [1, 0, 0]]
    t = fold_balance_table(C, [0, 1, 1], ["a", "b", "c"])
    row = t[(t["fold"] == 1) & (t["class"] == "c")].iloc[0]
    assert row.share_of_class == 1.0 and not row.train_has_class


def test_rejects_bad_input():
    sids, ds, C = _toy(3, n_ds=2, per_ds=6)
    with pytest.raises(ValueError):
        assign_stratified_section_folds(sids, ds, C[:-1], n_splits=5)
    with pytest.raises(ValueError):
        assign_stratified_section_folds(sids, ds, C, n_splits=1)


def test_prepare_pooled_stratified_branch(monkeypatch):
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PanCancer"))
    import PanCancer_train_validate_cv_UNIlabel as T
    classes = T.ontology_class_names()
    l2 = list(classes["L2"])
    maps = T.hierarchy_maps(classes)
    to12 = {n: classes["L12"][maps["l2_to_l12"][i]] for i, n in enumerate(l2)}
    to1 = {n: classes["L1"][maps["l12_to_l1"][i]] for i, n in enumerate(classes["L12"])}
    rng = np.random.default_rng(0)
    rows = []
    for d in ("BRCA", "CRC", "Lung"):
        for j in range(10):
            rows.append({"section_id": f"{d}::s{j}", "dataset": d, "h5ad": f"/x/{d}{j}.h5ad",
                         "native_col": "final_CT", "ok": True})
    index = pd.DataFrame(rows)

    def fake_load(path, **kw):
        n = int(rng.integers(50, 120))
        y = rng.choice(l2[:8], size=n, p=rng.dirichlet(np.ones(8)))
        y12 = np.asarray([to12[v] for v in y], dtype=object)
        return {"X": rng.normal(size=(n, 4)).astype(np.float32), "cell_ids": np.arange(n).astype(str),
                "y": y, "y_l1": np.asarray([to1[v] for v in y12], dtype=object), "y_l12": y12,
                "xy": rng.random((n, 2)) * 100, "n": n}
    monkeypatch.setattr(T, "load_section_arrays", fake_load)
    cv = T.prepare_pooled_cv_data(index, classes, cache_dir=None, cv_k=5,
                                  group_splitter="stratified_section", embedder="test")
    assert cv["fold_mode"] == "stratified_section"
    fo = cv["section_fold_of"]
    assert len(fo) == 30 and set(fo.values()) == set(range(5))
    g2s = cv["group_to_sample"]
    assert all(cv["fold_id_f"][i] == fo[g2s[int(g)]] for i, g in enumerate(cv["groups_f"]))
