"""2026.11.17 inner validation split: random = old behaviour, cohort_safe keeps cohorts and classes."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from pancancer_cv import inner_val_split


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    groups, ds, lab = [], [], []
    g = 0
    for cohort, n_sec, classes in (("GIST", 40, [0, 1, 2]), ("PDAC", 20, [0, 1, 3]), ("BRCA", 1, [0, 4]),
                                   ("Lung", 5, [0, 5])):
        for _ in range(n_sec):
            n = int(rng.integers(50, 120))
            groups += [g] * n; ds += [cohort] * n; lab += list(rng.choice(classes, n))
            g += 1
    # class 6 lives in a single PDAC section
    lab = np.asarray(lab); groups = np.asarray(groups)
    lab[np.flatnonzero(groups == 45)[:10]] = 6
    return groups, np.asarray(ds, dtype=object), lab


def test_random_mode_is_old_group_shuffle_split():
    groups, ds, lab = _toy()
    outer = np.arange(len(groups))
    it, iv, rep = inner_val_split(outer, groups, fraction=0.2, seed=44, mode="random", datasets=ds, labels=lab)
    it0, iv0 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=44).split(outer, groups=groups))
    assert (it == it0).all() and (iv == iv0).all()
    assert rep["mode"] == "random" and "lost_classes" in rep


def test_cohort_safe_keeps_single_section_cohort_and_rare_class():
    groups, ds, lab = _toy()
    outer = np.arange(len(groups))
    for seed in range(20):
        it, iv, rep = inner_val_split(outer, groups, fraction=0.2, seed=seed, mode="cohort_safe",
                                      datasets=ds, labels=lab)
        assert len(np.intersect1d(it, iv)) == 0 and len(it) + len(iv) == len(outer)
        assert (ds[it] == "BRCA").any() and not (ds[iv] == "BRCA").any()
        assert set(np.unique(lab[it])) == set(np.unique(lab)) and rep["lost_classes"] == []
        assert len(set(groups[it]) & set(groups[iv])) == 0          # whole sections only
        for cohort in ("GIST", "PDAC", "Lung"):
            assert (ds[it] == cohort).any()
    # roughly the requested share of sections goes to inner validation
    assert 10 <= len(rep["inner_val_groups"]) <= 15


def test_random_mode_can_lose_the_brca_section():
    groups, ds, lab = _toy()
    outer = np.arange(len(groups))
    lost = [inner_val_split(outer, groups, fraction=0.2, seed=s, mode="random", datasets=ds, labels=lab)[2]["lost_classes"]
            for s in range(60)]
    assert any(4 in l for l in lost)        # class 4 exists only in the one BRCA section
