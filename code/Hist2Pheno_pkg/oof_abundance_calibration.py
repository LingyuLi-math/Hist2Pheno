"""2026.11.15 Per-section abundance calibration on OOF (Hist2Prot-style calibration curves).

Hist2Prot reports per-cohort calibration of predicted vs measured abundance.
The Hist2Pheno analogue: for each held-out section, compare the true L2
fraction with (a) the mean OOF softmax and (b) the argmax fraction.
A slope < 1 for B_cell means the model under-calls B where B is abundant,
which is the training-cohort version of the NPC symptom.

    python code/Hist2Pheno_pkg/oof_abundance_calibration.py \
        --oof data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_predictions.npz \
        --oof-rows data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_rows.csv \
        --output data/PanCancer/calibration_A3

Writes ``section_fractions.csv`` (long table) and ``calibration_summary.csv``
(per class × cohort: slope through origin, Pearson, Spearman, Lin's CCC, mean
bias). Compare runs (A3 vs A8 vs A9) on the same table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def lins_ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    d = vx + vy + (mx - my) ** 2
    return float(2 * cov / d) if d > 0 else np.nan


def section_fractions(labels, probs, sections, names):
    n_cls = len(names)
    rows = []
    hard = probs.argmax(1)
    for sec in pd.unique(sections):
        m = sections == sec
        true = np.bincount(labels[m], minlength=n_cls) / m.sum()
        soft = probs[m].mean(0)
        argm = np.bincount(hard[m], minlength=n_cls) / m.sum()
        for j, c in enumerate(names):
            rows.append(dict(section_id=sec, cohort=str(sec).split("::")[0], cls=c, n_cells=int(m.sum()),
                             true_frac=true[j], soft_frac=soft[j], argmax_frac=argm[j]))
    return pd.DataFrame(rows)


def summarize(df, min_sections=3):
    out = []
    for (cohort, cls), g in list(df.groupby(["cohort", "cls"])) + [(("ALL", c), g) for c, g in df.groupby("cls")]:
        if len(g) < min_sections or g.true_frac.max() == 0:
            continue
        rec = dict(cohort=cohort, cls=cls, n_sections=len(g), mean_true=g.true_frac.mean())
        for kind in ("soft", "argmax"):
            x, y = g.true_frac.to_numpy(), g[f"{kind}_frac"].to_numpy()
            rec[f"{kind}_slope"] = float((x * y).sum() / max((x * x).sum(), 1e-12))
            rec[f"{kind}_bias"] = float((y - x).mean())
            rec[f"{kind}_ccc"] = lins_ccc(x, y)
            ok = np.std(x) > 0 and np.std(y) > 0
            rec[f"{kind}_pearson"] = float(pearsonr(x, y)[0]) if ok else np.nan
            rec[f"{kind}_spearman"] = float(spearmanr(x, y)[0]) if ok else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["cls", "cohort"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--oof-rows", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-sections", type=int, default=3)
    args = ap.parse_args(argv)
    with np.load(args.oof, allow_pickle=False) as z:
        labels = np.asarray(z["l2_labels"], int)
        probs = np.asarray(z["l2_probs"], np.float32)
        ids = np.asarray(z["row_indices"], int)
        names = z["l2_class_names"].astype(str).tolist() if "l2_class_names" in z else None
    rows = pd.read_csv(args.oof_rows)
    if not np.array_equal(rows["row_index"].to_numpy(), np.arange(len(rows))):
        raise ValueError("oof_rows.csv must be in global row order")
    if names is None:
        from pancancer_labels import ontology_class_names
        names = list(ontology_class_names()["L2"])
        print("WARNING: legacy OOF without l2_class_names; using the ontology order")
    sections = rows["section_id"].astype(str).to_numpy()[ids]
    df = section_fractions(labels, probs, sections, names)
    summ = summarize(df, args.min_sections)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "section_fractions.csv", index=False)
    summ.to_csv(args.output / "calibration_summary.csv", index=False)
    show = summ[(summ.cohort == "ALL")][["cls", "n_sections", "mean_true", "soft_slope", "soft_ccc",
                                         "argmax_slope", "argmax_ccc"]]
    print(show.round(3).to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
