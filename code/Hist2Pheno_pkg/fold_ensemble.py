"""2026.11.15 five-fold ensemble for external StarDist cohorts (UNI2).

``best_mlp_gpu.pt`` is one fold (A2 fold 1, A3 fold 0). With
``--inner-val-fraction 0.2`` that fold saw ~64 % of the training sections,
and the fold was picked by its own held-out score. This script scores the
same external slide with every ``hce_group_fold_{k}.pt`` through the normal
verified pipeline (each fold uses its own artifact scaler), then averages
the L2 softmax.

The per-fold table is also the cheapest stability check: if mean P(B_cell)
moves a lot between folds, the B estimate on this slide is not reliable.

    STAR_PROB_MIN=0 python code/Hist2Pheno_pkg/fold_ensemble.py NPC \
        --ckpt-dir data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate \
        --pred-base pred_uni2_A3_fold

Writes ``data/Visium/NPC/<pred-base>{k}/`` per fold and
``<pred-base>_ens/{sample}/{sample}_l2_probs_ensemble.npz`` plus
``fold_mean_probs.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort", help="NPC | CRC | HCC | HCC_NATURE")
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--pred-base", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args(argv)

    import external_validation as ev

    folds = sorted(args.ckpt_dir.glob("hce_group_fold_*.pt"))
    folds = [f for f in folds if not f.name.endswith(".artifact.pt")]
    if len(folds) < 2:
        raise SystemExit(f"need >=2 fold checkpoints in {args.ckpt_dir}")
    per_fold, names, classes, rows = {}, None, None, []
    for ckpt in folds:
        k = ckpt.stem.rsplit("_", 1)[-1]
        os.environ["UNI2_CKPT"] = str(ckpt.resolve())
        os.environ["UNI2_PRED_DIRNAME"] = f"{args.pred_base}{k}"
        cohort = ev.simple_cohort(args.cohort, embedder="uni2")
        samples = cohort.discover_samples(args.sample)
        cohort.cmd_predict(samples, force=False, allow_cpu=args.allow_cpu)
        for s in samples:
            with np.load(cohort.l2_probs_path(s), allow_pickle=False) as z:
                p = np.asarray(z["probs"], dtype=np.float32)
                c = [str(x) for x in z["class_names"]]
                o = np.asarray(z["obs_names"]).astype(str)
            if classes is None:
                classes = c
            if c != classes:
                raise ValueError("fold class order differs")
            ref = per_fold.setdefault(s, {"obs": o, "probs": []})
            if not np.array_equal(ref["obs"], o):
                order = pd.Index(o).get_indexer(ref["obs"])
                if (order < 0).any():
                    raise ValueError(f"{s}: fold {k} scored a different nucleus set")
                p = p[order]
            ref["probs"].append(p)
            rows.append({"sample": s, "fold": int(k), **dict(zip(classes, p.mean(0)))})
    os.environ["UNI2_PRED_DIRNAME"] = f"{args.pred_base}_ens"
    cohort = ev.simple_cohort(args.cohort, embedder="uni2")
    for s, rec in per_fold.items():
        ens = np.mean(np.stack(rec["probs"]), axis=0).astype(np.float32)
        out = cohort.pred_root / s / f"{s}_l2_probs_ensemble.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, probs=ens, class_names=np.asarray(classes, dtype=str),
                            obs_names=rec["obs"], n_folds=len(rec["probs"]))
        print(f"[ensemble] {s}: {len(rec['probs'])} folds -> {out}")
    tab = pd.DataFrame(rows)
    tab.to_csv(cohort.pred_root / "fold_mean_probs.csv", index=False)
    show = [c for c in ("B_cell", "Plasma", "CD4_T", "CD8_T", "Tregs", "Tumor") if c in tab]
    print(tab[["sample", "fold", *show]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
