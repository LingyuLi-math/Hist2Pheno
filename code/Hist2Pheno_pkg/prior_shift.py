"""2026.11.15 label-shift (prior) correction for external L2 softmax.

Hist2Pheno's L2 softmax carries the training prior. The pan-cancer pool is
~2.4 % B_cell and ~5.7 % Tregs (97 % of those Tregs come from GIST CODEX), so a
lymphocyte that the 7 µm crop cannot place is split across CD4_T / CD8_T /
Tregs / B_cell / Plasma roughly in training proportion. NPC is lymphoid-rich,
and its B fraction is well above 2.4 %.

``em_prior_shift`` is the EM of Saerens, Latinne & Decaestecker (Neural
Computation 14:21-41, 2002). It assumes only the class prior changed
(p(x|y) is the same in training and NPC). It cannot separate classes the
embedding cannot separate; it re-weights the mass that is already there.

``train_prior`` is the model's own mean OOF softmax on the training cohorts
(``oof_predictions.npz``), not the label frequency, because class-weighted CE
(``balanced``) shifts the prior the network actually learned.

    python code/Hist2Pheno_pkg/prior_shift.py \
        data/Visium/NPC/pred_uni2_A3_bestmlp_paired_v2_seed42/S/S_l2_probs.npz \
        --oof data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_predictions.npz

Writes ``S_l2_probs_priorshift.npz`` and ``S_l2_prior_table.csv`` next to the input.
Use the adjusted probabilities for abundance maps only. Do not overwrite
``pred_CT``; the run is unsupervised and cannot be checked without nucleus GT.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def em_prior_shift(probs, train_prior, *, max_iter=200, tol=1e-6, min_prior=1e-6):
    """Return ``(adjusted_probs, target_prior, n_iter)``."""
    p = np.asarray(probs, dtype=np.float64)
    pi0 = np.clip(np.asarray(train_prior, dtype=np.float64), min_prior, None)
    pi0 = pi0 / pi0.sum()
    if p.ndim != 2 or p.shape[1] != pi0.size:
        raise ValueError(f"probs {p.shape} vs prior {pi0.shape}")
    pi = pi0.copy()
    q = p
    for it in range(1, max_iter + 1):
        w = pi / pi0
        q = p * w[None, :]
        q /= np.clip(q.sum(axis=1, keepdims=True), 1e-300, None)
        new = np.clip(q.mean(axis=0), min_prior, None)
        new /= new.sum()
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return q.astype(np.float32), pi, it


def oof_mean_prior(oof_npz, head="l2"):
    with np.load(oof_npz, allow_pickle=False) as z:
        return np.asarray(z[f"{head}_probs"], dtype=np.float64).mean(axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probs_npz", type=Path)
    ap.add_argument("--oof", type=Path, required=True, help="same run's oof_predictions.npz")
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args(argv)
    with np.load(args.probs_npz, allow_pickle=True) as z:
        probs = np.asarray(z["probs"], dtype=np.float32)
        names = [str(x) for x in z["class_names"]]
        obs = np.asarray(z["obs_names"]).astype(str) if "obs_names" in z.files else None
    pi0 = oof_mean_prior(args.oof)
    if pi0.size != len(names):
        raise ValueError("OOF and external class counts differ; use the OOF of the same run")
    adj, pi, n_iter = em_prior_shift(probs, pi0, max_iter=args.max_iter)
    stem = args.probs_npz.name.replace("_l2_probs.npz", "")
    out = args.probs_npz.with_name(f"{stem}_l2_probs_priorshift.npz")
    np.savez_compressed(out, probs=adj, class_names=np.asarray(names, dtype=str),
                        obs_names=obs if obs is not None else np.arange(len(adj)).astype(str),
                        train_prior=pi0, target_prior=pi, n_iter=n_iter)
    table = pd.DataFrame({
        "class": names,
        "train_prior_oof_mean_softmax": pi0,
        "external_mean_softmax": probs.mean(axis=0),
        "external_argmax_fraction": np.bincount(probs.argmax(1), minlength=len(names)) / len(probs),
        "em_target_prior": pi,
        "em_argmax_fraction": np.bincount(adj.argmax(1), minlength=len(names)) / len(adj),
    })
    table.to_csv(args.probs_npz.with_name(f"{stem}_l2_prior_table.csv"), index=False)
    print(table.round(4).to_string(index=False))
    print(f"EM iterations {n_iter}; wrote {out}")


if __name__ == "__main__":
    main()
