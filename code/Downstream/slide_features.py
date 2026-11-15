"""2026.11.15 Slide-level "virtual phenotype" features from Hist2Pheno predictions.

Hist2Prot (Methods, "Protein-aware latent aggregator", Eq. 12) mean-pools the
per-cell virtual proteins inside each H&E region and feeds the region vectors
to a Transformer. The Hist2Pheno analogue pools per-nucleus *probabilities*:

    per nucleus   [P(L2) 20 | P(L12) 8 | nbr_comp 20 (A9+) | tissue 7 (A11)]
    per region    mean over nuclei in a square tile + log1p(nucleus count)

Soft probabilities are pooled, not argmax labels, so rare classes (B_cell) are
not erased by the argmax. Regions follow the UNI2 patch grid by default
(224 px at 0.5 µm/px = 112 µm), so region i lines up with UNI2 patch i when
both are tiled from the same origin.

Also writes an interpretable per-slide table (soft fractions, B/T
co-localisation, lymphoid-aggregate burden) for Cox / logistic baselines.

    python code/Downstream/slide_features.py \
        --pred data/TCGA_HNSC/pred_uni2_A11/<slide>/<slide>_l2_probs.npz \
        --coords data/TCGA_HNSC/pred_uni2_A11/<slide>/<slide>_nuclei.csv \
        --mpp 0.25 --out data/TCGA_HNSC/pheno_regions/<slide>.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_probs(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        blocks, names = [], []
        p2 = np.asarray(z["probs"], np.float32)
        l2_names = [str(x) for x in z["class_names"]]
        blocks.append(p2); names += [f"L2:{c}" for c in l2_names]
        if "l12_probs" in z.files:
            blocks.append(np.asarray(z["l12_probs"], np.float32))
            names += [f"L12:{c}" for c in z["l12_class_names"].astype(str)]
        if "nbr_comp_probs" in z.files:
            blocks.append(np.asarray(z["nbr_comp_probs"], np.float32))
            names += [f"NBR:{c}" for c in l2_names]
        if "tissue_comp_probs" in z.files:
            blocks.append(np.asarray(z["tissue_comp_probs"], np.float32))
            names += [f"TIS:{c}" for c in z["tissue_comp_names"].astype(str)]
        obs = np.asarray(z["obs_names"]).astype(str) if "obs_names" in z.files else None
    return np.concatenate(blocks, 1), names, l2_names, obs


def region_features(cell_feats, xy_um, region_um=112.0):
    """Mean-pool cell features per square region (Hist2Prot Eq. 12).

    Returns ``(F, centres_um, counts)``; empty regions are absent.
    """
    xy_um = np.asarray(xy_um, np.float64)
    ij = np.floor(xy_um / float(region_um)).astype(np.int64)
    keys, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    F = np.zeros((len(keys), cell_feats.shape[1]), np.float64)
    np.add.at(F, inv, cell_feats)
    F /= counts[:, None]
    F = np.concatenate([F, np.log1p(counts)[:, None]], 1).astype(np.float32)
    centres = (keys + 0.5) * float(region_um)
    return F, centres, counts


def slide_summary(cell_feats, names, region_F, region_names, counts):
    """Interpretable per-slide descriptors (one row)."""
    col = {n: i for i, n in enumerate(names)}
    out = {"n_nuclei": int(cell_feats.shape[0]), "n_regions": int(len(counts))}
    for n, i in col.items():
        if n.startswith("L2:") or n.startswith("L12:") or n.startswith("TIS:"):
            out[f"frac_{n}"] = float(cell_feats[:, i].mean())
    rc = {n: i for i, n in enumerate(region_names)}
    if "L2:B_cell" in rc and "L12:T_cell" in rc:
        b = region_F[:, rc["L2:B_cell"]]; t = region_F[:, rc["L12:T_cell"]]
        out["region_corr_B_T"] = float(np.corrcoef(b, t)[0, 1]) if b.std() > 0 and t.std() > 0 else np.nan
        # lymphoid-aggregate proxy: regions where B >= 10 % and B+T >= 50 %
        agg = (b >= 0.10) & (b + t >= 0.50)
        out["lymphoid_aggregate_region_frac"] = float(agg.mean())
    if "TIS:Lymphoid" in rc:
        out["lymphoid_compartment_region_frac"] = float((region_F[:, rc["TIS:Lymphoid"]] >= 0.5).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="*_l2_probs.npz from external_consistency")
    ap.add_argument("--coords", type=Path, required=True,
                    help="CSV with obs_name (optional), centroid_x, centroid_y in native pixels")
    ap.add_argument("--mpp", type=float, required=True, help="native µm/px of the H&E")
    ap.add_argument("--region-um", type=float, default=112.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    feats, names, _, obs = _load_probs(args.pred)
    c = pd.read_csv(args.coords)
    if obs is not None and "obs_name" in c:
        c = c.set_index(c["obs_name"].astype(str)).loc[obs]
    if len(c) != len(feats):
        raise ValueError(f"{len(c)} coordinates vs {len(feats)} nuclei")
    xy_um = c[["centroid_x", "centroid_y"]].to_numpy(float) * args.mpp
    F, centres, counts = region_features(feats, xy_um, args.region_um)
    region_names = names + ["log1p_nuclei"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, region_features=F, region_centres_um=centres, region_counts=counts,
                        feature_names=np.asarray(region_names, dtype=str), region_um=args.region_um)
    summ = slide_summary(feats, names, F, region_names, counts)
    pd.DataFrame([summ]).to_csv(args.out.with_suffix(".summary.csv"), index=False)
    print(f"{len(feats):,} nuclei -> {len(F):,} regions × {F.shape[1]} features -> {args.out}")


if __name__ == "__main__":
    main()
