#!/usr/bin/env python3
"""2026.11.17 BRCA 10x slide: Hist2Pheno arms against the pathologist spot map.

Reproduces the "10x breast slide against the pathologist map" table of
code/PanCancer/README.md, with two changes:

* capture disc = 55 µm (``visium_validation.capture_radius_px``), not the
  65 µm ``spot_diameter_fullres / 2`` used for the first table;
* every arm is scored with four label modes (``posthoc.label_modes``):
  argmax (= pred_CT), the run's ``decision_rule_l2_only.json``, EM prior
  shift on this slide, and kNN smoothing. The mode that matches the
  pathologist map best is the one to use for external maps.

Ground truth: ``data/Visium/BRCA/10x_data/metadata.tsv`` (``ID``,
``fine_annot_type``), collapsed to IDC / DCIS/LCIS / Tumor edge / Healthy.
A spot's call is the majority L2 label of the StarDist nuclei inside the disc;
spots without nuclei are left out (their number is reported).

Columns of the table (fractions of annotated spots):
  IDC_called_Tumor           IDC spots whose majority is Tumor
  DCIS_called_DCIS           DCIS/LCIS spots whose majority is DCIS
  cancer_called_neoplastic   IDC + DCIS/LCIS spots called Tumor or DCIS  (sensitivity)
  healthy_not_neoplastic     Healthy spots called neither Tumor nor DCIS (specificity)
  name_correct               among cancer spots called Tumor or DCIS, IDC->Tumor and DCIS->DCIS
  score                      mean of sensitivity, specificity and name_correct

    python code/External_val/BRCA/brca_10x_annotation_eval.py
    python code/External_val/BRCA/brca_10x_annotation_eval.py --arms spatial_k8 l12_links_1 --radius-um 65
    # 2026.11.17, fold models and ensembles from predict_fold_ensemble.sh BRCA
    python code/External_val/BRCA/brca_10x_annotation_eval.py --arms l12_links_1 \\
        fold4_l12_links_1_ivsafe_seed42 ens5_l12_links_1_ivsafe_seed42 ens5_l12_links_1_ivsafe_seed44 \\
        ens10_l12_links_1_ivsafe_seeds_42_44
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "code", REPO / "code" / "Hist2Pheno_pkg"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nuclei_to_spots import nuclei_to_spots  # noqa: E402
from posthoc import LABEL_MODES, label_modes  # noqa: E402
from visium_validation import CAPTURE_UM, SPACERANGER_SPOT_UM, read_tissue_positions  # noqa: E402

SAMPLE = "V1_Breast_Cancer_Block_A_Section_1_image"
DATA_10X = REPO / "data" / "Visium" / "BRCA" / "10x_data"
PAN = REPO / "data" / "PanCancer"
BRCA = REPO / "data" / "Visium" / "BRCA"
ARMS = (
    "legacy_nospatial", "adjacent_nospatial", "spatial_k8", "spatial_k8_gate",
    "l12_links_1", "l12_links_2", "l12_balanced", "gate_l12_balanced",
    "focal_gamma1", "balanced_sampler", "l12_balanced_nospatial",
)
REGIONS = ("IDC", "DCIS/LCIS", "Tumor edge", "Healthy")
NEOPLASTIC = ("Tumor", "DCIS")


def coarse_label(fine: str) -> str:
    name = str(fine)
    for prefix, region in (("DCIS/LCIS", "DCIS/LCIS"), ("IDC", "IDC"),
                           ("Tumor_edge", "Tumor edge"), ("Healthy", "Healthy")):
        if name.startswith(prefix):
            return region
    raise KeyError(f"unmapped fine_annot_type: {name}")


def annotated_spots(radius_um: float = CAPTURE_UM):
    """In-tissue annotated spots (barcode, x, y, region) and the disc radius in fullres px."""
    meta = pd.read_csv(DATA_10X / "metadata.tsv", sep="\t")
    pos = read_tissue_positions(DATA_10X / "spatial")
    spots = meta[["ID", "fine_annot_type"]].merge(pos, left_on="ID", right_on="barcode", how="inner")
    spots["region"] = spots["fine_annot_type"].map(coarse_label)
    sf = json.loads((DATA_10X / "spatial" / "scalefactors_json.json").read_text())
    radius = float(sf["spot_diameter_fullres"]) * radius_um / SPACERANGER_SPOT_UM / 2.0
    return spots.reset_index(drop=True), radius


def spot_majority(xy, labels, spots, radius_px):
    """Majority L2 label per spot ('' for spots without nuclei)."""
    names = sorted(set(map(str, labels)))
    cap = nuclei_to_spots(np.asarray(xy, float), spots[["x", "y"]].to_numpy(float), float(radius_px),
                          shape="circle", labels=np.asarray(labels).astype(str), class_names=names)
    return np.asarray(cap["dominant"], dtype=object), np.asarray(cap["n_nuclei"])


def score_calls(region, call) -> dict:
    region = np.asarray(region); call = np.asarray(call).astype(str)
    neo = np.isin(call, NEOPLASTIC)
    idc, dcis, healthy = region == "IDC", region == "DCIS/LCIS", region == "Healthy"
    cancer = idc | dcis
    named = cancer & neo
    correct = (idc & (call == "Tumor")) | (dcis & (call == "DCIS"))
    out = {
        "IDC_called_Tumor": float((call[idc] == "Tumor").mean()) if idc.any() else np.nan,
        "DCIS_called_DCIS": float((call[dcis] == "DCIS").mean()) if dcis.any() else np.nan,
        "cancer_called_neoplastic": float(neo[cancer].mean()) if cancer.any() else np.nan,
        "healthy_not_neoplastic": float((~neo[healthy]).mean()) if healthy.any() else np.nan,
        "name_correct": float(correct[named].sum() / named.sum()) if named.any() else np.nan,
    }
    out["score"] = float(np.nanmean([out["cancer_called_neoplastic"], out["healthy_not_neoplastic"],
                                     out["name_correct"]]))
    for r in REGIONS:
        top = pd.Series(call[region == r]).value_counts(normalize=True)
        out[f"{r}_top_call"] = f"{top.index[0]} {top.iloc[0]:.2f}" if len(top) else ""
    return out


def arm_paths(arm, data_tag, seed, pred_tag=""):
    # 2026.11.17, fold members and ensembles of predict_fold_ensemble.sh:
    #   fold<k>_<run> -> pred_uni2_fold<k>_<run>, ens<N>_<run> -> pred_uni2_ens<N>_<run>
    #   <run> = l12_links_1_ivsafe_seed42 (or ..._seeds_42_44); the decision rule / prior come from the first seed
    m = re.match(r"^(fold\d+|ens\d+)_(.+?)_(ivsafe_)?seeds?_?(\d+)", arm)
    if m:
        base_arm, ivsafe, first_seed = m.group(2), m.group(3), m.group(4)
        tag = f"{data_tag}_ivsafe" if ivsafe else data_tag
        result = PAN / f"result_uni2_ablation_{base_arm}_{tag}_paired_v2_seed{first_seed}"
        pred = BRCA / f"pred_uni2_{arm}{pred_tag}" / SAMPLE
        return result, pred / f"{SAMPLE}_stardist_pred.h5ad", pred / f"{SAMPLE}_l2_probs.npz"
    result = PAN / f"result_uni2_ablation_{arm}_{data_tag}_paired_v2_seed{seed}"
    pred = BRCA / f"pred_uni2_axisfix_{arm}{pred_tag}" / SAMPLE
    return result, pred / f"{SAMPLE}_stardist_pred.h5ad", pred / f"{SAMPLE}_l2_probs.npz"


def evaluate(arms=ARMS, *, data_tag="new1066axisfix", seed=42, radius_um=CAPTURE_UM,
             modes=LABEL_MODES, pred_tag="", write_labels=True):
    """One row per (arm, mode). Also writes ``{SAMPLE}_l2_label_modes.csv`` next
    to each prediction (nucleus label per mode) when ``write_labels``."""
    import anndata as ad

    spots, radius = annotated_spots(radius_um)
    print(f"{len(spots):,} annotated spots  radius {radius:.1f} px ({radius_um:g} µm)  "
          + "  ".join(f"{r} {int((spots.region == r).sum())}" for r in REGIONS), flush=True)
    rows = []
    for arm in arms:
        result, h5, npz = arm_paths(arm, data_tag, seed, pred_tag)
        if not h5.is_file() or not npz.is_file():
            print(f"skip {arm}: no prediction ({h5.parent})", flush=True)
            continue
        adata = ad.read_h5ad(h5, backed="r")
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        obs = np.asarray(adata.obs_names).astype(str)
        adata.file.close()
        with np.load(npz, allow_pickle=True) as z:
            probs = np.asarray(z["probs"], dtype=np.float32)
            names = [str(x) for x in z["class_names"]]
        if len(probs) != len(xy):
            raise RuntimeError(f"{arm}: probs {len(probs)} != nuclei {len(xy)}")
        labs = label_modes(probs, names, result, xy=xy, modes=modes)
        if write_labels:
            pd.DataFrame({"obs_name": obs, **labs}).to_csv(h5.with_name(f"{SAMPLE}_l2_label_modes.csv"), index=False)
        for mode, lab in labs.items():
            call, n = spot_majority(xy, lab, spots, radius)
            keep = n > 0
            row = {"arm": arm, "mode": mode, "n_spots": int(keep.sum()), "n_empty": int((~keep).sum())}
            row.update(score_calls(spots.region.to_numpy()[keep], call[keep]))
            rows.append(row)
        missing = [m for m in modes if m not in labs]
        if missing:
            print(f"{arm}: modes not available {missing} (no decision rule / OOF in {result.name})", flush=True)
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["score"], ascending=False).reset_index(drop=True)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--modes", nargs="+", default=list(LABEL_MODES), choices=LABEL_MODES)
    ap.add_argument("--data-tag", default="new1066axisfix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius-um", type=float, default=CAPTURE_UM, help="55 = capture area; 65 = first table")
    ap.add_argument("--pred-tag", default="", help="PROB_TAG of the prediction folders, e.g. _prob0.50")
    ap.add_argument("--out", type=Path, default=BRCA / "pred_uni2_axisfix_figures" / f"{SAMPLE}_annotation_eval.csv")
    a = ap.parse_args(argv)
    table = evaluate(a.arms, data_tag=a.data_tag, seed=a.seed, radius_um=a.radius_um,
                     modes=a.modes, pred_tag=a.pred_tag)
    if table.empty:
        print("no arm has a prediction; run code/External_val/BRCA/demo_uni2_axisfix.sh predict")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    cols = ["arm", "mode", "IDC_called_Tumor", "DCIS_called_DCIS", "cancer_called_neoplastic",
            "healthy_not_neoplastic", "name_correct", "score"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
