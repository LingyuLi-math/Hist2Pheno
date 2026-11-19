#!/usr/bin/env python3
"""2026.11.17 Fold (and seed) ensemble for external slides.

Hist2Pheno, as released, is the mean of the five fold models of one run (or
of several runs, e.g. two seeds), not the single fold whose own validation
score was highest. Each member is predicted by the normal pipeline
(``external_validation.py`` / ``brca_external.py`` with ``UNI2_CKPT`` set to
``hce_group_fold_<k>.pt``) into its own folder; this script

  seed-features  hard-links the verified feature h5ad of the first member into
                 the other member folders, so the UNI2 token pooling is done
                 once. The feature marker is rewritten for the new path with
                 the same source stamp; nothing else is trusted.
  ensemble       averages the L1 / L12 / L2 softmax of the members (same
                 nuclei, same class order, checked), writes
                 ``<sample>_stardist_pred.h5ad`` and ``<sample>_l2_probs.npz``
                 in the usual layout, plus ``<sample>_ensemble_report.csv``
                 (L2 composition per member and for the mean, and how often
                 each member agrees with the mean).

2026.11.18, hierarchy-consistent labels. The three heads are separate
softmaxes, so independent argmaxes can disagree (L2 Tumor under L12 Lymphoid).
``--consistency`` decides the written labels:
  l2_parent    (default) pred_CT = argmax of the mean L2 softmax; pred_sublineage
               and pred_lineage are its ontology parents.
  fused        pred_CT = argmax of log p2 + w12 log p12[parent] + w1 log p1[grandparent]
               (product of experts, as posthoc.fused_scores); parents follow it.
  independent  each head's own argmax (the earlier behaviour).
The head argmaxes are always kept as pred_sublineage_head / pred_lineage_head,
and ``check`` reports how often they disagree with the L2 parents for any
prediction folder (single model or ensemble).

The ensemble folder has no ``.pipeline.json`` marker: it is derived from
verified member predictions, whose markers stay in their own folders.
Notebooks that read the h5ad / npz directly (NPC, BRCA axisfix) work as is.

    python code/External_val/fold_ensemble_external.py seed-features --sample S --members D0 D1 D2 D3 D4
    python code/External_val/fold_ensemble_external.py ensemble --sample S --members D0 ... D4 --out DIR
    python code/External_val/fold_ensemble_external.py check --sample S --members DIR [DIR ...]
(each D is ``<pred root>/<sample>``)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent / "Hist2Pheno_pkg"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

HEADS = (("L1", "l1_probs", "l1_class_names", "pred_lineage"),
         ("L12", "l12_probs", "l12_class_names", "pred_sublineage"),
         ("L2", "probs", "class_names", "pred_CT"))


CONSISTENCY = ("l2_parent", "fused", "independent")


def _maps(npz):
    """l2 -> l12 and l12 -> l1 index maps for the class order stored in the npz."""
    from pancancer_labels import hierarchy_maps

    classes = {"L1": [str(c) for c in npz["l1_class_names"]],
               "L12": [str(c) for c in npz["l12_class_names"]],
               "L2": [str(c) for c in npz["class_names"]]}
    m = hierarchy_maps(classes)
    return np.asarray(m["l2_to_l12"]), np.asarray(m["l12_to_l1"])


def consistent_labels(p1, p12, p2, l2_to_l12, l12_to_l1, *, mode="l2_parent", w12=1.0, w1=0.5):
    """Index labels (l1, l12, l2) for one of CONSISTENCY."""
    if mode == "independent":
        return p1.argmax(1), p12.argmax(1), p2.argmax(1)
    if mode == "fused":
        eps = 1e-8
        score = (np.log(p2 + eps) + w12 * np.log(p12 + eps)[:, l2_to_l12]
                 + w1 * np.log(p1 + eps)[:, l12_to_l1[l2_to_l12]])
        y2 = score.argmax(1)
    elif mode == "l2_parent":
        y2 = p2.argmax(1)
    else:
        raise ValueError(f"consistency must be one of {CONSISTENCY}")
    y12 = l2_to_l12[y2]
    return l12_to_l1[y12], y12, y2


def disagreement(p1, p12, p2, l2_to_l12, l12_to_l1) -> dict:
    """How often the independent head argmaxes disagree with the L2 argmax's parents."""
    y2, y12, y1 = p2.argmax(1), p12.argmax(1), p1.argmax(1)
    return {"L12_head_vs_L2_parent": float((y12 != l2_to_l12[y2]).mean()),
            "L1_head_vs_L2_grandparent": float((y1 != l12_to_l1[l2_to_l12[y2]]).mean()),
            "L1_head_vs_L12_head_parent": float((y1 != l12_to_l1[y12]).mean())}


def check(sample: str, members) -> pd.DataFrame:
    rows = []
    for m in members:
        d = _load_member(Path(m), sample)
        a, b = _maps(d)
        rows.append({"folder": Path(m).parent.name, "n": int(len(d["probs"])),
                     **disagreement(d["l1_probs"], d["l12_probs"], d["probs"], a, b)})
    return pd.DataFrame(rows)


def _paths(member: Path, sample: str):
    member = Path(member)
    return {"feat": member / f"{sample}_stardist_features.h5ad",
            "feat_marker": member / f"{sample}_stardist_features.source.json",
            "pred": member / f"{sample}_stardist_pred.h5ad",
            "pred_marker": member / f"{sample}_stardist_pred.pipeline.json",
            "probs": member / f"{sample}_l2_probs.npz"}


def seed_features(sample: str, members) -> list[str]:
    """Link member 0's completed feature h5ad into members 1..n."""
    import pipeline_provenance as pp

    src = _paths(members[0], sample)
    run = json.loads(src["feat_marker"].read_text())
    if run.get("state") != "completed":
        raise RuntimeError(f"{src['feat_marker']}: features of the first member are not complete")
    sha = run["outputs"].get(str(src["feat"].resolve()))
    if sha is None or pp.sha256(src["feat"]) != sha:
        raise RuntimeError(f"{src['feat']}: does not match its marker; run predict for the first member again")
    done = []
    for member in members[1:]:
        dst = _paths(member, sample)
        if pp.complete(dst["feat_marker"], run["stamp"], [dst["feat"]]):
            continue
        dst["feat"].parent.mkdir(parents=True, exist_ok=True)
        tmp = dst["feat"].with_name(dst["feat"].name + ".seed_tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src["feat"], tmp)
        except OSError:
            shutil.copy2(src["feat"], tmp)
        os.replace(tmp, dst["feat"])
        record = dict(run)
        record["outputs"] = {str(dst["feat"].resolve()): sha}
        record["seeded_from"] = str(src["feat"].resolve())
        pp.atomic_json(dst["feat_marker"], record)
        done.append(str(dst["feat"]))
    return done


def _load_member(member: Path, sample: str):
    p = _paths(member, sample)
    if not p["probs"].is_file() or not p["pred"].is_file():
        raise FileNotFoundError(f"{member}: no prediction (run predict for this member)")
    with np.load(p["probs"], allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    for head, pk, ck, _ in HEADS:
        if pk not in d or ck not in d:
            raise KeyError(f"{p['probs']}: missing {pk}/{ck} (prediction older than the five-head npz)")
    return d


def archive_previous(sample: str, out: Path) -> Path | None:
    """2026.11.18, move results derived from an earlier mean out of the way.

    Everything in ``<pred root>/<sample>/`` except the three ensemble outputs
    (pred h5ad, l2_probs npz, ensemble report), and ``<pred root>/figures/``,
    goes to ``<pred root>/_previous_<time>/``. Nothing is deleted.
    """
    import time

    out = Path(out)
    root = out.parent
    keep = {_paths(out, sample)["pred"].name, _paths(out, sample)["probs"].name, f"{sample}_ensemble_report.csv"}
    stale = [f for f in out.iterdir() if f.name not in keep] if out.is_dir() else []
    if (root / "figures").is_dir():
        stale.append(root / "figures")
    if not stale:
        return None
    dest = root / f"_previous_{time.strftime('%Y%m%d_%H%M%S')}"
    (dest / sample).mkdir(parents=True, exist_ok=True)
    for f in stale:
        os.replace(f, (dest / f.name) if f.name == "figures" else (dest / sample / f.name))
    return dest


def ensemble(sample: str, members, out: Path, *, consistency="l2_parent", w12=1.0, w1=0.5,
             archive=True) -> pd.DataFrame:
    import anndata as ad
    import pipeline_provenance as pp

    members = [Path(m) for m in members]
    data = [_load_member(m, sample) for m in members]
    ref = data[0]
    names = ref["obs_names"].astype(str)
    for m, d in zip(members[1:], data[1:]):
        if not np.array_equal(d["obs_names"].astype(str), names):
            raise ValueError(f"{m}: nuclei differ from {members[0]} (STAR_PROB_MIN or features differ)")
        for _, _, ck, _ in HEADS:
            if not np.array_equal(d[ck].astype(str), ref[ck].astype(str)):
                raise ValueError(f"{m}: {ck} order differs")
        if "checkpoint_family" in d and "checkpoint_family" in ref and str(d["checkpoint_family"]) != str(ref["checkpoint_family"]):
            raise ValueError(f"{m}: checkpoint family differs (different heads / labels)")

    mean = {pk: np.mean([d[pk].astype(np.float64) for d in data], axis=0).astype(np.float32) for _, pk, _, _ in HEADS}
    base = ad.read_h5ad(_paths(members[0], sample)["pred"])
    if not np.array_equal(np.asarray(base.obs_names).astype(str), names):
        raise ValueError("member 0 h5ad and npz rows differ")
    obs = base.obs.copy()
    # 2026.11.18, written labels follow the ontology (see --consistency); head argmaxes kept beside them
    l2_to_l12, l12_to_l1 = _maps(ref)
    y1, y12, y2 = consistent_labels(mean["l1_probs"], mean["l12_probs"], mean["probs"], l2_to_l12, l12_to_l1,
                                    mode=consistency, w12=w12, w1=w1)
    idx = {"L1": y1, "L12": y12, "L2": y2}
    for head, pk, ck, col in HEADS:
        cls = ref[ck].astype(str)
        obs[col] = cls[idx[head]]
        obs[f"conf_{head}"] = mean[pk][np.arange(len(names)), idx[head]]
        if head != "L2":
            obs[f"{col}_head"] = cls[mean[pk].argmax(1)]
    gap = disagreement(mean["l1_probs"], mean["l12_probs"], mean["probs"], l2_to_l12, l12_to_l1)
    l2_names = ref["class_names"].astype(str)
    member_l2 = np.stack([d["probs"].argmax(1) for d in data])          # (n_members, n_cells)
    ens_l2 = y2
    obs["ensemble_agree_L2"] = (member_l2 == ens_l2[None, :]).mean(0).astype(np.float32)

    pred = ad.AnnData(obs=obs)
    pred.obsm["spatial_HE"] = np.asarray(base.obsm["spatial_HE"])
    uns = {k: v for k, v in base.uns.items() if k in ("classes", "stardist_prob_min", "spatial_k", "feature_sha256",
                                                     "row_identity_sha256")}
    member_info = []
    for m in members:
        try:
            u = ad.read_h5ad(_paths(m, sample)["pred"], backed="r").uns
            member_info.append({"dir": str(m), "checkpoint": str(u.get("checkpoint", "")),
                                "checkpoint_sha256": str(u.get("checkpoint_sha256", ""))})
        except Exception:
            member_info.append({"dir": str(m)})
    uns.update(ensemble_method="mean softmax over members", ensemble_n=len(members),
               ensemble_members=json.dumps(member_info),
               label_consistency=consistency, fused_w12=float(w12), fused_w1=float(w1),
               head_disagreement=json.dumps(gap))
    pred.uns.update(uns)
    out = Path(out)
    if archive:
        moved = archive_previous(sample, out)
        if moved is not None:
            print(f"earlier derived files (niche, confidence, figures, ...) moved to {moved}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    o = _paths(out, sample)
    pp.atomic_h5ad(pred, o["pred"])
    pp.atomic_npz(o["probs"], probs=mean["probs"], class_names=ref["class_names"].astype(str), obs_names=names,
                  l1_probs=mean["l1_probs"], l1_class_names=ref["l1_class_names"].astype(str),
                  l12_probs=mean["l12_probs"], l12_class_names=ref["l12_class_names"].astype(str),
                  ensemble_members=np.asarray(json.dumps(member_info)))

    rows = []
    for i, m in enumerate(members):
        comp = pd.Series(l2_names[member_l2[i]]).value_counts(normalize=True)
        rows.append({"member": m.parent.name, "agree_with_ensemble": float((member_l2[i] == ens_l2).mean()),
                     **disagreement(data[i]["l1_probs"], data[i]["l12_probs"], data[i]["probs"], l2_to_l12, l12_to_l1),
                     **{f"frac_{c}": float(comp.get(c, 0.0)) for c in l2_names}})
    comp = pd.Series(l2_names[ens_l2]).value_counts(normalize=True)
    rows.append({"member": f"ENSEMBLE ({consistency})", "agree_with_ensemble": 1.0, **gap,
                 **{f"frac_{c}": float(comp.get(c, 0.0)) for c in l2_names}})
    report = pd.DataFrame(rows)
    report.to_csv(out / f"{sample}_ensemble_report.csv", index=False)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed-features")
    s.add_argument("--sample", required=True)
    s.add_argument("--members", nargs="+", type=Path, required=True)
    e = sub.add_parser("ensemble")
    e.add_argument("--sample", required=True)
    e.add_argument("--members", nargs="+", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--consistency", choices=CONSISTENCY, default="l2_parent")
    e.add_argument("--w12", type=float, default=1.0, help="fused: weight of the L12 head")
    e.add_argument("--w1", type=float, default=0.5, help="fused: weight of the L1 head")
    e.add_argument("--no-archive", action="store_true",
                   help="keep earlier derived files in place (default: move them to _previous_<time>/)")
    c = sub.add_parser("check", help="head disagreement of existing prediction folders")
    c.add_argument("--sample", required=True)
    c.add_argument("--members", nargs="+", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        print(check(a.sample, a.members).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return 0
    if a.cmd == "seed-features":
        done = seed_features(a.sample, a.members)
        print(f"seeded features into {len(done)} member folder(s)")
        return 0
    report = ensemble(a.sample, a.members, a.out, consistency=a.consistency, w12=a.w12, w1=a.w1,
                      archive=not a.no_archive)
    show = ["member", "agree_with_ensemble", "L12_head_vs_L2_parent", "L1_head_vs_L2_grandparent"] + [
        c for c in report.columns if c.startswith("frac_") and report[c].max() >= 0.05]
    pd.set_option("display.width", 220)
    print(report[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {_paths(a.out, a.sample)['pred']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
