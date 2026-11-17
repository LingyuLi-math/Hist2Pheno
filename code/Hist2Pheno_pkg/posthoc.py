"""Post-hoc layer on top of trained Hist2Pheno heads.

2026.11.17 / 整合：本文件合并了原来的四个后处理脚本，原备注逐段保留：
  oof_evaluation.py            -> section 1  (OOF metrics, patient bootstrap)
  prior_shift.py               -> section 2  (Saerens EM label shift)
  oof_abundance_calibration.py -> section 3  (per-section abundance calibration)
  fold_ensemble.py             -> section 4  (five-fold external ensemble)
and adds section 5 (2026.11.17, new): decision rules that turn the three
softmax heads into **labels** with higher Accuracy / macro-F1 at the same
AUROC (hierarchy-consistent fusion, temperature, per-class bias fitted by
cross-fitting on OOF, optional spatial smoothing and abstention).

CLI (one entry point, the old flags are unchanged):

    python code/Hist2Pheno_pkg/posthoc.py evaluate RESULT_DIR [CHECKPOINT] [--bootstrap N]
    python code/Hist2Pheno_pkg/posthoc.py fit-decision RESULT_DIR [--objective macro_f1]
    python code/Hist2Pheno_pkg/posthoc.py decide PROBS_NPZ --rule RESULT_DIR/decision_rule.json
    python code/Hist2Pheno_pkg/posthoc.py prior-shift PROBS_NPZ --oof OOF_NPZ
    python code/Hist2Pheno_pkg/posthoc.py calibrate-abundance --oof ... --oof-rows ... --output ...
    python code/Hist2Pheno_pkg/posthoc.py fold-ensemble NPC --ckpt-dir ... --pred-base ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PRIOR_SHIFT_DOC = """2026.11.15 label-shift (prior) correction for external L2 softmax.

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

    python code/Hist2Pheno_pkg/posthoc.py prior-shift \
        data/Visium/NPC/pred_uni2_A3_bestmlp_paired_v2_seed42/S/S_l2_probs.npz \
        --oof data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_predictions.npz

Writes ``S_l2_probs_priorshift.npz`` and ``S_l2_prior_table.csv`` next to the input.
Use the adjusted probabilities for abundance maps only. Do not overwrite
``pred_CT``; the run is unsupervised and cannot be checked without nucleus GT.
"""
ABUNDANCE_DOC = """2026.11.15 Per-section abundance calibration on OOF (Hist2Prot-style calibration curves).

Hist2Prot reports per-cohort calibration of predicted vs measured abundance.
The Hist2Pheno analogue: for each held-out section, compare the true L2
fraction with (a) the mean OOF softmax and (b) the argmax fraction.
A slope < 1 for B_cell means the model under-calls B where B is abundant,
which is the training-cohort version of the NPC symptom.

    python code/Hist2Pheno_pkg/posthoc.py calibrate-abundance \
        --oof data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_predictions.npz \
        --oof-rows data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/oof_rows.csv \
        --output data/PanCancer/calibration_A3

Writes ``section_fractions.csv`` (long table) and ``calibration_summary.csv``
(per class × cohort: slope through origin, Pearson, Spearman, Lin's CCC, mean
bias). Compare runs (A3 vs A8 vs A9) on the same table.
"""
FOLD_ENSEMBLE_DOC = """2026.11.15 five-fold ensemble for external StarDist cohorts (UNI2).

``best_mlp_gpu.pt`` is one fold (A2 fold 1, A3 fold 0). With
``--inner-val-fraction 0.2`` that fold saw ~64 % of the training sections,
and the fold was picked by its own held-out score. This script scores the
same external slide with every ``hce_group_fold_{k}.pt`` through the normal
verified pipeline (each fold uses its own artifact scaler), then averages
the L2 softmax.

The per-fold table is also the cheapest stability check: if mean P(B_cell)
moves a lot between folds, the B estimate on this slide is not reliable.

    STAR_PROB_MIN=0 python code/Hist2Pheno_pkg/posthoc.py fold-ensemble NPC \
        --ckpt-dir data/PanCancer/result_uni2_A3_adjacent_spatial_k8mean_gate_paired_v2_seed42/cross_dataset_cv/A3_adjacent_spatial_k8mean_gate \
        --pred-base pred_uni2_A3_fold

Writes ``data/Visium/NPC/<pred-base>{k}/`` per fold and
``<pred-base>_ens/{sample}/{sample}_l2_probs_ensemble.npz`` plus
``fold_mean_probs.csv``.
"""


########################################################
# Section 1. (from oof_evaluation.py)
########################################################
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score  # noqa: E402


########################################################
# 2026.10.24, inspect true OOF by cohort / 明确支持数与队列内指标
########################################################


# 2026.11.17, helpers for section 1 / 新增辅助函数
_HEAD_KEYS = (("L1", "l1_head"), ("L12", "l12"), ("L2", "l2"))


def _head_classes(z, checkpoint=None):
    """{'L1','L12','L2'} -> class names, from the checkpoint artifact or the OOF npz."""
    if checkpoint is not None:
        from checkpoint_artifact import read_artifact
        return read_artifact(checkpoint, required=True)['metadata']['head_classes']
    out = {}
    for head, key in _HEAD_KEYS:
        if f"{key}_class_names" not in z.files:
            raise ValueError("Legacy OOF without class names; pass the checkpoint")
        out[head] = [str(x) for x in z[f"{key}_class_names"]]
    return out


def _balanced_accuracy(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    cls = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in cls])) if len(cls) else np.nan


def evaluate(result_dir,checkpoint=None):
    result_dir=Path(result_dir)
    rows=pd.read_csv(result_dir/'oof_rows.csv')
    z=np.load(result_dir/'oof_predictions.npz',allow_pickle=False)
    ids=z['row_indices'];order=np.argsort(ids)
    if len(ids)!=len(rows) or not np.array_equal(np.sort(ids),np.arange(len(rows))):
        raise ValueError('This is not complete unique-cell OOF; repeated holdout is not k-fold OOF')
    # 2026.11.17, class names from the OOF file when no checkpoint is given / 不再强制读取 checkpoint
    classes=_head_classes(z,checkpoint)
    metrics=[];per_class=[]
    groups={'ALL':np.ones(len(rows),bool)}
    groups.update({ds:rows.section_id.str.split('::').str[0].to_numpy()==ds for ds in rows.section_id.str.split('::').str[0].unique()})
    for head,key in [('L1','l1_head'),('L12','l12'),('L2','l2')]:
        y=z[key+'_labels'][order];prob=z[key+'_probs'][order];pred=prob.argmax(1)
        for cohort,mask in groups.items():
            yt,yp,pr=y[mask],pred[mask],prob[mask];aucs=[];aps=[]
            for c,name in enumerate(classes[head]):
                binary=yt==c;positive=int(binary.sum());negative=int((~binary).sum())
                auc=roc_auc_score(binary,pr[:,c]) if positive and negative else np.nan
                ap=average_precision_score(binary,pr[:,c]) if positive and negative else np.nan
                if np.isfinite(auc):aucs.append(auc);aps.append(ap)
                per_class.append(dict(cohort=cohort,head=head,cell_type=name,positive=positive,negative=negative,auroc=auc,auprc=ap))
            metrics.append(dict(cohort=cohort,head=head,n_cells=len(yt),n_evaluable_auc_classes=len(aucs),
                macro_f1_supported=f1_score(yt,yp,labels=np.unique(yt),average='macro',zero_division=0),
                macro_f1_all_classes=f1_score(yt,yp,labels=np.arange(len(classes[head])),average='macro',zero_division=0),
                macro_auroc=np.mean(aucs) if aucs else np.nan,macro_auprc=np.mean(aps) if aps else np.nan,
                # 2026.11.17, label metrics next to AUROC / 同时报告 accuracy 与 balanced accuracy
                accuracy=float(np.mean(yt==yp)) if len(yt) else np.nan,
                balanced_accuracy=_balanced_accuracy(yt,yp)))
    met,pcl=pd.DataFrame(metrics),pd.DataFrame(per_class)
    met.to_csv(result_dir/'oof_metrics_by_cohort.csv',index=False);pcl.to_csv(result_dir/'oof_metrics_per_class.csv',index=False)
    return met,pcl


def patient_bootstrap_f1(y,pred,patients,n_classes,*,n_bootstrap=500,seed=42):
    """Patient-cluster bootstrap using confusion matrices; no repeated cell inference."""
    if n_bootstrap<1:raise ValueError("n_bootstrap must be positive")
    patients=np.asarray(patients)
    unique=np.unique(patients)
    if len(unique)<2:raise ValueError('Patient CI needs at least two independent patients')
    matrices=np.stack([np.bincount(y[patients==p]*n_classes+pred[patients==p],minlength=n_classes*n_classes).reshape(n_classes,n_classes) for p in unique])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n_bootstrap):
        c=matrices[rng.integers(0,len(unique),size=len(unique))].sum(0)
        den=c.sum(0)+c.sum(1)
        values.append(float(np.divide(2*np.diag(c),den,out=np.zeros(n_classes,float),where=den>0).mean()))
    return np.quantile(values,[.025,.975])


def bootstrap_result(result_dir,checkpoint=None,n_bootstrap=500):
    result_dir=Path(result_dir);rows=pd.read_csv(result_dir/'oof_rows.csv')
    if 'patient_id' not in rows or rows.patient_id.isna().any():
        raise ValueError('Supply a real patient-map when training before computing patient CI')
    z=np.load(result_dir/'oof_predictions.npz',allow_pickle=False);order=np.argsort(z['row_indices'])
    if not np.array_equal(np.sort(z['row_indices']),np.arange(len(rows))):raise ValueError('Incomplete/duplicate OOF rows')
    classes=_head_classes(z,checkpoint);out=[]  # 2026.11.17, OOF class names when no checkpoint
    for head,key in [('L1','l1_head'),('L12','l12'),('L2','l2')]:
        y=z[key+'_labels'][order];pred=z[key+'_probs'][order].argmax(1)
        lo,hi=patient_bootstrap_f1(y,pred,rows.patient_id.to_numpy(),len(classes[head]),n_bootstrap=n_bootstrap)
        out.append(dict(head=head,metric='macro_F1_all_classes',lower_95=float(lo),upper_95=float(hi),bootstrap_unit='patient',replicates=n_bootstrap))
    result=pd.DataFrame(out);result.to_csv(result_dir/'oof_patient_bootstrap_f1.csv',index=False);return result


########################################################
# Section 2. (from prior_shift.py) — docstring kept in PRIOR_SHIFT_DOC
########################################################
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


def prior_shift_main(argv=None):
    ap = argparse.ArgumentParser(description=PRIOR_SHIFT_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
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


########################################################
# Section 3. (from oof_abundance_calibration.py) — docstring kept in ABUNDANCE_DOC
########################################################
from scipy.stats import pearsonr, spearmanr  # noqa: E402


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


def abundance_calibration_main(argv=None):
    ap = argparse.ArgumentParser(description=ABUNDANCE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
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


########################################################
# Section 4. (from fold_ensemble.py) — docstring kept in FOLD_ENSEMBLE_DOC
########################################################


def fold_ensemble_main(argv=None):
    ap = argparse.ArgumentParser(description=FOLD_ENSEMBLE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
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


########################################################
# Section 5. 2026.11.17, decision rules for Accuracy / F1 (new)
# AUROC 只看排序；Accuracy / F1 还取决于“怎样从概率得到标签”。
# 训练用 balanced 类权重 / 平衡采样时，网络学到的先验偏向稀有类，
# argmax 会把大量 Tumor/Epithelial 判成稀有类 -> 稀有类 precision 低、
# 整体 accuracy 低。三个 head 各自 argmax 还会互相矛盾（L1 Immune 与
# L2 免疫类之和不一致）。这里在 OOF 上拟合一个很小的决策层：
#   1) 层级融合：log p(L2) + w12·log p(L12 父类) + w1·log p(L1 祖父类)，
#      得到一个一致的 L2 分布，L12 / L1 由求和得到，三层标签不再矛盾；
#   2) 温度 T（只影响置信度 / 弃权，不改变 argmax）；
#   3) 每类偏置 b_c（坐标上升，直接最大化 macro-F1 / accuracy / 二者平均）；
#   4) 用 fold_ids 交叉拟合：在 k-1 个 fold 上拟合、在第 k 个 fold 上评估，
#      报告的提升不是在同一批细胞上“过拟合”出来的。
# 外部切片只存了 L2 softmax，所以同时拟合一个 l2_only 版本（w12 = w1 = 0）。
########################################################
OBJECTIVES = ("macro_f1", "accuracy", "f1_acc", "balanced_accuracy", "weighted_f1")


def _confusion(y, pred, n):
    return np.bincount(np.asarray(y) * n + np.asarray(pred), minlength=n * n).reshape(n, n)


def label_metrics(y, pred, n_classes) -> dict:
    """Accuracy, balanced accuracy, macro-F1 (classes present in y), weighted-F1."""
    c = _confusion(y, pred, n_classes).astype(np.float64)
    tp = np.diag(c); sup = c.sum(1); prd = c.sum(0)
    f1 = np.divide(2 * tp, sup + prd, out=np.zeros(n_classes), where=(sup + prd) > 0)
    rec = np.divide(tp, sup, out=np.zeros(n_classes), where=sup > 0)
    present = sup > 0
    n = max(sup.sum(), 1.0)
    return {
        "accuracy": float(tp.sum() / n),
        "balanced_accuracy": float(rec[present].mean()) if present.any() else np.nan,
        "macro_f1": float(f1[present].mean()) if present.any() else np.nan,
        "weighted_f1": float((f1 * sup).sum() / n),
        "per_class_f1": f1,
    }


def _objective_value(c, objective):
    """Objective from a confusion matrix (rows = truth)."""
    c = c.astype(np.float64)
    tp = np.diag(c); sup = c.sum(1); prd = c.sum(0)
    present = sup > 0
    acc = tp.sum() / max(sup.sum(), 1.0)
    if objective == "accuracy":
        return acc
    f1 = np.divide(2 * tp, sup + prd, out=np.zeros(len(tp)), where=(sup + prd) > 0)
    if objective == "macro_f1":
        return f1[present].mean()
    if objective == "weighted_f1":
        return (f1 * sup).sum() / max(sup.sum(), 1.0)
    if objective == "balanced_accuracy":
        return np.divide(tp, sup, out=np.zeros(len(tp)), where=present)[present].mean()
    if objective == "f1_acc":
        return 0.5 * (f1[present].mean() + acc)
    raise ValueError(f"objective must be one of {OBJECTIVES}")


def _log(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, None))


def fused_scores(p2, p12=None, p1=None, *, l2_to_l12=None, l12_to_l1=None, w12=0.0, w1=0.0):
    """Unnormalised log-score per L2 class from the three heads (product of experts)."""
    s = _log(p2)
    if w12 and p12 is not None:
        s = s + w12 * _log(p12)[:, np.asarray(l2_to_l12)]
    if w1 and p1 is not None:
        l2_to_l1 = np.asarray(l12_to_l1)[np.asarray(l2_to_l12)]
        s = s + w1 * _log(p1)[:, l2_to_l1]
    return s


def softmax_rows(s, temperature=1.0):
    z = np.asarray(s, dtype=np.float64) / float(temperature)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(scores, y, *, lo=0.05, hi=20.0, iters=40):
    """1-D golden-section search for T minimising NLL of softmax(scores / T)."""
    y = np.asarray(y)
    rows = np.arange(len(y))

    def nll(t):
        z = scores / t
        z = z - z.max(1, keepdims=True)
        return float(-(z[rows, y] - np.log(np.exp(z).sum(1))).mean())

    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(np.exp(c)), nll(np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = nll(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = nll(np.exp(d))
    return float(np.exp((a + b) / 2))


def fit_class_bias(scores, y, *, objective="macro_f1", grid=np.linspace(-4, 4, 33), sweeps=3,
                   init=None):
    """Per-class additive bias on log-scores that maximises ``objective``.

    Coordinate ascent: for class c, every candidate b_c only moves cells
    between c and the best other class, so each candidate costs one bincount.
    Returns ``(bias, objective_value)``.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, k = s.shape
    b = np.zeros(k) if init is None else np.asarray(init, dtype=np.float64).copy()
    grid = np.asarray(grid, dtype=np.float64)
    best_val = _objective_value(_confusion(y, (s + b).argmax(1), k), objective)
    for _ in range(int(sweeps)):
        improved = False
        for c in range(k):
            t = s + b
            t[:, c] = -np.inf
            other = t.argmax(1)
            other_val = t[np.arange(n), other]
            sc = s[:, c]
            vals = []
            for g in grid:
                pred = np.where(sc + g > other_val, c, other)
                vals.append(_objective_value(_confusion(y, pred, k), objective))
            j = int(np.argmax(vals))
            if vals[j] > best_val + 1e-9:
                best_val, b[c], improved = vals[j], grid[j], True
        if not improved:
            break
    return b, float(best_val)


@dataclass
class DecisionRule:
    """Everything needed to turn softmax heads into labels. JSON-serialisable."""
    classes_l2: list
    classes_l12: list
    classes_l1: list
    l2_to_l12: list
    l12_to_l1: list
    w12: float = 0.0
    w1: float = 0.0
    temperature: float = 1.0
    bias: list = field(default_factory=list)
    objective: str = "macro_f1"
    min_confidence: float = 0.0          # 0 = never abstain
    variant: str = "full"                # "full" (three heads) or "l2_only"

    def scores(self, p2, p12=None, p1=None):
        w12, w1 = (self.w12, self.w1) if self.variant == "full" else (0.0, 0.0)
        if self.variant == "full" and (p12 is None or p1 is None) and (w12 or w1):
            raise ValueError("this rule needs L12 and L1 probabilities; use the l2_only rule")
        s = fused_scores(p2, p12, p1, l2_to_l12=self.l2_to_l12, l12_to_l1=self.l12_to_l1, w12=w12, w1=w1)
        return s + (np.asarray(self.bias) if len(self.bias) else 0.0)

    def apply(self, p2, p12=None, p1=None) -> dict:
        """L2 / L12 / L1 indices (consistent along the hierarchy), calibrated
        L2 probabilities and confidence. Abstained cells get index -1."""
        s = self.scores(p2, p12, p1)
        q = softmax_rows(s, self.temperature)
        l2 = q.argmax(1)
        conf = q[np.arange(len(q)), l2]
        m12 = np.asarray(self.l2_to_l12); m1 = np.asarray(self.l12_to_l1)
        l12 = m12[l2]; l1 = m1[l12]
        if self.min_confidence > 0:
            low = conf < self.min_confidence
            l2 = np.where(low, -1, l2); l12 = np.where(low, -1, l12); l1 = np.where(low, -1, l1)
        return {"l2": l2, "l12": l12, "l1": l1, "probs": q.astype(np.float32), "confidence": conf.astype(np.float32)}

    def names(self, idx, level="l2"):
        table = {"l2": self.classes_l2, "l12": self.classes_l12, "l1": self.classes_l1}[level]
        return np.asarray([table[i] if i >= 0 else "Unassigned" for i in idx], dtype=object)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def _load_oof(result_dir):
    result_dir = Path(result_dir)
    z = np.load(result_dir / "oof_predictions.npz", allow_pickle=False)
    order = np.argsort(z["row_indices"])
    out = {k: np.asarray(z[k])[order] for k in
           ("l2_probs", "l12_probs", "l1_head_probs", "l2_labels", "l12_labels", "l1_head_labels")}
    out["fold"] = np.asarray(z["fold_ids"])[order] if "fold_ids" in z.files else None
    out["classes"] = _head_classes(z)
    rows_path = result_dir / "oof_rows.csv"
    if rows_path.is_file():
        rows = pd.read_csv(rows_path, usecols=["row_index", "section_id"], low_memory=False)
        rows = rows.set_index("row_index").loc[np.asarray(z["row_indices"])[order]]
        out["cohort"] = rows.section_id.astype(str).str.split("::").str[0].to_numpy()
        if out["fold"] is None and "fold" in pd.read_csv(rows_path, nrows=1):
            out["fold"] = pd.read_csv(rows_path, usecols=["fold"]).fold.to_numpy()[np.asarray(z["row_indices"])[order]]
    return out


def _fit_rule(p2, p12, p1, y, maps, classes, *, objective, variant, weights, max_fit, rng):
    """Grid over (w12, w1), bias by coordinate ascent, temperature by NLL."""
    n = len(y)
    idx = rng.choice(n, size=min(n, int(max_fit)), replace=False) if n > max_fit else np.arange(n)
    grid_w = [(0.0, 0.0)] if variant == "l2_only" else weights
    best = None
    for w12, w1 in grid_w:
        s = fused_scores(p2[idx], p12[idx] if p12 is not None else None, p1[idx] if p1 is not None else None,
                         l2_to_l12=maps["l2_to_l12"], l12_to_l1=maps["l12_to_l1"], w12=w12, w1=w1)
        bias, val = fit_class_bias(s, y[idx], objective=objective)
        if best is None or val > best[0]:
            best = (val, w12, w1, bias, s)
    val, w12, w1, bias, s = best
    temperature = fit_temperature(s + bias, y[idx])
    return DecisionRule(classes_l2=list(classes["L2"]), classes_l12=list(classes["L12"]),
                        classes_l1=list(classes["L1"]), l2_to_l12=list(map(int, maps["l2_to_l12"])),
                        l12_to_l1=list(map(int, maps["l12_to_l1"])), w12=float(w12), w1=float(w1),
                        temperature=float(temperature), bias=[float(x) for x in bias],
                        objective=objective, variant=variant)


DEFAULT_WEIGHTS = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0))


def fit_decision(result_dir, *, objective="macro_f1", max_fit=400_000, seed=42,
                 weights=DEFAULT_WEIGHTS, out=None) -> pd.DataFrame:
    """Fit rules on OOF, report cross-fitted label metrics, save the deployable rules.

    Writes ``decision_rule.json`` (three heads), ``decision_rule_l2_only.json``
    (external slides, which keep only L2 softmax) and ``decision_report.csv``.
    """
    from pancancer_labels import hierarchy_maps

    result_dir = Path(result_dir)
    d = _load_oof(result_dir)
    classes = d["classes"]
    maps = hierarchy_maps(classes)
    p2, p12, p1 = d["l2_probs"], d["l12_probs"], d["l1_head_probs"]
    y2, y12, y1 = d["l2_labels"].astype(int), d["l12_labels"].astype(int), d["l1_head_labels"].astype(int)
    k2, k12, k1 = len(classes["L2"]), len(classes["L12"]), len(classes["L1"])
    m12 = np.asarray(maps["l2_to_l12"]); m1 = np.asarray(maps["l12_to_l1"])
    folds = d["fold"]
    if folds is None:
        raise ValueError("OOF has no fold ids; cannot cross-fit")
    rng = np.random.default_rng(seed)
    preds = {name: np.empty(len(y2), dtype=np.int64) for name in ("full", "l2_only")}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        for variant in ("full", "l2_only"):
            rule = _fit_rule(p2[tr], p12[tr], p1[tr], y2[tr], maps, classes, objective=objective,
                             variant=variant, weights=weights, max_fit=max_fit, rng=rng)
            preds[variant][te] = rule.apply(p2[te], p12[te], p1[te])["l2"]
        print(f"  fold {int(f)} cross-fitted", flush=True)

    rows = []
    cohorts = {"ALL": np.ones(len(y2), bool)}
    if "cohort" in d:
        cohorts.update({c: d["cohort"] == c for c in pd.unique(d["cohort"])})
    l2_arg = p2.argmax(1)
    methods = {
        # name: (L2 pred, L12 pred, L1 pred)
        "argmax per head (current)": (l2_arg, p12.argmax(1), p1.argmax(1)),
        "argmax L2, parents derived": (l2_arg, m12[l2_arg], m1[m12[l2_arg]]),
        "decision l2_only (cross-fitted)": (preds["l2_only"], m12[preds["l2_only"]], m1[m12[preds["l2_only"]]]),
        "decision full (cross-fitted)": (preds["full"], m12[preds["full"]], m1[m12[preds["full"]]]),
    }
    for method, (a2, a12, a1) in methods.items():
        for cohort, mask in cohorts.items():
            for head, yy, pp, k in (("L2", y2, a2, k2), ("L12", y12, a12, k12), ("L1", y1, a1, k1)):
                m = label_metrics(yy[mask], pp[mask], k)
                m.pop("per_class_f1")
                rows.append({"method": method, "cohort": cohort, "head": head, "n_cells": int(mask.sum()), **m})
        consistent = float(np.mean((m12[a2] == a12) & (m1[a12] == a1)))
        rows.append({"method": method, "cohort": "ALL", "head": "hierarchy_consistent_fraction",
                     "n_cells": len(y2), "accuracy": consistent})
    report = pd.DataFrame(rows)
    per_class = pd.DataFrame({
        "class": classes["L2"],
        "support": np.bincount(y2, minlength=k2),
        "f1_argmax": label_metrics(y2, l2_arg, k2)["per_class_f1"],
        "f1_decision_l2_only": label_metrics(y2, preds["l2_only"], k2)["per_class_f1"],
        "f1_decision_full": label_metrics(y2, preds["full"], k2)["per_class_f1"],
        "pred_fraction_argmax": np.bincount(l2_arg, minlength=k2) / len(y2),
        "pred_fraction_decision_full": np.bincount(preds["full"], minlength=k2) / len(y2),
        "true_fraction": np.bincount(y2, minlength=k2) / len(y2),
    })
    out = Path(out) if out else result_dir
    out.mkdir(parents=True, exist_ok=True)
    for variant, name in (("full", "decision_rule.json"), ("l2_only", "decision_rule_l2_only.json")):
        rule = _fit_rule(p2, p12, p1, y2, maps, classes, objective=objective, variant=variant,
                         weights=weights, max_fit=max_fit, rng=rng)
        rule.save(out / name)
        print(f"  {name}: w12={rule.w12} w1={rule.w1} T={rule.temperature:.3f}", flush=True)
    report.to_csv(out / "decision_report.csv", index=False)
    # 2026.11.17, keep the cross-fitted labels for paired per-fold comparisons (posthoc.py compare)
    np.savez_compressed(out / "decision_crossfit_predictions.npz", l2_full=preds["full"],
                        l2_only=preds["l2_only"], fold=folds)
    per_class.to_csv(out / "decision_per_class_f1.csv", index=False)
    return report


def _macro_auroc(y, prob, max_rows=500_000, seed=0):
    """Mean one-vs-rest AUROC over classes with both labels present (rank based)."""
    from scipy.stats import rankdata

    y = np.asarray(y); prob = np.asarray(prob)
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
        y, prob = y[idx], prob[idx]
    vals = []
    for c in range(prob.shape[1]):
        pos = y == c
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 and n0:
            r = rankdata(prob[:, c])
            vals.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(vals)) if vals else np.nan


def run_fold_metrics(result_dir) -> pd.DataFrame:
    """Per-fold L2/L12 metrics of one run: AUROC, argmax and cross-fitted decision labels."""
    from pancancer_labels import hierarchy_maps

    d = _load_oof(result_dir)
    maps = hierarchy_maps(d["classes"])
    m12 = np.asarray(maps["l2_to_l12"])
    k2, k12 = len(d["classes"]["L2"]), len(d["classes"]["L12"])
    dec_path = Path(result_dir) / "decision_crossfit_predictions.npz"
    dec = np.load(dec_path)["l2_full"] if dec_path.is_file() else None
    rows = []
    for f in np.unique(d["fold"]):
        m = d["fold"] == f
        y2, y12, p2 = d["l2_labels"][m].astype(int), d["l12_labels"][m].astype(int), d["l2_probs"][m]
        a = label_metrics(y2, p2.argmax(1), k2)
        rec = {"fold": int(f), "n_cells": int(m.sum()),
               "L2_macro_auroc": _macro_auroc(y2, p2, seed=int(f)),
               "L2_accuracy_argmax": a["accuracy"], "L2_macro_f1_argmax": a["macro_f1"],
               "L12_macro_f1_argmax": label_metrics(y12, d["l12_probs"][m].argmax(1), k12)["macro_f1"]}
        if dec is not None:
            b = label_metrics(y2, dec[m], k2)
            rec.update({"L2_accuracy_decision": b["accuracy"], "L2_macro_f1_decision": b["macro_f1"],
                        "L12_macro_f1_decision": label_metrics(y12, m12[dec[m]], k12)["macro_f1"]})
        rows.append(rec)
    return pd.DataFrame(rows)


ARM_ORDER = ("legacy_nospatial", "adjacent_nospatial", "spatial_k8", "spatial_k8_gate", "l12_links_1",
             "l12_links_2", "balanced_sampler", "focal_gamma1", "l12_balanced", "gate_l12_balanced",
             "l12_balanced_nospatial", "spatial_k4", "spatial_k16")


def _arm_of(name, data_tag=None):
    """(arm, seed) from a run folder name; descriptive names when data_tag is given."""
    import re

    if data_tag:
        m = re.match(rf"^result_uni2_(?:ablation_)?(.+?)_(?:stratsec_)?{re.escape(data_tag)}_paired_v2_seed(\d+)$", name)
        if m:
            return m.group(1), int(m.group(2))
    m_arm = re.match(r"result_uni2_((?:A|K)\d+)_", name)
    m_seed = re.search(r"seed(\d+)$", name)
    if not m_arm:
        return None, None
    return m_arm.group(1), (int(m_seed.group(1)) if m_seed else -1)


def compare_runs(root, pattern, reference="spatial_k8", out=None, data_tag=None) -> pd.DataFrame:
    """Ablation table: every run folder matching ``pattern`` under ``root``.

    Seeds of the same arm are averaged per fold, then each arm is compared
    with ``reference`` fold by fold (same fold plan), so the table shows the
    mean paired difference, its SD over folds and how many folds improved.
    """
    root = Path(root)
    per = []
    for rd in sorted(root.glob(pattern)):
        if not (rd / "oof_predictions.npz").is_file():
            print(f"  skip {rd.name}: no oof_predictions.npz", flush=True)
            continue
        arm, seed = _arm_of(rd.name, data_tag)
        if arm is None:
            continue
        fm = run_fold_metrics(rd)
        fm["arm"] = arm
        fm["seed"] = seed
        fm["run"] = rd.name
        per.append(fm)
        print(f"  {rd.name}: {len(fm)} folds", flush=True)
    if not per:
        raise SystemExit(f"no finished runs match {root}/{pattern}")
    per = pd.concat(per, ignore_index=True)
    metrics = [c for c in per.columns if c.startswith(("L2_", "L12_"))]
    by_fold = per.groupby(["arm", "fold"])[metrics].mean()
    seeds = per.groupby("arm")["seed"].nunique()
    rows = []
    for arm, g in by_fold.groupby(level="arm"):
        g = g.droplevel("arm")
        rec = {"arm": arm, "n_seeds": int(seeds[arm]), "n_folds": len(g)}
        for c in metrics:
            rec[f"{c}_mean"] = float(g[c].mean())
            rec[f"{c}_sd_over_folds"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
        if reference in by_fold.index.get_level_values("arm") and arm != reference:
            ref = by_fold.loc[reference]
            common = g.index.intersection(ref.index)
            for c in metrics:
                diff = (g.loc[common, c] - ref.loc[common, c]).to_numpy()
                rec[f"{c}_diff_vs_{reference}"] = float(diff.mean()) if len(diff) else np.nan
                rec[f"{c}_diff_sd"] = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
                rec[f"{c}_folds_better"] = f"{int((diff > 0).sum())}/{len(diff)}"
        rows.append(rec)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    table = pd.DataFrame(rows).sort_values("arm", key=lambda s: s.map(lambda a: order.get(a, 99)))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        per.to_csv(Path(out).with_name(Path(out).stem + "_per_fold.csv"), index=False)
    return table


def smooth_probs_knn(probs, xy, *, k=8, alpha=0.3):
    """Cell-type map smoothing: (1-alpha)·p + alpha·mean of the k nearest cells' p.

    Spatially coherent maps for figures; alpha=0 returns ``probs``. Choose
    alpha on a section with GT before using it for numbers.
    """
    if alpha <= 0:
        return np.asarray(probs, dtype=np.float32)
    from scipy.spatial import cKDTree

    p = np.asarray(probs, dtype=np.float32)
    _, nbr = cKDTree(np.asarray(xy, float)).query(xy, k=int(k) + 1)
    return ((1 - alpha) * p + alpha * p[nbr[:, 1:]].mean(1)).astype(np.float32)


def decide_npz(probs_npz, rule_path, *, smooth_alpha=0.0, xy=None, out=None) -> pd.DataFrame:
    """Apply an ``l2_only`` rule to an external ``{sample}_l2_probs.npz``.

    Writes ``{sample}_l2_decision.csv`` (obs_name, L2/L12/L1 label, confidence)
    next to the input. ``pred_CT`` in the h5ad is not touched.
    """
    rule = DecisionRule.load(rule_path)
    if rule.variant != "l2_only" and (rule.w12 or rule.w1):
        raise ValueError("external npz has only L2 softmax; use decision_rule_l2_only.json")
    with np.load(probs_npz, allow_pickle=True) as z:
        p = np.asarray(z["probs"], dtype=np.float32)
        names = [str(x) for x in z["class_names"]]
        obs = np.asarray(z["obs_names"]).astype(str) if "obs_names" in z.files else np.arange(len(p)).astype(str)
    if names != list(rule.classes_l2):
        raise ValueError("class order of the npz differs from the rule")
    if smooth_alpha > 0:
        if xy is None:
            raise ValueError("smoothing needs nucleus coordinates")
        p = smooth_probs_knn(p, xy, alpha=smooth_alpha)
    res = rule.apply(p)
    df = pd.DataFrame({"obs_name": obs, "L2": rule.names(res["l2"], "l2"), "L12": rule.names(res["l12"], "l12"),
                       "L1": rule.names(res["l1"], "l1"), "confidence": res["confidence"]})
    out = Path(out) if out else Path(probs_npz).with_name(Path(probs_npz).name.replace("_l2_probs.npz", "_l2_decision.csv"))
    df.to_csv(out, index=False)
    return df


########################################################
# CLI (2026.11.17, one entry point for sections 1-5)
########################################################
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = {"prior-shift": prior_shift_main, "calibrate-abundance": abundance_calibration_main,
              "fold-ensemble": fold_ensemble_main}
    if argv and argv[0] in legacy:
        return legacy[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="OOF metrics by cohort (was oof_evaluation.py)")
    e.add_argument("result_dir", type=Path); e.add_argument("checkpoint", type=Path, nargs="?")
    e.add_argument("--bootstrap", type=int, default=0)
    f = sub.add_parser("fit-decision", help="fit decision rules on OOF, cross-fitted report")
    f.add_argument("result_dir", type=Path)
    f.add_argument("--objective", choices=OBJECTIVES, default="macro_f1")
    f.add_argument("--max-fit", type=int, default=400_000, help="cells used per fit (uniform subsample)")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--out", type=Path, default=None)
    cmp_ = sub.add_parser("compare", help="ablation table, paired per fold against a reference arm")
    cmp_.add_argument("--root", type=Path, required=True)
    cmp_.add_argument("--pattern", required=True, help="glob of run folders, e.g. result_uni2_*_new1066axisfix_paired_v2_seed*")
    cmp_.add_argument("--reference", default="spatial_k8")
    cmp_.add_argument("--data-tag", default=None, help="read descriptive arm names from folders ending _<tag>_paired_v2_seed<N>")
    cmp_.add_argument("--out", type=Path, default=None)
    dcd = sub.add_parser("decide", help="apply decision_rule_l2_only.json to an external *_l2_probs.npz")
    dcd.add_argument("probs_npz", type=Path); dcd.add_argument("--rule", type=Path, required=True)
    for name in legacy:
        sub.add_parser(name, help="same flags as the old script")
    a = ap.parse_args(argv)
    if a.cmd == "evaluate":
        print(evaluate(a.result_dir, a.checkpoint)[0].to_string(index=False))
        if a.bootstrap:
            print(bootstrap_result(a.result_dir, a.checkpoint, a.bootstrap).to_string(index=False))
    elif a.cmd == "fit-decision":
        rep = fit_decision(a.result_dir, objective=a.objective, max_fit=a.max_fit, seed=a.seed, out=a.out)
        show = rep[(rep.cohort == "ALL")]
        print(show.pivot_table(index="method", columns="head", values=["accuracy", "macro_f1"]).round(4).to_string())
    elif a.cmd == "compare":
        tab = compare_runs(a.root, a.pattern, a.reference, a.out, data_tag=a.data_tag)
        cols = ["arm", "n_seeds", "L2_macro_auroc_mean", "L2_accuracy_argmax_mean", "L2_macro_f1_argmax_mean"]
        cols += [c for c in ("L2_accuracy_decision_mean", "L2_macro_f1_decision_mean",
                             f"L2_macro_f1_decision_diff_vs_{a.reference}",
                             f"L2_macro_f1_decision_folds_better") if c in tab]
        print(tab[cols].round(4).to_string(index=False))
    elif a.cmd == "decide":
        df = decide_npz(a.probs_npz, a.rule)
        print(df["L2"].value_counts(normalize=True).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
