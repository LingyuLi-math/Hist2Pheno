"""Read-only UNI2 A2/A3 diagnostics. Softmax mass is not a calibrated cell count.

Run with the same strategy's checkpoint, OOF and NPC prediction archive.
No labels, probabilities or model weights are changed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from checkpoint_artifact import read_artifact, file_digest


def validate_probabilities(p, names):
    p = np.asarray(p)
    if p.ndim != 2 or not len(p) or p.shape[1] != len(names):
        raise ValueError('Probability shape does not match class order')
    if len(set(names)) != len(names) or not np.isfinite(p).all():
        raise ValueError('Duplicate classes or non-finite probabilities')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=5e-4):
        raise ValueError('Invalid softmax probabilities')


def summarize_target(p, names, members, labels=None):
    columns = [names.index(c) for c in members]
    score = p[:, columns].sum(1)
    hard = np.isin(p.argmax(1), columns)
    out = dict(n_cells=len(p), hard_count=int(hard.sum()),
               hard_fraction=float(hard.mean()), mean_softmax_mass=float(score.mean()),
               sum_softmax_mass=float(score.sum()),
               score_quantiles=dict(zip(['q00', 'q50', 'q90', 'q99', 'q100'],
                                        np.quantile(score, [0, .5, .9, .99, 1]).tolist())))
    if labels is not None:
        truth = np.isin(labels, columns)
        tp = int((truth & hard).sum())
        out.update(positive_support=int(truth.sum()), negative_support=int((~truth).sum()),
                   prevalence=float(truth.mean()), true_positive=tp,
                   precision=tp / int(hard.sum()) if hard.any() else 0.,
                   recall=tp / int(truth.sum()) if truth.any() else None,
                   f1=2 * tp / (int(truth.sum()) + int(hard.sum()))
                   if truth.any() or hard.any() else None,
                   auroc=float(roc_auc_score(truth, score)) if truth.any() and (~truth).any() else None,
                   average_precision=float(average_precision_score(truth, score)) if truth.any() and (~truth).any() else None)
        out['missed_truth_predicted_as'] = pd.Series(
            np.asarray(names)[p[truth & ~hard].argmax(1)]).value_counts().to_dict()
    return out


def run(args):
    artifact = read_artifact(args.checkpoint, required=True)
    meta = artifact['metadata']
    from pipeline_provenance import checkpoint_family
    family = checkpoint_family(meta)
    if meta.get('embedder') not in (None, 'uni2'):
        raise ValueError('This diagnostic expects a UNI2 checkpoint')
    names = [str(x) for x in meta['head_classes']['L2']]
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': artifact['weight_sha256'],
              'checkpoint_fold': meta.get('fold'), 'spatial_config': meta.get('spatial_config'),
              'loss_weights': meta.get('loss_weights'), 'hierarchy_config': meta.get('hierarchy_config'),
              'notes': [
                  'Softmax mass is a model score, not validated abundance or a cell count.',
                  'NPC has no nucleus ground truth here; external recall cannot be inferred.',
                  'Pooled OOF combines fold models; its calibration does not guarantee NPC FPR/FDR.',
                  'Use all StarDist nuclei (STAR_PROB_MIN=0) for the requested comparison.']}
    targets = {'B_cell': ['B_cell'], 'Plasma': ['Plasma'], 'B_Plasma': ['B_cell', 'Plasma']}
    for members in targets.values():
        if not set(members).issubset(names):
            raise ValueError('Checkpoint does not contain the expected B/Plasma classes')
    with np.load(args.pred_probs, allow_pickle=False) as z:
        if 'checkpoint_family' in z and str(z['checkpoint_family'].item()) != family:
            raise ValueError('NPC archive belongs to a different model family (e.g. A2 vs A3)')
        pn = z['class_names'].astype(str).tolist()
        p = np.asarray(z['probs'])
        validate_probabilities(p, pn)
        if pn != names:
            raise ValueError('NPC class order differs from checkpoint; do not compare these files')
        obs = z['obs_names'].astype(str) if 'obs_names' in z else None
        if obs is not None and (len(obs) != len(p) or len(set(obs)) != len(obs)):
            raise ValueError('NPC row IDs are incomplete or duplicated')
        report['external'] = {k: summarize_target(p, names, v) for k, v in targets.items()}
        if 'l12_probs' in z:
            p12 = np.asarray(z['l12_probs']); n12 = z['l12_class_names'].astype(str).tolist()
            validate_probabilities(p12, n12)
            if len(p12) != len(p) or n12 != list(meta['head_classes']['L12']):
                raise ValueError('L12 rows/class order differs from checkpoint')
            mapping = np.asarray(meta['hierarchy_maps']['l2_to_l12'], dtype=int)
            if mapping.shape != (len(names),) or (mapping < 0).any() or (mapping >= len(n12)).any():
                raise ValueError('Invalid checkpoint hierarchy')
            q12 = p @ np.eye(len(n12), dtype=np.float32)[mapping]
            b = n12.index('B_Plasma')
            report['external_hierarchy'] = dict(
                l12_head_B_Plasma_fraction=float((p12.argmax(1) == b).mean()),
                aggregated_l2_B_Plasma_fraction=float((q12.argmax(1) == b).mean()),
                l12_head_B_Plasma_mean_score=float(p12[:, b].mean()),
                l2_B_Plasma_mean_score=float(q12[:, b].mean()),
                head_vs_aggregate_argmax_disagreement=float((p12.argmax(1) != q12.argmax(1)).mean()))
        else:
            report['notes'].append('Legacy L2-only archive: rerun predict --force after the patch to inspect L12 disagreement.')
    with np.load(args.oof, allow_pickle=False) as z:
        if 'checkpoint_family' in z and str(z['checkpoint_family'].item()) != family:
            raise ValueError('OOF archive belongs to a different model family (e.g. A2 vs A3)')
        y = np.asarray(z['l2_labels'], dtype=int); oof = np.asarray(z['l2_probs'])
        ids = np.asarray(z['row_indices'], dtype=int)
        if 'l2_class_names' in z and z['l2_class_names'].astype(str).tolist() != names:
            raise ValueError('OOF class order differs from selected checkpoint')
        if 'l2_class_names' not in z:
            report['notes'].append('Legacy OOF has no embedded class order; checkpoint order is assumed. Verify both belong to the same run.')
        validate_probabilities(oof, names)
        if y.shape != (len(oof),) or (y < 0).any() or (y >= len(names)).any():
            raise ValueError('Invalid OOF labels')
        if ids.shape != y.shape or not np.array_equal(np.sort(ids), np.arange(len(y))):
            raise ValueError('Expected complete, unique-cell OOF row coverage')
        fold_ids = np.asarray(z['fold_ids']) if 'fold_ids' in z else None
    report['oof'] = {k: summarize_target(oof, names, v, y) for k, v in targets.items()}
    # Artifact indices are global rows; OOF arrays are stored in fold-concatenated order.
    global_y = y[np.argsort(ids)]
    support = {}
    partitions = {}
    for key in ('train_rows', 'val_rows', 'outer_val_rows'):
        rows = np.asarray(meta.get(key, []), dtype=int)
        if rows.size and ((rows < 0).any() or (rows >= len(y)).any()):
            raise ValueError('Checkpoint indices exceed OOF rows; likely mixed runs')
        if len(np.unique(rows)) != len(rows):
            raise ValueError('Duplicate checkpoint partition indices')
        partitions[key] = rows
        support[key] = {k: int(np.isin(global_y[rows], [names.index(c) for c in v]).sum())
                        for k, v in targets.items()}
    for a, b in [('train_rows', 'val_rows'), ('train_rows', 'outer_val_rows')]:
        if np.intersect1d(partitions[a], partitions[b]).size:
            raise ValueError(f'Checkpoint leakage: {a} overlaps {b}')
    report['selected_checkpoint_partition_support'] = support
    grouped = []
    if args.oof_rows:
        rows = pd.read_csv(args.oof_rows)
        if len(rows) != len(y) or not np.array_equal(rows['row_index'].to_numpy(), np.arange(len(y))):
            raise ValueError('oof_rows.csv is not in global row order')
        if fold_ids is None and 'fold' in rows:
            fold_ids = rows['fold'].to_numpy()[ids]
        cohorts = rows['section_id'].astype(str).str.split('::').str[0].to_numpy()[ids]
        group_sets = [('cohort', cohorts)]
        patient = rows.get('patient_id')
        if patient is None or patient.isna().any():
            report['notes'].append('No complete patient metadata: section folds cannot establish patient-level independence.')
        else:
            a = patient.iloc[partitions['train_rows']].astype(str)
            b = patient.iloc[partitions['outer_val_rows']].astype(str)
            report['selected_checkpoint_shared_train_outer_patients'] = sorted(set(a) & set(b))
    else:
        group_sets = []
    if fold_ids is not None:
        if len(fold_ids) != len(y):
            raise ValueError('OOF fold IDs are not row-aligned')
        group_sets.append(('fold', fold_ids))
    for kind, labels in group_sets:
        for label in np.unique(labels):
            mask = labels == label
            for target, members in targets.items():
                values = summarize_target(oof[mask], names, members, y[mask])
                grouped.append(dict(group_kind=kind, group=str(label), target=target,
                                    **{k: v for k, v in values.items() if not isinstance(v, dict)}))
    label_audits = []
    if args.label_h5ad:
        import anndata as ad
        from pancancer_labels import native_to_pancancer
        for path in args.label_h5ad:
            data = ad.read_h5ad(path, backed='r')
            try:
                labels = data.obs[args.native_col].astype(str)
                for label, count in labels.value_counts().items():
                    mapped = native_to_pancancer(label)
                    label_audits.append(dict(file=str(path), native_label=label, count=int(count),
                                             kept=mapped is not None, mapped_L2=mapped[2] if mapped else ''))
            finally:
                data.file.close()
    args.output.mkdir(parents=True, exist_ok=True)
    report['inputs_sha256'] = {str(path): file_digest(path) for path in [args.pred_probs, args.oof]}
    (args.output / 'bcell_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if grouped:
        pd.DataFrame(grouped).to_csv(args.output / 'bcell_oof_by_cohort_fold.csv', index=False)
    if label_audits:
        pd.DataFrame(label_audits).to_csv(args.output / 'native_label_audit.csv', index=False)
    print(args.output / 'bcell_audit.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pred-probs', type=Path, required=True)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--oof-rows', type=Path)
    parser.add_argument('--label-h5ad', nargs='*', type=Path, default=[])
    parser.add_argument('--native-col', default='final_CT')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
