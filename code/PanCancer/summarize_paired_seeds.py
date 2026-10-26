#!/usr/bin/env python3
########################################################
# 2026.10.26, multi-seed summary / 固定拆分后报告均值、标准差与配对差值
########################################################
import argparse
from pathlib import Path
import pandas as pd
from pred_statistic_visual import oof_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/PanCancer'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--models', nargs='+', default=['A0','A1','A2','A3'])
    args = p.parse_args()
    tags = {'A0':'A0_legacy_nospatial','A1':'A1_adjacent_nospatial',
            'A2':'A2_adjacent_spatial_k8mean','A3':'A3_adjacent_spatial_k8mean_gate'}
    rows, reference = [], None
    for seed in args.seeds:
        for model in args.models:
            result = args.data_root/f'result_uni2_{tags[model]}_paired_v2_seed{seed}'
            identity = oof_identity({'result':result})
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f'Cell identities or folds differ: {result}')
            frame = pd.read_csv(result/'validation_internal_metrics.csv')
            if len(frame) != 1:
                raise ValueError(f'Expected one OOF summary row: {result}')
            rows.append({'seed':seed,'model':model,**frame.iloc[0].to_dict()})
    data = pd.DataFrame(rows)
    metrics = [c for c in data if c.startswith('oof_') and pd.api.types.is_numeric_dtype(data[c])]
    out = args.data_root/'paired_v2_seed_summary'
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'per_seed.csv',index=False)
    summary = data.groupby('model')[metrics].agg(['mean','std','count'])
    summary.to_csv(out/'mean_std.csv')
    print(summary.to_string())
    if {'A2','A3'}.issubset(args.models):
        a = data.pivot(index='seed', columns='model', values='oof_l2_macro_auc')
        delta = (a.A3-a.A2).rename('A3_minus_A2')
        delta.to_csv(out/'paired_A3_minus_A2.csv')
        print('A3-A2 paired L2 AUROC:',delta.to_dict(),'mean=',delta.mean(),'std=',delta.std())


if __name__ == '__main__':
    main()
