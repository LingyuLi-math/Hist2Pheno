########################################################
# 2026.10.24, inspect true OOF by cohort / 明确支持数与队列内指标
########################################################
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from checkpoint_artifact import read_artifact


def evaluate(result_dir,checkpoint):
    result_dir=Path(result_dir)
    rows=pd.read_csv(result_dir/'oof_rows.csv')
    z=np.load(result_dir/'oof_predictions.npz',allow_pickle=False)
    ids=z['row_indices'];order=np.argsort(ids)
    if len(ids)!=len(rows) or not np.array_equal(np.sort(ids),np.arange(len(rows))):
        raise ValueError('This is not complete unique-cell OOF; repeated holdout is not k-fold OOF')
    classes=read_artifact(checkpoint,required=True)['metadata']['head_classes']
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
                macro_auroc=np.mean(aucs) if aucs else np.nan,macro_auprc=np.mean(aps) if aps else np.nan))
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


def bootstrap_result(result_dir,checkpoint,n_bootstrap=500):
    result_dir=Path(result_dir);rows=pd.read_csv(result_dir/'oof_rows.csv')
    if 'patient_id' not in rows or rows.patient_id.isna().any():
        raise ValueError('Supply a real patient-map when training before computing patient CI')
    z=np.load(result_dir/'oof_predictions.npz',allow_pickle=False);order=np.argsort(z['row_indices'])
    if not np.array_equal(np.sort(z['row_indices']),np.arange(len(rows))):raise ValueError('Incomplete/duplicate OOF rows')
    classes=read_artifact(checkpoint,required=True)['metadata']['head_classes'];out=[]
    for head,key in [('L1','l1_head'),('L12','l12'),('L2','l2')]:
        y=z[key+'_labels'][order];pred=z[key+'_probs'][order].argmax(1)
        lo,hi=patient_bootstrap_f1(y,pred,rows.patient_id.to_numpy(),len(classes[head]),n_bootstrap=n_bootstrap)
        out.append(dict(head=head,metric='macro_F1_all_classes',lower_95=float(lo),upper_95=float(hi),bootstrap_unit='patient',replicates=n_bootstrap))
    result=pd.DataFrame(out);result.to_csv(result_dir/'oof_patient_bootstrap_f1.csv',index=False);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--bootstrap',type=int,default=0)
    a=p.parse_args();print(evaluate(a.result_dir,a.checkpoint)[0].to_string(index=False))
    if a.bootstrap:print(bootstrap_result(a.result_dir,a.checkpoint,a.bootstrap).to_string(index=False))
