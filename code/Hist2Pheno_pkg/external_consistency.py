########################################################
# 2026.10.28, strict External_val pipeline / 所有消费者先校验完整上游，不按文件存在跳过
########################################################
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    from . import pipeline_provenance as pp
except ImportError:
    import pipeline_provenance as pp


def feature_source(cohort,sample):
    paths=cohort.sample_paths(sample);cache=cohort.uni_root/'.provenance'
    return {'version':pp.VERSION,'method':cohort.method,'patch_size':cohort.patch_size,
        'geometry':cohort.geometry(sample),'csv':pp.file_record(paths['csv'],cache),
        'tokens':pp.feature_manifest(paths['emb'],cache),
        'he':pp.file_record(paths['he'],cache),
        'extraction_implementation':pp.file_record(cohort.feature_script,cache),
        'extraction_manifest':pp.file_record(paths['emb']/'embedding_manifest.json',cache),
        'matching_implementation':pp.implementations()}


def build_features(cohort,sample,force=False):
    paths=cohort.sample_paths(sample);path=paths['feat'];marker=path.with_suffix('.source.json')
    with pp.file_lock(str(path)+'.lock'):
        with pp.file_lock(paths['emb']/'.prepare.lock'):
            if not extraction_complete(cohort,sample):cohort.cmd_features([sample],force=True)
        source=feature_source(cohort,sample);stamp=pp.fingerprint(source)
        if not force and pp.complete(marker,stamp,[path]):return path
        # Historical builder retained for matching/annotations; never allow its existence shortcut.
        cohort._legacy_build_one_h5ad(sample,force=True)
        import anndata as ad
        data=ad.read_h5ad(path);pp.finite_matrix(data.X)
        rows=pp.row_signature(data.obs_names,data.obsm['spatial_HE'])
        if feature_source(cohort,sample)!=source:
            raise RuntimeError('Embedding/CSV/geometry changed while building features; finish extraction and retry')
        data.uns['feature_source_sha256']=stamp;data.uns['row_identity_sha256']=rows
        pp.atomic_h5ad(data,path)
        pp.publish(marker,stamp,[path],source=source,row_signature=rows,n_cells=data.n_obs)
        return path


def load_model(cohort,allow_cpu):
    cache=cohort.uni_root/'.provenance'
    key=pp.fingerprint(pp.file_record(cohort.checkpoint,cache),
        pp.file_record(str(cohort.checkpoint)+'.artifact.pt',cache),bool(allow_cpu))
    cached=getattr(cohort,'_verified_model_cache',None)
    if cached is None or cached[0]!=key:
        device=cohort._device(allow_cpu)
        model,scaler,classes=cohort._load_model(device)
        cached=(key,model,scaler,classes,device);cohort._verified_model_cache=cached
    return cached[1:]


def predict_one(cohort,sample,*,force=False,allow_cpu=False):
    import anndata as ad
    paths=cohort.sample_paths(sample);dest=paths['pred'];probpath=cohort.l2_probs_path(sample)
    marker=dest.with_suffix('.pipeline.json')
    with pp.file_lock(str(dest)+'.lock'):
        feat=build_features(cohort,sample)
        model,scaler,classes,device=load_model(cohort,allow_cpu)
        minimum=cohort._stardist_prob_min()
        header=ad.read_h5ad(feat,backed='r')
        try:
            mask=np.ones(header.n_obs,dtype=bool)
            if minimum>0:
                if 'probability' not in header.obs:raise KeyError('Missing nucleus probability column')
                mask=np.asarray(header.obs['probability'],dtype=float)>=minimum
            if not mask.any():raise ValueError('No nuclei pass current filter')
            names=np.asarray(header.obs_names[mask],dtype=str)
            xy=np.asarray(header.obsm['spatial_HE'],dtype=np.float64)[mask]
            rows=pp.row_signature(names,xy)
        finally:header.file.close()
        desc=pp.model_descriptor(model,scaler,classes)
        cache=cohort.uni_root/'.provenance'
        source={'feature':pp.file_record(feat,cache),'model':desc,'filter_min':minimum,
            'model_files':[pp.file_record(cohort.checkpoint,cache),pp.file_record(str(cohort.checkpoint)+'.artifact.pt',cache)],
            'rows':rows,'geometry':cohort.geometry(sample),'implementation':pp.implementations()}
        stamp=pp.fingerprint(source)
        run=pp.complete(marker,stamp,[dest,probpath])
        if run and not force:return run
        pp.atomic_json(marker,{"version":pp.VERSION,"state":"running","stamp":stamp,"outputs":{}})
        data=ad.read_h5ad(feat);data,minimum=cohort._subset_by_stardist_prob(data,sample)
        if pp.row_signature(data.obs_names,data.obsm['spatial_HE'])!=rows:raise RuntimeError('Rows changed while loading features')
        X=np.asarray(cohort._pipeline_base().adata_X_to_dense(data.X),dtype=np.float32)
        pp.finite_matrix(X)
        if X.shape[1]!=scaler.n_features_in_:raise ValueError('Embedding dimension differs from checkpoint scaler')
        Xs=np.asarray(scaler.transform(X),dtype=np.float32);pp.finite_matrix(Xs,'scaled features')
        neighbors=cohort._context_request(model,sample,xy) if getattr(model,'use_spatial_context',False) else None
        probs=cohort._pipeline_plot().mlp_collect_five_head_softmax_probs(model,Xs,device,neighbor_index=neighbors)
        pp.validate_probabilities(probs,classes,len(X))
        obs=data.obs.copy()
        for head,key in (('L1','l1'),('L12','l12'),('L2','l2')):
            values=np.asarray(probs[key],dtype=np.float32)
            col={'L1':'pred_lineage','L12':'pred_sublineage','L2':'pred_CT'}[head]
            obs[col]=np.asarray(classes[head])[values.argmax(1)];obs['conf_'+head]=values.max(1)
        pred=ad.AnnData(obs=obs);pred.obsm['spatial_HE']=xy
        pred.uns.update(pipeline_run_id=stamp,feature_sha256=source['feature']['sha256'],
            checkpoint_sha256=desc['checkpoint_sha256'],checkpoint=str(cohort.checkpoint),
            scaler_sha256=desc['scaler_sha256'],row_identity_sha256=rows,
            classes=classes,stardist_prob_min=float(minimum),
            spatial_k=int(model.spatial_fusion.config.k) if getattr(model,'use_spatial_context',False) else 0)
        # Check feature root again before any completed transaction is published.
        if pp.fingerprint(feature_source(cohort,sample))!=data.uns['feature_source_sha256']:
            raise RuntimeError('External input changed during prediction')
        pp.atomic_h5ad(pred,dest)
        probabilities_sha=pp.fingerprint(np.asarray(probs['l2'],dtype=np.float32),np.asarray(classes['L2'],dtype=str),names)
        pp.atomic_npz(probpath,probs=np.asarray(probs['l2'],dtype=np.float32),
            class_names=np.asarray(classes['L2'],dtype=str),obs_names=names,
            row_identity_sha256=np.asarray(rows),pipeline_run_id=np.asarray(stamp))
        return pp.publish(marker,stamp,[dest,probpath],source=source,row_signature=rows,
            obs_ids_sha256=pp.fingerprint(names),n_cells=len(X),model=desc,probabilities_sha256=probabilities_sha)


def probabilities(cohort,sample,*,force=False,allow_cpu=False):
    run=predict_one(cohort,sample,force=force,allow_cpu=allow_cpu)
    with np.load(cohort.l2_probs_path(sample),allow_pickle=False) as z:
        if str(z['pipeline_run_id'].item())!=run['stamp']:raise ValueError('Probability version mismatch')
        if str(z['row_identity_sha256'].item())!=run['row_signature']:raise ValueError('Probability row mismatch')
        result=np.asarray(z['probs'],dtype=np.float32).copy(),list(z['class_names'].astype(str))
    paths=cohort.sample_paths(sample)
    if pp.complete(paths['pred'].with_suffix('.pipeline.json'),run['stamp'],[paths['pred'],cohort.l2_probs_path(sample)]) is None:
        raise RuntimeError('Prediction transaction changed while reading probabilities; retry')
    return result


def preflight(cohort,samples,*,allow_cpu=False):
    for sample in samples:predict_one(cohort,sample,allow_cpu=allow_cpu)


def extraction_source(cohort,sample):
    p=cohort.sample_paths(sample);cache=cohort.uni_root/'.provenance'
    return {'geometry':cohort.geometry(sample),'csv':pp.file_record(p['csv'],cache),
        'he':pp.file_record(p['he'],cache),'method':cohort.method,
        'extractor':pp.file_record(cohort.feature_script,cache)}


def extraction_complete(cohort,sample):
    p=cohort.sample_paths(sample);marker=p['emb']/"embedding_manifest.json"
    try:
        old=json.loads(marker.read_text());source=extraction_source(cohort,sample)
        if old.get('state')!='completed' or old.get('version')!=pp.VERSION:return False
        if old.get('method')!=cohort.method or old.get('patch_size')!=cohort.patch_size:return False
        if old.get('inputs')!={'image':source['he'],'positions':source['csv']}:return False
        geo=source['geometry']
        if old.get('scale_image')!=bool(geo['scale_image']) or abs(old.get('scale',-1)-float(geo['scale']))>1e-7:return False
        if old.get('extractor_sha256')!=source['extractor']['sha256']:return False
        weight=old.get('encoder_weights')
        if weight and pp.file_record(weight['path'],cohort.uni_root/'.provenance')!=weight:return False
        return old.get('tokens')==pp.feature_manifest(p['emb'],cohort.uni_root/'.provenance')
    except (OSError,ValueError,RuntimeError,KeyError,TypeError):return False


def record_extraction(cohort,sample):
    p=cohort.sample_paths(sample)
    pp.atomic_json(p['emb'].parent/(cohort.method.lower()+'_extraction.json'),
        {'version':pp.VERSION,'source':extraction_source(cohort,sample),
         'tokens':pp.feature_manifest(p['emb'],cohort.uni_root/'.provenance')})
