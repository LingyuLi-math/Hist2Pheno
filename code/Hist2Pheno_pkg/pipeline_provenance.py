########################################################
# 2026.10.28, one provenance chain / 特征、预测、概率、评价和下游行身份统一校验
########################################################
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys
import tempfile
import uuid
import numpy as np

_PKG = Path(__file__).resolve().parent
_PAN = _PKG.parent/'PanCancer'
if str(_PAN) not in sys.path:
    sys.path.insert(0,str(_PAN))
from feature_integrity import atomic_json, file_lock, feature_manifest, sha256

VERSION = 'consistent-chain-v1'
_LABEL_CODE_STAMP = None


def canonical(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):
        if value.dtype.kind in 'OUS':
            h=hashlib.sha256()
            for item in value.reshape(-1):
                encoded=str(item).encode('utf-8');h.update(len(encoded).to_bytes(8,'little'));h.update(encoded)
            return {'shape':list(value.shape),'dtype':'str','sha256':h.hexdigest()}
        array=np.ascontiguousarray(value)
        return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(memoryview(array).cast('B') if array.size else b'').hexdigest()}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in sorted(value.items(),key=lambda kv:str(kv[0]))}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if hasattr(value,'__dict__'):return canonical(vars(value))
    return value


def fingerprint(*values):
    return hashlib.sha256(json.dumps(canonical(values),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def file_record(path,cache_dir=None):
    """Hash changed files, including in-place overwrites; no process-lifetime memo."""
    p=Path(path).resolve()
    st=p.stat();stat=[st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino]
    if cache_dir is None:return {'path':str(p),'sha256':sha256(p)}
    dest=Path(cache_dir)/'file_hashes'/(fingerprint(str(p))+'.json')
    with file_lock(str(dest)+'.lock'):
        try:old=json.loads(dest.read_text())
        except (OSError,ValueError):old={}
        if old.get('stat')!=stat:
            value=sha256(p);after=p.stat()
            if stat!=[after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino]:
                raise RuntimeError(f'Input changed while hashing: {p}')
            old={'stat':stat,'sha256':value};atomic_json(dest,old)
        return {'path':str(p),'sha256':old['sha256']}


def implementations():
    names=('pipeline_provenance.py','external_consistency.py','external_validation.py',
           'model.py','base.py','plot.py','spatial_context.py','checkpoint_artifact.py',
           'embedding_alignment.py','../PanCancer/PanCancer_train_validate_cv_UNIlabel.py')
    return {name:sha256(_PKG/name) for name in names if (_PKG/name).is_file()}


def scaler_signature(scaler):
    return fingerprint({n:np.asarray(getattr(scaler,n)) for n in
        ('mean_','var_','scale_','n_samples_seen_','n_features_in_') if hasattr(scaler,n)})


def mapping_descriptor():
    module=importlib.import_module('pancancer_labels')
    actual=module if hasattr(module,'NATIVE_TO_V1') else importlib.import_module(module.map_native_series.__module__)
    builder=importlib.import_module(actual.unified_of.__module__)
    # 2026.10.28, notebook kernels may outlive annotation updates / 清除旧映射与 lru 缓存
    global _LABEL_CODE_STAMP
    stamp=(sha256(actual.__file__),sha256(builder.__file__),sha256(actual.DEFAULT_ONTOLOGY_XLSX))
    if stamp!=_LABEL_CODE_STAMP:
        for target in (builder,actual):
            source=target.__spec__.loader.get_source(target.__name__)
            exec(compile(source,target.__file__,'exec'),target.__dict__)
        _LABEL_CODE_STAMP=(sha256(actual.__file__),sha256(builder.__file__),sha256(actual.DEFAULT_ONTOLOGY_XLSX))
    effective=[(str(k),actual.native_to_pancancer(k)) for k in sorted(actual.NATIVE_TO_V1,key=str)]
    return {'effective_mapping':fingerprint(effective),'ontology':sha256(actual.DEFAULT_ONTOLOGY_XLSX),
            'sheet':actual.DEFAULT_ONTOLOGY_SHEET,
            'implementation':{Path(m.__file__).name:sha256(m.__file__) for m in (actual,builder)}}


def checkpoint_family(metadata):
    keys=('training_seed','split_seed','loss_weights','data_fingerprint','fold_mode',
          'spatial_config','spatial_mode','hierarchy_config','hierarchy_maps','head_classes')
    return fingerprint({k:metadata.get(k) for k in keys},(metadata.get('initialization') or {}).get('method'))


def model_descriptor(model,scaler,classes):
    try:device=str(next(model.parameters()).device)
    except (AttributeError,StopIteration):device='unknown'
    runtime={name:str(getattr(sys.modules.get(name),'__version__','unknown')) for name in ('numpy','torch','sklearn')}
    runtime['device']=device
    return {'runtime':runtime,'checkpoint_sha256':getattr(model,'checkpoint_weight_sha256',None),
        'metadata':fingerprint(getattr(model,'checkpoint_metadata',{})),
        'family':checkpoint_family(getattr(model,'checkpoint_metadata',{})),
        'scaler_sha256':scaler_signature(scaler),'classes':classes,
        'spatial_config':vars(model.spatial_fusion.config) if getattr(model,'use_spatial_context',False) else None,
        'spatial_mode':getattr(getattr(model,'spatial_fusion',None),'mode',None)}


def finite_matrix(X,name='features'):
    if getattr(X,'ndim',None)!=2 or len(X)==0:raise ValueError(f'{name} must be a nonempty matrix')
    for start in range(0,len(X),4096):
        if not np.isfinite(np.asarray(X[start:start+4096])).all():
            raise ValueError(f'{name} contains NaN/Inf at rows {start}:{start+4096}')


def row_signature(names,xy):
    names=np.asarray(names).astype(str);xy=np.asarray(xy,dtype=np.float64)
    if len(set(names))!=len(names):raise ValueError('Duplicate cell IDs')
    if xy.shape!=(len(names),2) or not np.isfinite(xy).all():raise ValueError('Invalid spatial coordinates/row identity')
    return fingerprint(names,xy)


def validate_probabilities(probs,classes,n):
    for head,key in (('L1','l1'),('L12','l12'),('L2','l2')):
        a=np.asarray(probs[key]);finite_matrix(a,f'{head} probabilities')
        if a.shape!=(n,len(classes[head])):raise ValueError(f'{head} probability shape/class order mismatch')
        if (a<0).any() or (a>1).any() or not np.allclose(a.sum(1),1,atol=5e-4):
            raise ValueError(f'Invalid {head} probability normalization')


def atomic_h5ad(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # pandas 3 nullable strings must remain readable by older AnnData installations.
    import pandas as pd
    data=data.copy()
    for frame in (data.obs,data.var):
        for col in frame.columns:
            dtype=frame[col].dtype
            nullable=isinstance(dtype,pd.StringDtype)
            categorical=isinstance(dtype,pd.CategoricalDtype) and isinstance(frame[col].cat.categories.dtype,pd.StringDtype)
            if nullable or categorical:
                frame[col]=pd.Series(frame[col].astype(object).fillna(""),index=frame.index,dtype=object)
    data.obs.index=pd.Index(np.asarray(data.obs_names,dtype=object),dtype=object,name=data.obs_names.name)
    data.var.index=pd.Index(np.asarray(data.var_names,dtype=object),dtype=object,name=data.var_names.name)
    fd,tmp=tempfile.mkstemp(prefix=path.stem+'.',suffix='.h5ad',dir=path.parent);os.close(fd)
    try:data.write_h5ad(tmp);os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.stem+'.',suffix='.npz',dir=path.parent);os.close(fd)
    try:
        with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def complete(marker,stamp,outputs):
    try:
        run=json.loads(Path(marker).read_text())
        if run.get('version')!=VERSION or run.get('state')!='completed' or run.get('stamp')!=stamp:return None
        expected={str(Path(p).resolve()) for p in outputs}
        if expected!=set(run.get('outputs',{})):return None
        for p,d in run['outputs'].items():
            if not Path(p).is_file() or sha256(p)!=d:return None
        return run
    except (OSError,ValueError,KeyError,TypeError):return None


def publish(marker,stamp,outputs,**metadata):
    record={'version':VERSION,'state':'completed','stamp':stamp,'outputs':{str(Path(p).resolve()):sha256(p) for p in outputs},**metadata}
    atomic_json(marker,record);return record


def inference_coordinates(model,xy,*,unit=None,mpp=None):
    """Explicit incoming units; never interpret coordinates_um as an input unit."""
    cfg=model.spatial_fusion.config
    physical=bool(cfg.radii_um or cfg.geometry or cfg.weighting=='gaussian' or cfg.coordinates_um or cfg.mpp)
    if not physical:return np.asarray(xy),vars(cfg).copy()
    unit=unit or os.environ.get('HIST2PHENO_INFERENCE_COORDINATES_UNIT','pixel')
    if unit not in ('pixel','um'):raise ValueError('Coordinate unit must be pixel or um')
    if unit=='um':factor=1.
    else:
        mpp=mpp if mpp is not None else os.environ.get('HIST2PHENO_INFERENCE_MPP')
        if mpp is None:raise ValueError('Pixel coordinates need this section\'s calibrated mpp')
        factor=float(mpp)
        if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid inference mpp')
    config=vars(cfg).copy();config.update(coordinates_um=True,mpp=None)
    return np.asarray(xy,dtype=np.float64)*factor,config


def verify_prediction(path):
    """Read-only consumer guard; stale legacy/partial outputs cannot enter plots."""
    import anndata as ad
    path=Path(path);marker=path.with_suffix('.pipeline.json')
    run=json.loads(marker.read_text())
    if run.get('version')!=VERSION or str(path.resolve()) not in run.get('outputs',{}):
        raise ValueError(f'Unversioned prediction: {path}; run predict first')
    if run.get('state')!='completed':raise ValueError('Prediction transaction is not complete')
    for p,d in run['outputs'].items():
        if not Path(p).is_file() or sha256(p)!=d:raise ValueError(f'Prediction outputs changed: {p}; rerun predict')
    source=run['source'];cache=path.parent/'.provenance'
    if 'inputs' in source:
        if not inputs_current(source['inputs'],cache):raise ValueError('Prediction inputs changed; rerun predict')
    else:
        for record in [source['feature'],*source['model_files']]:
            if file_record(record['path'],cache)!=record:raise ValueError('Prediction feature/model changed')
        feature=Path(source['feature']['path']);feature_run=json.loads(feature.with_suffix('.source.json').read_text())
        feature_source=feature_run['source']
        if complete(feature.with_suffix('.source.json'),feature_run['stamp'],[feature]) is None:
            raise ValueError('Feature transaction is not complete')
        token=feature_source['tokens']
        if feature_manifest(token['directory'],cache)!=token:raise ValueError('Upstream token version changed')
        for key in ('csv','he','extraction_implementation','extraction_manifest'):
            if file_record(feature_source[key]['path'],cache)!=feature_source[key]:raise ValueError('Feature root input changed')
    if source['implementation']!=implementations():raise ValueError('Pipeline implementation changed')
    if source.get('predict_implementation') and source['predict_implementation']!=sha256(_PAN/'PanCancer_predict_visualize.py'):
        raise ValueError('Prediction implementation changed')
    return run


def read_verified_prediction(path):
    import anndata as ad
    verify_prediction(path)
    return ad.read_h5ad(path)


def capture_inputs(files=(),tokens=(),cache_dir=None,optional_tokens=()):
    records=[file_record(p,cache_dir) if Path(p).is_file() else {'path':str(Path(p).resolve()),'missing':True} for p in files]
    def token_record(p):
        record=feature_manifest(p,cache_dir)
        record['extraction_sha256']=file_record(Path(p)/'embedding_manifest.json',cache_dir)['sha256']
        return record
    token_records=[token_record(p) for p in tokens if p is not None]
    known={str(Path(x['directory']).resolve()) for x in token_records}
    for directory in optional_tokens:
        directory=Path(directory).resolve()
        if str(directory) in known:continue
        if directory.is_dir() and any(directory.glob('*.pth')):token_records.append(token_record(directory))
        else:token_records.append({'directory':str(directory),'missing':True})
    return {'files':records,'tokens':token_records,'mapping':mapping_descriptor()}


def inputs_current(record,cache_dir):
    try:
        if record['mapping']!=mapping_descriptor():return False
        for value in record['files']:
            if value.get('missing'):
                if Path(value['path']).is_file():return False
            elif file_record(value['path'],cache_dir)!=value:return False
        for value in record['tokens']:
            if value.get('missing'):
                if Path(value['directory']).is_dir() and any(Path(value['directory']).glob('*.pth')):return False
            else:
                current=feature_manifest(value['directory'],cache_dir)
                current['extraction_sha256']=file_record(Path(value['directory'])/'embedding_manifest.json',cache_dir)['sha256']
                if current!=value:return False
        return True
    except (OSError,ValueError,RuntimeError,KeyError):return False


def ground_truth_sources(data,cache_dir=None):
    """The supplied annotated h5ad is authoritative; declared upstream annotations must match."""
    raw=data.uns.get('ground_truth_sources_json','[]')
    records=json.loads(str(raw))
    if not isinstance(records,list):raise ValueError('ground_truth_sources_json must be a list of path/SHA records')
    paths=[]
    for record in records:
        if file_record(record['path'],cache_dir)!=record:
            raise ValueError('Ground-truth annotation source changed; rebuild the annotated h5ad before inference')
        paths.append(record['path'])
    return paths



def publish_derived_h5ad(data,path,prediction_path,*,probabilities,class_names,obs_names,parameters,algorithm):
    """Persist niche results only if probabilities, rows and current prediction agree."""
    run=verify_prediction(prediction_path)
    supplied=fingerprint(np.asarray(probabilities,dtype=np.float32),np.asarray(class_names,dtype=str),np.asarray(obs_names,dtype=str))
    if supplied!=run.get('probabilities_sha256'):
        raise ValueError('Derived calculation used probabilities/cells from another prediction version')
    source={'upstream_pipeline_run_id':run['stamp'],'probabilities_sha256':supplied,
            'parameters':canonical(parameters),'algorithm':file_record(algorithm)}
    stamp=fingerprint(source)
    data.uns['upstream_pipeline_run_id']=run['stamp'];data.uns['derived_run_id']=stamp
    data.uns['derived_source_json']=json.dumps(source,sort_keys=True)
    atomic_h5ad(data,path)
    # Revalidate before publishing derived completion; interrupted upstream writes are rejected.
    if verify_prediction(prediction_path)['stamp']!=run['stamp']:raise RuntimeError('Prediction changed during derived calculation')
    publish(Path(path).with_suffix('.derived.json'),stamp,[path],source=source,prediction_path=str(Path(prediction_path).resolve()))


def read_verified_derived(path):
    import anndata as ad
    path=Path(path);record=json.loads(path.with_suffix('.derived.json').read_text());source=record['source']
    if complete(path.with_suffix('.derived.json'),fingerprint(source),[path]) is None:
        raise ValueError('Invalid derived output')
    current=verify_prediction(record['prediction_path'])
    if current['stamp']!=source['upstream_pipeline_run_id']:raise ValueError('Derived output belongs to an older prediction')
    if file_record(source['algorithm']['path'])!=source['algorithm']:raise ValueError('Derived algorithm changed; recompute')
    return ad.read_h5ad(path)
