########################################################
# 2026.10.24, modular spatial context / 模块化物理邻域与流式上下文
# Cached fixed aggregation avoids (N,K,D) storage and full-pool GPU copies.
########################################################
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import tempfile
import numpy as np
import torch
from scipy.spatial import cKDTree
from checkpoint_artifact import digest


@dataclass(frozen=True)
class SpatialConfig:
    k: int = 8
    radii_um: tuple = ()
    mpp: float | None = None
    coordinates_um: bool = False
    weighting: str = 'uniform'
    sigma_um: float = 16.
    gate: bool = False
    geometry: bool = False
    cache_dir: str = '/tmp/hist2pheno_context'
    chunk_size: int = 1024
    # Explicit migration ablation; not used by default.
    allow_missing_context: bool = False

    def __post_init__(self):
        if self.k < 1 or self.chunk_size < 1:raise ValueError('k/chunk_size must be positive')
        if self.weighting not in ('uniform','gaussian'):raise ValueError('Unknown weighting')
        if self.mpp is not None and (not np.isfinite(self.mpp) or self.mpp<=0):raise ValueError('mpp must be positive')
        if any(not np.isfinite(r) or r<=0 for r in self.radii_um):raise ValueError('Radii must be positive µm')
        if (self.radii_um or self.weighting=='gaussian' or self.geometry) and not(self.coordinates_um or self.mpp):
            raise ValueError('Physical neighborhoods need coordinates_um=True or calibrated mpp')
        if not np.isfinite(self.sigma_um) or self.sigma_um<=0:raise ValueError('sigma_um must be positive')


def config_of(value=None,k=8):
    if isinstance(value,SpatialConfig):return value
    return SpatialConfig(**(value or {'k':k}))


def graphs(xy,groups,config):
    cfg=config_of(config);xy=np.asarray(xy,dtype=np.float64)
    if not np.isfinite(xy).all():raise ValueError('Non-finite spatial coordinates')
    xy=xy*(1. if cfg.coordinates_um else (cfg.mpp or 1.))
    # Per-row mpp conversion is done by the caller before entering this function.
    n=len(xy);result=[]
    for radius in cfg.radii_um or (np.inf,):
        idx=np.full((n,cfg.k),-1,np.int64);dist=np.full((n,cfg.k),np.inf,np.float32)
        for group in np.unique(groups):
            rows=np.flatnonzero(np.asarray(groups)==group)
            d,j=cKDTree(xy[rows]).query(xy[rows],k=min(cfg.k+1,len(rows)),distance_upper_bound=radius)
            d=np.asarray(d).reshape(len(rows),-1);j=np.asarray(j).reshape(len(rows),-1)
            # Exclude self by identity, not by deleting column 0 (duplicate coordinates).
            for local,row in enumerate(rows):
                good=(j[local]<len(rows)) & (j[local]!=local) & np.isfinite(d[local])
                take=j[local,good][:cfg.k];count=len(take)
                idx[row,:count]=rows[take];dist[row,:count]=d[local,good][:cfg.k]
        result.append((idx,dist))
    return result


def restrict(graph,train,val):
    idx,dist=graph;idx=idx.copy();dist=dist.copy();part=np.full(len(idx),-1,np.int8)
    part[train]=0;part[val]=1
    valid=idx>=0;safe=np.maximum(idx,0)
    valid &= (part[:,None]==part[safe]) & (part[:,None]>=0)
    idx[~valid]=-1;dist[~valid]=np.inf
    return idx,dist


def graph_weights(idx,dist,cfg):
    valid=idx>=0
    w=valid.astype(np.float32)
    if cfg.weighting=='gaussian':w*=np.exp(-.5*(dist/cfg.sigma_um)**2)
    denom=w.sum(1,keepdims=True)
    return w/np.maximum(denom,1e-30)


def prepare_runtime(cv,result,config):
    cfg=config_of(config);n,d=cv['X_f'].shape
    # A long-lived CV object may be reused for a different k/radius or coordinates.
    # An existing graph alone is not evidence that it matches the new request.
    graph_key=digest(asdict(cfg),np.asarray(cv['X_coords_f']),
                     np.asarray(cv.get('groups_f',np.zeros(n))))
    if cv.get('_context_graph_key') != graph_key:
        cv['_context_graphs']=graphs(cv['X_coords_f'],cv.get('groups_f',np.zeros(n)),cfg)
        cv['_context_graph_key']=graph_key
    gs=cv['_context_graphs']
    tr=result['train_global_indices'];va=result['val_global_indices']
    if np.intersect1d(np.asarray(cv['groups_f'])[tr],np.asarray(cv['groups_f'])[va]).size:
        gs=[restrict(g,tr,va) for g in gs]
    mode=result.get('spatial_mode','mean')
    scale=result['fold_scaler'];runtime={'config':asdict(cfg),'scaler':scale,'mode':mode}
    if mode=='attention':
        if cfg.geometry:raise ValueError('Attention geometry is not implemented; disable geometry or use mean')
        if len(gs)!=1:raise ValueError('Attention currently accepts one scale; use mean for multiscale')
        runtime.update(raw_X=cv['X_f'],graph=gs[0]);return runtime
    if mode!='mean':raise ValueError('Supported spatial modes are mean and attention')
    key=digest(cv['_data_fingerprint'],asdict(cfg),*[a for g in gs for a in g])
    folder=Path(cfg.cache_dir);folder.mkdir(parents=True,exist_ok=True)
    # 2026.11.16, optional float16 cache (HIST2PHENO_STORE_DTYPE); dtype is part of the file name
    sdt=np.dtype(os.environ.get('HIST2PHENO_STORE_DTYPE','float32').lower())
    if sdt not in (np.float32,np.float16):raise ValueError('HIST2PHENO_STORE_DTYPE must be float32 or float16')
    tag='' if sdt==np.float32 else '.f16'
    path=folder/(key+tag+'.npy');geom_path=folder/(key+'.geometry.npy')
    if not path.is_file() or not geom_path.is_file():
        import shutil
        required=n*len(gs)*d*sdt.itemsize+n*len(gs)*3*4
        if shutil.disk_usage(folder).free < required+10*1024**2:
            raise OSError(f'Context cache needs {required/1e9:.2f} GB free; choose a larger cache directory')
        fd,tmp=tempfile.mkstemp(prefix=key,dir=folder);os.close(fd)
        cache=np.lib.format.open_memmap(tmp,mode='w+',dtype=sdt,shape=(n,len(gs),d))
        geo=np.zeros((n,len(gs)*3),np.float32)
        try:
            for s,(idx,dist) in enumerate(gs):
                for start in range(0,n,cfg.chunk_size):
                    end=min(n,start+cfg.chunk_size);ii=idx[start:end];dd=dist[start:end]
                    w=graph_weights(ii,dd,cfg)
                    values=np.asarray(cv['X_f'][np.maximum(ii,0)],np.float32)
                    cache[start:end,s]=(values*w[:,:,None]).sum(1)
                    empty=(w.sum(1)==0)
                    cache[start:end,s][empty]=cv['X_f'][start:end][empty]
                    counts=(ii>=0).sum(1);finite=np.where(ii>=0,dd,0)
                    mean_d=finite.sum(1)/np.maximum(counts,1)
                    # Unitless count, log distance and density geometry.
                    geo[start:end,3*s:3*s+3]=np.column_stack([np.log1p(counts),np.log1p(mean_d),np.log1p(counts/(np.pi*np.maximum(mean_d,1)**2))])
            cache.flush();del cache
            os.replace(tmp,path)
            fd,gtmp=tempfile.mkstemp(prefix=key,dir=folder);os.close(fd)
            with open(gtmp,'wb') as f:np.save(f,geo)
            os.replace(gtmp,geom_path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    print(f"  context cache ready {path.name}", flush=True)
    runtime.update(contexts=np.load(path,mmap_mode='r'),geometry=np.load(geom_path,mmap_mode='r'))
    return runtime


def runtime_batch(runtime,rows,device):
    cfg=config_of(runtime['config']);rows=rows.detach().cpu().numpy();s=runtime['scaler']
    if runtime['mode']=='mean':
        c=np.array(runtime['contexts'][rows],dtype=np.float32,copy=True)
        # 2026.11.20, scale_rows = the same affine scaling for a StandardScaler; a ProjectedScaler
        # (train-fold PCA) projects raw-width rows first. 按折 PCA 时先投影再标准化。
        # c=(c-s.mean_.astype(np.float32))/s.scale_.astype(np.float32)
        from projected_scaler import scale_rows
        c=scale_rows(s,c)
        return {'contexts':torch.as_tensor(c,device=device),
                'geometry':torch.as_tensor(np.array(runtime['geometry'][rows],copy=True),device=device)}
    idx,dist=runtime['graph'];ii=idx[rows];dd=dist[rows];valid=ii>=0
    c=np.asarray(runtime['raw_X'][np.maximum(ii,0)],np.float32)
    # c=(c-s.mean_.astype(np.float32))/s.scale_.astype(np.float32)
    from projected_scaler import scale_rows  # 2026.11.20, see mean mode above
    c=scale_rows(s,c)
    # Isolated nodes fall back to the center, never to another section's first node.
    empty=~valid.any(1)
    # c[empty,0]=(runtime['raw_X'][rows[empty]]-s.mean_)/s.scale_
    if empty.any():c[empty,0]=scale_rows(s,np.asarray(runtime['raw_X'][rows[empty]],np.float32))
    valid[empty,0]=True;dd=dd.copy();dd[empty,0]=0
    return {'neighbors':torch.as_tensor(c,device=device),'mask':torch.as_tensor(valid,device=device),
            'distances':torch.as_tensor(dd,device=device)}


def inference_runtime(X,xy,scaler,config,mode='mean',groups=None):
    cfg=config_of(config);n=len(X)
    if os.environ.get('HIST2PHENO_CONTEXT_CACHE'):
        from dataclasses import replace
        cfg=replace(cfg,cache_dir=os.environ['HIST2PHENO_CONTEXT_CACHE'])
    cv={'X_f':X,'X_coords_f':xy,'groups_f':np.zeros(n) if groups is None else groups,
        '_data_fingerprint':digest(np.asarray(X),np.asarray(xy))}
    r={'train_global_indices':np.arange(n),'val_global_indices':np.array([],dtype=int),
       'fold_scaler':scaler,'spatial_mode':mode}
    return prepare_runtime(cv,r,cfg)


# Future shallow GNNs: register a sampled/block encoder, never dense N×N attention.
# Caller owns graph sampling; no optional torch-geometric dependency is imported.
GRAPH_ENCODERS={}
def register_graph_encoder(name,factory):
    if name in GRAPH_ENCODERS:raise ValueError('Graph encoder already registered')
    GRAPH_ENCODERS[name]=factory

def create_graph_encoder(name,**kwargs):
    if name not in GRAPH_ENCODERS:raise ValueError('Register a graph encoder factory first')
    return GRAPH_ENCODERS[name](**kwargs)


def estimate_graph_cost(n=5_500_000,k=8,d=1024,h=128,layers=2,gflops=5000.,bandwidth_gbs=20.):
    """Lower-bound compute/IO estimates, NOT an observed runtime prediction."""
    edges=n*k
    flops=2*n*d*h+max(0,layers-1)*2*n*h*h+layers*2*edges*h
    return {'directed_edges':edges,'int64_edges_GB':edges*8/1e9,
            'fp32_hidden_per_layer_GB':n*h*4/1e9,'aggregation_FLOPs':flops,
            'ideal_compute_seconds':flops/(gflops*1e9),
            'one_embedding_read_seconds':n*d*4/(bandwidth_gbs*1e9),
            'note':'Excludes backward, optimizer, sampling, CPU/Disk IO, kernels; benchmark on target farm GPU.'}


def prediction_context(model, X_scaled, neighbors):
    """Shared host-streamed inference for old indices or modern physical requests."""
    if not getattr(model,'use_spatial_context',False):return None
    if neighbors is None:raise ValueError('Spatial model needs neighbor indices or {"xy": coordinates}')
    if isinstance(neighbors,dict):
        if 'runtime' in neighbors:return neighbors['runtime']
        scaler=getattr(model,'checkpoint_scaler',None)
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler();scaler.mean_=np.zeros(X_scaled.shape[1]);scaler.scale_=np.ones(X_scaled.shape[1]);scaler.var_=scaler.scale_;scaler.n_features_in_=len(scaler.mean_);scaler.n_samples_seen_=len(X_scaled)
        raw=scaler.inverse_transform(X_scaled)
        cfg=neighbors.get("config_override",model.spatial_fusion.config)
        return inference_runtime(raw,neighbors['xy'],scaler,cfg,
            model.spatial_fusion.mode,groups=neighbors.get('groups'))
    cfg=model.spatial_fusion.config
    if cfg.radii_um or cfg.geometry or cfg.weighting!='uniform':
        raise ValueError('Physical context requires {"xy": coordinates}, not a plain kNN index')
    return {'legacy_X':X_scaled,'legacy_idx':np.asarray(neighbors)}


def prediction_batch(context,rows,device):
    if context is None:return None
    if 'legacy_X' not in context:return runtime_batch(context,rows,device)
    ii=context['legacy_idx'][rows.detach().cpu().numpy()]
    # Host gather: copies only B×K×D to GPU, never N×D per batch.
    return torch.as_tensor(np.asarray(context['legacy_X'][ii],np.float32),device=device)
