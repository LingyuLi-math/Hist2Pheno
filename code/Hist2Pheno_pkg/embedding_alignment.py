########################################################
# 2026.10.28, refresh recorded UNI tokens too / 有原始 token 路径时禁止沿用陈旧 h5ad.X
########################################################
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import pipeline_provenance as pp
except ImportError:
    import pipeline_provenance as pp


def align_tokens(h5ad,adata,directory,cache_dir):
    from base import load_hist_embeddings,match_embeddings,infer_pth_prefix_from_dir
    cache=Path(cache_dir);directory=Path(directory)
    if not directory.is_dir():raise FileNotFoundError(f'Recorded embedding directory missing: {directory}')
    xy=np.asarray(adata.obsm.get('spatial_HE',adata.obsm.get('spatial')),dtype=np.float64)
    prefix=str(adata.uns.get('pth_prefix') or infer_pth_prefix_from_dir(directory) or '')
    tolerance=float(adata.uns.get('match_tolerance',1.0))
    source={'file':pp.file_record(h5ad,cache),'tokens':pp.feature_manifest(directory,cache),
            'prefix':prefix,'tolerance':tolerance,'rows':pp.row_signature(adata.obs_names,xy),
            'implementation':pp.implementations()}
    stamp=pp.fingerprint(source);path=cache/'uni_aligned'/(stamp+'.npy');marker=path.with_suffix('.json')
    with pp.file_lock(str(path)+'.lock'):
        if pp.complete(marker,stamp,[path]):return np.load(path,mmap_mode='r')
        cells=pd.DataFrame({'centroid_x':xy[:,0],'centroid_y':xy[:,1],'celltype':'unlabeled'})
        embeddings=load_hist_embeddings(directory,celltype_df=cells,coord_cols=('centroid_x','centroid_y'),tolerance=tolerance,pth_prefix=prefix,strict=True)
        matches=match_embeddings(cells,embeddings,tolerance=tolerance,coord_cols=('centroid_x','centroid_y'))
        if not matches:raise ValueError('No current UNI tokens matched')
        dim=len(np.asarray(matches[0]['embedding']).reshape(-1));array=np.full((len(cells),dim),np.nan,dtype=np.float32)
        for match in matches:
            value=np.asarray(match['embedding'],dtype=np.float32).reshape(-1)
            if len(value)!=dim or not np.isfinite(value).all():raise ValueError('Invalid/mixed-dimension embedding')
            array[int(match['idx'])]=value
        if pp.feature_manifest(directory,cache)!=source['tokens']:raise RuntimeError('Tokens changed during alignment')
        path.parent.mkdir(parents=True,exist_ok=True)
        import tempfile,os
        fd,temp=tempfile.mkstemp(suffix='.npy',dir=path.parent);os.close(fd)
        try:np.save(temp,array);os.replace(temp,path)
        finally:Path(temp).unlink(missing_ok=True)
        pp.publish(marker,stamp,[path],source=source)
        return np.load(path,mmap_mode='r')
