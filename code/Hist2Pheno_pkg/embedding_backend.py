########################################################
# 2026.10.29, encoder-specific paths/protocols / 各 encoder 独立目录、缓存、权重与协议
########################################################
from pathlib import Path
import re
import numpy as np
from embedding_store import available, extraction_metadata

SPECS={'uni':('UNI',16),'uni2':('UNI2',14),'virchow2':('Virchow2',14)}


def spec(embedder):
    try:return SPECS[str(embedder).lower()]
    except KeyError:raise ValueError('embedder must be uni, uni2 or virchow2')


def embedding_dir(h5ad_path,uns,embedder,*,stardist=False):
    method,patch=spec(embedder);parent=Path(h5ad_path).parent;sample=parent.name
    rel=Path('ImgEmbeddings_all_stardist' if stardist else 'ImgEmbeddings_all')/f'sc_pth_{patch}_{patch}'
    candidates=[];recorded=str(uns.get('embedding_dir','') or '')
    if recorded:
        swapped=re.sub(r'project_all_(?:UNI2|UNI|Virchow2)(?=/|$)','project_all_'+method,recorded)
        swapped=re.sub(r'sc_pth_\d+_\d+(?=/|$)',f'sc_pth_{patch}_{patch}',swapped)
        if stardist:swapped=swapped.replace('/ImgEmbeddings_all/','/ImgEmbeddings_all_stardist/')
        candidates.append(Path(swapped))
    candidates += [parent/f'{sample}_project_all_{method}'/rel,parent/f'project_all_{method}'/rel]
    for directory in dict.fromkeys(candidates):
        if not available(directory):
            continue
        meta=extraction_metadata(directory)
        found=meta.get('method')
        # 2026.11.01, leftover *.pth without a manifest are not a current extract
        if found is None:
            continue
        if found!=method:
            raise ValueError(f'{embedder} requested but token manifest is {found}: {directory}')
        return directory
    raise FileNotFoundError(f'No {method} embeddings for {h5ad_path}; looked at {candidates}')


def protocol(directory):
    from feature_integrity import feature_manifest
    root=Path(directory);manifest=feature_manifest(root,root/'.provenance')
    meta=extraction_metadata(root);weight=meta.get('encoder_weights')
    if meta.get('method') in ('UNI','Virchow2') and not weight:
        raise ValueError('Embeddings need the actual encoder weight SHA; re-extract unverified legacy data')
    preprocessing=dict(meta.get('encoder_protocol',{}))
    if meta.get('method') in ('UNI','UNI2','Virchow2') and not preprocessing:
        raise ValueError('Missing encoder preprocessing protocol; re-extract with the current encoder script')
    preprocessing.pop('requested_revision',None)  # Alias/tag spelling is not feature semantics.
    return {'method':meta.get('method'),'patch_size':meta.get('patch_size'),
            'encoder_sha256':(weight or {}).get('sha256'),
            'preprocessing':preprocessing,
            'dimension':manifest.get('dimension')}


def aligned(h5ad,adata,cache_dir,embedder,*,directory=None):
    from embedding_alignment import align_tokens
    from base import infer_pth_prefix_from_dir
    directory=Path(directory) if directory else embedding_dir(h5ad,adata.uns,embedder)
    method,_=spec(embedder)
    if extraction_metadata(directory).get('method')!=method:raise ValueError('Embedding method mismatch')
    current=protocol(directory)
    if embedder=='uni' and current['dimension']!=1024:
        raise ValueError('UNI patch16 pipeline requires 1024 dimensions')
    if embedder=='virchow2' and current['dimension']!=1280:
        raise ValueError('Virchow2 patch14 pipeline requires 1280 dimensions; do not reuse another crop/token layout')
    # Old annotation h5ad may record another encoder's prefix; use this store's identity.
    saved=adata.uns.get('pth_prefix');had='pth_prefix' in adata.uns
    adata.uns['pth_prefix']=infer_pth_prefix_from_dir(directory) or ''
    try:return align_tokens(h5ad,adata,directory,Path(cache_dir)/embedder)
    finally:
        if had:adata.uns['pth_prefix']=saved
        else:adata.uns.pop('pth_prefix',None)


def validate_model(model,scaler,embedder):
    meta=getattr(model,'checkpoint_metadata',{})
    declared=meta.get('embedder')
    if declared is not None and declared!=embedder:raise ValueError(f'Checkpoint encoder {declared} != requested {embedder}')
    if embedder in ('uni','virchow2'):
        if declared!=embedder or not meta.get('embedding_protocol'):
            raise ValueError(f'Use a newly trained {embedder} A2 checkpoint/artifact; another encoder or unversioned legacy weights are incompatible')
        expected=1024 if embedder=='uni' else 1280
        if int(scaler.n_features_in_)!=expected:raise ValueError(f'{embedder} scaler must have {expected} features')


def validate_tokens(model,directory,embedder):
    expected=getattr(model,'checkpoint_metadata',{}).get('embedding_protocol')
    # Existing paired_v2 UNI2 artifacts predate this field; retain their checked legacy path.
    if embedder=='uni2' and not expected:return
    if embedder not in ('uni','uni2','virchow2'):return
    if not expected or protocol(directory)!=expected:
        raise ValueError(f'Prediction embedding encoder/weights/preprocessing differ from the {embedder} training artifact')
