########################################################
# 2026.10.24, optional batch/sharded feature IO / 可选批量特征存储适配器
# The original Image_feature_extraction.py and .pth interfaces stay unchanged.
########################################################
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import h5py


def extract_to_h5(items, embed_batch, output, *, batch_size=128, metadata=None, overwrite=False):
    """items yields (cell_id,x,y,patch); embed_batch(patches) returns (B,D).

    The callback must use the original encoder/preprocessing (no crop-size change).
    Store checkpoint/preprocessing identity in metadata; use a new output path.
    """
    if batch_size<1:raise ValueError('batch_size must be positive')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and not overwrite:raise FileExistsError(output)
    fd,tmp=tempfile.mkstemp(prefix='.'+output.name,dir=output.parent);os.close(fd)
    count=0;seen=set()
    try:
        with h5py.File(tmp,'w') as f:
            f.attrs['schema']='Hist2Pheno.embedding.v1';f.attrs['metadata']=json.dumps(metadata or {},sort_keys=True)
            pending=[]
            def flush(batch):
                nonlocal count
                vec=embed_batch([r[3] for r in batch])
                if hasattr(vec,'detach'):vec=vec.detach().float().cpu().numpy()
                vec=np.asarray(vec,dtype=np.float32)
                if vec.ndim!=2 or len(vec)!=len(batch) or not np.isfinite(vec).all():raise ValueError('Expected finite (B,D) embeddings')
                ids=[str(r[0]) for r in batch]
                if len(set(ids))!=len(ids) or any(i in seen for i in ids):raise ValueError('Duplicate cell_id')
                seen.update(ids)
                if 'embeddings' not in f:
                    f.create_dataset('embeddings',shape=(0,vec.shape[1]),maxshape=(None,vec.shape[1]),dtype='f4',chunks=(min(batch_size,1024),vec.shape[1]),compression='lzf')
                    f.create_dataset('cell_id',shape=(0,),maxshape=(None,),dtype=h5py.string_dtype('utf-8'),chunks=True)
                    f.create_dataset('xy',shape=(0,2),maxshape=(None,2),dtype='f8',chunks=True)
                end=count+len(batch)
                for k in ('embeddings','cell_id','xy'):f[k].resize(end,axis=0)
                f['embeddings'][count:end]=vec;f['cell_id'][count:end]=ids
                f['xy'][count:end]=np.asarray([[r[1],r[2]] for r in batch],dtype=float)
                count=end
            for item in items:
                pending.append(item)
                if len(pending)==batch_size:flush(pending);pending=[]
            if pending:flush(pending)
            if count==0:raise ValueError('No valid image patches')
            f.attrs['n_cells']=count
        os.replace(tmp,output)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return output


def iter_h5_embeddings(path,chunk_size=8192):
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    with h5py.File(path,'r') as f:
        for start in range(0,len(f['embeddings']),chunk_size):
            end=start+chunk_size
            yield f['embeddings'][start:end],f['cell_id'].asstr()[start:end],f['xy'][start:end]
