########################################################
# 2026.10.24, bind model/scaler/protocol / 模型、标准化器与验证协议绑定
# Bare state_dict stays compatible; .artifact.pt is the validated companion.
########################################################
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch


def digest(*values):
    h = hashlib.sha256()
    for v in values:
        if isinstance(v, np.ndarray):
            h.update(str((v.shape, str(v.dtype))).encode())
            if v.dtype.kind in 'OUS':
                for row in v.reshape(-1):
                    h.update(json.dumps(str(row)).encode()); h.update(b'\0')
            else:
                a = np.ascontiguousarray(v)
                h.update(memoryview(a).cast('B'))
        else:
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
        h.update(b'\xff')
    return h.hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
    return h.hexdigest()


def atomic_save(obj, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent);os.close(fd)
    try:torch.save(obj,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def companion(path):return Path(str(path)+'.artifact.pt')


def scaler_state(scaler):
    state={k:torch.as_tensor(np.asarray(getattr(scaler,k)).copy()) for k in
            ('mean_','var_','scale_','n_samples_seen_')} | {'n_features_in_':int(scaler.n_features_in_)}
    # 2026.11.20, train-fold PCA (projected_scaler.ProjectedScaler) is saved with the scaler
    # 按折拟合的 PCA 投影与 scaler 一起保存，外部预测时自动使用同一投影。
    if getattr(scaler,'proj_components_',None) is not None:
        from projected_scaler import STATE_KEYS
        state |= {k:torch.as_tensor(np.asarray(getattr(scaler,k)).copy()) for k in STATE_KEYS}
    return state


def restore_scaler(state):
    from sklearn.preprocessing import StandardScaler
    # 2026.11.20, a state with proj_components_ restores the ProjectedScaler
    if 'proj_components_' in state:
        from projected_scaler import restore
        return restore({k:(v.cpu().numpy() if isinstance(v,torch.Tensor) else v) for k,v in state.items()})
    s=StandardScaler()
    for k,v in state.items():setattr(s,k,v.cpu().numpy() if isinstance(v,torch.Tensor) else v)
    return s


def save_artifact(path, scaler, metadata, protocol):
    atomic_save({'version':1,'complete':True,'weight_sha256':file_digest(path),
                 'scaler':scaler_state(scaler),'metadata':metadata,'protocol':protocol}, companion(path))


def read_artifact(path, required=False):
    p=companion(path)
    if not p.is_file():
        if required:raise ValueError(f'Missing validated artifact: {p}; retrain or explicitly migrate legacy weights')
        return None
    a=torch.load(p,map_location='cpu',weights_only=True)
    if not a.get('complete') or a['weight_sha256'] != file_digest(path):
        raise ValueError(f'Incomplete/stale checkpoint artifact: {path}')
    return a


def checkpoint_scaler(path, fallback=None, allow_legacy=False):
    a=read_artifact(path)
    if a is not None:return restore_scaler(a['scaler'])
    if allow_legacy and fallback is not None:return fallback
    raise ValueError('Scaler must come from the checkpoint artifact; implicit refitting is disabled')


def copy_checkpoint(src,dst):
    a=read_artifact(src,required=True)
    if Path(src).resolve()==Path(dst).resolve():return
    shutil.copy2(src,dst)
    atomic_save(a,companion(dst))


def reusable(path, protocol):
    """Incomplete/incompatible folds restart; never call best-so-far a completed fold."""
    if not Path(path).is_file():return False
    try:a=read_artifact(path,required=True)
    except ValueError:return False
    return a['protocol']==protocol
