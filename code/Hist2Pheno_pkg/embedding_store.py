########################################################
# 2026.10.28, section-level chunked embeddings / 每切片一个 HDF5，保持坐标与行身份
########################################################
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import h5py

FILENAME='embeddings.h5'
SCHEMA='hist2pheno-embeddings-v1'
# Filled from the pinned baseline and delivered extractor; unrelated future changes are not accepted.
BASELINE_EXTRACTOR_SHA='11d6296c57318eb5592eefd3a6161ff29e196005b7b7b94570ea8b79e7396497'
DELIVERED_EXTRACTOR_SHA='1c8abd777833ee2e1433045de4a8d98aae0d4ed404f62c6819285b2dcdeb1c22'


def extraction_metadata(directory):
    try:return json.loads((Path(directory)/'embedding_manifest.json').read_text())
    except (OSError,ValueError):return {}


def hdf5_path(directory):
    p=Path(directory)
    if p.is_file():return p
    return p/FILENAME


def uses_hdf5(directory):
    p=Path(directory)
    if p.is_file():return p.suffix in ('.h5','.hdf5')
    meta=extraction_metadata(p)
    if meta.get('storage_format')=='pth':return False
    return meta.get('storage_format')=='hdf5' or (p/FILENAME).is_file()


def available(directory):
    p=Path(directory)
    return uses_hdf5(p) or (p.is_dir() and (any(p.glob('*.pth')) or extraction_metadata(p).get('state')=='writing'))


def compatible_extractor(meta,current_sha):
    return meta.get('extractor_sha256')==current_sha or (
        current_sha==DELIVERED_EXTRACTOR_SHA and meta.get('extractor_sha256')==BASELINE_EXTRACTOR_SHA)


class EmbeddingWriter:
    """Buffer at most chunk_rows vectors; publish only an explicitly committed file."""
    def __init__(self,directory,*,dtype='float32',chunk_rows=1024,compression='lzf'):
        if dtype not in ('float32','float16','float64'):raise ValueError('Unsupported embedding dtype')
        if chunk_rows<1:raise ValueError('chunk_rows must be positive')
        if compression not in ('lzf','gzip',None,'none'):raise ValueError('Unknown compression')
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        fd,self.temp=tempfile.mkstemp(prefix='.embeddings.',suffix='.h5',dir=self.root);os.close(fd)
        self.file=h5py.File(self.temp,'w');self.file.attrs.update(schema=SCHEMA,complete=False)
        self.dtype=np.dtype(dtype);self.chunk_rows=int(chunk_rows);self.compression=None if compression in (None,'none') else compression
        self.buffer=[];self.count=0;self.dim=None;self.committed=False

    def append(self,vector,*,cell_id,xy,input_row=-1,legacy_name='',component_shapes=None):
        vector=np.asarray(vector).reshape(-1)
        if vector.size==0 or not np.isfinite(vector).all():raise ValueError('Invalid embedding')
        with np.errstate(over='ignore',invalid='ignore'):value=vector.astype(self.dtype)
        if not np.isfinite(value).all():raise ValueError('Embedding overflows requested storage dtype')
        xy=np.asarray(xy,dtype=np.float64)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid coordinates')
        if self.dim is None:
            self.dim=len(value);strings=h5py.string_dtype('utf-8')
            self.file.create_dataset('embeddings',shape=(0,self.dim),maxshape=(None,self.dim),chunks=(self.chunk_rows,self.dim),dtype=self.dtype,compression=self.compression,shuffle=True)
            self.file.create_dataset('coordinates',shape=(0,2),maxshape=(None,2),chunks=(self.chunk_rows,2),dtype='f8')
            for name in ('cell_id','legacy_name'):self.file.create_dataset(name,shape=(0,),maxshape=(None,),chunks=(self.chunk_rows,),dtype=strings)
            self.file.create_dataset('input_row',shape=(0,),maxshape=(None,),chunks=(self.chunk_rows,),dtype='i8')
        if len(value)!=self.dim:raise ValueError('Mixed embedding dimensions')
        self.buffer.append((value,str(cell_id),xy,int(input_row),str(legacy_name)))
        if len(self.buffer)>=self.chunk_rows:self.flush()

    def flush(self):
        if not self.buffer:return
        start=self.count;end=start+len(self.buffer)
        for name in self.file:
            d=self.file[name];d.resize((end,*d.shape[1:]))
        self.file['embeddings'][start:end]=np.stack([r[0] for r in self.buffer])
        self.file['cell_id'][start:end]=[r[1] for r in self.buffer]
        self.file['coordinates'][start:end]=np.stack([r[2] for r in self.buffer])
        self.file['input_row'][start:end]=[r[3] for r in self.buffer]
        self.file['legacy_name'][start:end]=[r[4] for r in self.buffer]
        self.count=end;self.buffer.clear()

    def commit(self,metadata=None):
        self.flush()
        if not self.count:raise ValueError('No embeddings written')
        self.file.attrs['metadata_json']=json.dumps(metadata or {},sort_keys=True)
        self.file.attrs['complete']=True;self.file.flush();self.file.close()
        os.replace(self.temp,self.root/FILENAME);self.committed=True
        return self.root/FILENAME

    def __enter__(self):return self
    def __exit__(self,*args):
        if self.file.id.valid:self.file.close()
        if not self.committed:Path(self.temp).unlink(missing_ok=True)


class EmbeddingTable:
    """Coordinates in memory; vectors are opened/read once per selected batch."""
    def __init__(self,directory,*,validate=True):
        self.path=hdf5_path(directory)
        if validate:
            from feature_integrity import feature_manifest
            feature_manifest(self.path.parent,self.path.parent/'.provenance')
        self._signature=self._stat()
        with h5py.File(self.path,'r') as f:
            if f.attrs.get('schema')!=SCHEMA or not f.attrs.get('complete'):raise ValueError('Incomplete/unsupported embedding store')
            self.coordinates=np.asarray(f['coordinates'],dtype=np.float64)
            self.count,self.dim=f['embeddings'].shape
            if len(self.coordinates)!=self.count or any(len(f[n])!=self.count for n in ('cell_id','input_row','legacy_name')):
                raise ValueError('HDF5 row identity/shape mismatch')
            self.dtype=str(f['embeddings'].dtype);self.metadata=json.loads(f.attrs.get('metadata_json','{}'))
        if self._stat()!=self._signature:raise RuntimeError('Embedding store changed during header read')
    def _stat(self):
        st=self.path.stat();return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def __len__(self):return self.count
    def read_rows(self,rows):
        rows=np.asarray(rows,dtype=np.int64)
        if rows.ndim!=1 or ((rows<0)|(rows>=self.count)).any():raise IndexError('Invalid embedding rows')
        if self._stat()!=self._signature:raise RuntimeError('Embedding generation changed; reopen store')
        result=np.empty((len(rows),self.dim),dtype=np.float32)
        with h5py.File(self.path,'r') as f:
            for start in range(0,len(rows),4096):
                selected=rows[start:start+4096];unique,inverse=np.unique(selected,return_inverse=True)
                result[start:start+len(selected)]=np.asarray(f['embeddings'][unique],dtype=np.float32)[inverse]
        if self._stat()!=self._signature:raise RuntimeError('Embedding generation changed during vector read')
        if not np.isfinite(result).all():raise ValueError('HDF5 embeddings contain NaN/Inf')
        return result


def descriptor(directory,cache_dir):
    from pipeline_provenance import file_record
    path=hdf5_path(directory);table=EmbeddingTable(path,validate=False)
    record=file_record(path,cache_dir)
    return {'version':3,'storage':'hdf5','directory':str(path.parent.resolve()),'count':len(table),
            'dimension':table.dim,'dtype':table.dtype,'fingerprint':record['sha256']}


def migrate_pth(source,destination=None,*,archive_pth=False,chunk_rows=1024,compression='lzf'):
    """Lossless float32 conversion; validated sources only, never deletes legacy files."""
    import hashlib
    import shutil
    from contextlib import ExitStack
    from datetime import datetime,timezone
    import pipeline_provenance as pp
    from base import _load_pth_embedding_file,_PTH_FLAT_BASENAME
    source=Path(source).resolve();destination=Path(destination or source).resolve()
    if uses_hdf5(source):raise ValueError('Source already selects HDF5; do not merge active formats')
    if destination==source and not archive_pth:raise ValueError('In-place conversion requires archive_pth=True; otherwise choose a separate destination')
    with ExitStack() as stack:
        for root in sorted(set((source,destination))):stack.enter_context(pp.file_lock(root/'.extraction.lock'))
        old_tokens=pp.feature_manifest(source,source/'.provenance')
        old=extraction_metadata(source)
        if (destination/FILENAME).exists():raise FileExistsError(destination/FILENAME)
        paths=list(source.glob('*.pth'))
        def inventory():return [(p.name,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in source.glob('*.pth')]
        before=inventory();prefix=None;expected=hashlib.sha256()
        moved=[]
        try:
            pp.atomic_json(destination/'embedding_manifest.json',{'state':'writing','version':pp.VERSION,'storage_format':'hdf5'})
            with EmbeddingWriter(destination,chunk_rows=chunk_rows,compression=compression) as writer:
                for i,path in enumerate(paths):
                    match=_PTH_FLAT_BASENAME.match(path.name)
                    if not match:raise ValueError(f'Cannot recover coordinates from {path.name}')
                    current,x,y=match.groups()
                    if prefix is None:prefix=current
                    if current!=prefix:raise ValueError('Mixed filename prefixes; split stores explicitly')
                    vector=np.asarray(_load_pth_embedding_file(path)).reshape(-1)
                    if vector.dtype not in (np.dtype('float16'),np.dtype('float32')):
                        raise ValueError('Lossless migration supports float16/float32 sources; do not silently downcast float64')
                    vector=vector.astype(np.float32);expected.update(memoryview(np.ascontiguousarray(vector)).cast('B'))
                    writer.append(vector,cell_id=path.stem,xy=(float(x),float(y)),input_row=i,legacy_name=path.stem)
                if inventory()!=before:raise RuntimeError('Source tokens changed during conversion')
                writer.commit({'prefix':prefix,'method':old.get('method'),'migration_source':old_tokens})
            actual=hashlib.sha256()
            with h5py.File(destination/FILENAME,'r') as f:
                for start in range(0,len(paths),chunk_rows):actual.update(memoryview(np.asarray(f['embeddings'][start:start+chunk_rows],dtype=np.float32)).cast('B'))
            if actual.digest()!=expected.digest():raise RuntimeError('HDF5 round-trip verification failed')
            if destination==source:
                archive=source/'previous_embeddings'/('converted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                archive.mkdir(parents=True)
                for path in paths:
                    shutil.move(str(path),str(archive/path.name));moved.append((archive/path.name,path))
            record=dict(old,storage_format='hdf5',storage_dtype='float32',tokens=descriptor(destination,destination/'.provenance'),
                state='completed',format_migration={'source_tokens':old_tokens,'lossless_float32':True})
            pp.atomic_json(destination/'embedding_manifest.json',record)
            # Do not leave an apparently completed result if source validation no longer holds.
            pp.feature_manifest(destination,destination/'.provenance')
            return destination/FILENAME
        except Exception:
            for archived,original in reversed(moved):
                if archived.exists():shutil.move(str(archived),str(original))
            failed=destination/FILENAME
            if failed.exists():
                quarantine=destination/'previous_embeddings'/('failed_conversion_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.h5')
                quarantine.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(failed),str(quarantine))
            if destination==source:pp.atomic_json(source/'embedding_manifest.json',old)
            raise

