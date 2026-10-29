########################################################
# 2026.10.28, section-level chunked embeddings / 每切片一个标准 h5ad，保持坐标与行身份
########################################################
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import h5py

FILENAME='embeddings.h5ad'
LEGACY_FILENAME='embeddings.h5'
SCHEMA='hist2pheno-embeddings-h5ad-v1'
LEGACY_SCHEMA='hist2pheno-embeddings-v1'
PREVIOUS_EXTRACTOR_SHA='1c8abd777833ee2e1433045de4a8d98aae0d4ed404f62c6819285b2dcdeb1c22'
# Only reviewed pre-relabeling UNI/UNI2 extractors are compatible; the 2026.10.29 partial merge must be re-extracted.
BASELINE_EXTRACTOR_SHA='11d6296c57318eb5592eefd3a6161ff29e196005b7b7b94570ea8b79e7396497'
DELIVERED_EXTRACTOR_SHA='bea817c0cd05331b2044486ea393bfac73b81c2c75f1a710f4d04ae7d9dd7b9d'


def extraction_metadata(directory):
    try:return json.loads((Path(directory)/'embedding_manifest.json').read_text())
    except (OSError,ValueError):return {}


def storage_format(directory):
    """Resolve the manifest-selected generation; never merge multiple formats."""
    p=Path(directory)
    if p.is_file():
        return 'h5ad' if p.suffix=='.h5ad' else ('hdf5' if p.suffix in ('.h5','.hdf5') else 'pth')
    selected=extraction_metadata(p).get('storage_format')
    if selected in ('h5ad','hdf5','pth'):return selected
    if (p/FILENAME).is_file():return 'h5ad'
    if (p/LEGACY_FILENAME).is_file():return 'hdf5'
    return 'pth'


def hdf5_path(directory):
    # Historical helper name retained: h5ad is also backed by HDF5.
    p=Path(directory)
    if p.is_file():return p
    return p/(LEGACY_FILENAME if storage_format(p)=='hdf5' else FILENAME)


def uses_hdf5(directory):
    """Compatibility name: True for h5ad and previous custom HDF5 stores."""
    return storage_format(directory) in ('h5ad','hdf5')


def available(directory):
    p=Path(directory)
    return uses_hdf5(p) or (p.is_dir() and (any(p.glob('*.pth')) or extraction_metadata(p).get('state')=='writing'))


def compatible_extractor(meta,current_sha):
    return meta.get('extractor_sha256')==current_sha or (
        current_sha==DELIVERED_EXTRACTOR_SHA and meta.get('method') in ('UNI','UNI2') and meta.get('extractor_sha256') in (BASELINE_EXTRACTOR_SHA,PREVIOUS_EXTRACTOR_SHA))


class EmbeddingWriter:
    """Buffer at most chunk_rows vectors; publish only an explicitly committed file."""
    def __init__(self,directory,*,dtype='float32',chunk_rows=1024,compression='lzf'):
        if dtype not in ('float32','float16','float64'):raise ValueError('Unsupported embedding dtype')
        if chunk_rows<1:raise ValueError('chunk_rows must be positive')
        if compression not in ('lzf','gzip',None,'none'):raise ValueError('Unknown compression')
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        fd,self.temp=tempfile.mkstemp(prefix='.embeddings.',suffix='.h5ad',dir=self.root);os.close(fd)
        self.file=h5py.File(self.temp,'w');self.file.attrs.update(schema=SCHEMA,complete=False)
        self.dtype=np.dtype(dtype);self.chunk_rows=int(chunk_rows);self.compression=None if compression in (None,'none') else compression
        self.buffer=[];self.count=0;self.dim=None;self.committed=False;self.component_shapes=None

    def append(self,vector,*,cell_id,xy,input_row=-1,legacy_name='',component_shapes=None):
        vector=np.asarray(vector).reshape(-1)
        if vector.size==0 or not np.isfinite(vector).all():raise ValueError('Invalid embedding')
        with np.errstate(over='ignore',invalid='ignore'):value=vector.astype(self.dtype)
        if not np.isfinite(value).all():raise ValueError('Embedding overflows requested storage dtype')
        xy=np.asarray(xy,dtype=np.float64)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid coordinates')
        if component_shapes is not None:
            shapes=[list(shape) for shape in component_shapes]
            if sum(int(np.prod(shape)) for shape in shapes)!=len(value):
                raise ValueError('Tensor component shapes do not match embedding dimension')
            if self.component_shapes is None:self.component_shapes=shapes
            elif self.component_shapes!=shapes:raise ValueError('Tensor component layout changed between cells')
        if self.dim is None:
            self.dim=len(value);strings=h5py.string_dtype('utf-8')
            self.file.create_dataset('X',shape=(0,self.dim),maxshape=(None,self.dim),chunks=(self.chunk_rows,self.dim),dtype=self.dtype,compression=self.compression,shuffle=True)
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
        self.file['X'][start:end]=np.stack([r[0] for r in self.buffer])
        self.file['cell_id'][start:end]=[r[1] for r in self.buffer]
        self.file['coordinates'][start:end]=np.stack([r[2] for r in self.buffer])
        self.file['input_row'][start:end]=[r[3] for r in self.buffer]
        self.file['legacy_name'][start:end]=[r[4] for r in self.buffer]
        self.count=end;self.buffer.clear()

    def commit(self,metadata=None):
        """Finalize AnnData encodings without loading the N×D matrix into RAM."""
        self.flush()
        if not self.count:raise ValueError('No embeddings written')
        import pandas as pd
        try:
            from anndata.io import write_elem
        except ImportError:  # anndata < 0.11
            from anndata._io.specs import write_elem
        meta=dict(metadata or {})
        if self.component_shapes is not None:
            meta.update(component_shapes=self.component_shapes,flatten_order='C')
        else:
            meta.setdefault('component_shapes',[[self.dim]])
            meta.setdefault('flatten_order','C')
        self.file.attrs['metadata_json']=json.dumps(meta,sort_keys=True)
        self.file.attrs.update({'encoding-type':'anndata','encoding-version':'0.1.0'})
        def encode(dataset,kind='array'):
            dataset.attrs.update({'encoding-type':kind,'encoding-version':'0.2.0'})
        encode(self.file['X'])
        obs=self.file.create_group('obs')
        obs.attrs.update({'encoding-type':'dataframe','encoding-version':'0.2.0',
                          '_index':'cell_id','column-order':np.asarray(['input_row','legacy_name'],dtype=h5py.string_dtype())})
        for name in ('cell_id','input_row','legacy_name'):
            self.file.move(name,'obs/'+name)
            encode(obs[name],'string-array' if name!='input_row' else 'array')
        # Only D feature names are materialized; all cell metadata stays on disk.
        write_elem(self.file,'var',pd.DataFrame(index=pd.Index([f'embedding_{j}' for j in range(self.dim)],name='feature',dtype=object)))
        obsm=self.file.create_group('obsm')
        obsm.attrs.update({'encoding-type':'dict','encoding-version':'0.1.0'})
        self.file.move('coordinates','obsm/spatial');encode(obsm['spatial'])
        for name in ('layers','obsp','varm','varp'):write_elem(self.file,name,{})
        write_elem(self.file,'uns',{'embedding_store_json':json.dumps(meta,sort_keys=True),
                                  'embedding_store_schema':SCHEMA})
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
            schema=f.attrs.get('schema')
            if schema not in (SCHEMA,LEGACY_SCHEMA) or not f.attrs.get('complete'):
                raise ValueError('Incomplete/unsupported embedding store')
            modern=schema==SCHEMA
            if modern and f.attrs.get('encoding-type')!='anndata':raise ValueError('Invalid AnnData encoding')
            self.matrix_key='X' if modern else 'embeddings'
            self.coordinates=np.asarray(f['obsm/spatial' if modern else 'coordinates'],dtype=np.float64)
            self.count,self.dim=f[self.matrix_key].shape
            names=('obs/cell_id','obs/input_row','obs/legacy_name') if modern else ('cell_id','input_row','legacy_name')
            if self.coordinates.shape!=(self.count,2) or any(len(f[n])!=self.count for n in names):
                raise ValueError('Embedding row identity/shape mismatch')
            self.dtype=str(f[self.matrix_key].dtype);self.metadata=json.loads(f.attrs.get('metadata_json','{}'))
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
                result[start:start+len(selected)]=np.asarray(f[self.matrix_key][unique],dtype=np.float32)[inverse]
        if self._stat()!=self._signature:raise RuntimeError('Embedding generation changed during vector read')
        if not np.isfinite(result).all():raise ValueError('HDF5 embeddings contain NaN/Inf')
        return result


def descriptor(directory,cache_dir):
    from pipeline_provenance import file_record
    path=hdf5_path(directory);table=EmbeddingTable(path,validate=False)
    record=file_record(path,cache_dir)
    return {'version':3,'storage':storage_format(path),'directory':str(path.parent.resolve()),'count':len(table),
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
    if storage_format(source)=='hdf5':
        return migrate_legacy_hdf5(source,destination,archive_pth=archive_pth,chunk_rows=chunk_rows,compression=compression)
    if storage_format(source)=='h5ad':raise ValueError('Source already selects h5ad; do not merge active formats')
    if destination==source and not archive_pth:raise ValueError('In-place conversion requires archive_pth=True; otherwise choose a separate destination')
    with ExitStack() as stack:
        for root in sorted(set((source,destination))):stack.enter_context(pp.file_lock(root/'.extraction.lock'))
        old_tokens=pp.feature_manifest(source,source/'.provenance')
        old=extraction_metadata(source)
        if (destination/FILENAME).exists():raise FileExistsError(destination/FILENAME)
        if destination!=source and (destination/'embedding_manifest.json').exists():raise FileExistsError(destination/'embedding_manifest.json')
        paths=list(source.glob('*.pth'))
        def inventory():return [(p.name,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in source.glob('*.pth')]
        before=inventory();prefix=None;expected=hashlib.sha256()
        moved=[]
        try:
            pp.atomic_json(destination/'embedding_manifest.json',{'state':'writing','version':pp.VERSION,'storage_format':'h5ad'})
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
                for start in range(0,len(paths),chunk_rows):actual.update(memoryview(np.asarray(f['X'][start:start+chunk_rows],dtype=np.float32)).cast('B'))
            if actual.digest()!=expected.digest():raise RuntimeError('HDF5 round-trip verification failed')
            if destination==source:
                archive=source/'previous_embeddings'/('converted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                archive.mkdir(parents=True)
                for path in paths:
                    shutil.move(str(path),str(archive/path.name));moved.append((archive/path.name,path))
            record=dict(old,storage_format='h5ad',storage_dtype='float32',tokens=descriptor(destination,destination/'.provenance'),
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
                quarantine=destination/'previous_embeddings'/('failed_conversion_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.h5ad')
                quarantine.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(failed),str(quarantine))
            if destination==source:pp.atomic_json(source/'embedding_manifest.json',old)
            raise



def migrate_legacy_hdf5(source,destination=None,*,archive_pth=False,chunk_rows=1024,compression='lzf'):
    """2026.10.28, old embeddings.h5 -> standard h5ad / 不重新运行 encoder。"""
    import hashlib
    import shutil
    from contextlib import ExitStack
    from datetime import datetime,timezone
    import pipeline_provenance as pp
    source=Path(source).resolve();destination=Path(destination or source).resolve()
    if destination==source and not archive_pth:
        raise ValueError('In-place conversion requires archive_pth=True; old HDF5 will be archived')
    with ExitStack() as stack:
        for root in sorted(set((source,destination))):stack.enter_context(pp.file_lock(root/'.extraction.lock'))
        source_tokens=pp.feature_manifest(source,source/'.provenance')
        old=extraction_metadata(source);old_path=source/LEGACY_FILENAME
        table=EmbeddingTable(old_path,validate=False)
        if table.dtype not in ('float16','float32'):raise ValueError('Lossless conversion supports float16/float32 only')
        if (destination/FILENAME).exists():raise FileExistsError(destination/FILENAME)
        if destination!=source and (destination/'embedding_manifest.json').exists():raise FileExistsError(destination/'embedding_manifest.json')
        source_record=pp.file_record(old_path,source/'.provenance')
        archived=None
        try:
            pp.atomic_json(destination/'embedding_manifest.json',{'state':'writing','storage_format':'h5ad','version':pp.VERSION})
            expected=hashlib.sha256()
            with EmbeddingWriter(destination,chunk_rows=chunk_rows,compression=compression) as writer:
                with h5py.File(old_path,'r') as f:
                    for start in range(0,len(table),chunk_rows):
                        end=min(len(table),start+chunk_rows)
                        values=table.read_rows(np.arange(start,end))
                        expected.update(memoryview(np.ascontiguousarray(values)).cast('B'))
                        ids=f['cell_id'].asstr()[start:end];names=f['legacy_name'].asstr()[start:end]
                        rows=f['input_row'][start:end]
                        for j,vector in enumerate(values):
                            writer.append(vector,cell_id=ids[j],xy=table.coordinates[start+j],input_row=rows[j],legacy_name=names[j])
                if source_record!=pp.file_record(old_path,source/'.provenance'):raise RuntimeError('Source HDF5 changed during conversion')
                writer.commit(dict(table.metadata,format_migration_source=source_tokens))
            actual=hashlib.sha256()
            with h5py.File(destination/FILENAME,'r') as f:
                for start in range(0,len(table),chunk_rows):
                    actual.update(memoryview(np.asarray(f['X'][start:start+chunk_rows],dtype=np.float32)).cast('B'))
            if actual.digest()!=expected.digest():raise RuntimeError('h5ad round-trip verification failed')
            if destination==source:
                archived=source/'previous_embeddings'/('converted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))/LEGACY_FILENAME
                archived.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(old_path),str(archived))
            record=dict(old,state='completed',storage_format='h5ad',storage_dtype='float32',
                        tokens=descriptor(destination,destination/'.provenance'),
                        format_migration={'source_tokens':source_tokens,'lossless_float32':True})
            pp.atomic_json(destination/'embedding_manifest.json',record)
            pp.feature_manifest(destination,destination/'.provenance')
            return destination/FILENAME
        except Exception:
            if archived is not None and archived.exists():shutil.move(str(archived),str(old_path))
            failed=destination/FILENAME
            if failed.exists():
                quarantine=destination/'previous_embeddings'/('failed_conversion_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.h5ad')
                quarantine.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(failed),str(quarantine))
            if destination==source:pp.atomic_json(source/'embedding_manifest.json',old)
            raise
