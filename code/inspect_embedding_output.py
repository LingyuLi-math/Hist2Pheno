#!/usr/bin/env python3
"""Inspect one embedding directory without loading its N×D feature matrix or deleting files."""
from pathlib import Path
import argparse
import json
import os
import sys


def count_suffix(directory,suffix):
    if not directory.is_dir():return 0
    with os.scandir(directory) as entries:
        return sum(entry.is_file(follow_symlinks=False) and entry.name.endswith(suffix) for entry in entries)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--verify',action='store_true',help='Also verify completed provenance and hashes; may read large source files on first use')
    parser.add_argument('--include-archives',action='store_true',help='Count archived .pth recursively; may take time for millions of files')
    args=parser.parse_args(argv);root=args.directory.resolve()
    if not root.is_dir():parser.error(f'No such directory: {root}')
    marker=root/'embedding_manifest.json'
    try:meta=json.loads(marker.read_text())
    except (OSError,ValueError):meta={}
    target=root/'embeddings.h5ad'
    info={'directory':str(root),'manifest_state':meta.get('state','missing_or_invalid'),
          'storage_format':meta.get('storage_format'), 'final_h5ad_exists':target.is_file(),
          'root_pth_files':count_suffix(root,'.pth'),
          'existing_png_crops':count_suffix(Path(str(root)+'_image'),'.png'),
          'legacy_archive_exists':(root/'previous_embeddings').is_dir(),
          'temporary_h5ad_files':sum(1 for _ in root.glob('.embeddings.*.h5ad')),
          'provenance_verified':False}
    if target.is_file():
        import h5py
        with h5py.File(target,'r') as f:
            info['standard_anndata']=f.attrs.get('encoding-type')=='anndata' and 'X' in f
            if 'X' in f:info.update(shape=list(f['X'].shape),dtype=str(f['X'].dtype))
    if meta.get('state')!='completed':info['note']='Current extraction is not complete; a final file, if present, may belong to an older generation.'
    else:info['note']='Existing PNG crops and previous_embeddings archives are not deleted automatically.'
    if args.include_archives:
        info['archived_pth_files']=sum(1 for _ in (root/'previous_embeddings').rglob('*.pth'))
    if args.verify:
        sys.path.insert(0,str(Path(__file__).resolve().parent/'Hist2Pheno_pkg'))
        from pipeline_provenance import feature_manifest
        feature_manifest(root,root/'.provenance')
        info['provenance_verified']=True
    print(json.dumps(info,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
