#!/usr/bin/env python3
"""2026.10.28, migrate validated per-cell .pth to one lossless section HDF5."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'Hist2Pheno_pkg'))
from embedding_store import migrate_pth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--destination',type=Path,default=None)
    p.add_argument('--archive-pth',action='store_true',help='Required in-place; move legacy files into previous_embeddings, never delete')
    p.add_argument('--chunk-rows',type=int,default=1024)
    p.add_argument('--compression',choices=('lzf','gzip','none'),default='lzf')
    a=p.parse_args()
    print(migrate_pth(a.source,a.destination,archive_pth=a.archive_pth,chunk_rows=a.chunk_rows,compression=a.compression))


if __name__=='__main__':main()
