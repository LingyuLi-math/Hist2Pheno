#!/usr/bin/env python3
########################################################
# 2026.10.28, validate completed embeddings / 校验完成清单，不以文件数量判定成功
########################################################
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'Hist2Pheno_pkg'))
import pipeline_provenance as pp
from embedding_store import extraction_metadata,compatible_extractor

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    try:
        metadata=extraction_metadata(args.directory)
        current=pp.sha256(Path(__file__).resolve().parent/'Image_feature_extraction.py')
        if not compatible_extractor(metadata,current):raise ValueError('Extractor version changed or missing')
        tokens=pp.feature_manifest(args.directory,args.directory/'.provenance')
        if not tokens['count']:raise ValueError('Empty embeddings')
        print(tokens['count'])
    except Exception as exc:
        print(f'Embeddings require refresh: {exc}',file=sys.stderr)
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
