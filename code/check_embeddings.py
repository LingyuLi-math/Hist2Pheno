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
    parser.add_argument('--method')
    parser.add_argument('--patch-size',type=int)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--positions',type=Path)
    parser.add_argument('--scale-image',choices=('True','False'))
    parser.add_argument('--scale',type=float)
    parser.add_argument('--encoder-weights',type=Path)
    parser.add_argument('--encoder-revision')
    args=parser.parse_args()
    try:
        metadata=extraction_metadata(args.directory)
        if metadata.get('method') in ('UNI','Virchow2') and not metadata.get('encoder_protocol'):
            raise ValueError('Missing encoder protocol; extraction must be refreshed')
        current=pp.sha256(Path(__file__).resolve().parent/'Image_feature_extraction.py')
        if not compatible_extractor(metadata,current):raise ValueError('Extractor version changed or missing')
        for name,expected in (('method',args.method),('patch_size',args.patch_size),('scale',args.scale)):
            if expected is not None and metadata.get(name)!=expected:raise ValueError(f'Requested {name} changed')
        if args.scale_image is not None and metadata.get('scale_image')!=(args.scale_image=='True'):raise ValueError('Requested scale_image changed')
        for name,path in (('image',args.image),('positions',args.positions)):
            if path is not None and metadata.get('inputs',{}).get(name)!=pp.file_record(path):raise ValueError(f'Requested {name} changed')
        if args.encoder_weights is not None and metadata.get('encoder_weights')!=pp.file_record(args.encoder_weights):raise ValueError('Requested encoder weights changed')
        if args.encoder_revision is not None and metadata.get('encoder_protocol',{}).get('requested_revision')!=args.encoder_revision:raise ValueError('Requested encoder revision changed')
        tokens=pp.feature_manifest(args.directory,args.directory/'.provenance')
        if not tokens['count']:raise ValueError('Empty embeddings')
        print(tokens['count'])
    except Exception as exc:
        print(f'Embeddings require refresh: {exc}',file=sys.stderr)
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
