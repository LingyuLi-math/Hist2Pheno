#!/usr/bin/env python
"""Tile-streamed StarDist H&E segmentation; coordinates use level-0 pixels."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
from seg_common import extract_instances, grid_starts, max_rss_gb, tile_cores


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True, help='Must be empty or absent')
    p.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent / 'models/2D_versatile_he')
    p.add_argument('--tile', type=int, default=2048)
    p.add_argument('--overlap', type=int, default=256)
    p.add_argument('--prob', type=float, help='Default: bundled model threshold')
    p.add_argument('--nms', type=float, help='Default: bundled model threshold')
    p.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                   help='Optional level-0 pixel region; outputs retain whole-slide coordinates')
    p.add_argument('--max-tiles', type=int, help='Process only the first N planned tiles (partial run)')
    p.add_argument('--mpp', type=float, help='Override microns per pixel; otherwise read slide metadata')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check-only', action='store_true', help='Read metadata and print plan; no inference or output writes')
    a = p.parse_args()
    if a.tile < 128 or not 0 <= a.overlap < a.tile:
        p.error('require tile >= 128 and 0 <= overlap < tile')
    if a.max_tiles is not None and a.max_tiles <= 0:
        p.error('--max-tiles must be positive')
    for name in ('prob', 'nms'):
        value = getattr(a, name)
        if value is not None and not 0 <= value <= 1:
            p.error(f'--{name} must be between 0 and 1')
    if a.mpp is not None and (not np.isfinite(a.mpp) or a.mpp <= 0):
        p.error('--mpp must be positive and finite')
    return a


def main():
    a = arguments()
    import openslide
    started = time.perf_counter()
    with openslide.OpenSlide(str(a.image.resolve(strict=True))) as slide:
        width, height = slide.dimensions
        x, y, rw, rh = a.roi or (0, 0, width, height)
        if min(x, y) < 0 or min(rw, rh) <= 0 or x + rw > width or y + rh > height:
            raise ValueError('--roi must be inside the full-resolution image')
        tiles = tile_cores(grid_starts(rw, a.tile, a.tile-a.overlap),
                           grid_starts(rh, a.tile, a.tile-a.overlap), rw, rh, a.tile)
        for t in tiles:
            for key in ('x0', 'x1', 'cx0', 'cx1'):
                t[key] += x
            for key in ('y0', 'y1', 'cy0', 'cy1'):
                t[key] += y
        planned = len(tiles)
        if a.max_tiles is not None:
            tiles = tiles[:a.max_tiles]
        thresholds = json.loads((a.model_dir / 'thresholds.json').read_text())
        prob = thresholds['prob'] if a.prob is None else a.prob
        nms = thresholds['nms'] if a.nms is None else a.nms
        for name in ('config.json', 'weights_best.h5'):
            if not (a.model_dir / name).is_file():
                raise FileNotFoundError(a.model_dir / name)
        mpp_x = a.mpp or slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = a.mpp or slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        mpp_x = float(mpp_x) if mpp_x is not None else None
        mpp_y = float(mpp_y) if mpp_y is not None else None
        plan = dict(input=str(a.image.resolve()), dimensions=[width, height],
                    roi=[x, y, rw, rh], tile=a.tile, overlap=a.overlap,
                    planned_tiles=planned, selected_tiles=len(tiles),
                    partial=(a.roi is not None or len(tiles) < planned),
                    prob=prob, nms=nms, mpp_x=mpp_x, mpp_y=mpp_y,
                    output=str(a.out_dir.resolve()), model=str(a.model_dir.resolve()))
        print(json.dumps(plan, indent=2), flush=True)
        required = ('tensorflow', 'stardist', 'csbdeep', 'cv2', 'pyarrow', 'skimage', 'PIL')
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            print('Missing Python dependencies: ' + ', '.join(missing), flush=True)
            if 'cv2' in missing:
                print('Install OpenCV in the selected environment; see README_embryo.md.', flush=True)
        if a.check_only:
            print('Metadata check passed. No segmentation was run.', flush=True)
            if missing:
                raise SystemExit(2)
            return
        if missing:
            raise SystemExit('Install missing dependencies before running segmentation.')
        if a.out_dir.exists() and any(a.out_dir.iterdir()):
            raise FileExistsError(f'Output is not empty: {a.out_dir}; use a new directory')
        # Import heavy dependencies only after metadata / argument validation.
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import random
        random.seed(a.seed)
        np.random.seed(a.seed)
        import tensorflow as tf
        import stardist
        import pyarrow as pa
        import pyarrow.parquet as pq
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
        tf.random.set_seed(a.seed)
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        model = StarDist2D(None, name=a.model_dir.resolve().name,
                           basedir=str(a.model_dir.resolve().parent))
        print(f'GPU devices: {len(gpus)}; normalization: tile joint-channel 1/99.8 percentiles', flush=True)
        a.out_dir.mkdir(parents=True, exist_ok=True)
        # Incremental outputs keep memory independent of total nucleus count.
        schema = pa.schema([('id', pa.int64())] + [(k, pa.float64()) for k in
                            ('cx_px', 'cy_px', 'area_px', 'poly_area_px', 'x0', 'y0', 'x1', 'y1')])
        count, polygons, next_id = 0, 0, 1
        with pq.ParquetWriter(a.out_dir / 'instances.parquet', schema) as parquet, \
                (a.out_dir / 'nuclei_coordinates.csv').open('w', newline='') as cf, \
                (a.out_dir / 'instances.geojson').open('w') as gf, \
                (a.out_dir / 'tile_times.csv').open('w', newline='') as tfh:
            coords, timing = csv.writer(cf), csv.writer(tfh)
            coords.writerow(['id', 'cx_px', 'cy_px', 'area_px'])
            timing.writerow(['tile', 'x0', 'y0', 'seconds', 'n_nuclei'])
            gf.write('{"type":"FeatureCollection","features":[')
            for k, t in enumerate(tiles):
                tick = time.perf_counter()
                rgba = slide.read_region((t['x0'], t['y0']), 0,
                                         (t['x1']-t['x0'], t['y1']-t['y0']))
                # Composite transparent slide pixels onto white, not black.
                from PIL import Image
                rgb = Image.new('RGB', rgba.size, 'white')
                rgb.paste(rgba, mask=rgba.getchannel('A'))
                crop = np.asarray(rgb).astype('float32')
                labels, _ = model.predict_instances(normalize(crop, 1, 99.8),
                                                    prob_thresh=prob, nms_thresh=nms)
                instances, next_id, _, _ = extract_instances(labels, t['x0'], t['y0'], t, next_id)
                rows = []
                for ins in instances:
                    row = dict(id=ins['id'], cx_px=ins['cx'], cy_px=ins['cy'],
                               area_px=ins['area'], poly_area_px=ins['poly_area'],
                               **{key: ins[key] for key in ('x0', 'y0', 'x1', 'y1')})
                    rows.append(row)
                    coords.writerow([row[key] for key in ('id', 'cx_px', 'cy_px', 'area_px')])
                    poly = ins['poly']
                    if poly is not None:
                        signed = np.sum(poly[:, 0]*np.roll(poly[:, 1], -1) -
                                        np.roll(poly[:, 0], -1)*poly[:, 1])
                        if signed < 0:
                            poly = poly[::-1]
                        ring = np.vstack([poly, poly[:1]]).tolist()
                        feature = dict(type='Feature', id=str(ins['id']),
                                       properties=row,
                                       geometry=dict(type='Polygon', coordinates=[ring]))
                        gf.write((',' if polygons else '') + json.dumps(feature))
                        polygons += 1
                if rows:
                    parquet.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(instances)
                elapsed = time.perf_counter()-tick
                timing.writerow([k, t['x0'], t['y0'], round(elapsed, 3), len(instances)])
                cf.flush()
                tfh.flush()
                print(f'tile {k+1}/{len(tiles)}: {elapsed:.1f}s, nuclei={len(instances)}, total={count}', flush=True)
            gf.write(']}')
        contract = dict(coordinate_unit='fullres_he_pixel', origin='top_left',
                        x='column', y='row', mpp_x=mpp_x, mpp_y=mpp_y,
                        note='GeoJSON coordinates are slide pixels, not geographic coordinates')
        (a.out_dir / 'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
        meta = dict(**plan, status='complete', n_instances=count, n_polygons=polygons,
                    n_degenerate_polygons=count-polygons, seed=a.seed,
                    normalization='tile, joint-channel 1/99.8 percentiles, no clipping',
                    elapsed_s=round(time.perf_counter()-started, 2), rss_max_gb=max_rss_gb(),
                    versions=dict(tensorflow=tf.__version__, stardist=stardist.__version__,
                                  openslide=openslide.__version__), n_gpu=len(gpus))
        # Completion marker is written last; interrupted outputs are incomplete.
        (a.out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
        print(f'DONE: {count} nuclei; output: {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
