#!/usr/bin/env python
"""StarDist nuclei on one H&E TIFF. No Space Ranger directory required.

Same model and thresholds as the Visium HD run: 2D_versatile_he,
prob_thresh=0.01, nms_thresh=0.3, 4096 px tiles, 256 px overlap,
per-tile percentile normalization. The whole image is read into memory
and every nucleus whose centroid falls on the slide is kept.

Coordinates are full-resolution pixels of this TIFF.
cx_px is the column (x), cy_px is the row (y), origin at the top-left.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seg_common import extract_instances, grid_starts, max_rss_gb, tile_cores, write_outputs


def mpp_from_tiff(page):
    """µm per pixel from TIFF resolution tags, or None when the tags are absent."""
    try:
        xr = page.tags['XResolution'].value
        unit = int(page.tags['ResolutionUnit'].value)
    except Exception:
        return None
    pixels_per_unit = (xr[0] / xr[1]) if isinstance(xr, tuple) else float(xr)
    if pixels_per_unit <= 0:
        return None
    if unit == 2:      # inch
        return 25400.0 / pixels_per_unit
    if unit == 3:      # centimetre
        return 10000.0 / pixels_per_unit
    return None


def as_rgb(img, axes):
    """Return uint/float array shaped (H, W, 3)."""
    axes = (axes or '').upper()
    img = np.asarray(img)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise SystemExit(f'expected a 2D H&E image, got shape {img.shape} axes={axes!r}')
    if axes == 'CYX' or (not axes and img.shape[0] in (3, 4) and img.shape[1] > 4 and img.shape[2] > 4):
        img = np.moveaxis(img, 0, -1)
    elif axes not in ('', 'YXC', 'YXS'):
        raise SystemExit(f'unsupported TIFF axes {axes!r}, shape {img.shape}')
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise SystemExit(f'H&E image must have 3 channels, got shape {img.shape}')
    return img


def load_tiff(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        img = series.asarray(level=0)
        axes = getattr(series, 'axes', '')
        mpp = mpp_from_tiff(tf.pages[0])
    return as_rgb(img, axes), mpp


def main():
    ap = argparse.ArgumentParser(description='Detect nuclei in one H&E TIFF with StarDist.')
    ap.add_argument('--image', required=True, help='H&E .tif or .tiff')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--overlap', type=int, default=256)
    ap.add_argument('--prob', type=float, default=0.01)
    ap.add_argument('--nms', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import random
    random.seed(a.seed)
    np.random.seed(a.seed)
    import tensorflow as tf
    tf.random.set_seed(a.seed)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    import stardist
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    T0 = time.perf_counter()
    img, mpp = load_tiff(a.image)
    load_s = time.perf_counter() - T0
    H, W = img.shape[:2]
    print(f'image {W} x {H} px, {img.nbytes/1e9:.2f} GB in memory, mpp={mpp}', flush=True)

    step = a.tile - a.overlap
    tiles = tile_cores(grid_starts(W, a.tile, step, 0),
                       grid_starts(H, a.tile, step, 0), W, H, a.tile)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model = StarDist2D(None, name='2D_versatile_he', basedir=model_dir)

    instances, next_id, infer_total, tile_log = [], 1, 0.0, []
    n_rej_sum, n_x3_sum = 0, 0
    for k, t in enumerate(tiles):
        ts = time.perf_counter()
        crop = img[t['y0']:t['y1'], t['x0']:t['x1']].astype('float32')
        x = normalize(crop, 1, 99.8)
        labels, _ = model.predict_instances(x, prob_thresh=a.prob, nms_thresh=a.nms)
        dt = time.perf_counter() - ts
        infer_total += dt
        tile_log.append((k, t['x0'], t['y0'], dt, int(labels.max())))
        ins, next_id, n_rej, n_x3 = extract_instances(labels, t['x0'], t['y0'], t, next_id)
        instances += ins
        n_rej_sum += n_rej
        n_x3_sum += n_x3
        if k % 20 == 0 or k + 1 == len(tiles):
            print(f'tile {k + 1}/{len(tiles)} dt={dt:.1f}s kept={len(instances)}', flush=True)

    section = os.path.splitext(os.path.basename(a.image))[0]
    total_s = time.perf_counter() - T0
    gpu_peak = None
    try:
        gpu_peak = round(tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e9, 2)
    except Exception:
        pass
    write_outputs(a.out_dir, section, instances, mpp, dict(
        tool='stardist', model='2D_versatile_he', input=os.path.abspath(a.image),
        prob_thresh=a.prob, nms_thresh=a.nms, tile=a.tile, overlap=a.overlap,
        normalization='tile', norm_impl='joint_channels,no_clip',
        capture_only=False, full_slide=True,
        n_tiles=len(tiles), min_area_px=0,
        n_rejected_min_area=n_rej_sum, n_poly_excess_gt3=n_x3_sum, seed=a.seed,
        versions=dict(tensorflow=tf.__version__, stardist=stardist.__version__),
        timing_s=dict(total=round(total_s, 1), image_load=round(load_s, 1),
                      inference_sum=round(infer_total, 1)),
        gpu_peak_gb=gpu_peak,
        n_gpu=len(tf.config.list_physical_devices('GPU')),
        coordinate_note='cx_px = column/x, cy_px = row/y, origin top-left, full-resolution TIFF pixels',
    ), img.shape)
    csv_path = os.path.join(a.out_dir, 'nuclei_coordinates.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'cx_px', 'cy_px', 'area_px'])
        for i in instances:
            w.writerow([i['id'], f"{i['cx']:.2f}", f"{i['cy']:.2f}", f"{i['area']:.1f}"])
    with open(os.path.join(a.out_dir, 'tile_times.csv'), 'w') as f:
        f.write('tile,x0,y0,infer_s,max_label\n')
        for r in tile_log:
            f.write('%d,%d,%d,%.2f,%d\n' % r)
    print(f'DONE {len(instances)} nuclei, {total_s/60:.1f} min, RSS {max_rss_gb():.1f} GB', flush=True)
    print(f'coordinates: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
