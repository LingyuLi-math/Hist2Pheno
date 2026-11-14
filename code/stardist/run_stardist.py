#!/usr/bin/env python
"""StarDist 2D_versatile_he 全片 H&E 核分割（GPU，分块）。

官方用法（stardist 文档 + 10x Nuclei Segmentation and Custom Binning 教程）：
  normalize(img, 1, 99.8) -> predict_instances(prob_thresh, nms_thresh)
--norm tile              = 教程做法：每 tile 各自 1/99.8 分位数（默认）
--norm global_tissue     = 全片 in_tissue 像素（Space Ranger 组织掩膜）分位数，一次、共用
--norm global_wholeslide = 全片含空白背景的分位数（正式规格 level：粗心用户实际所得）
默认 tile 网格限制在 capture area bbox 内（--full-slide 关闭）。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from seg_common import (capture_bbox, extract_instances, global_percentiles,
                        grid_starts, load_btf, max_rss_gb, read_mpp,
                        restrict_tiles, tile_cores, tissue_area, write_outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--section-dir', required=True)
    ap.add_argument('--section', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--tile', type=int, default=4096)
    ap.add_argument('--overlap', type=int, default=256)
    ap.add_argument('--prob', type=float, default=0.01)
    ap.add_argument('--nms', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grid-offset', type=int, default=0,
                    help='tile 网格平移（px）——确定性推理臂的正确扰动带')
    ap.add_argument('--norm', choices=['tile', 'global_tissue', 'global_wholeslide'],
                    default='tile')
    ap.add_argument('--full-slide', action='store_true',
                    help='默认关闭：tile 网格限制在 capture bbox 内')
    ap.add_argument('--start-tile', type=int, default=0)
    ap.add_argument('--max-tiles', type=int, default=None)
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
    mpp = read_mpp(a.section_dir)
    img, load_s = load_btf(a.image)
    H, W = img.shape[:2]

    bbox = None if a.full_slide else capture_bbox(a.section_dir, mpp=mpp)

    t_norm = time.perf_counter()
    glo = ghi = None
    if a.norm in ('global_tissue', 'global_wholeslide'):
        glo, ghi = global_percentiles(img, a.section_dir, 1.0, 99.8,
                                      tissue_only=(a.norm == 'global_tissue'))
    t_norm = time.perf_counter() - t_norm

    step = a.tile - a.overlap
    tiles = tile_cores(grid_starts(W, a.tile, step, a.grid_offset),
                       grid_starts(H, a.tile, step, a.grid_offset), W, H, a.tile)
    n_tiles_fullslide = len(tiles)
    if bbox:
        tiles = restrict_tiles(tiles, bbox)
    if a.max_tiles:
        tiles = tiles[a.start_tile:a.start_tile + a.max_tiles]
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model = StarDist2D(None, name='2D_versatile_he', basedir=model_dir)
    t_model = time.perf_counter() - T0

    instances, next_id, infer_total, tile_log = [], 1, 0.0, []
    n_rej_sum, n_x3_sum = 0, 0
    for k, t in enumerate(tiles):
        ts = time.perf_counter()
        crop = img[t['y0']:t['y1'], t['x0']:t['x1']].astype('float32')
        if a.norm == 'tile':
            x = normalize(crop, 1, 99.8)
        else:
            x = np.clip((crop - glo) / (ghi - glo), 0, 1)
        labels, _ = model.predict_instances(x, prob_thresh=a.prob, nms_thresh=a.nms)
        dt = time.perf_counter() - ts
        infer_total += dt
        tile_log.append((k, t['x0'], t['y0'], dt, int(labels.max())))
        ins, next_id, n_rej, n_x3 = extract_instances(labels, t['x0'], t['y0'], t, next_id)
        instances += ins
        n_rej_sum += n_rej
        n_x3_sum += n_x3
        if k % 20 == 0:
            print(f'tile {k}/{len(tiles)} dt={dt:.1f}s kept={len(instances)}', flush=True)

    total_s = time.perf_counter() - T0
    gpu_peak = None
    try:
        gpu_peak = round(tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e9, 2)
    except Exception:
        pass
    outdir = os.path.join(a.out_root,
                          a.section if a.norm == 'tile' else f'{a.section}_norm-{a.norm}')
    write_outputs(outdir, a.section, instances, mpp, dict(
        tool='stardist', model='2D_versatile_he',
        prob_thresh=a.prob, nms_thresh=a.nms, tile=a.tile, overlap=a.overlap,
        grid_offset=a.grid_offset,
        normalization=a.norm,
        # review3 披露（未改计算，见报告 needs-decision）：tile 档 = csbdeep normalize(axis=None)
        # 三通道**联合**分位数、不 clip；global 档 = **逐通道**分位数 + clip[0,1] —— 两档
        # 除统计范围外还差这两点，norm 主效应含此混杂。
        norm_impl=('joint_channels,no_clip' if a.norm == 'tile' else 'per_channel,clip01'),
        global_p_lo=[round(float(v), 2) for v in glo] if glo is not None else None,
        global_p_hi=[round(float(v), 2) for v in ghi] if ghi is not None else None,
        capture_only=not a.full_slide, capture_bbox=bbox,
        n_tiles=len(tiles), n_tiles_fullslide=n_tiles_fullslide,
        min_area_px=0,
        n_rejected_min_area=n_rej_sum, n_poly_excess_gt3=n_x3_sum, seed=a.seed,
        checks_applicable={'n_poly_excess_gt3': False,
                           'n_rejected_min_area': False},
        checks_notes='excess>3 对星凸多边形按构造近恒为零（非质量信号）；min_area 生成期=0',
        **tissue_area(a.section_dir),
        versions=dict(tensorflow=tf.__version__, stardist=stardist.__version__),
        timing_s=dict(total=round(total_s, 1), image_load=round(load_s, 1),
                      norm_init=round(t_norm, 1), model_init=round(t_model, 1),
                      inference_sum=round(infer_total, 1)),
        gpu_peak_gb=gpu_peak,
        n_gpu=len(tf.config.list_physical_devices('GPU')),
    ), img.shape)
    with open(os.path.join(outdir, 'tile_times.csv'), 'w') as f:
        f.write('tile,x0,y0,infer_s,max_label\n')
        for r in tile_log:
            f.write('%d,%d,%d,%.2f,%d\n' % r)
    print(f'DONE {a.section} [{a.norm}, prob={a.prob}]: {len(instances)} instances, '
          f'{total_s/60:.1f} min, RSS {max_rss_gb():.1f} GB, GPU peak {gpu_peak} GB', flush=True)


if __name__ == '__main__':
    main()
