"""2026.11.15 Macenko stain normalisation for H&E slides (Hist2Prot preprocessing).

Hist2Prot normalises stain before segmentation and feature extraction; in its
ablation, removing stain normalisation lowered PCC by 0.022–0.027 (HNSC,
LUAD). Hist2Pheno currently feeds raw RGB to UNI2. Independent
implementation of Macenko et al. (ISBI 2009); no Hist2Prot code is reused.

Two uses:
  1. Quick test on one external slide (e.g. NPC): normalise it to a training
     reference, re-extract UNI2 on the normalised TIFF, predict with the same
     checkpoint, and compare B / lymphoid fractions with the raw run.
  2. Proper use: normalise every training and external slide to one
     reference, then re-extract UNI2 and retrain (needed for a fair model).

    # fit a reference from a few training tiles (PNG/TIFF crops, tissue-rich)
    python code/stain_normalize.py fit --images ref1.png ref2.png --out data/stain_ref_macenko.json
    # normalise a slide (tiled, BigTIFF out; resolution tags copied)
    python code/stain_normalize.py apply --ref data/stain_ref_macenko.json \
        --slide data/Visium/NPC/HE_images/20210809-C-AH4199551.tif \
        --out   data/Visium/NPC/HE_images_macenko/20210809-C-AH4199551.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IO = 240.0          # background intensity
BETA = 0.15         # OD threshold for tissue pixels
ALPHA = 1.0         # percentile for robust extreme angles


def rgb_to_od(rgb):
    return -np.log((np.asarray(rgb, np.float64) + 1.0) / IO)


def estimate_stain(rgb, max_pixels=2_000_000, seed=0):
    """Return (H&E stain matrix 3×2, 99th-percentile concentrations 2)."""
    od = rgb_to_od(rgb).reshape(-1, 3)
    od = od[(od > BETA).all(1)]
    if len(od) < 100:
        raise ValueError("too few tissue pixels to estimate stain vectors")
    if len(od) > max_pixels:
        od = od[np.random.default_rng(seed).choice(len(od), max_pixels, replace=False)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    plane = vecs[:, 1:3]                                     # two largest
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [ALPHA, 100 - ALPHA])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    H, E = (v1, v2) if v1[0] > v2[0] else (v2, v1)          # haematoxylin has larger red OD
    stains = np.stack([H, E], 1)
    stains /= np.linalg.norm(stains, axis=0, keepdims=True)
    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    return stains, np.percentile(conc, 99, axis=1)


def normalise(rgb, src_stains, src_max, ref_stains, ref_max):
    shape = rgb.shape
    od = rgb_to_od(rgb).reshape(-1, 3).T
    conc = np.linalg.lstsq(src_stains, od, rcond=None)[0]
    conc *= (ref_max / np.maximum(src_max, 1e-6))[:, None]
    out = IO * np.exp(-(ref_stains @ conc))
    return np.clip(out.T.reshape(shape), 0, 255).astype(np.uint8)


def _read_rgb(path):
    import tifffile
    from PIL import Image
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff", ".btf", ".svs"):
        arr = tifffile.imread(p, level=0) if p.suffix.lower() == ".svs" else tifffile.imread(p)
    else:
        arr = np.asarray(Image.open(p).convert("RGB"))
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr[..., :3]).astype(np.uint8)


def cmd_fit(args):
    rgbs = [_read_rgb(p) for p in args.images]
    stacked = np.concatenate([r.reshape(-1, 3) for r in rgbs])[None]
    stains, mx = estimate_stain(stacked)
    args.out.write_text(json.dumps({"method": "macenko", "stains": stains.tolist(), "max_conc": mx.tolist(),
                                    "images": [str(p) for p in args.images]}, indent=2))
    print(f"reference stains\n{np.round(stains, 4)}\nmax conc {np.round(mx, 4)} -> {args.out}")


def cmd_apply(args):
    import tifffile
    ref = json.loads(args.ref.read_text())
    ref_s, ref_m = np.asarray(ref["stains"]), np.asarray(ref["max_conc"])
    rgb = _read_rgb(args.slide)
    # source stains from a down-sampled view (whole slide, robust and fast)
    step = max(1, int(np.ceil(np.sqrt(rgb.shape[0] * rgb.shape[1] / 4e6))))
    src_s, src_m = estimate_stain(rgb[::step, ::step])
    out = np.empty_like(rgb)
    T = int(args.tile)
    for y in range(0, rgb.shape[0], T):
        for x in range(0, rgb.shape[1], T):
            tile = rgb[y:y + T, x:x + T]
            bg = (tile.astype(np.int16).min(-1) > 220)
            norm_tile = normalise(tile, src_s, src_m, ref_s, ref_m)
            norm_tile[bg] = tile[bg]                       # keep white background untouched
            out[y:y + T, x:x + T] = norm_tile
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    with tifffile.TiffFile(args.slide) as tf:
        page = tf.pages[0]
        for tag in ("XResolution", "YResolution", "ResolutionUnit"):
            if tag in page.tags:
                kw.setdefault("_tags", {})[tag] = page.tags[tag].value
    res = kw.get("_tags", {})
    resolution = (res["XResolution"], res["YResolution"]) if "XResolution" in res else None
    unit = res.get("ResolutionUnit")
    tifffile.imwrite(args.out, out, bigtiff=True, tile=(512, 512), compression="zlib",
                     photometric="rgb", resolution=resolution,
                     resolutionunit=unit if unit is not None else None)
    print(f"source stains\n{np.round(src_s, 4)}\nmax conc {np.round(src_m, 4)}\nwrote {args.out}")
    print("NOTE: OME-XML PhysicalSize is not copied; pass the µm/px explicitly when extracting features.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--images", nargs="+", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True); f.set_defaults(fn=cmd_fit)
    a = sub.add_parser("apply"); a.add_argument("--ref", type=Path, required=True)
    a.add_argument("--slide", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    a.add_argument("--tile", type=int, default=2048); a.set_defaults(fn=cmd_apply)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
