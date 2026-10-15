#!/usr/bin/env python3
"""Convert large HE JPEGs to pyramidal tiled BigTIFF for QuPath.

QuPath / Java ImageIO often cannot open single-plane JPEG with side length
> 32767 (these pathology HE are ~48k–52k). Write a JPEG-compressed tile
pyramid next to each source as ``{stem}_qupath.tif``.

Usage (from repo root)::

    conda run --no-capture-output -n SeededNTM python -u \\
      code/External_val/BRCA/jpg_to_qupath_pyramid_tif.py

    # optional:
    python -u code/External_val/BRCA/jpg_to_qupath_pyramid_tif.py \\
      --he-dir data/Visium/BRCA/HE_images --force
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import TiffWriter

Image.MAX_IMAGE_PIXELS = None

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_HE = _REPO / "data" / "Visium" / "BRCA" / "HE_images"
_TILE = 512


def n_pyramid_levels(height: int, width: int, tile: int = _TILE) -> int:
    side = min(height, width)
    if side <= tile:
        return 0
    return max(1, int(math.floor(math.log2(side / tile))))


def convert_one(src: Path, dst: Path, *, tile: int = _TILE, force: bool = False) -> Path:
    if dst.is_file() and dst.stat().st_size > 1_000_000 and not force:
        print(f"skip (exists): {dst.name}", flush=True)
        return dst

    print(f"convert {src.name} ...", flush=True)
    with Image.open(src) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]
    nlevels = n_pyramid_levels(height, width, tile)
    opts = dict(
        photometric="rgb",
        planarconfig="contig",
        tile=(tile, tile),
        compression="jpeg",
    )
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if tmp.is_file():
        tmp.unlink()
    with TiffWriter(tmp, bigtiff=True) as tif:
        tif.write(arr, subifds=nlevels, **opts)
        for level in range(nlevels):
            mag = 2 ** (level + 1)
            tif.write(arr[::mag, ::mag], subfiletype=1, **opts)
    tmp.replace(dst)
    print(
        f"  → {dst.name}  ({dst.stat().st_size / 1e9:.2f} GB, "
        f"{width}×{height}, levels={nlevels + 1})",
        flush=True,
    )
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--he-dir",
        type=Path,
        default=_DEFAULT_HE,
        help=f"HE_images directory (default: {_DEFAULT_HE})",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob under he-dir (default: *.jpg)",
    )
    parser.add_argument("--tile", type=int, default=_TILE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing *_qupath.tif",
    )
    args = parser.parse_args(argv)

    he_dir = args.he_dir.resolve()
    if not he_dir.is_dir():
        print(f"ERROR: HE dir not found: {he_dir}", file=sys.stderr)
        return 1

    sources = sorted(he_dir.glob(args.pattern))
    if not sources:
        print(f"ERROR: no files matching {args.pattern!r} in {he_dir}", file=sys.stderr)
        return 1

    print(f"HE dir: {he_dir}")
    print(f"inputs: {len(sources)}  tile={args.tile}  force={args.force}")
    for src in sources:
        dst = src.with_name(src.stem + "_qupath.tif")
        convert_one(src, dst, tile=args.tile, force=args.force)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
