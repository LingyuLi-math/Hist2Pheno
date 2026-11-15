#!/usr/bin/env python3
"""StarDist on one GPU for a list of prepared SPORA H&E tifs.

The parent shell sets CUDA_VISIBLE_DEVICES to a single card and launches one
worker per GPU. This file does not clear that variable. The older
``StarDist_nuclei_segmente.py`` forces CPU; do not use it for this cohort.

Each finished tissue gets ``<tissue>_stardist.csv`` with ``centroid_x`` = row
and ``centroid_y`` = column, the same axis order as the GT csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import tifffile

        image = np.asarray(tifffile.imread(path))
    except ImportError:
        from PIL import Image

        image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] != 3:
        raise ValueError(f"{path} expected RGB, got {image.shape}")
    return image


def _load_model():
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained("2D_versatile_he")


def _segment(model, image: np.ndarray, prob: float, nms: float, tilesize: int):
    from csbdeep.utils import normalize

    norm = normalize(image, 1, 99.8, axis=(0, 1, 2))
    height, width = norm.shape[:2]
    if max(height, width) > tilesize:
        overlap = max(32, int(tilesize * 0.1))
        labels, details = model.predict_instances_big(
            norm,
            axes="YXC",
            block_size=tilesize,
            min_overlap=overlap,
            prob_thresh=prob,
            nms_thresh=nms,
        )
    else:
        labels, details = model.predict_instances(
            norm, prob_thresh=prob, nms_thresh=nms
        )
    del labels
    points = np.asarray(details["points"], dtype=np.float64)
    probability = np.asarray(details["prob"], dtype=np.float32)
    if len(points) == 0:
        return pd.DataFrame(columns=["centroid_x", "centroid_y", "probability"])
    return pd.DataFrame(
        {
            "centroid_x": points[:, 0],
            "centroid_y": points[:, 1],
            "probability": probability,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True, help="Text file, one *_he.tif per line.")
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.3)
    parser.add_argument("--tilesize", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"StarDist worker CUDA_VISIBLE_DEVICES={gpu or '(unset)'}", flush=True)
    paths = [Path(line.strip()) for line in args.list.read_text().splitlines() if line.strip()]
    failed = 0
    model = None
    for tif in paths:
        tissue = tif.name.removesuffix("_he.tif")
        out = tif.parent / f"{tissue}_stardist.csv"
        if out.is_file() and not args.force:
            print(f"skip {tissue}", flush=True)
            continue
        try:
            if model is None:
                model = _load_model()
            table = _segment(model, _read_rgb(tif), args.prob, args.nms, args.tilesize)
            table.to_csv(out, index=False)
            print(f"wrote {tissue}: {len(table)} nuclei", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {tissue}: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
