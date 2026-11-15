#!/usr/bin/env python3
"""UNI2-h GT tokens for one GPU's share of prepared SPORA tissues.

Same extractor flags as ``code/Hist2Pheno_pkg/run_uni2_patch14.sh``:
patch 14, ``scale_image=False`` (coordinates are already 1 µm pixels),
embeddings.h5ad, no PNG crops. A finished store is skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _done(pth: Path) -> bool:
    if not pth.is_dir():
        return False
    probe = subprocess.run(
        [sys.executable, str(_REPO / "code" / "check_embeddings.py"), str(pth)],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    args = parser.parse_args(argv)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"UNI2 worker CUDA_VISIBLE_DEVICES={gpu or '(unset)'}", flush=True)
    failed = 0
    for line in args.list.read_text().splitlines():
        tif = Path(line.strip())
        if not tif.name:
            continue
        tissue = tif.name.removesuffix("_he.tif")
        pos = tif.parent / f"{tissue}_cells_with_pixel.csv"
        out_root = tif.parent / f"{tissue}_project_all_UNI2" / "ImgEmbeddings_all"
        pth = out_root / "sc_pth_14_14"
        if _done(pth):
            print(f"skip {tissue}", flush=True)
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(_REPO / "code" / "Image_feature_extraction.py"),
            "--dataset", tissue,
            "--position_path", str(pos),
            "--rawimage_path", str(tif),
            "--scale_image", "False",
            "--method", "UNI2",
            "--patch_size", "14",
            "--output_img", str(out_root / "sc_pth_14_14_image"),
            "--output_pth", str(pth),
            "--logging", str(out_root) + "/",
            "--embedding-storage", "h5ad",
            "--no_save_patch_images",
        ]
        print(f"uni2 {tissue}", flush=True)
        if subprocess.call(cmd) != 0:
            failed += 1
            print(f"FAIL {tissue}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
