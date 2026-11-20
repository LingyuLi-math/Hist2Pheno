#!/usr/bin/env python3
"""UNI2-h GT tokens for one GPU's share of prepared SPORA tissues.

Same extractor flags as ``code/Hist2Pheno_pkg/run_uni2_patch14.sh``:
patch 14, ``scale_image=False`` (coordinates are already 1 µm pixels),
embeddings.h5ad, no PNG crops. A finished store is skipped.

2026.11.20, ``--target-mpp 0.5``: resize the 1 µm/px H&E by 2 (scale_image=True,
scale = native / target) so the 14 px crop is 7 µm, the same physical size as
every other cohort (it was 14 µm). A finished store with another scale is moved
to ``sc_pth_14_14_<native>mpp_archive_<date>`` (kept, not deleted) and re-extracted.
SPORA 的 14 px 小块统一到 0.5 µm/px（7 µm）；旧特征移到 archive 文件夹保留。

2026.11.20 (2), 0.5 µm/px is now the default for every caller (run_spora_pancancer.sh uni2,
fix-axis, refresh-schurch, uni2-mpp05). The 1 µm/px crops are no longer produced.
默认即 0.5 µm/px；不再生成 1 µm/px 的旧特征。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
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


def _manifest_scale(pth: Path) -> float:
    """2026.11.20, resize factor the finished store was cut at (1.0 when scale_image was False)."""
    try:
        meta = json.loads((pth / "embedding_manifest.json").read_text())
    except (OSError, ValueError):
        return float("nan")
    return float(meta.get("scale", 1.0)) if meta.get("scale_image") else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    # 2026.11.20, physical scale of the crops (None = the original 1 µm/px crops)
    # parser.add_argument("--target-mpp", type=float, default=None,
    #                     help="resize so the 14 px crop is at this µm/px (0.5 = 7 µm, as the other cohorts)")
    # 2026.11.20 (2), default 0.5 µm/px: every cohort's 14 px crop is 7 µm
    parser.add_argument("--target-mpp", type=float, default=0.5,
                        help="resize so the 14 px crop is at this µm/px (default 0.5 = 7 µm, as the other cohorts)")
    parser.add_argument("--native-mpp", type=float, default=1.0, help="µm/px of the prepared SPORA H&E")
    args = parser.parse_args(argv)
    scale = None if args.target_mpp is None else args.native_mpp / args.target_mpp
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
        # if _done(pth):
        #     print(f"skip {tissue}", flush=True)
        #     continue
        # 2026.11.20, a finished store counts only at the requested scale
        if _done(pth) and (scale is None or abs(_manifest_scale(pth) - scale) < 1e-6):
            print(f"skip {tissue}", flush=True)
            continue
        if scale is not None and pth.exists():
            archive = pth.with_name(f"sc_pth_14_14_{args.native_mpp:g}mpp_archive_{time.strftime('%Y%m%d')}")
            if archive.exists():
                archive = archive.with_name(archive.name + time.strftime("_%H%M%S"))
            shutil.move(str(pth), str(archive))
            print(f"archived {pth.name} (scale {_manifest_scale(archive):g}) -> {archive.name}", flush=True)
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(_REPO / "code" / "Image_feature_extraction.py"),
            "--dataset", tissue,
            "--position_path", str(pos),
            "--rawimage_path", str(tif),
            # "--scale_image", "False",
            "--scale_image", "False" if scale is None else "True",  # 2026.11.20
            "--method", "UNI2",
            "--patch_size", "14",
            "--output_img", str(out_root / "sc_pth_14_14_image"),
            "--output_pth", str(pth),
            "--logging", str(out_root) + "/",
            "--embedding-storage", "h5ad",
            "--no_save_patch_images",
        ]
        if scale is not None:  # 2026.11.20, 1 µm/px x 2 -> 0.5 µm/px
            cmd += ["--scale", f"{scale:g}"]
        print(f"uni2 {tissue}" + ("" if scale is None else f" (x{scale:g} -> {args.target_mpp:g} µm/px)"), flush=True)
        if subprocess.call(cmd) != 0:
            failed += 1
            print(f"FAIL {tissue}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
