"""2026.11.15 check that StarDist nuclei and Visium spots share one pixel frame.

The abundance cell in ``Pred_visual_npc_uni2_A*.ipynb`` reads
``patient{1,2}/tissue_positions_list.csv`` (Space Ranger full-res pixels) and
overlays them on nuclei from ``HE_images/{sample}.tif``. That is only valid if
the TIFF *is* the full-res image Space Ranger aligned. The NPC README says the
HE slides came without tissue_positions / scalefactors, so this is not given.

Low or negative Spearman for every type (tumor 0.25, fibroblast -0.32,
B -0.06 on patient1) is what a shifted, flipped or rescaled frame looks like.
This script prints the checks that would expose that:

1. hires image size / tissue_hires_scalef vs the TIFF size (should match);
2. nuclei per in-tissue spot vs per out-of-tissue spot (in >> out);
3. same statistic after x/y flips and a transpose (the identity should win);
4. a PNG overlay of spots on the HE thumbnail.

    python code/External_val/NPC/check_visium_alignment.py --sample 20210809-C-AH4199551 --visium patient1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]


def nuclei_per_spot(nuc_xy, spot_xy, radius):
    d, i = cKDTree(spot_xy).query(nuc_xy, k=1)
    n = np.bincount(i[d <= radius], minlength=len(spot_xy))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--visium", required=True, help="patient1 | patient2")
    ap.add_argument("--data", type=Path, default=REPO / "data/Visium/NPC")
    ap.add_argument("--prob-min", type=float, default=0.0)
    args = ap.parse_args(argv)
    vdir = args.data / args.visium
    he = args.data / "HE_images" / f"{args.sample}.tif"
    csv = args.data / "StarDist_Segment" / args.sample / f"{args.sample}_Float_prob0.01_nms_0.3.csv"
    sf = json.loads((vdir / "scalefactors_json.json").read_text())
    pos = pd.read_csv(vdir / "tissue_positions_list.csv", header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"])
    import tifffile
    with tifffile.TiffFile(he) as tif:
        shape = tif.pages[0].shape
    H, W = int(shape[0]), int(shape[1])
    print(f"HE TIFF {he.name}: W={W} H={H}")
    for name in ("tissue_hires_image.png", "tissue_lowres_image.png"):
        img = vdir / "spatial" / name if (vdir / "spatial" / name).is_file() else vdir / name
        key = "tissue_hires_scalef" if "hires" in name else "tissue_lowres_scalef"
        if img.is_file() and key in sf:
            from PIL import Image
            w, h = Image.open(img).size
            print(f"{name}: {w}x{h} / {key}={sf[key]:.6f} -> implied full-res "
                  f"{w / sf[key]:.0f}x{h / sf[key]:.0f}  (TIFF {W}x{H})")
    print(f"spot_diameter_fullres={sf['spot_diameter_fullres']:.2f} px")
    print(f"spot pxl_col range {pos.pxl_col.min():.0f}-{pos.pxl_col.max():.0f} (x, TIFF W={W}); "
          f"pxl_row range {pos.pxl_row.min():.0f}-{pos.pxl_row.max():.0f} (y, TIFF H={H})")

    nuc = pd.read_csv(csv, usecols=["centroid_x", "centroid_y", "probability"])
    nuc = nuc.loc[nuc.probability >= args.prob_min]
    xy = nuc[["centroid_x", "centroid_y"]].to_numpy(float)
    r = float(sf["spot_diameter_fullres"]) / 2
    spot = pos[["pxl_col", "pxl_row"]].to_numpy(float)
    tissue = pos.in_tissue.to_numpy() == 1
    variants = {
        "identity": spot,
        "flip_x": np.c_[W - spot[:, 0], spot[:, 1]],
        "flip_y": np.c_[spot[:, 0], H - spot[:, 1]],
        "transpose": spot[:, ::-1],
    }
    print(f"\nnuclei (prob>={args.prob_min}) = {len(xy):,}")
    print(f"{'frame':<10} {'in-tissue mean':>15} {'out mean':>10} {'in/out':>8} {'in-tissue empty%':>17}")
    for k, s in variants.items():
        n = nuclei_per_spot(xy, s, r)
        a, b = n[tissue].mean(), n[~tissue].mean() if (~tissue).any() else np.nan
        print(f"{k:<10} {a:15.2f} {b:10.2f} {a / max(b, 1e-9):8.1f} {100 * (n[tissue] == 0).mean():16.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        img = tifffile.imread(he)
        step = max(1, max(H, W) // 2000)
        thumb = img[::step, ::step]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.imshow(thumb, extent=(0, W, H, 0))
        ax.scatter(spot[tissue, 0], spot[tissue, 1], s=2, c="lime", alpha=0.5, label="in_tissue")
        ax.scatter(spot[~tissue, 0], spot[~tissue, 1], s=1, c="red", alpha=0.2, label="out")
        ax.legend(loc="upper right")
        ax.set_title(f"{args.sample} spots on HE")
        out = REPO / "code/External_val/NPC" / f"{args.sample}_visium_alignment_check.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"\nwrote {out}")
    except Exception as exc:  # overlay is optional
        print(f"overlay skipped: {exc}")


if __name__ == "__main__":
    main()
