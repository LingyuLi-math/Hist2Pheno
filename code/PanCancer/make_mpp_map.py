"""2026.11.15 Build data/PanCancer/mpp_map.csv (section_id, mpp_um) for A10.

``spatial_HE`` is in native H&E pixels of each section (the UNI2 .pth names
are matched on it). A10 converts it to µm per section, so the 50 µm
neighbour cap and the density geometry mean the same thing in every cohort.

Native µm/px per cohort comes from ``run_uni2_patch14.sh``: there
``SCALE = native_mpp / 0.5`` (resize onto the 0.5 µm/px UNI2 grid), so
native_mpp = 0.5 * SCALE. CODEX cohorts use SCALE=False (H&E at 0.5 µm/px).

    python code/PanCancer/make_mpp_map.py                         # defaults below
    python code/PanCancer/make_mpp_map.py --set BRCA=0.4200       # override one cohort

[需核实] Before training A10, check one section per cohort, e.g. the OME
PhysicalSizeX of the H&E TIFF, or the median StarDist nucleus area in µm².
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MPP = {
    "Lung": 0.5 * 0.425,   # 0.2125 µm/px Xenium H&E
    "BRCA": 0.5 * 0.84,    # 0.42
    "CRC": 0.5 * 0.548,    # 0.274
    "HCC": 0.5,
    "PDAC": 0.5,
    "GIST": 0.5,
    "GBM": 0.5,
}

# SPORA sections share the CRC / GBM names but not the Xenium CRC pixel size.
# pan_organ wins over dataset. The prepared TIFF is the 1 µm OME-Zarr
# (he/1_0mpp), and spatial_HE is on that grid, so A10 uses 1.0 µm/px.
PAN_ORGAN_MPP = {
    "codex_crc_spora": 1.0,
    "codex_glioma_spora": 1.0,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=REPO / "data/PanCancer/section_index.csv")
    ap.add_argument("--out", type=Path, default=REPO / "data/PanCancer/mpp_map.csv")
    ap.add_argument("--set", nargs="*", default=[], help="COHORT=mpp overrides")
    ap.add_argument("--set-organ", nargs="*", default=[], help="pan_organ=mpp overrides (e.g. codex_crc_spora=0.377)")
    args = ap.parse_args(argv)
    mpp = dict(DEFAULT_MPP)
    for item in args.set:
        k, v = item.split("=", 1)
        mpp[k] = float(v)
    index = pd.read_csv(args.index)
    if "section_id" not in index:
        raise ValueError(f"{args.index} has no section_id column")
    cohort = index["dataset"] if "dataset" in index else index.section_id.str.split("::").str[0]
    out = pd.DataFrame({"section_id": index.section_id.astype(str), "cohort": cohort.astype(str)})
    out["pan_organ"] = index["pan_organ"].astype(str) if "pan_organ" in index else ""
    out = out.drop_duplicates("section_id")
    organ_mpp = dict(PAN_ORGAN_MPP)
    for item in args.set_organ:
        k, v = item.split("=", 1)
        organ_mpp[k] = float(v)
    out["mpp_um"] = out.pan_organ.map(organ_mpp).fillna(out.cohort.map(mpp))
    out.loc[out.pan_organ.isin(organ_mpp), "cohort"] = out.cohort + "/" + out.pan_organ
    missing = sorted(out.loc[out.mpp_um.isna(), "cohort"].unique())
    if missing:
        # Not one of the 7 training cohorts; training raises if it ever needs them.
        print(f"WARNING: skipped cohorts without µm/px {missing}; pass --set COHORT=mpp to keep them")
        out = out.loc[out.mpp_um.notna()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[["section_id", "mpp_um"]].to_csv(args.out, index=False)
    print(out.groupby("cohort").mpp_um.agg(["first", "size"]).rename(columns={"first": "mpp_um", "size": "n_sections"}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
