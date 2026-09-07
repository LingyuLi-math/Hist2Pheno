###########################################################
# 2026.09.03, CODEX GBM (GD label)
# 2026.09.07, for gbm, add function to build nuclei-level annotation table: cell type + primary SN (+ Rec GD)
###########################################################
#!/usr/bin/env python3
"""Build a nucleus-level annotation table: cell type + primary SN (+ Rec GD).

One row per nucleus. Spatial niche comes from the 16 µm bin with the largest
``intersect_ratio``. Rec GD/nonGD labels are 8 µm barcodes; they are mapped to
a 16 µm parent (row//2, col//2) and attached to that primary bin.

    conda activate SeededNTM
    python -u code/CODEX_gbm/build_nuclei_sn_gd_annotation.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gbm_paths import (  # noqa: E402
    DEFAULT_GD_LABEL_TXT,
    DEFAULT_NUCLEI_SN_GD_XLSX,
    annotation_xlsx_path,
    bin_nuclei_mapping_xlsx_path,
    normalize_gbm_cell_type,
    sample_ids,
)

_UM8_RE = re.compile(r"^s_008um_(\d+)_(\d+)-(\d+)$")
ANNO_USECOLS = ["cellID", "cell_type", "subcluster"]
MAP_USECOLS = [
    "cell_id",
    "index",
    "SN",
    "intersect_ratio",
    "cell_area",
    "cell_type",
    "subcluster",
]


def _fill_subcluster(value: object) -> str:
    if pd.isna(value) or str(value).strip() in ("", "nan"):
        return "LowQ"
    return str(value).strip()


def um8_to_um16(barcode: object) -> str | None:
    """Visium HD 2×2 aggregate: 16 µm (i, j) covers 8 µm rows [2i, 2i+1]."""
    match = _UM8_RE.match(str(barcode).strip())
    if not match:
        return None
    row, col, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
    return f"s_016um_{row // 2:05d}_{col // 2:05d}-{suffix}"


def load_nuclei(sample: str) -> pd.DataFrame:
    anno = pd.read_excel(annotation_xlsx_path(sample), usecols=ANNO_USECOLS)
    out = pd.DataFrame(
        {
            "cell_id": anno["cellID"].astype(str).str.strip(),
            "cell_type_raw": anno["cell_type"],
            "cell_type": anno["cell_type"].map(normalize_gbm_cell_type),
            "subcluster": anno["subcluster"].map(_fill_subcluster),
            "sample": sample,
        }
    )
    if out["cell_id"].duplicated().any():
        raise ValueError(f"{sample} nuclei cellID is not unique")
    return out


def load_primary_sn(sample: str) -> pd.DataFrame:
    mapping = pd.read_excel(bin_nuclei_mapping_xlsx_path(sample), usecols=MAP_USECOLS)
    mapping["cell_id"] = mapping["cell_id"].astype(str).str.strip()
    n_rows, n_cells = len(mapping), mapping["cell_id"].nunique()
    primary = (
        mapping.sort_values("intersect_ratio", ascending=False)
        .drop_duplicates("cell_id", keep="first")
        .rename(
            columns={
                "index": "bin_barcode",
                "SN": "spatial_niche",
                "intersect_ratio": "sn_intersect_ratio",
                "cell_area": "sn_cell_area",
                "cell_type": "mapping_cell_type",
                "subcluster": "mapping_subcluster",
            }
        )
    )
    print(
        f"  {sample} mapping: {n_rows:,} overlaps → {n_cells:,} nuclei "
        f"with a primary 16 µm bin"
    )
    return primary[
        [
            "cell_id",
            "bin_barcode",
            "spatial_niche",
            "sn_intersect_ratio",
            "sn_cell_area",
            "mapping_cell_type",
            "mapping_subcluster",
        ]
    ]


def load_gd_by_um16(txt_path: Path) -> pd.DataFrame:
    gd = pd.read_csv(txt_path, sep="\t")
    if list(gd.columns)[:2] != ["Barcode", "Label"]:
        raise KeyError(f"{txt_path.name} needs Barcode, Label; got {list(gd.columns)}")
    gd = gd.copy()
    gd["bin16"] = gd["Barcode"].map(um8_to_um16)
    missing = gd["bin16"].isna().sum()
    if missing:
        raise ValueError(f"{missing} GD barcodes were not s_008um_*")
    agg = (
        gd.groupby("bin16")
        .agg(
            GD_n_8um=("Label", "size"),
            GD_n_GD=("Label", lambda s: int((s == "GD").sum())),
            GD_n_nonGD=("Label", lambda s: int((s == "nonGD").sum())),
        )
        .reset_index()
        .rename(columns={"bin16": "bin_barcode"})
    )
    # Any labeled 8 µm child that is GD → GD; else nonGD.
    agg["GD_label"] = agg["GD_n_GD"].gt(0).map({True: "GD", False: "nonGD"})
    print(
        f"  GD txt: {len(gd):,} 8 µm bins → {len(agg):,} 16 µm parents "
        f"(GD parents={(agg['GD_label']=='GD').sum():,})"
    )
    return agg


def build_sample(sample: str, gd16: pd.DataFrame | None) -> pd.DataFrame:
    print(f"\n{sample}")
    nuclei = load_nuclei(sample)
    primary = load_primary_sn(sample)
    out = nuclei.merge(primary, on="cell_id", how="left")
    if gd16 is not None:
        out = out.merge(gd16, on="bin_barcode", how="left")
    else:
        out["GD_label"] = pd.NA
        out["GD_n_8um"] = pd.NA
        out["GD_n_GD"] = pd.NA
        out["GD_n_nonGD"] = pd.NA
    n_sn = out["spatial_niche"].notna().sum()
    n_gd = out["GD_label"].notna().sum() if "GD_label" in out.columns else 0
    print(f"  nuclei={len(out):,}  primary SN={n_sn:,}  GD attached={n_gd:,}")
    if n_gd:
        print(f"  GD_label: {out['GD_label'].value_counts(dropna=False).to_dict()}")
    cols = [
        "sample",
        "cell_id",
        "cell_type",
        "cell_type_raw",
        "subcluster",
        "spatial_niche",
        "bin_barcode",
        "sn_intersect_ratio",
        "sn_cell_area",
        "GD_label",
        "GD_n_8um",
        "GD_n_GD",
        "GD_n_nonGD",
        "mapping_cell_type",
        "mapping_subcluster",
    ]
    return out[cols]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gd-txt", type=Path, default=DEFAULT_GD_LABEL_TXT)
    p.add_argument("--out", type=Path, default=DEFAULT_NUCLEI_SN_GD_XLSX)
    args = p.parse_args()

    gd16 = load_gd_by_um16(args.gd_txt) if args.gd_txt.is_file() else None
    if gd16 is None:
        print(f"No GD txt at {args.gd_txt}; Rec GD_label will be empty")

    frames = {}
    for sample in sample_ids():
        attach_gd = gd16 if "Recurrent" in sample else None
        frames[sample] = build_sample(sample, attach_gd)

    rules = pd.DataFrame(
        {
            "rule": [
                "One row per nucleus (cell_id from single-nuclei cellID).",
                "cell_type / subcluster ground truth = 2_Single_Nuclei mapping xlsx.",
                "Primary 16 µm bin = max intersect_ratio row in 3_Mapping_bin_nuclei.",
                "spatial_niche = SN of that primary bin (not 1_Bin SpatialNiches directly).",
                "GD txt is Rec-only 8 µm barcodes; parent 16 µm = s_016um_{row//2}_{col//2}.",
                "GD_label on a nucleus = GD if any labeled 8 µm child of the primary 16 µm bin is GD.",
                "Ini has no GD file; GD_* columns are empty.",
                "Hist2Pheno cell-type heads still train on subcluster / cell_type; SN and GD are extra labels.",
            ]
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        frames["P174511_Initial"].to_excel(writer, sheet_name="Ini", index=False)
        frames["P179161_Recurrent"].to_excel(writer, sheet_name="Rec", index=False)
        rules.to_excel(writer, sheet_name="join_rules", index=False)
    print(f"\nWrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
