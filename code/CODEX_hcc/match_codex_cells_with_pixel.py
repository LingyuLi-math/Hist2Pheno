####################################
# 2026.08.12 LLY
####################################
## Analog of code/Xenium_lung/match_HEanno_with_sample_pix.py for CODEX HCC (s4769).
##
## Prebuild the CSV inputs consumed by HCC_train_validate_cv_UNIlabel_single.ipynb:
##   - {MATCHED_HE}_cells_with_pixel.csv
##   - {MATCHED_HE}_cells_matched_by_stardist.csv
##
## StarDist centroids must come from data/CODEX/HCC/StarDist_Segment only
## (do not rebuild from UNI .pth filenames).
##
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#
## All ALIGNED=='Y' regions that have CODEX annotation CSVs:
# conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py
#
## Single HE key:
# conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py \
#   --he-key awy-98938_aligned_0d535a74
#
## Skip StarDist matching (GT cells_with_pixel only):
# conda run -n SeededNTM python code/CODEX_hcc/match_codex_cells_with_pixel.py --no-stardist

"""Build CODEX HCC GT / StarDist cell tables with HE pixel coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
for _p in (_SCRIPT_DIR, _PKG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from s4769_img_cell_mapping import (  # noqa: E402
    DEFAULT_CODEX_HCC_DIR,
    S4769_SAMPLE,
    has_codex_celltype_annotation,
    load_codex_celltype_hierarchy,
    load_codex_he_alignment,
)
from base import load_cell_pixcoords, match_celltype2stardist  # noqa: E402

DEFAULT_STARDIST_ROOT = _REPO_ROOT / "data/CODEX/HCC/StarDist_Segment"
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"

CELLS_WITH_PIXEL_SUFFIX = "_cells_with_pixel.csv"
CELLS_MATCHED_STARDIST_SUFFIX = "_cells_matched_by_stardist.csv"

# Notebook / h5ad column naming (fine / intermediate / coarse).
HCC_COLUMN_RENAME = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
    "final_lineage": "celltype_level1",
}


def sample_he_dir(
    he_key: str,
    *,
    sample: str = S4769_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    return Path(base_dir) / sample / "HE" / he_key


def cells_with_pixel_path(he_dir: Path, he_key: str) -> Path:
    return he_dir / f"{he_key}{CELLS_WITH_PIXEL_SUFFIX}"


def cells_matched_stardist_path(he_dir: Path, he_key: str) -> Path:
    return he_dir / f"{he_key}{CELLS_MATCHED_STARDIST_SUFFIX}"


def stardist_csv_path(stardist_root: Path | str, he_key: str) -> Path:
    """Prepared StarDist table under data/CODEX/HCC/StarDist_Segment/{he_key}/."""
    return Path(stardist_root) / he_key / f"{he_key}{STARDIST_CSV_SUFFIX}"


def build_cells_with_pixel(
    acq_id: str,
    he_key: str,
    *,
    sample: str = S4769_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_HCC_DIR,
    hierarchy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge CODEX cell_data + cell_types and attach L2→L12→L1 hierarchy labels."""
    hierarchy_df = (
        hierarchy_df
        if hierarchy_df is not None
        else load_codex_celltype_hierarchy(sample=sample, base_dir=base_dir)
    )
    hierarchy_map = hierarchy_df.set_index("celltype_level2")

    acq_dir = Path(base_dir) / sample / acq_id
    cell_data_csv = acq_dir / f"{acq_id}.cell_data.csv"
    cell_types_csv = acq_dir / f"{acq_id}.cell_types.csv"
    if not cell_data_csv.is_file():
        raise FileNotFoundError(cell_data_csv)
    if not cell_types_csv.is_file():
        raise FileNotFoundError(cell_types_csv)

    cells = (
        pd.read_csv(cell_data_csv)
        .merge(pd.read_csv(cell_types_csv), on="CELL_ID", how="inner")
        .rename(
            columns={
                "CELL_ID": "cell_id",
                "X": "X_pix_HE",
                "Y": "Y_pix_HE",
                "ANNOTATION_LABEL": "final_CT",
            }
        )
    )
    cells["final_CT"] = cells["final_CT"].astype(str).str.strip()
    cells = cells[cells["final_CT"].isin(hierarchy_map.index)].copy()
    cells["final_sublineage"] = cells["final_CT"].map(hierarchy_map["celltype_level1"])
    cells["final_lineage"] = cells["final_CT"].map(hierarchy_map["celltype_level0"])
    cells["x_centroid"] = cells["X_pix_HE"]
    cells["y_centroid"] = cells["Y_pix_HE"]
    cells["tma"] = he_key
    return cells.reset_index(drop=True)


def build_cells_matched_by_stardist(
    cells_with_pixel: pd.DataFrame | Path | str,
    he_key: str,
    *,
    stardist_root: Path | str = DEFAULT_STARDIST_ROOT,
) -> pd.DataFrame:
    """Match GT hierarchy labels onto prepared StarDist nuclei (StarDist_Segment only)."""
    star_path = stardist_csv_path(stardist_root, he_key)
    if not star_path.is_file():
        raise FileNotFoundError(
            "Prepared StarDist CSV not found (rebuild-from-.pth is disabled). "
            f"Expected: {star_path}"
        )

    if isinstance(cells_with_pixel, (str, Path)):
        celltype_codex = load_cell_pixcoords(
            str(cells_with_pixel),
            column_rename=HCC_COLUMN_RENAME,
            auto_rename=False,
        )
    else:
        tmp = cells_with_pixel.copy()
        celltype_codex = tmp.rename(columns=HCC_COLUMN_RENAME)

    star_raw = pd.read_csv(star_path)
    matched = match_celltype2stardist(
        celltype_codex,
        star_raw,
        celltype_pixel_coords_cols=("X_pix_HE", "Y_pix_HE"),
        stardist_pixel_coords_cols=("centroid_x", "centroid_y"),
    )
    matched = matched.dropna(subset=["centroid_x", "centroid_y"])
    restore = {
        tgt: src
        for src, tgt in HCC_COLUMN_RENAME.items()
        if tgt in matched.columns
    }
    return matched.rename(columns=restore).reset_index(drop=True)


def list_aligned_annotated_regions(
    *,
    sample: str = S4769_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_HCC_DIR,
) -> pd.DataFrame:
    """ALIGNED=='Y' HE↔CODEX rows that also have on-disk CODEX annotation CSVs."""
    alignment = load_codex_he_alignment(sample=sample, base_dir=base_dir)
    rows = []
    for _, row in alignment.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"])
        he_key = str(row["MATCHED_HE"]).strip()
        if not he_key or he_key.lower() == "nan":
            continue
        if not has_codex_celltype_annotation(acq_id, sample=sample, base_dir=base_dir):
            continue
        rows.append({"MATCHED_HE": he_key, "CODEX_ACQUISITION_ID": acq_id})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def process_region(
    acq_id: str,
    he_key: str,
    *,
    sample: str = S4769_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_HCC_DIR,
    stardist_root: Path | str = DEFAULT_STARDIST_ROOT,
    hierarchy_df: pd.DataFrame | None = None,
    write_stardist: bool = True,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Write GT (+ optional StarDist-matched) CSVs under HE/{MATCHED_HE}/."""
    he_dir = sample_he_dir(he_key, sample=sample, base_dir=base_dir)
    he_dir.mkdir(parents=True, exist_ok=True)

    out_gt = cells_with_pixel_path(he_dir, he_key)
    if out_gt.is_file() and not overwrite:
        print(f"  skip existing {out_gt.name}")
        cells = pd.read_csv(out_gt)
    else:
        cells = build_cells_with_pixel(
            acq_id,
            he_key,
            sample=sample,
            base_dir=base_dir,
            hierarchy_df=hierarchy_df,
        )
        cells.to_csv(out_gt, index=False)
        print(
            f"  wrote {out_gt.name} n={len(cells):,} | "
            f"L2={cells['final_CT'].nunique()}, "
            f"L12={cells['final_sublineage'].nunique()}, "
            f"L1={cells['final_lineage'].nunique()}"
        )

    written = {"cells_with_pixel": out_gt}
    if not write_stardist:
        return written

    out_star = cells_matched_stardist_path(he_dir, he_key)
    if out_star.is_file() and not overwrite:
        print(f"  skip existing {out_star.name}")
        written["cells_matched_by_stardist"] = out_star
        return written

    matched = build_cells_matched_by_stardist(
        cells,
        he_key,
        stardist_root=stardist_root,
    )
    matched.to_csv(out_star, index=False)
    print(f"  wrote {out_star.name} n={len(matched):,}")
    written["cells_matched_by_stardist"] = out_star
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prebuild CODEX HCC *_cells_with_pixel.csv and "
            "*_cells_matched_by_stardist.csv for the training notebook."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_CODEX_HCC_DIR,
        help="Michael_data_transfer root (default: repo data/CODEX/HCC/...)",
    )
    parser.add_argument(
        "--stardist-root",
        type=Path,
        default=DEFAULT_STARDIST_ROOT,
        help="Prepared StarDist_Segment root (required CSV source).",
    )
    parser.add_argument(
        "--he-key",
        action="append",
        default=None,
        help="MATCHED_HE key(s); default = all ALIGNED annotated regions.",
    )
    parser.add_argument(
        "--no-stardist",
        action="store_true",
        help="Only write *_cells_with_pixel.csv",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip regions whose output CSV already exists.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    hierarchy_df = load_codex_celltype_hierarchy(base_dir=base_dir)
    print(hierarchy_df.nunique().rename("n_classes").to_string())

    if args.he_key:
        alignment = load_codex_he_alignment(base_dir=base_dir)
        regions = []
        for he_key in args.he_key:
            rows = alignment[alignment["MATCHED_HE"].astype(str) == str(he_key)]
            if rows.empty:
                raise KeyError(f"No CODEX_ACQUISITION_ID for MATCHED_HE={he_key!r}")
            regions.append(
                {
                    "MATCHED_HE": he_key,
                    "CODEX_ACQUISITION_ID": str(rows.iloc[0]["CODEX_ACQUISITION_ID"]),
                }
            )
        region_df = pd.DataFrame(regions)
    else:
        region_df = list_aligned_annotated_regions(base_dir=base_dir)

    print(f"Regions to process: {len(region_df)}")
    for _, row in region_df.iterrows():
        he_key = row["MATCHED_HE"]
        acq_id = row["CODEX_ACQUISITION_ID"]
        print(f"\n[{he_key}] acq={acq_id}")
        process_region(
            acq_id,
            he_key,
            base_dir=base_dir,
            stardist_root=args.stardist_root,
            hierarchy_df=hierarchy_df,
            write_stardist=not args.no_stardist,
            overwrite=not args.no_overwrite,
        )


if __name__ == "__main__":
    main()
