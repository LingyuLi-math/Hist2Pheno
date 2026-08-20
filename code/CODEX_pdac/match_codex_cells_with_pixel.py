####################################
# 2026.08.20 LLY
####################################
## Analog of code/CODEX_hcc/match_codex_cells_with_pixel.py for CODEX PDAC (s1167).
##
## Sample key is ACQUISITION_ID (no MATCHED_HE). Outputs live under
##   s1167/{ACQUISITION_ID}/
##     {ACQUISITION_ID}_cells_with_pixel.csv
##     {ACQUISITION_ID}_cells_matched_by_stardist.csv
##
## Default: 278 Pancreas TMA cores with cell-type CSV (c001 all, c003 130/149).
## Unannotated PDAC cores (195: 19 in c003 + all of c005/c007) are Incomplete_Cases
## and are not processed here.
##
# cd /home/lingyu/ssd2/Python/Hist2Pheno
#
# conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py
#
# conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py \
#   --acq-id Charvill-94_c001_v001_r001_reg001
#
# conda run -n SeededNTM python code/CODEX_pdac/match_codex_cells_with_pixel.py --no-stardist

"""Build CODEX PDAC GT / StarDist cell tables with HE pixel coordinates."""

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

from s1167_img_cell_mapping import (  # noqa: E402
    DEFAULT_CODEX_DIR,
    DEFAULT_STARDIST_RESULT_DIR,
    S1167_SAMPLE,
    acq_dir,
    cell_data_path,
    cell_types_path,
    has_codex_celltype_annotation,
    load_s1167_celltype_hierarchy,
    load_s1167_metadata,
)
from base import load_cell_pixcoords, match_celltype2stardist  # noqa: E402

DEFAULT_STARDIST_ROOT = DEFAULT_STARDIST_RESULT_DIR
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"

CELLS_WITH_PIXEL_SUFFIX = "_cells_with_pixel.csv"
CELLS_MATCHED_STARDIST_SUFFIX = "_cells_matched_by_stardist.csv"

# Notebook / h5ad column naming (fine / intermediate / coarse), same as HCC.
PDAC_COLUMN_RENAME = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
    "final_lineage": "celltype_level1",
}
HCC_COLUMN_RENAME = PDAC_COLUMN_RENAME


def sample_he_dir(
    acq_id: str,
    *,
    sample: str = S1167_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_DIR,
) -> Path:
    """PDAC analog of HCC ``HE/{MATCHED_HE}/``: the acquisition folder itself."""
    return acq_dir(acq_id, sample=sample, base_dir=base_dir)


def cells_with_pixel_path(he_dir: Path, acq_id: str) -> Path:
    return he_dir / f"{acq_id}{CELLS_WITH_PIXEL_SUFFIX}"


def cells_matched_stardist_path(he_dir: Path, acq_id: str) -> Path:
    return he_dir / f"{acq_id}{CELLS_MATCHED_STARDIST_SUFFIX}"


def stardist_csv_path(stardist_root: Path | str, acq_id: str) -> Path:
    """Prepared StarDist table under StarDist_Segment_pdac/pdac_result/{acq}/."""
    return Path(stardist_root) / acq_id / f"{acq_id}{STARDIST_CSV_SUFFIX}"


def build_cells_with_pixel(
    acq_id: str,
    he_key: str | None = None,
    *,
    sample: str = S1167_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_DIR,
    hierarchy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge CODEX cell_data + cell_types and attach L2→L12→L1 hierarchy labels."""
    he_key = he_key or acq_id
    hierarchy_df = (
        hierarchy_df
        if hierarchy_df is not None
        else load_s1167_celltype_hierarchy(sample=sample, base_dir=base_dir)
    )
    hierarchy_map = hierarchy_df.set_index("celltype_level2")

    cell_data_csv = cell_data_path(acq_id, sample=sample, base_dir=base_dir)
    cell_types_csv = cell_types_path(acq_id, sample=sample, base_dir=base_dir)
    if not cell_data_csv.is_file():
        raise FileNotFoundError(cell_data_csv)
    if cell_types_csv is None:
        raise FileNotFoundError(
            acq_dir(acq_id, sample=sample, base_dir=base_dir) / f"{acq_id}.cell_types.csv"
        )

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
    """Match GT hierarchy labels onto prepared StarDist nuclei."""
    star_path = stardist_csv_path(stardist_root, he_key)
    if not star_path.is_file():
        raise FileNotFoundError(
            "Prepared StarDist CSV not found. "
            f"Expected: {star_path}"
        )

    if isinstance(cells_with_pixel, (str, Path)):
        celltype_codex = load_cell_pixcoords(
            str(cells_with_pixel),
            column_rename=PDAC_COLUMN_RENAME,
            auto_rename=False,
        )
    else:
        tmp = cells_with_pixel.copy()
        celltype_codex = tmp.rename(columns=PDAC_COLUMN_RENAME)

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
        for src, tgt in PDAC_COLUMN_RENAME.items()
        if tgt in matched.columns
    }
    return matched.rename(columns=restore).reset_index(drop=True)


def list_aligned_annotated_regions(
    *,
    sample: str = S1167_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_DIR,
) -> pd.DataFrame:
    """Pancreas TMA cores that have on-disk CODEX cell-type CSVs (278).

    HCC analog of ``ALIGNED=='Y'`` + ``Annotation=='Y'``. ``MATCHED_HE`` is an
    alias of ``ACQUISITION_ID`` so the training CLI can share path helpers.
    """
    meta = load_s1167_metadata(
        sample=sample,
        base_dir=base_dir,
        cohort="PDAC",
        annotated_only=True,
        with_he_only=True,
    )
    rows = []
    for aid in meta["ACQUISITION_ID"].astype(str).str.strip():
        if not has_codex_celltype_annotation(aid, sample=sample, base_dir=base_dir):
            continue
        rows.append({"MATCHED_HE": aid, "ACQUISITION_ID": aid, "CODEX_ACQUISITION_ID": aid})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def list_incomplete_pdac_regions(
    *,
    sample: str = S1167_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_DIR,
) -> pd.DataFrame:
    """Pancreas TMA cores without cell-type CSV (195 Incomplete_Cases)."""
    meta = load_s1167_metadata(
        sample=sample,
        base_dir=base_dir,
        cohort="PDAC",
        annotated_only=False,
        with_he_only=True,
    )
    rows = []
    for aid in meta["ACQUISITION_ID"].astype(str).str.strip():
        if has_codex_celltype_annotation(aid, sample=sample, base_dir=base_dir):
            continue
        rows.append({"MATCHED_HE": aid, "ACQUISITION_ID": aid, "CODEX_ACQUISITION_ID": aid})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def process_region(
    acq_id: str,
    he_key: str | None = None,
    *,
    sample: str = S1167_SAMPLE,
    base_dir: Path | str = DEFAULT_CODEX_DIR,
    stardist_root: Path | str = DEFAULT_STARDIST_ROOT,
    hierarchy_df: pd.DataFrame | None = None,
    write_stardist: bool = True,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Write GT (+ optional StarDist-matched) CSVs under ``s1167/{ACQUISITION_ID}/``."""
    he_key = he_key or acq_id
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
            "Prebuild CODEX PDAC *_cells_with_pixel.csv and "
            "*_cells_matched_by_stardist.csv for the 278 annotated cores."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_CODEX_DIR,
        help="Michael_data_transfer root (default: repo data/CODEX/HCC/...)",
    )
    parser.add_argument(
        "--stardist-root",
        type=Path,
        default=DEFAULT_STARDIST_ROOT,
        help="Prepared StarDist pdac_result root.",
    )
    parser.add_argument(
        "--acq-id",
        action="append",
        default=None,
        help="ACQUISITION_ID(s); default = all 278 annotated PDAC cores.",
    )
    parser.add_argument(
        "--he-key",
        action="append",
        default=None,
        help="Alias of --acq-id (HCC CLI compatibility).",
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
    hierarchy_df = load_s1167_celltype_hierarchy(base_dir=base_dir)
    print(hierarchy_df.nunique().rename("n_classes").to_string())

    requested = list(args.acq_id or []) + list(args.he_key or [])
    if requested:
        region_df = pd.DataFrame(
            {
                "MATCHED_HE": requested,
                "ACQUISITION_ID": requested,
                "CODEX_ACQUISITION_ID": requested,
            }
        )
    else:
        region_df = list_aligned_annotated_regions(base_dir=base_dir)

    print(f"Regions to process: {len(region_df)}")
    for _, row in region_df.iterrows():
        acq_id = str(row["ACQUISITION_ID"])
        print(f"\n[{acq_id}]")
        process_region(
            acq_id,
            acq_id,
            base_dir=base_dir,
            stardist_root=args.stardist_root,
            hierarchy_df=hierarchy_df,
            write_stardist=not args.no_stardist,
            overwrite=not args.no_overwrite,
        )


if __name__ == "__main__":
    main()
