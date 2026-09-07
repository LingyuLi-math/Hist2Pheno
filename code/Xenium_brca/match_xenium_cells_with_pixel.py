"""Build Xenium BRCA GT / StarDist cell tables with HE pixel coordinates.

Analog of ``code/CODEX_hcc/match_codex_cells_with_pixel.py``.

Writes under ``data/Xemium/BRCA/Cases/{rep1,rep2}/``:

- ``{sample}_cells_with_pixel.csv``
- ``{sample}_cells_matched_by_stardist.csv``

  python code/Xenium_brca/match_xenium_cells_with_pixel.py
  python code/Xenium_brca/match_xenium_cells_with_pixel.py --sample rep1
"""

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

from base import load_cell_pixcoords, match_celltype2stardist  # noqa: E402
from brca_paths import (  # noqa: E402
    BRCA_COLUMN_RENAME,
    DEFAULT_CASES_ROOT,
    DEFAULT_HIERARCHY_XLSX,
    DEFAULT_STARDIST_ROOT,
    he_tif_path,
    load_brca_celltype_hierarchy,
    sample_config,
    sample_dir,
    sample_ids,
    stardist_csv_path as brca_stardist_csv_path,
)

HCC_COLUMN_RENAME = BRCA_COLUMN_RENAME


def cells_with_pixel_path(he_dir: Path | str, he_key: str) -> Path:
    """HCC-compatible signature: ``(sample_dir, sample)``."""
    return Path(he_dir) / f"{he_key}_cells_with_pixel.csv"


def cells_matched_stardist_path(he_dir: Path | str, he_key: str) -> Path:
    return Path(he_dir) / f"{he_key}_cells_matched_by_stardist.csv"


def stardist_csv_path(stardist_root: Path | str, he_key: str) -> Path:
    """HCC-compatible signature: ``(stardist_root, sample)``."""
    return brca_stardist_csv_path(he_key, stardist_root)


def list_aligned_annotated_regions() -> pd.DataFrame:
    """rep1/rep2 as MATCHED_HE keys (HCC train-script compatible)."""
    ids = sample_ids()
    return pd.DataFrame({"MATCHED_HE": ids, "CODEX_ACQUISITION_ID": ids})

###########################################################
# 2026.09.03, Xenium BRCA (LY celltype sheet)
# 2026.09.04, for brca, add function to build cells with pixel coordinates
###########################################################
def build_cells_with_pixel(
    sample: str,
    *,
    hierarchy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join supervised Cluster onto the LY hierarchy.

    Drops ``Unlabeled`` and red-excluded L2 types
    (``T_Cell_&_Tumor_Hybrid``, ``Stromal_&_T_Cell_Hybrid``,
    ``Perivascular-Like``) via ``load_brca_celltype_hierarchy``.

    Expects ``X_pix_HE`` / ``Y_pix_HE`` already in working HE ``*.tif`` pixels
    (Explorer ``he_imagealignment`` + OME→tif scale from preprocess).
    """
    cfg = sample_config(sample)
    src = Path(cfg["cell_info_csv"])
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed cell table: {src}\n"
            "Run Data_process_HEcelltype_BRCA.ipynb first."
        )
    hierarchy_df = (
        hierarchy_df
        if hierarchy_df is not None
        else load_brca_celltype_hierarchy()
    )
    hierarchy_map = hierarchy_df.set_index("celltype_level2")
    cells = pd.read_csv(src)
    if "Cluster" not in cells.columns:
        raise KeyError(f"{src.name} needs Cluster")
    cells = cells.copy()
    cells["final_CT"] = cells["Cluster"].astype(str).str.strip()
    cells = cells[cells["final_CT"].isin(hierarchy_map.index)].copy()
    cells["final_sublineage"] = cells["final_CT"].map(hierarchy_map["celltype_level1"])
    cells["final_lineage"] = cells["final_CT"].map(hierarchy_map["celltype_level0"])
    cells["tma"] = sample
    required = {"cell_id", "X_pix_HE", "Y_pix_HE", "x_centroid", "y_centroid"}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise KeyError(f"{src.name} missing {missing}")
    import tifffile

    with tifffile.TiffFile(he_tif_path(sample)) as tiff:
        he_height, he_width = tiff.pages[0].shape[:2]
    n_labeled = len(cells)
    inside = (
        (cells["X_pix_HE"] >= 0)
        & (cells["X_pix_HE"] < he_width)
        & (cells["Y_pix_HE"] >= 0)
        & (cells["Y_pix_HE"] < he_height)
    )
    cells = cells.loc[inside].copy()
    print(
        f"  inside HE TIFF {he_width}x{he_height}: "
        f"{len(cells):,}/{n_labeled:,} labeled cells"
    )
    return cells.reset_index(drop=True)


def build_cells_matched_by_stardist(
    cells_with_pixel: pd.DataFrame | Path | str,
    sample: str,
    *,
    stardist_root: Path | str = DEFAULT_STARDIST_ROOT,
    max_distance: float | None = 50.0,
) -> pd.DataFrame:
    star_path = stardist_csv_path(stardist_root, sample)
    if not star_path.is_file():
        raise FileNotFoundError(f"Prepared StarDist CSV not found: {star_path}")
    if isinstance(cells_with_pixel, (str, Path)):
        celltype_gt = load_cell_pixcoords(
            str(cells_with_pixel),
            column_rename=BRCA_COLUMN_RENAME,
            auto_rename=False,
        )
    else:
        celltype_gt = cells_with_pixel.rename(columns=BRCA_COLUMN_RENAME)
    star_raw = pd.read_csv(star_path)
    matched = match_celltype2stardist(
        celltype_gt,
        star_raw,
        celltype_pixel_coords_cols=("X_pix_HE", "Y_pix_HE"),
        stardist_pixel_coords_cols=("centroid_x", "centroid_y"),
    )
    matched = matched.dropna(subset=["centroid_x", "centroid_y"])
    if max_distance is not None and "matched_distance" in matched.columns:
        n_before = len(matched)
        matched = matched.loc[matched["matched_distance"] <= float(max_distance)].copy()
        print(
            f"  kept matched_distance <= {max_distance:g} px: "
            f"{len(matched):,}/{n_before:,}"
        )
    restore = {
        tgt: src
        for src, tgt in BRCA_COLUMN_RENAME.items()
        if tgt in matched.columns
    }
    return matched.rename(columns=restore).reset_index(drop=True)


def process_sample(
    sample: str,
    *,
    cases_root: Path | str = DEFAULT_CASES_ROOT,
    stardist_root: Path | str = DEFAULT_STARDIST_ROOT,
    hierarchy_df: pd.DataFrame | None = None,
    write_stardist: bool = True,
    overwrite: bool = True,
    max_distance: float | None = 50.0,
) -> dict[str, Path]:
    he_dir = sample_dir(sample, cases_root)
    he_dir.mkdir(parents=True, exist_ok=True)
    out_gt = cells_with_pixel_path(he_dir, sample)
    if out_gt.is_file() and not overwrite:
        print(f"  skip existing {out_gt.name}")
        cells = pd.read_csv(out_gt)
    else:
        cells = build_cells_with_pixel(sample, hierarchy_df=hierarchy_df)
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
    out_sd = cells_matched_stardist_path(he_dir, sample)
    if out_sd.is_file() and not overwrite:
        print(f"  skip existing {out_sd.name}")
        written["cells_matched_stardist"] = out_sd
        return written
    matched = build_cells_matched_by_stardist(
        cells if overwrite or not out_gt.is_file() else out_gt,
        sample,
        stardist_root=stardist_root,
        max_distance=max_distance,
    )
    matched.to_csv(out_sd, index=False)
    print(f"  wrote {out_sd.name} n={len(matched):,}")
    written["cells_matched_stardist"] = out_sd
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Xenium BRCA GT + StarDist cell tables.")
    p.add_argument("--sample", choices=sample_ids(), default=None)
    p.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    p.add_argument("--stardist-root", type=Path, default=DEFAULT_STARDIST_ROOT)
    p.add_argument("--hierarchy-xlsx", type=Path, default=DEFAULT_HIERARCHY_XLSX)
    p.add_argument("--no-stardist", action="store_true")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Drop StarDist matches farther than this many HE pixels (default 50). "
        "Pass a negative value to keep all greedy matches.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy = load_brca_celltype_hierarchy(args.hierarchy_xlsx)
    samples = [args.sample] if args.sample else sample_ids()
    print(f"Hierarchy L2={len(hierarchy)} from {args.hierarchy_xlsx.name}")
    for sample in samples:
        print(f"\n{sample}")
        process_sample(
            sample,
            cases_root=args.cases_root,
            stardist_root=args.stardist_root,
            hierarchy_df=hierarchy,
            write_stardist=not args.no_stardist,
            overwrite=not args.no_overwrite,
            max_distance=None if args.max_distance < 0 else args.max_distance,
        )


if __name__ == "__main__":
    main()
