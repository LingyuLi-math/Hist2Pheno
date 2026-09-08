"""Build CODEX GBM GT / StarDist cell tables with microscope HE pixel coordinates.

Analog of ``code/CODEX_hcc/match_codex_cells_with_pixel.py`` /
``code/Xenium_brca/match_xenium_cells_with_pixel.py``.

Writes under ``data/CODEX/GBM/Cases/{P174511_Initial,P179161_Recurrent}/``:

- ``{sample}_cells_with_pixel.csv``
- ``{sample}_cells_matched_by_stardist.csv``

    python code/CODEX_gbm/match_codex_cells_with_pixel.py
    python code/CODEX_gbm/match_codex_cells_with_pixel.py --sample P174511_Initial
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


##################################################
# 2026.09.08, revise the hierarchy xlsx and update cell level labels
##################################################
from base import load_cell_pixcoords, match_celltype2stardist  # noqa: E402
from gbm_paths import (  # noqa: E402
    DEFAULT_CASES_ROOT,
    DEFAULT_HIERARCHY_XLSX,
    DEFAULT_STARDIST_ROOT,
    GBM_COLUMN_RENAME,
    fill_gbm_spatial_niche,    # 2026.09.08, revise the hierarchy
    filter_usable_gbm_cell_labels,  # 2026.09.09, drop Unknown/LowQ
    filter_usable_gbm_sn,      # 2026.09.08, revise the hierarchy
    he_tif_path,
    load_gbm_celltype_hierarchy,
    sample_config,
    sample_dir,
    sample_ids,
    stardist_csv_path as gbm_stardist_csv_path,
)

HCC_COLUMN_RENAME = GBM_COLUMN_RENAME


def cells_with_pixel_path(he_dir: Path | str, he_key: str) -> Path:
    """HCC-compatible signature: ``(sample_dir, sample)``."""
    return Path(he_dir) / f"{he_key}_cells_with_pixel.csv"


def cells_matched_stardist_path(he_dir: Path | str, he_key: str) -> Path:
    return Path(he_dir) / f"{he_key}_cells_matched_by_stardist.csv"


def stardist_csv_path(stardist_root: Path | str, he_key: str) -> Path:
    """HCC-compatible signature: ``(stardist_root, sample)``."""
    return gbm_stardist_csv_path(he_key, stardist_root)


def list_aligned_annotated_regions() -> pd.DataFrame:
    """Ini/Rec as MATCHED_HE keys (HCC train-script compatible)."""
    ids = sample_ids()
    return pd.DataFrame({"MATCHED_HE": ids, "CODEX_ACQUISITION_ID": ids})


def build_cells_with_pixel(
    sample: str,
    *,
    hierarchy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join preprocessed nuclei onto the 3-level GBM hierarchy.

    ``final_CT`` = ``subcluster`` (fine), ``final_sublineage`` = ``spatial_niche``
    (intermediate), ``final_lineage`` = ``cell_type`` (coarse). Rows whose
    triple is absent from the trainable hierarchy (Unknown / LowQ cell labels,
    or LowQ / missing SN) are dropped.
    """
    cfg = sample_config(sample)
    src = Path(cfg["cell_info_csv"])
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed cell table: {src}\n"
            "Run Data_process_HEcelltype_GBM.ipynb first."
        )
    hierarchy_df = (
        hierarchy_df
        if hierarchy_df is not None
        else load_gbm_celltype_hierarchy()
    )
    need = {"cell_type", "subcluster", "spatial_niche", "bin_barcode"}
    cells = pd.read_csv(src)
    missing_src = sorted(need - set(cells.columns))
    if missing_src:
        raise KeyError(f"{src.name} needs {missing_src}")
    n_raw = len(cells)
    cells = filter_usable_gbm_sn(cells)
    n_after_sn = len(cells)
    cells = filter_usable_gbm_cell_labels(cells)
    print(
        f"  drop SN LowQ/missing (and empty bin_barcode): "
        f"{n_raw - n_after_sn:,}/{n_raw:,} → {n_after_sn:,}; "
        f"drop Unknown/LowQ cell labels: "
        f"{n_after_sn - len(cells):,} → {len(cells):,}"
    )
    cells = cells.copy()
    cells["final_CT"] = cells["subcluster"].map(
        lambda v: str(v).strip() if pd.notna(v) else ""
    )
    cells["final_sublineage"] = cells["spatial_niche"].map(fill_gbm_spatial_niche)
    cells["final_lineage"] = cells["cell_type"].map(
        lambda v: str(v).strip() if pd.notna(v) else ""
    )
    keys = hierarchy_df[
        ["celltype_level2", "celltype_level1", "celltype_level0"]
    ].drop_duplicates()
    n_before = len(cells)
    cells = cells.merge(
        keys,
        left_on=["final_CT", "final_sublineage", "final_lineage"],
        right_on=["celltype_level2", "celltype_level1", "celltype_level0"],
        how="inner",
    )
    cells = cells.drop(
        columns=["celltype_level2", "celltype_level1", "celltype_level0"]
    )
    print(
        f"  hierarchy triples kept {len(cells):,}/{n_before:,} "
        f"(drop Unknown/LowQ and SN LowQ/missing)"
    )
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
            column_rename=GBM_COLUMN_RENAME,
            auto_rename=False,
        )
    else:
        celltype_gt = cells_with_pixel.rename(columns=GBM_COLUMN_RENAME)
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
        for src, tgt in GBM_COLUMN_RENAME.items()
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
            f"L1={cells['final_lineage'].nunique()} | "
            f"bin_barcode non-null={cells['bin_barcode'].notna().all()}"
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
    p = argparse.ArgumentParser(description="CODEX GBM GT + StarDist cell tables.")
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
    hierarchy = load_gbm_celltype_hierarchy(args.hierarchy_xlsx)
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
