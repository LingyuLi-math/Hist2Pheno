"""
2026.04.02 Lingyu create this script.
PCF2HE2StarDist alignment runner.

This script is adapted from the corresponding block in `NCRT_valid.ipynb`:
  - build stardist/qupath/codex/cellpix coordinate paths
  - call `make_PCF2HE2StarDist_alignment(...)`
  - save outputs to `data/he_cell_coords/`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd  # type: ignore[reportMissingImports]


def build_paths(
    *,
    python_root: str,
    stardist_data: str,
    qupath_corner: str,
    codex_meta_celltype_final: str,
    transfer_data: str,
    therapy_data: str,
    parent_value: str,
) -> dict[str, str]:
    # Matches the notebook convention:
    #   path = '/home/lingyu/ssd2/Python/'
    #   f'{path}Hist2Pheno/data/CODEX/ESCC/...'
    base_data_dir = f"{python_root}Hist2Pheno/data/CODEX/ESCC"

    stardist_coords_path = f"{base_data_dir}/{stardist_data}/{therapy_data}_Float_prob0.01_nms_0.3.csv"
    qupath_corner_path = f"{base_data_dir}/{qupath_corner}/{therapy_data}_ROI_Corners.csv"
    codex_meta_celltype_path = codex_meta_celltype_final
    celltype_pixel_NCRT_path = f"{base_data_dir}/{transfer_data}/{therapy_data}_CellPixCoords_all.csv"

    save_path = f"{base_data_dir}/he_cell_coords/{therapy_data}_CellPixCoords_{parent_value}_StarDist.csv"
    save_path4ViT = (
        f"{base_data_dir}/he_cell_coords/{therapy_data}_CellPixCoords_{parent_value}_StarDist_ViT.csv"
    )

    return {
        "stardist_coords_path": stardist_coords_path,
        "qupath_corner_path": qupath_corner_path,
        "codex_meta_celltype_path": codex_meta_celltype_path,
        "celltype_pixel_NCRT_path": celltype_pixel_NCRT_path,
        "save_path": save_path,
        "save_path4ViT": save_path4ViT,
    }


def main() -> None:
    CODE_DIR = Path(__file__).resolve().parent
    PKG_DIR = CODE_DIR.parent / "Hist2Pheno_pkg"

    # Make Hist2Pheno_pkg importable so `from base import ...` works.
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    if PKG_DIR.is_dir() and str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))

    from base import make_PCF2HE2StarDist_alignment  # type: ignore
    from plot import plot_PCF_HE_counts, plot_WSI_counts  # type: ignore

    parser = argparse.ArgumentParser(
        description="Run make_PCF2HE2StarDist_alignment using paths matching NCRT_valid.ipynb."
    )
    parser.add_argument("--python_root", type=str, default="/home/lingyu/ssd2/Python/", help="Base python root")
    parser.add_argument("--therapy_data", type=str, default="NCRT", help="e.g. NCRT/SA/NCT/NICT")
    parser.add_argument("--parent_value", type=str, default="tumor1", help="e.g. tumor1/tumor3/all")

    # Dataset folders under `data/<...>`
    parser.add_argument("--stardist_data", type=str, default="StarDist_Segment")
    parser.add_argument("--qupath_corner", type=str, default="QupathCorners")
    parser.add_argument("--transfer_data", type=str, default="he_cell_coords")
    parser.add_argument("--save_plots", action="store_true", help="Save WSI/PCF-HE comparison plots")

    # codex meta celltype file (usually shared across datasets)
    parser.add_argument(
        "--codex_meta_celltype_final",
        type=str,
        default="/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv",
    )

    args = parser.parse_args()

    paths = build_paths(
        python_root=args.python_root,
        stardist_data=args.stardist_data,
        qupath_corner=args.qupath_corner,
        codex_meta_celltype_final=args.codex_meta_celltype_final,
        transfer_data=args.transfer_data,
        therapy_data=args.therapy_data,
        parent_value=args.parent_value,
    )

    # ---- The core block from NCRT_valid.ipynb (parameterized) ----
    stardist_coords_path = paths["stardist_coords_path"]
    qupath_corner_path = paths["qupath_corner_path"]
    codex_meta_celltype_path = paths["codex_meta_celltype_path"]
    celltype_pixel_NCRT_path = paths["celltype_pixel_NCRT_path"]
    save_path = paths["save_path"]
    save_path4ViT = paths["save_path4ViT"]

    # Preflight: GroundTruth must exist and TumorID must match therapy_data (e.g. SA_tumor3).
    gt_path = Path(celltype_pixel_NCRT_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"GroundTruth file not found: {celltype_pixel_NCRT_path}")

    gt_df = pd.read_csv(gt_path)
    if "TumorID" not in gt_df.columns:
        raise ValueError(f"'TumorID' column missing in: {celltype_pixel_NCRT_path}")

    if args.parent_value != "all":
        expected_tid = f"{args.therapy_data}_{args.parent_value}"
        n_expected = int((gt_df["TumorID"].astype(str) == expected_tid).sum())
        if n_expected == 0:
            sample_ids = sorted(gt_df["TumorID"].dropna().astype(str).unique().tolist())[:20]
            raise ValueError(
                f"No rows found for TumorID '{expected_tid}' in ground-truth CSV.\n"
                f"Sample TumorID values: {sample_ids}"
            )

    celltype_anno_df, roi_cell_counts, star_coords_in_roi, celltype_pixel_NCRT = (
        make_PCF2HE2StarDist_alignment(
            stardist_coords_path,
            qupath_corner_path,
            codex_meta_celltype_path,
            celltype_pixel_NCRT_path,
            therapy_data=args.therapy_data,
            parent_value=args.parent_value,
            save_path=save_path,
            save_path4ViT=save_path4ViT,
            celltype_pixel_NCRT_df=gt_df,
        )
    )

    base_data_dir = f"{args.python_root}Hist2Pheno/data/CODEX/ESCC"
    save_dir = f"{base_data_dir}/{args.therapy_data}/result"
    if args.save_plots:
        os.makedirs(save_dir, exist_ok=True)
        _ = plot_WSI_counts(
            celltype_anno_df,
            roi_cell_counts,
            prefix=f"{args.therapy_data}_",
            figure_size=(7, 5),
            save_path=f"{save_dir}/cellcounts_compare.pdf",
        )
        plot_PCF_HE_counts(
            star_coords_in_roi,
            celltype_pixel_NCRT,
            ROI=f"{args.therapy_data}_{args.parent_value}",
            figure_size=(10, 5),
            save_path=f"{save_dir}/cellcounts_compare_{args.parent_value}.jpg",
        )
    else:
        _ = plot_WSI_counts(
            celltype_anno_df,
            roi_cell_counts,
            prefix=f"{args.therapy_data}_",
            figure_size=(7, 5),
            save_path=None,
        )
        plot_PCF_HE_counts(
            star_coords_in_roi,
            celltype_pixel_NCRT,
            ROI=f"{args.therapy_data}_{args.parent_value}",
            figure_size=(10, 5),
            save_path=None,
        )
                
    # Basic summary; downstream notebooks/scripts can directly use saved csvs.
    print("Done make_PCF2HE2StarDist_alignment")
    print(f"  stardist_coords_path: {stardist_coords_path}")
    print(f"  qupath_corner_path: {qupath_corner_path}")
    print(f"  save_path: {save_path}")
    print(f"  save_path4ViT: {save_path4ViT}")
    print(f"  celltype_anno_df shape: {getattr(celltype_anno_df, 'shape', None)}")
    print(f"  celltype_pixel_NCRT shape: {getattr(celltype_pixel_NCRT, 'shape', None)}")
    print(f"  roi_cell_counts shape: {getattr(roi_cell_counts, 'shape', None)}")


if __name__ == "__main__":
    main()


# =========================
# Example run command (script)
# =========================
#
# cd /home/lingyu/ssd2/Python/Hist2Pheno
# conda activate SeededNTM
#
# python code/PCF2HE2StarDist_alignment.py \
#   --therapy_data NCRT \
#   --parent_value tumor1 \
#   --stardist_data StarDist_Segment \
#   --qupath_corner QupathCorners \
#   --transfer_data he_cell_coords \
#   --codex_meta_celltype_final data/codex_meta_celltype_final.csv \
#   --save_plots

