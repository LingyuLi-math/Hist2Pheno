"""
CLI: PCF -> HE alignment + quick spatial preview

This script reproduces the key outputs from the NCT/NCRT/NICT notebooks:
  1) Save HE-space cell centroid CSVs (PCF->HE aligned) + 4ViT CSV
  2) Save spatial distribution scatter plots (jpg) for the given ROI parent

Typical usage (example for NCT tumor1):
  python data_process_visualize.py \
    --dataset NCT \
    --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14" \
    --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
    --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14/export/NCT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_15-28/export/NCT-measurements-tumor-15-28.csv" \
    --parent_value tumor1
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Make sure we can import project modules when running from any directory.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PARENT = os.path.dirname(CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if CODE_PARENT not in sys.path:
    sys.path.insert(0, CODE_PARENT)
PKG_DIR = os.path.join(CODE_PARENT, "Hist2Pheno_pkg")
if os.path.isdir(PKG_DIR) and PKG_DIR not in sys.path:
    # Prefer package-style module location when present.
    sys.path.insert(0, PKG_DIR)

try:
    from base import get_celltype_coords, make_PCF2HE_alignment  # type: ignore
    from plot import (  # type: ignore
        plot_celltype_spatial_distribution,
        plot_celltype_distribution,
        plot_cell_area_histogram,
    )
except ModuleNotFoundError:
    from Hist2Pheno_pkg.base import get_celltype_coords, make_PCF2HE_alignment  # type: ignore
    from Hist2Pheno_pkg.plot import (  # type: ignore
        plot_celltype_spatial_distribution,
        plot_celltype_distribution,
        plot_cell_area_histogram,
    )


DATASET_DEFAULTS = {
    # PCF / HE segment indices under segment_project_dir/data/<n>/server.json
    "NCRT": {"pcf_segment": "1", "he_segment": "3"},
    "NCT": {"pcf_segment": "1", "he_segment": "2"},
    "NICT": {"pcf_segment": "1", "he_segment": "2"},
    "SA": {"pcf_segment": "1", "he_segment": "3"},
}


class _Tee:
    """Write all output to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            flush = getattr(s, "flush", None)
            if callable(flush):
                flush()


def _parse_csvs(csvs: str) -> Sequence[str]:
    if "," in csvs:
        parts = [p.strip() for p in csvs.split(",") if p.strip()]
        if not parts:
            raise ValueError("--coords_csvs is empty")
        return parts
    return [csvs]


def _default_output_csvs(
    *,
    data_root: str,
    dataset_key: str,
    parent_value: Optional[str],
) -> tuple[str, str]:
    he_dir = os.path.join(data_root, "he_cell_coords")
    os.makedirs(he_dir, exist_ok=True)

    if parent_value is None:
        out_csv = os.path.join(he_dir, f"{dataset_key}_CellPixCoords_all.csv")
        out_csv4vit = os.path.join(he_dir, f"{dataset_key}_CellPixCoords4ViT_all.csv")
    else:
        out_csv = os.path.join(he_dir, f"{dataset_key}_CellPixCoords_{parent_value}.csv")
        out_csv4vit = os.path.join(he_dir, f"{dataset_key}_CellPixCoords4ViT_{parent_value}.csv")

    return out_csv, out_csv4vit


def _save_spatial_jpg(
    *,
    dataset_key: str,
    data_root: str,
    save_result: str,
    celltype_anno_df: "pd.DataFrame",
    coords_df: "pd.DataFrame",
    therapy_data: str,
    parent_value: str,
    x_col: str,
    y_col: str,
    figsize=(12, 10),
    alpha: float = 0.6,
    s: float = 0.6,
) -> None:
    out_dir = os.path.join(data_root, dataset_key, save_result)
    os.makedirs(out_dir, exist_ok=True)

    # Notebook convention:
    #   celltype_NCT_tumor1 = get_celltype_coords(..., value='NCT_tumor1', parent_value='tumor1')
    plot_value = f"{therapy_data}_{parent_value}"
    celltype_for_plot = get_celltype_coords(
        celltype=celltype_anno_df,
        ncrt_anno_df=coords_df,
        value=plot_value,
        parent_value=parent_value,
    )

    plot_celltype_spatial_distribution(
        celltype_for_plot,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype",
        figsize=figsize,
        alpha=alpha,
        s=s,
        format="jpg",
        save_path=os.path.join(out_dir, f"celltype_spatial_{parent_value}_level2.jpg"),
    )

    plot_celltype_spatial_distribution(
        celltype_for_plot,
        x_col=x_col,
        y_col=y_col,
        celltype_col="celltype_level1",
        figsize=figsize,
        alpha=alpha,
        s=s,
        format="jpg",
        save_path=os.path.join(out_dir, f"celltype_spatial_{parent_value}_level1.jpg"),
    )


def _infer_hist_part_label_from_path(p: str) -> str:
    """
    Infer a label like:
      tumor1-14  -> "1_14"
      tumor-15-28 -> "15_28"
      measurements-3sample -> "3"
    """
    import re

    base = os.path.basename(p)
    nums = re.findall(r"\d+", base)
    if not nums:
        return "part"
    if len(nums) >= 2:
        return f"{nums[0]}_{nums[1]}"
    return nums[0]


def _save_distribution_pdfs(
    *,
    dataset_key: str,
    data_root: str,
    save_result: str,
    therapy_data: str,
    parent_value: str,
    celltype_anno_df: "pd.DataFrame",
    coords_df: "pd.DataFrame",
    celltype_col: str,
) -> None:
    out_dir = os.path.join(data_root, dataset_key, save_result)
    os.makedirs(out_dir, exist_ok=True)

    plot_value = f"{therapy_data}_{parent_value}"
    celltype_for_parent = get_celltype_coords(
        celltype=celltype_anno_df,
        ncrt_anno_df=coords_df,
        value=plot_value,
        parent_value=parent_value,
    )

    suffix = "level2" if celltype_col == "celltype" else "level1"
    save_path = os.path.join(out_dir, f"celltype_dist_{parent_value}_{suffix}.pdf")

    plot_celltype_distribution(
        celltype_for_parent,
        celltype_col=celltype_col,
        format="pdf",
        save_path=save_path,
    )


def _save_area_histograms(
    *,
    dataset_key: str,
    data_root: str,
    save_result: str,
    coords_dfs: Sequence["pd.DataFrame"],
    coords_csv_paths: Sequence[str],
    area_col: str = "Cell: Area µm^2",
) -> None:
    out_dir = os.path.join(data_root, dataset_key, save_result)
    os.makedirs(out_dir, exist_ok=True)

    # Combined histogram
    combined_df = pd.concat(list(coords_dfs), ignore_index=True) if len(coords_dfs) else pd.DataFrame()
    if not combined_df.empty:
        plot_cell_area_histogram(
            combined_df,
            col_name=area_col,
            bins=100,
            save_path=os.path.join(out_dir, "cellarea_dist.pdf"),
        )

    # Part histograms (match notebook naming style; use first two parts)
    if len(coords_dfs) >= 2:
        for i in [0, 1]:
            label = _infer_hist_part_label_from_path(coords_csv_paths[i])
            plot_cell_area_histogram(
                coords_dfs[i],
                col_name=area_col,
                bins=100,
                save_path=os.path.join(out_dir, f"cellarea_dist_{label}.pdf"),
            )
    elif len(coords_dfs) == 1:
        label = _infer_hist_part_label_from_path(coords_csv_paths[0])
        plot_cell_area_histogram(
            coords_dfs[0],
            col_name=area_col,
            bins=100,
            save_path=os.path.join(out_dir, f"cellarea_dist_{label}.pdf"),
        )


def run(
    *,
    dataset_key: str,
    segment_project_dir: str,
    celltype_anno_csv: str,
    coords_csvs: Sequence[str],
    parent_value: Optional[str],
    therapy_data_override: Optional[str],
    data_root: str,
    save_result: str,
    save_spatial_jpg: bool,
    output_csv: Optional[str],
    output_csv_4vit: Optional[str],
    print_head_n: int,
    sanity_check: bool,
    verbose: bool,
    save_distribution_pdf: bool,
    save_area_histogram_pdf: bool,
    also_save_all_parents_csv: bool,
    coords_csvs_all: Optional[Sequence[str]],
) -> "pd.DataFrame":
    if dataset_key not in DATASET_DEFAULTS:
        raise ValueError(f"--dataset must be one of {list(DATASET_DEFAULTS)}")

    # TumorID in codex_meta_celltype_final.csv matches --dataset (e.g. SA_tumor3 for --dataset SA).
    therapy_data = therapy_data_override if therapy_data_override is not None else dataset_key

    pcf_segment = DATASET_DEFAULTS[dataset_key]["pcf_segment"]
    he_segment = DATASET_DEFAULTS[dataset_key]["he_segment"]

    if verbose:
        print(f"[{dataset_key}] segment_project_dir={segment_project_dir}")
        print(f"[{dataset_key}] pcf_segment={pcf_segment}, he_segment={he_segment}")

    celltype_anno_df = pd.read_csv(celltype_anno_csv)

    # ROI coords (for the tumor-specific run)
    coords_dfs = [pd.read_csv(p) for p in coords_csvs]
    coords_df = pd.concat(coords_dfs, ignore_index=True)

    # All-parents coords (for parent_value=None run)
    coords_dfs_all = None
    coords_df_all = None
    if also_save_all_parents_csv:
        # If provided, use the full coords list; otherwise reuse ROI coords.
        coords_csvs_all_effective = coords_csvs_all or coords_csvs
        coords_dfs_all = [pd.read_csv(p) for p in coords_csvs_all_effective]
        coords_df_all = pd.concat(coords_dfs_all, ignore_index=True)

    if save_spatial_jpg and parent_value is not None:
        _save_spatial_jpg(
            dataset_key=dataset_key,
            data_root=data_root,
            save_result=save_result,
            celltype_anno_df=celltype_anno_df,
            coords_df=coords_df,
            therapy_data=therapy_data,
            parent_value=parent_value,
            x_col="Centroid X µm",
            y_col="Centroid Y µm",
        )

    # Save distribution (plot_celltype_distribution) + area histogram (plot_cell_area_histogram)
    if save_distribution_pdf and parent_value is not None:
        _save_distribution_pdfs(
            dataset_key=dataset_key,
            data_root=data_root,
            save_result=save_result,
            therapy_data=therapy_data,
            parent_value=parent_value,
            celltype_anno_df=celltype_anno_df,
            coords_df=coords_df,
            celltype_col="celltype",
        )
        _save_distribution_pdfs(
            dataset_key=dataset_key,
            data_root=data_root,
            save_result=save_result,
            therapy_data=therapy_data,
            parent_value=parent_value,
            celltype_anno_df=celltype_anno_df,
            coords_df=coords_df,
            celltype_col="celltype_level1",
        )

    if save_area_histogram_pdf:
        _save_area_histograms(
            dataset_key=dataset_key,
            data_root=data_root,
            save_result=save_result,
            coords_dfs=coords_dfs,
            coords_csv_paths=list(coords_csvs),
        )

    if output_csv is None or output_csv_4vit is None:
        default_csv, default_csv4vit = _default_output_csvs(
            data_root=data_root,
            dataset_key=dataset_key,
            parent_value=parent_value,
        )
        output_csv = output_csv or default_csv
        output_csv_4vit = output_csv_4vit or default_csv4vit

    # Also save "all parents" CSVs (no ROI suffix) when parent_value is provided.
    output_csv_all, output_csv_4vit_all = None, None
    if also_save_all_parents_csv and parent_value is not None:
        output_csv_all, output_csv_4vit_all = _default_output_csvs(
            data_root=data_root,
            dataset_key=dataset_key,
            parent_value=None,
        )

        if verbose:
            print(f"[{dataset_key}] also saving all-parents CSVs:")
            print(f"  {output_csv_all}")
            print(f"  {output_csv_4vit_all}")

        _ = make_PCF2HE_alignment(
            celltype_anno_df=celltype_anno_df,
            celltype_coords_df=coords_df_all,
            therapy_data=dataset_key,
            segment_path=segment_project_dir,
            segment_batch1=pcf_segment,
            segment_batch2=he_segment,
            value=therapy_data,
            parent_value=None,
            save_path=output_csv_all,
            save_path4ViT=output_csv_4vit_all,
        )

    # ROI-specific run (the returned dataframe).
    celltype_pixel_df = make_PCF2HE_alignment(
        celltype_anno_df=celltype_anno_df,
        celltype_coords_df=coords_df,
        therapy_data=dataset_key,
        segment_path=segment_project_dir,
        segment_batch1=pcf_segment,
        segment_batch2=he_segment,
        value=therapy_data,
        parent_value=parent_value,
        save_path=output_csv,
        save_path4ViT=output_csv_4vit,
    )

    if verbose:
        print(f"[{dataset_key}] saved: {output_csv}")
        print(f"[{dataset_key}] saved: {output_csv_4vit}")

    print(f"Cell numbers with {dataset_key} {parent_value if parent_value else 'all'}:", celltype_pixel_df.shape[0])
    print(celltype_pixel_df.head(print_head_n))

    return celltype_pixel_df


def main() -> None:
    ap = argparse.ArgumentParser(description="NCT/NCRT/NICT/SA PCF->HE alignment + plots + CSVs")
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    ap.add_argument(
        "--segment_project_dir",
        required=True,
        help="e.g. /.../data/NCT/NCT_project -seg_1-14 (must contain ./data/<pcf|he_segment>/server.json)",
    )
    ap.add_argument("--celltype_anno_csv", required=True, help="codex_meta_celltype_final.csv")
    ap.add_argument(
        "--coords_csvs",
        required=True,
        help="QuPath measurements CSVs. Use comma-separated paths to concat.",
    )
    ap.add_argument(
        "--coords_csvs_all",
        default=None,
        help="Optional full coords CSV list (comma-separated) used for the all-parents run (parent_value=None).",
    )
    ap.add_argument("--parent_value", default=None, help="ROI parent filter like 'tumor1'. Omit for all.")
    ap.add_argument(
        "--therapy-data",
        "--therapy-prefix-value",
        default=None,
        dest="therapy_data",
        help="Override TumorID therapy prefix in codex_meta (default: same as --dataset). "
        "--therapy-prefix-value is a deprecated alias.",
    )

    ap.add_argument(
        "--data-root",
        default=os.path.join(CODE_PARENT, "..", "data", "CODEX", "ESCC"),
        help="ESCC data dir (default: <repo>/data/CODEX/ESCC)",
    )
    ap.add_argument("--save-result", default="result", help="Subfolder under data/<dataset>/<save-result> for jpg plots")
    ap.add_argument("--save_spatial_jpg", action="store_true", help="Save celltype spatial jpg (level1/level2) for parent_value")
    ap.add_argument("--no-save_spatial_jpg", action="store_true", help="Disable spatial jpg saving")

    ap.add_argument(
        "--no-save-distribution-pdf",
        action="store_true",
        help="Disable celltype distribution PDFs (plot_celltype_distribution).",
    )
    ap.add_argument(
        "--no-save-area-histogram-pdf",
        action="store_true",
        help="Disable cell area histogram PDFs (plot_cell_area_histogram).",
    )

    ap.add_argument(
        "--no-also-save-all-parents-csv",
        action="store_true",
        help="When --parent_value is set, also-save (global) <therapy>_CellPixCoords.csv and <therapy>_CellPixCoords4ViT.csv by default. Use this flag to disable that extra run.",
    )

    ap.add_argument("--output-csv", default=None, help="Optional override output csv path")
    ap.add_argument("--output-csv-4vit", default=None, help="Optional override output 4ViT csv path")

    ap.add_argument("--print-head-n", type=int, default=5)
    ap.add_argument("--no-sanity-check", action="store_true", help="Currently unused (kept for compatibility).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-log-to-file", action="store_true", help="Disable saving stdout/stderr to a timestamped .log file under result/")

    args = ap.parse_args()

    # Setup logging: write stdout/stderr both to console and to a timestamped log file.
    log_f = None
    if not args.no_log_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.abspath(args.data_root), args.dataset, args.save_result)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{timestamp}.log")
        log_f = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_f)
        sys.stderr = _Tee(sys.stderr, log_f)
        print(f"[log] writing stdout/stderr to: {log_path}")

    coords_csvs = _parse_csvs(args.coords_csvs)
    coords_csvs_all = _parse_csvs(args.coords_csvs_all) if args.coords_csvs_all else None
    save_spatial_jpg = args.save_spatial_jpg and not args.no_save_spatial_jpg
    save_distribution_pdf = not args.no_save_distribution_pdf
    save_area_histogram_pdf = not args.no_save_area_histogram_pdf
    also_save_all_parents_csv = (args.parent_value is not None) and (not args.no_also_save_all_parents_csv)

    try:
        run(
            dataset_key=args.dataset,
            segment_project_dir=args.segment_project_dir,
            celltype_anno_csv=args.celltype_anno_csv,
            coords_csvs=coords_csvs,
            parent_value=args.parent_value,
            therapy_data_override=args.therapy_data,
            data_root=os.path.abspath(args.data_root),
            save_result=args.save_result,
            save_spatial_jpg=save_spatial_jpg,
            save_distribution_pdf=save_distribution_pdf,
            save_area_histogram_pdf=save_area_histogram_pdf,
            also_save_all_parents_csv=also_save_all_parents_csv,
            coords_csvs_all=coords_csvs_all,
            output_csv=args.output_csv,
            output_csv_4vit=args.output_csv_4vit,
            print_head_n=args.print_head_n,
            sanity_check=not args.no_sanity_check,
            verbose=args.verbose,
        )
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()


# -------------------------
# Example scripts (commented out)
# -------------------------
#
# NCRT (PCF=data/1, HE=data/3)
# python data_process_visualize.py \
#   --dataset NCRT \
#   --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCRT/NCRT_project - seg 1-14" \
#   --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
#   --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCRT/NCRT_project - seg 1-14/export/NCRT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCRT/NCRT_project - seg 15-28/export/NCRT-measurements-tumor-15-28.csv" \
#   --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCRT/NCRT_project - seg 1-14/export/NCRT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCRT/NCRT_project - seg 15-28/export/NCRT-measurements-tumor-15-28.csv" \
#   --parent_value tumor1 \
#   --save_spatial_jpg
#
# NCT (PCF=data/1, HE=data/2)
# python data_process_visualize.py \
#   --dataset NCT \
#   --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14" \
#   --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
#   --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14/export/NCT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_15-28/export/NCT-measurements-tumor-15-28.csv" \
#   --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_1-14/export/NCT-measurements-tumor1-14.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NCT/NCT_project -seg_15-28/export/NCT-measurements-tumor-15-28.csv" \
#   --parent_value tumor1 \
#   --save_spatial_jpg
#
# NICT (tumor6)
# - coords_csvs: only tumor6 part (for ROI visualizations and ROI-specific CSVs)
# - coords_csvs_all: full 1-28 parts (for the global NICT_CellPixCoords.csv / 4ViT CSV)
# python data_process_visualize.py \
#   --dataset NICT \
#   --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6" \
#   --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
#   --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6/export/NICT-measurements-tumor6.csv" \
#   --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_1-5&7-14/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_6/export/NICT-measurements-tumor6.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/NICT/NICT_project - seg_15-28/export/measurements.csv" \
#   --parent_value tumor6 \
#   --save_spatial_jpg
#
# SA (PCF=data/1, HE=data/3); codex_meta TumorID uses SA_tumor* (same as --dataset SA)
# python data_process_visualize.py \
#   --dataset SA \
#   --segment_project_dir "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3" \
#   --celltype_anno_csv "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/codex_meta_celltype_final.csv" \
#   --coords_csvs "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3/export/measurements-3sample.csv" \
#   --coords_csvs_all "/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_tumor3/export/measurements-3sample.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor1-2&4-10/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor11-20/export/measurements.csv,/home/lingyu/ssd2/Python/Hist2Pheno/data/CODEX/ESCC/SA/new_project_seg_all - tumor21-28/export/measurements.csv" \
#   --parent_value tumor3 \
#   --save_spatial_jpg
