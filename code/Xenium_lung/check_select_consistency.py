## 2026.06.22 check zarr-derived centroids vs HE-annotated cells (Complete_Cases)
##
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# conda run -n SeededNTM python code/Xenium_lung/check_select_consistency.py
#
# conda run -n SeededNTM python code/Xenium_lung/check_select_consistency.py --sample VUHD113

"""Compare zarr-derived centroids vs HE-annotated cells (aligned with notebook check_consistance)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_ROOT = Path(
    "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/Xemiun/weiqin/"
    "SpatialPF-NGenetics/Spatial-PF-Processed"
)
DEFAULT_LABELS_ALL = (
    _DATA_ROOT / "Data/Complete_Cases/Cell_spatial_coords_um_pix_from_zarr_all.csv"
)
DEFAULT_LABELS_HE = (
    _DATA_ROOT
    / "Annotation/HE_Annotations/cells_partitioned_by_annotation_sample_match_with_pixel.csv"
)
DEFAULT_CASES_DIR = _DATA_ROOT / "Data/Complete_Cases"

SELECT4_SAMPLES = ["VUHD113", "VUILD107MA", "VUILD102LA", "VUILD96LA"]
EXPECTED_ZARR_COUNTS = {
    "VUHD113": 14_940,
    "VUILD107MA": 72_153,
    "VUILD102LA": 27_973,
    "VUILD96LA": 46_277,
}


def check_consistency(
    sample_name: str,
    labels_all_path: str | Path,
    labels_he_path: str | Path,
) -> bool:
    """Mirror notebook ``check_consistance``; return True if matched coords are all equal."""
    df_labels_all = pd.read_csv(labels_all_path)
    df_paper = pd.read_csv(labels_he_path)

    labels_vu = df_labels_all.loc[
        df_labels_all["sample"] == sample_name,
        ["cell_id", "x_centroid", "y_centroid"],
    ]
    paper_vu = df_paper.loc[
        df_paper["sample"] == sample_name,
        ["cell_id", "x_centroid", "y_centroid"],
    ]

    cmp = labels_vu.merge(
        paper_vu,
        on="cell_id",
        suffixes=("_labels", "_paper"),
        how="outer",
        indicator=True,
    )
    print("Counts: All_labels", len(labels_vu), "| HE_annotations", len(paper_vu))
    print(cmp["_merge"].value_counts(), sep="\n")

    both = cmp[cmp["_merge"] == "both"]
    if both.empty:
        print("Matched 0 cells | All equal: False")
        print(f"{'*' * 20} {sample_name} done {'*' * 20}")
        return False

    dx = both["x_centroid_labels"] - both["x_centroid_paper"]
    dy = both["y_centroid_labels"] - both["y_centroid_paper"]
    all_equal = bool(np.allclose(dx, 0) and np.allclose(dy, 0))
    print(
        f"Matched {len(both):,} cells | max |dx|={dx.abs().max():.2e} "
        f"max |dy|={dy.abs().max():.2e} | All equal: {all_equal}"
    )
    print(f"{'*' * 20} {sample_name} done {'*' * 20}")
    return all_equal

def summary_table(
    labels_all_path: str | Path,
    labels_he_path: str | Path,
    *,
    samples: list[str] | None = None,
) -> pd.DataFrame:
    """Per-sample merge stats and whether matched centroids are all equal."""
    df_z = pd.read_csv(labels_all_path)
    df_h = pd.read_csv(labels_he_path)
    sample_list = samples or sorted(set(df_z["sample"].unique()) & set(df_h["sample"].unique()))

    rows = []
    for s in sample_list:
        labels_vu = df_z.loc[df_z["sample"] == s, ["cell_id", "x_centroid", "y_centroid"]]
        paper_vu = df_h.loc[df_h["sample"] == s, ["cell_id", "x_centroid", "y_centroid"]]
        cmp = labels_vu.merge(
            paper_vu,
            on="cell_id",
            suffixes=("_labels", "_paper"),
            how="outer",
            indicator=True,
        )
        both = cmp[cmp["_merge"] == "both"]
        if both.empty:
            all_equal = False
            max_dx = max_dy = np.nan
        else:
            dx = both["x_centroid_labels"] - both["x_centroid_paper"]
            dy = both["y_centroid_labels"] - both["y_centroid_paper"]
            all_equal = bool(np.allclose(dx, 0) and np.allclose(dy, 0))
            max_dx = float(dx.abs().max())
            max_dy = float(dy.abs().max())

        z_ids = set(labels_vu["cell_id"])
        h_ids = set(paper_vu["cell_id"])
        rows.append(
            {
                "sample": s,
                "zarr_cells": len(labels_vu),
                "he_cells": len(paper_vu),
                "both": len(both),
                "left_only": len(z_ids - h_ids),
                "right_only": len(h_ids - z_ids),
                "max_abs_dx": max_dx,
                "max_abs_dy": max_dy,
                "all_equal": all_equal,
                "expected_zarr": EXPECTED_ZARR_COUNTS.get(s),
            }
        )
    return pd.DataFrame(rows)


def list_case_samples(cases_dir: Path) -> list[str]:
    if not cases_dir.is_dir():
        return []
    return sorted(p.name for p in cases_dir.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-all", type=Path, default=DEFAULT_LABELS_ALL)
    parser.add_argument("--labels-he", type=Path, default=DEFAULT_LABELS_HE)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--sample", action="append", default=None, help="Run one sample only (repeatable)")
    parser.add_argument("--summary-only", action="store_true", help="Print summary table only")
    args = parser.parse_args()

    samples = args.sample or list_case_samples(args.cases_dir)
    if not samples:
        raise SystemExit(f"No sample folders under {args.cases_dir}")

    summary = summary_table(args.labels_all, args.labels_he, samples=samples)
    print("=== Summary ===")
    print(summary.to_string(index=False))

    false_samples = summary.loc[~summary["all_equal"], "sample"].tolist()
    if false_samples:
        print(f"\nAll equal: False -> {false_samples}")
    else:
        print("\nAll equal: True for all listed samples")

    if args.summary_only:
        return

    print("\n=== Per-sample check ===")
    results: dict[str, bool] = {}
    for sample in samples:
        print()
        results[sample] = check_consistency(sample, args.labels_all, args.labels_he)

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed (All equal: False): {failed}")
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
