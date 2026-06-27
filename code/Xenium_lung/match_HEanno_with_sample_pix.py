## 20260518 LLY
## 20260621 align with Data_process_visual_xenium_all.ipynb (cells 14–15)
## 20260622 accept single Path or multiple case roots in write_per_sample_cells
##          conda run -n SeededNTM python code/Xenium_lung/match_HEanno_with_sample_pix.py
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI

## Complete_Cases + Incomplete_Cases
# conda run -n SeededNTM python code/Xenium_lung/match_HEanno_with_sample_pix.py

## Complete_Cases
# conda run -n SeededNTM python code/Xenium_lung/match_HEanno_with_sample_pix.py \
#   --cases-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases

## Incomplete_Cases
# conda run -n SeededNTM python code/Xenium_lung/match_HEanno_with_sample_pix.py \
#   --cases-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases


"""Match Weiqin HE cells to paper sample IDs and write pixel-coordinate table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd

from sample_mapping import (
    DEFAULT_PAPER_SHEET,
    load_select_samples_from_paper_xlsx,
    match_weiqin_cells_to_paper_samples,
)
from xenium_coords import add_he_pixel_columns

_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
)
WEIQIN_CELLS = _DATA_ROOT / "Annotation/HE_Annotations/cells_partitioned_by_annotation.csv"
PAPER_XLSX = _DATA_ROOT / "Annotation/HE_Annotations/Results/41588_2025_2080_MOESM5_ESM.xlsx"
CELLS_OUT = (
    _DATA_ROOT
    / "Annotation/HE_Annotations/cells_partitioned_by_annotation_sample_match_with_pixel.csv"
)

COMPLETE_CASES = _DATA_ROOT / "Data/Complete_Cases"
INCOMPLETE_CASES = _DATA_ROOT / "Data/Incomplete_Cases"
DEFAULT_CASES_ROOTS = (COMPLETE_CASES, INCOMPLETE_CASES)
PER_SAMPLE_SUFFIX = "_cells_partitioned_by_annotation_sample_match_with_pixel.csv"


def _coerce_cases_roots(
    cases_roots: Path | str | tuple[Path | str, ...] | list[Path | str],
) -> tuple[Path, ...]:
    """Accept one root or many; (Path(x)) without trailing comma is a single root."""
    if isinstance(cases_roots, (str, Path)):
        return (Path(cases_roots),)
    return tuple(Path(p) for p in cases_roots)


def list_case_sample_dirs(
    cases_roots: Path | str | tuple[Path | str, ...] | list[Path | str] = DEFAULT_CASES_ROOTS,
) -> dict[str, Path]:
    """Map paper sample folder name -> its case directory (Complete or Incomplete)."""
    out: dict[str, Path] = {}
    for root in _coerce_cases_roots(cases_roots):
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir():
                out[p.name] = p
    return out


def per_sample_path(sample_dir: Path, sample_id: str) -> Path:
    return sample_dir / f"{sample_id}{PER_SAMPLE_SUFFIX}"


def write_per_sample_cells(
    cells: pd.DataFrame,
    *,
    cases_roots: Path | str | tuple[Path | str, ...] | list[Path | str] = DEFAULT_CASES_ROOTS,
    sample_dirs: dict[str, Path] | None = None,
) -> list[Path]:
    """Write one CSV per sample into Complete_Cases/{sample}/ or Incomplete_Cases/{sample}/."""
    roots = _coerce_cases_roots(cases_roots)
    sample_dirs = sample_dirs or list_case_sample_dirs(roots)
    if not sample_dirs:
        print("WARNING: no case folders found, skip per-sample write")
        return []

    written: list[Path] = []
    print(f"Writing per-sample CSVs for up to {len(sample_dirs)} case folder(s) ...")
    for sample_id in sorted(cells["sample"].unique()):
        sub = cells.loc[cells["sample"] == sample_id]
        case_dir = sample_dirs.get(sample_id)
        if case_dir is None:
            print(f"  WARNING: no Complete/Incomplete folder for {sample_id}, skip")
            continue
        out_path = per_sample_path(case_dir, sample_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        rel_root = case_dir.parent.name
        print(f"  {sample_id}: {len(sub):,} cells -> {rel_root}/{sample_id}/{out_path.name}")
        written.append(out_path)
    return written


def build_sample_matched_cells_with_pixel(
    *,
    weiqin_cells_path: Path = WEIQIN_CELLS,
    paper_xlsx_path: Path = PAPER_XLSX,
    sheet_name: str = DEFAULT_PAPER_SHEET,
    select_samples: list[str] | None = None,
    cases_roots: Path | str | tuple[Path | str, ...] | list[Path | str] = DEFAULT_CASES_ROOTS,
) -> pd.DataFrame:
    """Mirror notebook: match paper samples, rename IDs, add HE pixel columns."""
    select_samples = select_samples or load_select_samples_from_paper_xlsx(
        paper_xlsx_path, sheet_name=sheet_name
    )
    print(f"Paper samples selected: {len(select_samples)}")

    df_all = pd.read_csv(weiqin_cells_path)
    matched, weiqin_ids, unknown_paper = match_weiqin_cells_to_paper_samples(
        df_all, select_samples
    )

    if unknown_paper:
        print("WARNING: paper samples without mapping:", unknown_paper)

    missing_weiqin = sorted(set(weiqin_ids) - set(df_all["sample"].unique()))
    if missing_weiqin:
        print("WARNING: no Weiqin rows for sample id:", missing_weiqin)

    print("Per-cell rows by sample (after rename):")
    print(matched.groupby("sample").size().to_string())
    print(f"Total before pixel columns: {len(matched):,} rows × {matched.shape[1]} cols")

    return add_he_pixel_columns(matched, cases_roots=_coerce_cases_roots(cases_roots))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cells_partitioned_by_annotation_sample_match_with_pixel.csv"
    )
    parser.add_argument("--weiqin-cells", type=Path, default=WEIQIN_CELLS)
    parser.add_argument("--paper-xlsx", type=Path, default=PAPER_XLSX)
    parser.add_argument("--sheet-name", default=DEFAULT_PAPER_SHEET)
    parser.add_argument("--output", type=Path, default=CELLS_OUT)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Case root(s) for per-sample CSV output (repeatable). "
            f"Default: {COMPLETE_CASES.name} + {INCOMPLETE_CASES.name}"
        ),
    )
    parser.add_argument(
        "--write-per-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write {sample}_cells_partitioned_by_annotation_sample_match_with_pixel.csv "
            "into each sample folder under Complete_Cases / Incomplete_Cases"
        ),
    )
    args = parser.parse_args()

    cases_roots = (
        _coerce_cases_roots(args.cases_dir)
        if args.cases_dir
        else DEFAULT_CASES_ROOTS
    )

    cells = build_sample_matched_cells_with_pixel(
        weiqin_cells_path=args.weiqin_cells,
        paper_xlsx_path=args.paper_xlsx,
        sheet_name=args.sheet_name,
        cases_roots=cases_roots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} ({len(cells):,} rows, {len(cells.columns)} cols)")

    if args.write_per_sample:
        write_per_sample_cells(cells, cases_roots=cases_roots)


if __name__ == "__main__":
    main()
