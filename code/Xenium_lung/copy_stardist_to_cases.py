## 2026.06.23 LLY
## /home/lingyu/data/Python/Collaborate/esccAI/data/Xenium/lung/StarDist_Segment
## /home/lingyu/ssd2/Python/Collaborate/esccAI/data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/

## Copy StarDist CSV from StarDist_Segment into Complete_Cases / Incomplete_Cases
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI

## All samples
# conda run -n SeededNTM python code/Xenium_lung/copy_stardist_to_cases.py

## Optional: limit to specific samples
# conda run -n SeededNTM python code/Xenium_lung/copy_stardist_to_cases.py --sample VUHD113

## Optional: do not overwrite existing CSV
# conda run -n SeededNTM python code/Xenium_lung/copy_stardist_to_cases.py --no-replace

"""Copy {sample}_Float_prob0.01_nms_0.3.csv into per-sample case folders."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from match_HEanno_with_sample_pix import COMPLETE_CASES, INCOMPLETE_CASES

STARDIST_SOURCE = Path(
    "/home/lingyu/data/Python/Collaborate/esccAI/data/Xenium/lung/StarDist_Segment"
)
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"


def stardist_source_csv(stardist_root: Path, sample: str) -> Path:
    return stardist_root / sample / f"{sample}{STARDIST_CSV_SUFFIX}"


def resolve_case_dir(sample: str) -> Path | None:
    """Return Complete_Cases/{sample} or Incomplete_Cases/{sample}, else None."""
    for root in (COMPLETE_CASES, INCOMPLETE_CASES):
        case_dir = root / sample
        if case_dir.is_dir():
            return case_dir
    return None


def copy_stardist_csvs(
    *,
    stardist_root: Path = STARDIST_SOURCE,
    replace: bool = True,
    samples: list[str] | None = None,
) -> list[Path]:
    """Copy StarDist CSVs into matching Complete/Incomplete case folders.

    Parameters
    ----------
    replace
        If True, overwrite existing destination CSV. If False, skip when present.
    """
    if not stardist_root.is_dir():
        raise FileNotFoundError(f"StarDist source not found: {stardist_root}")

    sample_ids = samples or sorted(
        p.name for p in stardist_root.iterdir() if p.is_dir()
    )
    copied: list[Path] = []
    print(f"StarDist source: {stardist_root}")
    print(f"Processing {len(sample_ids)} sample folder(s), replace={replace}")

    for sample in sample_ids:
        src = stardist_source_csv(stardist_root, sample)
        if not src.is_file():
            print(f"  WARNING: missing source CSV for {sample}: {src.name}, skip")
            continue

        case_dir = resolve_case_dir(sample)
        if case_dir is None:
            print(f"  WARNING: {sample} not in Complete_Cases or Incomplete_Cases, skip")
            continue

        dst = case_dir / src.name
        if dst.is_file() and not replace:
            print(f"  SKIP (exists): {sample} -> {case_dir.parent.name}/{sample}/{dst.name}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rel = f"{case_dir.parent.name}/{sample}/{dst.name}"
        action = "overwrote" if dst.exists() and replace else "copied"
        print(f"  {sample}: {action} -> {rel} ({src.stat().st_size:,} bytes)")
        copied.append(dst)

    print(f"Done: {len(copied)} file(s) copied")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy {sample}_Float_prob0.01_nms_0.3.csv from StarDist_Segment "
            "into Complete_Cases or Incomplete_Cases per sample folder name."
        )
    )
    parser.add_argument(
        "--stardist-root",
        type=Path,
        default=STARDIST_SOURCE,
        help=f"StarDist_Segment root (default: {STARDIST_SOURCE})",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Limit to sample id(s); default: all folders under --stardist-root",
    )
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite destination CSV if it already exists (default: True)",
    )
    args = parser.parse_args()
    copy_stardist_csvs(
        stardist_root=args.stardist_root,
        replace=args.replace,
        samples=args.sample,
    )


if __name__ == "__main__":
    main()
