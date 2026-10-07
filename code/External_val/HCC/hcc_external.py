#!/usr/bin/env python3
"""HCC external validation. The pipeline is Hist2Pheno_pkg.external_validation.

The HE files are JPEG. They have no OME PhysicalSize. The 96 dpi tag is a
Windows Photo Viewer placeholder and is not a pixel size. Pass the scanner
µm/px as MPP_UM before any step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import ExternalCohort


def _mpp_um() -> float:
    raw = os.environ.get("MPP_UM", "").strip()
    if not raw:
        raise SystemExit(
            "Set MPP_UM to the scanner µm/px before running.\n"
            "data/Visium/HCC/HE_images/*.jpg has no OME PhysicalSize. "
            "The 96 dpi tag is not a pixel size.\n"
            "Example: MPP_UM=0.5 bash code/External_val/HCC/demo.sh dry-run"
        )
    value = float(raw)
    if value <= 0:
        raise SystemExit(f"MPP_UM must be positive, got {value}")
    return value


COHORT = ExternalCohort(
    name="HCC",
    data_root=_REPO / "data" / "Visium" / "HCC",
    repo=_REPO,
    he_suffix=".jpg",
    mpp_um=_mpp_um(),
)

PRED_ROOT = COHORT.pred_root
discover_samples = COHORT.discover_samples
sample_paths = COHORT.sample_paths
geometry = COHORT.geometry
composition_table = COHORT.composition_table
ensure_l2_probs = COHORT.ensure_l2_probs
cmd_plot = COHORT.cmd_plot


def main(argv: list[str] | None = None) -> int:
    return COHORT.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
