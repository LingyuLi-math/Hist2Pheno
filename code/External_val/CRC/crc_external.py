#!/usr/bin/env python3
"""CRC external validation. The pipeline is Hist2Pheno_pkg.external_validation.

The HE file is an OME BigTIFF (``.btf``). ``ome_um_per_px`` reads
PhysicalSizeX / PhysicalSizeY from the first page ImageDescription.
The TIFF 96 dpi tag is a placeholder and is not a pixel size.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import ExternalCohort

COHORT = ExternalCohort(
    name="CRC",
    data_root=_REPO / "data" / "Visium" / "CRC",
    repo=_REPO,
    he_suffix=".btf",
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
