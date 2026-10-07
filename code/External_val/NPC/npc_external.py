#!/usr/bin/env python3
"""NPC external validation. The pipeline is Hist2Pheno_pkg.external_validation."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import ExternalCohort

COHORT = ExternalCohort(
    name="NPC",
    data_root=_REPO / "data" / "Visium" / "NPC",
    repo=_REPO,
)

# Names the notebook and demo.sh already call.
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
