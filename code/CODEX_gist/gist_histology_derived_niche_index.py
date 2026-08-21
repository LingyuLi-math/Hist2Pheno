## 2026.08.20 GIST niche-index API (s1167 GIST TMA)
"""Bind s1167 TLS / SRI / TNI helpers to ``pan_organ='codex_gist'``."""

from __future__ import annotations

import sys
from pathlib import Path

_PDAC_DIR = Path(__file__).resolve().parent.parent / "CODEX_pdac"
if str(_PDAC_DIR) not in sys.path:
    sys.path.insert(0, str(_PDAC_DIR))

from s1167_histology_derived_niche_index import configure as _configure

_configure("codex_gist")

from s1167_histology_derived_niche_index import *  # noqa: F401,F403
from s1167_histology_derived_niche_index import (  # noqa: E402
    DEFAULT_SPATIAL_RADIUS_UM,
    DEFAULT_UM_PER_HE_PIXEL,
    ORGAN_CONFIGS,
)

PAN_ORGAN = "codex_gist"
DEFAULT_DEMO_SAMPLE = ORGAN_CONFIGS[PAN_ORGAN].demo_sample
DEFAULT_HCC_SPATIAL_RADIUS_UM = DEFAULT_SPATIAL_RADIUS_UM
DEFAULT_HCC_UM_PER_HE_PIXEL = DEFAULT_UM_PER_HE_PIXEL
DEFAULT_HE_ROOT = DEFAULT_CASES_ROOT
