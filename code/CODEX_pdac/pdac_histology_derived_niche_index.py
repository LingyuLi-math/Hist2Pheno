## 2026.08.20 PDAC niche-index API (s1167 Pancreas TMA)
"""Bind s1167 TLS / SRI / TNI helpers to ``pan_organ='codex_pdac'``."""

from s1167_histology_derived_niche_index import configure as _configure

_configure("codex_pdac")

from s1167_histology_derived_niche_index import *  # noqa: F401,F403
from s1167_histology_derived_niche_index import (  # noqa: E402
    DEFAULT_SPATIAL_RADIUS_UM,
    DEFAULT_UM_PER_HE_PIXEL,
    ORGAN_CONFIGS,
)

PAN_ORGAN = "codex_pdac"
DEFAULT_DEMO_SAMPLE = ORGAN_CONFIGS[PAN_ORGAN].demo_sample
DEFAULT_HCC_SPATIAL_RADIUS_UM = DEFAULT_SPATIAL_RADIUS_UM
DEFAULT_HCC_UM_PER_HE_PIXEL = DEFAULT_UM_PER_HE_PIXEL
DEFAULT_HE_ROOT = DEFAULT_CASES_ROOT
