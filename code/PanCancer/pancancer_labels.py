"""Compatibility import. The implementation is ``Hist2Pheno_pkg.pancancer_labels``."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from Hist2Pheno_pkg.pancancer_labels import (  # noqa: E402
    DEFAULT_ONTOLOGY_SHEET,
    DEFAULT_ONTOLOGY_XLSX,
    hierarchy_l2_to_l1,
    hierarchy_maps,
    load_pancancer_ontology,
    map_native_series,
    native_to_pancancer,
    ontology_class_names,
    remap_obs_frame,
)

__all__ = [
    "DEFAULT_ONTOLOGY_SHEET",
    "DEFAULT_ONTOLOGY_XLSX",
    "hierarchy_l2_to_l1",
    "load_pancancer_ontology",
    "map_native_series",
    "native_to_pancancer",
    "ontology_class_names",
    "remap_obs_frame",
]
