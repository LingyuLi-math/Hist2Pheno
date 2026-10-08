"""Compatibility import. The implementation is ``Hist2Pheno_pkg.pancancer_cv``."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from Hist2Pheno_pkg.pancancer_cv import (  # noqa: E402
    CELL_STRAT_ABLATION_TAG,
    CELL_STRAT_RESULT_DIR,
    CELL_STRATIFIED_SPLITTERS,
    PREASSIGNED_SPLITTERS,
    assign_cell_stratified_folds,
    assign_section_folds,
    cache_path_for,
    fold_plan_frame,
    is_cell_stratified_splitter,
    ok_mask,
    summarize_cell_stratified_folds,
)

__all__ = [
    "CELL_STRAT_ABLATION_TAG",
    "CELL_STRAT_RESULT_DIR",
    "CELL_STRATIFIED_SPLITTERS",
    "PREASSIGNED_SPLITTERS",
    "assign_cell_stratified_folds",
    "assign_section_folds",
    "cache_path_for",
    "fold_plan_frame",
    "is_cell_stratified_splitter",
    "ok_mask",
    "summarize_cell_stratified_folds",
]
