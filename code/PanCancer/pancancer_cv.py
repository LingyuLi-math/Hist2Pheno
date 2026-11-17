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
    assign_stratified_section_folds,
    cache_path_for,
    fold_balance_score,
    fold_balance_table,
    is_stratified_section_splitter,
    section_label_counts,
    splitter_family,
    STRATIFIED_SECTION_SPLITTERS,
    fold_plan_frame,
    load_existing_folds,
    is_cell_stratified_splitter,
    ok_mask,
    summarize_cell_stratified_folds,
    INNER_VAL_MODES,  # 2026.11.17
    inner_val_split,  # 2026.11.17
)

__all__ = [
    "CELL_STRAT_ABLATION_TAG",
    "CELL_STRAT_RESULT_DIR",
    "CELL_STRATIFIED_SPLITTERS",
    "PREASSIGNED_SPLITTERS",
    "assign_cell_stratified_folds",
    "assign_section_folds",
    "assign_stratified_section_folds",
    "fold_balance_score",
    "fold_balance_table",
    "is_stratified_section_splitter",
    "section_label_counts",
    "splitter_family",
    "STRATIFIED_SECTION_SPLITTERS",
    "cache_path_for",
    "fold_plan_frame",
    "load_existing_folds",
    "is_cell_stratified_splitter",
    "ok_mask",
    "summarize_cell_stratified_folds",
    "INNER_VAL_MODES",
    "inner_val_split",
]
