#!/usr/bin/env python3
"""Map native cohort labels onto ``celltype_pancancer`` L1 / L12 / L2 heads."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg" / "dataset"
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from build_pancancer_celltype_v1 import (  # noqa: E402
    L1_ORDER,
    L12_ORDER,
    NATIVE_TO_V1,
    pancancer_keeps_native,
    unified_of,
    _strip,
)

DEFAULT_ONTOLOGY_XLSX = _DATASET_DIR / "PanCancerCellType.xlsx"
DEFAULT_ONTOLOGY_SHEET = "celltype_pancancer"


@lru_cache(maxsize=1)
def load_pancancer_ontology(
    xlsx: Path | str | None = None,
    sheet: str = DEFAULT_ONTOLOGY_SHEET,
) -> pd.DataFrame:
    path = Path(xlsx) if xlsx is not None else DEFAULT_ONTOLOGY_XLSX
    df = pd.read_excel(path, sheet)
    train = df[df["trainable"].astype(str).str.upper() == "Y"].copy()
    train = train[train["L2"].astype(str).str.strip().str.lower() != "other"]
    if len(train) != 20:
        raise ValueError(f"{path} sheet {sheet!r} expected 20 trainable L2, got {len(train)}")
    return train.reset_index(drop=True)


def ontology_class_names(xlsx: Path | str | None = None) -> dict[str, list[str]]:
    """Stable head order from the workbook (L1 5 / L12 8 / L2 20)."""
    onto = load_pancancer_ontology(xlsx)
    l2 = onto["L2"].astype(str).tolist()
    l12 = [x for x in L12_ORDER if x in set(onto["L12"].astype(str))]
    l1 = [x for x in L1_ORDER if x in set(onto["L1"].astype(str))]
    return {"L1": l1, "L12": l12, "L2": l2}


def hierarchy_l2_to_l1(classes: dict[str, list[str]] | None = None) -> "object":
    """L2 index → L1 index for the full ontology (including classes absent in a subset)."""
    import numpy as np

    classes = classes or ontology_class_names()
    onto = load_pancancer_ontology()
    l2_to_l1 = dict(zip(onto["L2"].astype(str), onto["L1"].astype(str)))
    l1_idx = {n: i for i, n in enumerate(classes["L1"])}
    out = np.full(len(classes["L2"]), -1, dtype=np.int64)
    for i, l2 in enumerate(classes["L2"]):
        parent = l2_to_l1.get(str(l2))
        if parent not in l1_idx:
            raise KeyError(f"L2 {l2!r} parent {parent!r} not in L1 {classes['L1']}")
        out[i] = int(l1_idx[parent])
    if np.any(out < 0):
        raise ValueError("incomplete L2→L1 ontology map")
    return out


def native_to_pancancer(native) -> tuple[str, str, str] | None:
    """Return ``(L1, L12, L2)`` if this native is kept for pan-cancer training."""
    key = _strip(native)
    if not key or key not in NATIVE_TO_V1:
        return None
    if not pancancer_keeps_native(key):
        return None
    return unified_of(key)


def map_native_series(native: pd.Series) -> pd.Series:
    """Map a native-label column; works when the column is pandas Categorical.

    ``Series.map`` on a Categorical of tuples/None otherwise builds a MultiIndex
    and crashes on dropped (None) categories.
    """
    return native.astype(object).map(native_to_pancancer)


def remap_obs_frame(
    obs: pd.DataFrame,
    *,
    native_col: str = "final_CT",
) -> pd.DataFrame:
    """Keep rows whose native L2 is in ``celltype_pancancer``; write pan-cancer heads.

    Output columns: ``final_CT`` (L2), ``final_sublineage`` (L12), ``final_lineage`` (L1),
    plus ``native_CT`` (original label).
    """
    if native_col not in obs.columns:
        if native_col != "celltype" and "celltype" in obs.columns:
            native_col = "celltype"
        else:
            raise KeyError(f"obs missing native column {native_col!r}; have {list(obs.columns)}")
    mapped = obs[native_col].astype(object).map(native_to_pancancer)
    keep = mapped.notna()
    out = obs.loc[keep].copy()
    triples = mapped.loc[keep].tolist()
    out["native_CT"] = out[native_col].map(_strip)
    out["final_lineage"] = [t[0] for t in triples]
    out["final_sublineage"] = [t[1] for t in triples]
    out["final_CT"] = [t[2] for t in triples]
    return out
