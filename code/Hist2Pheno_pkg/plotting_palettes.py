## 2026.08.13, add plotting_palettes for HCC and Lung dataset, 
##             add pan_organ and scheme_for_pan_organ and resolve_effective_scheme for CODEX ESCC and Xenium Lung dataset
## 2026.08.14, add clinical_group_order(pan_organ=...) for Xenium lung vs CODEX HCC
## 2026.08.14, add CODEX_HCC palettes, clinical group order by pan_organ for CODEX HCC dataset


"""Canonical dataset-aware palette registry and color resolvers.

The public API returns RGBA tuples so plotting callers can use one consistent
representation. Legacy constants retain their original hex/RGBA values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

GRAY_RGBA = (0.5, 0.5, 0.5, 1.0)


##########################################
# 2026.08.13, add CODEX_ESCC palettes
##########################################
# CODEX ESCC palettes copied from ``plot.py``. Historical NCRT-named constants
# remain compatibility APIs because this palette was originally exposed under
# the clinical cohort name.
_LEVEL0_SPATIAL_DISTINCT_RGBA = {
    "Immune": (82/255, 182/255, 174/255, 1.0),
    "Stromal": (240/255, 180/255, 70/255, 1.0),
    "Tumor": (255/255, 248/255, 0/255, 1.0),
    "Epithelial": (114/255, 9/255, 101/255, 1.0),
}
_LEVEL01_SPATIAL_DISTINCT_RGBA = {
    "B": (234/255, 107/255, 168/255, 1.0),
    "T": (47/255, 14/255, 181/255, 1.0),
    "Stromal": (240/255, 180/255, 70/255, 1.0),
    "Myeloid": (82/255, 182/255, 174/255, 1.0),
    "Immune": (82/255, 182/255, 174/255, 1.0),
    "Tumor": (255/255, 248/255, 0/255, 1.0),
    "Epithelial": (114/255, 9/255, 101/255, 1.0),
}
_LEVEL1_SPATIAL_DISTINCT_RGBA = {
    "Epithelial": (114/255, 9/255, 101/255, 1.0),
    "Tumor": (255/255, 248/255, 0/255, 1.0),
    "Stromal": (240/255, 180/255, 70/255, 1.0),
    "Myeloid": (82/255, 182/255, 174/255, 1.0),
    "CD8+T": (47/255, 14/255, 181/255, 1.0),
    "CD4+T": (5/255, 247/255, 15/255, 1.0),
    "B": (234/255, 107/255, 168/255, 1.0),
}
_LEVEL12_SPATIAL_DISTINCT_RGBA = {
    "B_other": (234/255, 107/255, 168/255, 1.0),
    "B_plasma": (196/255, 69/255, 140/255, 1.0),
    "CD4_Tconv": (5/255, 247/255, 15/255, 1.0),
    "CD4_Treg": (10/255, 143/255, 60/255, 1.0),
    "CD8_Other": (47/255, 14/255, 181/255, 1.0),
    "CD8_Effector": (26/255, 188/255, 156/255, 1.0),
    "Stromal_other": (240/255, 180/255, 70/255, 1.0),
    "Macrophage": (82/255, 182/255, 174/255, 1.0),
    "Dendritic": (142/255, 68/255, 173/255, 1.0),
    "Myeloid_other": (230/255, 126/255, 34/255, 1.0),
    "Tumor": (255/255, 248/255, 0/255, 1.0),
    "Epithelial": (114/255, 9/255, 101/255, 1.0),
}
_NCRT_FINE_CELLTYPE_ORDER = [
    "Treg", "Bn", "B_other", "CD4_Tn", "Endothelial", "CAF_other",
    "Bm_switched", "CD8_Tn", "Tfh_CXCL13", "CAF_ap", "pDC", "CD4_Tex",
    "Th17", "Macro_other", "B_proliferating", "CD4_Tcm", "Neutrophil",
    "Tfh_CXCR5", "Tfh_CXCL13_CXCR5", "HEV", "Monocyte", "Muscle&mCAF",
    "CD8_Trm", "CD4_Tem", "CD8_Tpex", "Plasma", "CD8_Tem", "CD8_Teff",
    "DC_mature", "cDC2", "cDC1", "CD8_Trm_ex", "CD8_Tex",
    "non_specific", "Macro_M1", "Low_quality", "Epithelial",
    "Tumor_PDL1pos_MHCIpos", "Tumor_PDL1neg_MHCIpos", "Bm_unswitched",
    "Tumor_PDL1neg_MHCIneg", "Tumor_PDL1pos_MHCIneg",
    "Epithelial", "Stromal", "Myeloid", "CD8+T", "Tumor", "CD4+T", "B",
]

##########################################
# 2026.08.13, add Xenium_lung palettes
##########################################
# Xenium lung palettes copied from ``plot_HEanno_spatial_labels.py``.
Cell_Type_COLORS = {
    "AT1": "#1f77b4", "AT2": "#aec7e8", "Activated Fibrotic FBs": "#ff7f0e",
    "Adventitial FBs": "#ffbb78", "Alveolar FBs": "#2ca02c",
    "Alveolar Macrophages": "#98df8a", "Arteriole": "#d62728",
    "B cells": "#ff9896", "Basal": "#9467bd", "Basophils": "#c5b0d5",
    "CD4+ T-cells": "#8c564b", "CD8+ T-cells": "#c49c94",
    "Capillary": "#e377c2", "Goblet": "#f7b6d2",
    "Inflammatory FBs": "#7f7f7f", "Interstitial Macrophages": "#c7c7c7",
    "KRT5-/KRT17+": "#bcbd22", "Langerhans cells": "#dbdb8d",
    "Lymphatic": "#17becf", "Macrophages - IFN-activated": "#9edae5",
    "Mast": "#393b79", "Mesothelial": "#5254a3", "Migratory DCs": "#6b6ecf",
    "Monocytes/MDMs": "#9c9ede", "Multiciliated": "#637939",
    "Myofibroblasts": "#8ca252", "NK/NKT": "#b5cf6b",
    "Neutrophils": "#cedb9c", "PNEC": "#8c6d31", "Plasma": "#bd9e39",
    "Proliferating AT2": "#e7ba52", "Proliferating Airway": "#e7cb94",
    "Proliferating B cells": "#843c39", "Proliferating FBs": "#ad494a",
    "Proliferating Myeloid": "#d6616b", "Proliferating NK/NKT": "#e7969c",
    "Proliferating T-cells": "#7b4173", "RASC": "#a55194",
    "SMCs/Pericytes": "#ce6dbd", "SPP1+ Macrophages": "#de9ed6",
    "Secretory": "#1f77b4", "Subpleural FBs": "#aec7e8",
    "Transitional AT2": "#ff7f0e", "Tregs": "#ffbb78",
    "Venous": "#2ca02c", "cDCs": "#98df8a", "pDCs": "#d62728",
}
CNICHE_COLORS = {
    "C1": "#003399", "C2": "#666633", "C3": "#CC0033", "C4": "#99CC66",
    "C5": "#9999FF", "C6": "#66CCCC", "C7": "#FF9966", "C8": "#993366",
    "C9": "#996633", "C10": "#000000", "C11": "#66CCFF", "C12": "#CCCC00",
}
TNICHE_COLORS = {
    "T1": "#99FFCC", "T2": "#000000", "T3": "#808000", "T4": "#FFCC99",
    "T5": "#33CC33", "T6": "#993300", "T7": "#003300", "T8": "#0066CC",
    "T9": "#FF99FF", "T10": "#CC0066", "T11": "#330033", "T12": "#99CC00",
}
LINEAGE_COLORS = {
    "Epithelial": "#8103fb", "Immune": "#2adddc",
    "Endothelial": "#d4df8a", "Mesenchymal": "#f80505",
}
SUBLINEAGE_COLORS = {
    "Alveolar": "#FF7F0E", "Airway": "#1F77B4", "Myeloid": "#9467BD",
    "Lymphoid": "#8C564B", "Endothelial": LINEAGE_COLORS["Endothelial"],
    "Mesenchymal": LINEAGE_COLORS["Mesenchymal"],
}
NICHE_PALETTES = {
    "CNiche": CNICHE_COLORS, "TNiche": TNICHE_COLORS,
    "final_lineage": LINEAGE_COLORS, "final_sublineage": SUBLINEAGE_COLORS,
    "final_CT": Cell_Type_COLORS,
}

##########################################
# 2026.08.13, add CODEX_HCC palettes
##########################################
# CODEX HCC palettes copied from ``s4769_plot.py``.
Cell_Type_COLORS_CODEX_hcc_level2 = {
    "Epithelium (INOS+)": "#17becf", "Epithelium (INOS-)": "#1f77b4",
    "Fibroblasts": "#98df8a", "Endothelial cells": "#2ca02c",
    "CD4 T cells": "#ff7f0e", "CD8 T cells": "#ffbb78",
    "Macrophages": "#d62728", "Macrophages M2-like": "#ff9896",
    "Neutrophils": "#e377c2", "B cells": "#f7b6d2",
    "Dendritic cells": "#8c564b", "INFg+": "#c49c94",
}
Cell_Type_COLORS_CODEX_hcc_level1 = {
    "Stromal": "#98df8a", "T cells": "#ff7f0e", "Myeloid": "#d62728",
    "Endothelial": "#2ca02c", "Epithelial": "#1f77b4", "B cells": "#f7b6d2",
}
Cell_Type_COLORS_CODEX_hcc_level0 = {
    "Stromal": "#98df8a", "Immune": "#d62728",
    "Endothelial": "#2ca02c", "Epithelial": "#1f77b4",
}
DEFAULT_CODEX_HCC_EXCLUDED_LABELS = ("Unknown", "Stroma Uncharacterized")
DEFAULT_CELLTYPE_FILTER = DEFAULT_CODEX_HCC_EXCLUDED_LABELS
DEFAULT_CELLTYPE_FILTER_CODEX_hcc = DEFAULT_CODEX_HCC_EXCLUDED_LABELS

##########################################
# 2026.09.03 Xenium BRCA palettes (Janesick supervised + LY hierarchy)
##########################################
Cell_Type_COLORS_Xenium_brca_level2 = {
    "DCIS_1": "#FE664D", "DCIS_2": "#fb34cd",
    "Myoepi_ACTA2+": "#009203", "Myoepi_KRT15+": "#66c102",
    "Invasive_Tumor": "#ff002a", "Prolif_Invasive_Tumor": "#8e0119",
    "T_Cell_&_Tumor_Hybrid": "#CB6035", "Stromal": "#e5e022",
    "Stromal_&_T_Cell_Hybrid": "#C1A029", "CD4+_T_Cells": "#4fa9ff",
    "CD8+_T_Cells": "#1068be", "B_Cells": "#565DFD",
    "Macrophages_1": "#10686f", "Macrophages_2": "#3694a8",
    "IRF7+_DCs": "#9f50f9", "LAMP3+_DCs": "#AB76AE",
    "Mast_Cells": "#999999", "Perivascular-Like": "#515151",
    "Endothelial": "#01257b", "Unlabeled": "#ffa5aa",
    "DCIS 1": "#FE664D", "DCIS 2": "#fb34cd",
    "Myoepi ACTA2+": "#009203", "Myoepi KRT15+": "#66c102",
    "Invasive Tumor": "#ff002a", "Prolif Invasive Tumor": "#8e0119",
    "T Cell & Tumor Hybrid": "#CB6035", "Stromal & T Cell Hybrid": "#C1A029",
    "CD4+ T Cells": "#4fa9ff", "CD8+ T Cells": "#1068be",
    "B Cells": "#565DFD", "Macrophages 1": "#10686f",
    "Macrophages 2": "#3694a8", "IRF7+ DCs": "#9f50f9",
    "LAMP3+ DCs": "#AB76AE", "Mast Cells": "#999999",
}
Cell_Type_COLORS_Xenium_brca_level1 = {
    "DCIS": "#fb34cd", "Invasive tumor": "#ff002a", "Myoepithelial": "#009203",
    "Hybrid": "#CB6035",
    "Tumor–T-cell hybrid": "#CB6035", "Tumor-T-cell hybrid": "#CB6035",
    "Stromal–T-cell hybrid": "#C1A029", "Stromal-T-cell hybrid": "#C1A029",
    "T cells": "#4fa9ff", "B cells": "#565DFD",
    "Myeloid": "#10686f", "Fibroblasts": "#e5e022",
    "Endothelial": "#01257b", "Perivascular": "#515151",
}
##########################################
# 2026.09.03 CODEX GBM (WangLab Visium HD) palettes
##########################################
Cell_Type_COLORS_CODEX_gbm_level2 = {
    "AC-like": "#e41a1c", "MES-like": "#ff7f00", "OC-like": "#a65628",
    "NPC-like": "#f781bf", "G1S": "#e7298a", "G2M": "#984ea3",
    "Mac_Tmr": "#377eb8", "Mac_SPP1": "#4daf4a", "Mac_other": "#a6d854",
    "Mac_SEPP1": "#66c2a5", "Lymphocyte": "#984ea3",
    "Oligodendrocyte": "#ffff33",
    "Vascular": "#377eb8", "CAF": "#fc8d62", "Collagen_fibrils": "#8da0cb",
    "lowQ_vas": "#999999", "Unknown": "#bdbdbd", "LowQ": "#d9d9d9",
}
Cell_Type_COLORS_CODEX_gbm_level1 = {
    "Tumor": "#e41a1c", "Myeloid": "#377eb8", "Lymph": "#984ea3",
    "Oligo": "#ffff33", "Vascular": "#fc8d62",
    "Unknown": "#bdbdbd", "LowQ": "#d9d9d9",
}
# GBM has no coarser layer than cell_type; coarse palette aliases L1.
Cell_Type_COLORS_CODEX_gbm_level0 = dict(Cell_Type_COLORS_CODEX_gbm_level1)
DEFAULT_CODEX_GBM_EXCLUDED_LABELS = ("Unknown", "LowQ")
DEFAULT_CELLTYPE_FILTER_CODEX_gbm = DEFAULT_CODEX_GBM_EXCLUDED_LABELS

Cell_Type_COLORS_Xenium_brca_level0 = {
    "Epithelial": "#ff002a", "Immune": "#4fa9ff",
    "Stromal": "#e5e022", "Endothelial": "#01257b",
}

##########################################
# 2026.08.20, add CODEX_PDAC palettes (s1167 Pancreas TMA)
##########################################
Cell_Type_COLORS_CODEX_pdac_level2 = {
    "Epithelial cells": "#17becf",
    "Fibroblasts": "#98df8a",
    "Cytotoxic T cells": "#d62728",
    "Helper T cells": "#ff7f0e",
    "Tregs": "#ffbb78",
    "Macrophages": "#9467bd",
    "Neutrophils": "#bcbd22",
    "Dendritic cells": "#8c564b",
    "B cells": "#f7b6d2",
    "Plasma cells": "#e377c2",
    "Lymphatic Endothelial cells": "#aec7e8",
}
Cell_Type_COLORS_CODEX_pdac_level1 = {
    "Stromal": "#98df8a",
    "T cells": "#ff7f0e",
    "Myeloid": "#d62728",
    "Endothelial": "#2ca02c",
    "Epithelial": "#1f77b4",
    "B cells": "#f7b6d2",
}
Cell_Type_COLORS_CODEX_pdac_level0 = {
    "Stromal": "#98df8a",
    "Immune": "#d62728",
    "Endothelial": "#2ca02c",
    "Epithelial": "#1f77b4",
}
DEFAULT_CODEX_PDAC_EXCLUDED_LABELS = ("Other", "Unannotated")

##########################################
# 2026.08.20, add CODEX_GIST palettes (s1167 GIST TMA, 550 annotated)
##########################################
Cell_Type_COLORS_CODEX_gist_level2 = {
    **Cell_Type_COLORS_CODEX_pdac_level2,
    "Endothelial cells": "#1f77b4",
    "Stromal cells": "#2ca02c",
    "T cells": "#c44e52",
    "Monocytes": "#c49c94",
    "DCs": "#8c564b",
}
Cell_Type_COLORS_CODEX_gist_level1 = dict(Cell_Type_COLORS_CODEX_pdac_level1)
Cell_Type_COLORS_CODEX_gist_level0 = dict(Cell_Type_COLORS_CODEX_pdac_level0)
DEFAULT_CODEX_GIST_EXCLUDED_LABELS = ("Other", "Unannotated")



def normalize_color_rgba(color: Any, alpha: float | None = None) -> tuple[float, float, float, float]:
    """Normalize any matplotlib-compatible color to an RGBA float tuple."""
    rgba = to_rgba(color, alpha=alpha)
    return tuple(float(x) for x in rgba)


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Legacy Xenium helper for hexadecimal colors."""
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0, float(alpha),
    )


def _tab_colors(*, darken_ncrt: bool = False) -> list[tuple[float, float, float, float]]:
    pools = [plt.cm.tab20(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))]
    if darken_ncrt:
        pools.append(plt.cm.Set1(np.linspace(0, 1, 10)))
    colors = np.vstack(pools)
    if darken_ncrt:
        for i, (r, g, b, a) in enumerate(colors):
            brightness = (r + g + b) / 3.0
            if brightness > 0.75:
                factor = 0.6 if brightness > 0.85 else 0.75
                colors[i] = (r * factor, g * factor, b * factor, a)
    return [normalize_color_rgba(color) for color in colors]


def build_ncrt_spatial_color_map(celltype_col: str = "celltype") -> dict[str, tuple[float, float, float, float]]:
    """Build the legacy NCRT-named map for the canonical ``codex_escc`` scheme."""
    tier = normalize_tier(celltype_col, dataset="codex_escc")
    if tier == "coarse":
        return dict(_LEVEL0_SPATIAL_DISTINCT_RGBA)
    if tier == "lineage_bucket":
        return dict(_LEVEL01_SPATIAL_DISTINCT_RGBA)
    if tier == "lineage":
        result = dict(_LEVEL1_SPATIAL_DISTINCT_RGBA)
        result.update(_LEVEL0_SPATIAL_DISTINCT_RGBA)
        return result
    if tier == "intermediate":
        return dict(_LEVEL12_SPATIAL_DISTINCT_RGBA)
    colors = _tab_colors(darken_ncrt=True)
    while len(colors) < len(_NCRT_FINE_CELLTYPE_ORDER):
        colors.append(GRAY_RGBA)
    result = dict(zip(_NCRT_FINE_CELLTYPE_ORDER, colors))
    result.update(_LEVEL1_SPATIAL_DISTINCT_RGBA)
    result.update(_LEVEL12_SPATIAL_DISTINCT_RGBA)
    return result


_DATASET_ALIASES = {
    "codex_escc": "codex_escc", "escc": "codex_escc", "ncrt": "codex_escc",
    "xenium": "xenium_lung", "xenium_lung": "xenium_lung",
    "lung": "xenium_lung", "codex": "codex_hcc", "hcc": "codex_hcc",
    "codex_hcc": "codex_hcc",
    "pdac": "codex_pdac", "codex_pdac": "codex_pdac", "pancreas": "codex_pdac",
    "gist": "codex_gist", "codex_gist": "codex_gist", "gist_tma": "codex_gist",
    "brca": "xenium_brca", "xenium_brca": "xenium_brca", "breast": "xenium_brca",
    "gbm": "codex_gbm", "codex_gbm": "codex_gbm", "glioma": "codex_gbm",
}
_SCHEME_DEFAULTS = {
    "codex_escc": ("codex_escc", None), "ncrt": ("codex_escc", None),
    "xenium_lung_fine": ("xenium_lung", "fine"),
    "xenium_lung_intermediate": ("xenium_lung", "intermediate"),
    "xenium_lung_coarse": ("xenium_lung", "coarse"),
    "xenium_lung_cniche": ("xenium_lung", "cniche"),
    "xenium_lung_tniche": ("xenium_lung", "tniche"),
    "xenium": ("xenium_lung", "auto"), "xenium_auto": ("xenium_lung", "auto"),
    "xenium_ct": ("xenium_lung", "fine"),
    "xenium_lineage": ("xenium_lung", "lineage"),
    "codex_hcc": ("codex_hcc", "auto"), "codex_hcc_fine": ("codex_hcc", "fine"),
    "codex_hcc_intermediate": ("codex_hcc", "intermediate"),
    "codex_hcc_coarse": ("codex_hcc", "coarse"),
    "codex_pdac": ("codex_pdac", "auto"), "codex_pdac_fine": ("codex_pdac", "fine"),
    "codex_pdac_intermediate": ("codex_pdac", "intermediate"),
    "codex_pdac_coarse": ("codex_pdac", "coarse"),
    "codex_gist": ("codex_gist", "auto"), "codex_gist_fine": ("codex_gist", "fine"),
    "codex_gist_intermediate": ("codex_gist", "intermediate"),
    "codex_gist_coarse": ("codex_gist", "coarse"),
    "xenium_brca": ("xenium_brca", "auto"),
    "xenium_brca_fine": ("xenium_brca", "fine"),
    "xenium_brca_intermediate": ("xenium_brca", "intermediate"),
    "xenium_brca_coarse": ("xenium_brca", "coarse"),
    "codex_gbm": ("codex_gbm", "auto"),
    "codex_gbm_fine": ("codex_gbm", "fine"),
    "codex_gbm_intermediate": ("codex_gbm", "intermediate"),
    "codex_gbm_coarse": ("codex_gbm", "coarse"),
}


def is_known_scheme(scheme: str | None) -> bool:
    """Return True when ``scheme`` is a registered palette scheme ID."""
    if scheme is None:
        return False
    return str(scheme).strip().lower() in _SCHEME_DEFAULTS


def normalize_dataset_id(dataset: str | None = None, *, scheme: str | None = None) -> str:
    """Return one of ``codex_escc``, ``xenium_lung``, or ``codex_hcc``.

    ``ncrt`` and ``escc`` remain backward-compatible aliases for
    ``codex_escc``.
    """
    scheme_key = str(scheme).strip().lower() if scheme else ""
    if scheme_key in _SCHEME_DEFAULTS:
        return _SCHEME_DEFAULTS[scheme_key][0]
    key = str(dataset or "codex_escc").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DATASET_ALIASES:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return _DATASET_ALIASES[key]


##########################################
# 2026.08.13, add normalize_pan_organ for CODEX ESCC and Xenium Lung dataset
##########################################
def normalize_pan_organ(pan_organ: str | None) -> str | None:
    """Normalize the optional cross-dataset organ selector.

    This is a small wrapper around :func:`normalize_dataset_id` that preserves
    ``None`` for backward compatibility.
    """
    if pan_organ is None:
        return None
    return normalize_dataset_id(pan_organ)


# s1167 TMA cores are ~2k×2k px with a few thousand cells. The HCC / Xenium
# defaults (s=0.6 maps, s=0.25–0.5 StarDist) look like single pixels on TMA.
TMA_PAN_ORGANS = frozenset({"codex_pdac", "codex_gist"})
DEFAULT_SPATIAL_POINT_SIZE = 0.6
DEFAULT_STARDIST_MAP_POINT_SIZE = 0.25
DEFAULT_SPATIAL_OVERVIEW_POINT_SIZE = 0.5
TMA_SPATIAL_POINT_SIZE = 12.0
TMA_SPATIAL_OVERVIEW_POINT_SIZE = 5.0


def default_spatial_point_size(
    pan_organ: str | None = None,
    *,
    overview: bool = False,
    stardist_map: bool = False,
) -> float:
    """Matplotlib scatter ``s`` for spatial cell maps.

    PDAC / GIST TMA cores use a larger marker than HCC / Xenium whole slides.
    Pass an explicit ``spatial_point_size`` at the call site to override.
    """
    organ = None
    if pan_organ is not None:
        try:
            organ = normalize_pan_organ(pan_organ)
        except ValueError:
            organ = str(pan_organ).strip().lower()
    if organ in TMA_PAN_ORGANS:
        return TMA_SPATIAL_OVERVIEW_POINT_SIZE if overview else TMA_SPATIAL_POINT_SIZE
    if overview:
        return DEFAULT_SPATIAL_OVERVIEW_POINT_SIZE
    if stardist_map:
        return DEFAULT_STARDIST_MAP_POINT_SIZE
    return DEFAULT_SPATIAL_POINT_SIZE


_PAN_ORGAN_HEAD_SCHEMES: dict[str, dict[str, str]] = {
    # CODEX ESCC uses a single scheme; tier is selected via column/tier hints.
    "codex_escc": {
        "l2": "codex_escc",
        "l1": "codex_escc",
        "l12": "codex_escc",
        "l3": "codex_escc",
        "l4": "codex_escc",
    },
    "xenium_lung": {
        "l2": "xenium_lung_fine",
        "l1": "xenium_lung_coarse",
        "l12": "xenium_lung_intermediate",
        "l3": "xenium_lung_CNiche",
        "l4": "xenium_lung_TNiche",
    },
    "codex_hcc": {
        "l2": "codex_hcc_fine",
        "l1": "codex_hcc_coarse",
        "l12": "codex_hcc_intermediate",
        # HCC does not use L3/L4 niche tiers; default to the intermediate palette
        # so callers that reuse the extra-tier loops remain stable.
        "l3": "codex_hcc_intermediate",
        "l4": "codex_hcc_intermediate",
    },
    "codex_pdac": {
        "l2": "codex_pdac_fine",
        "l1": "codex_pdac_coarse",
        "l12": "codex_pdac_intermediate",
        "l3": "codex_pdac_intermediate",
        "l4": "codex_pdac_intermediate",
    },
    "codex_gist": {
        "l2": "codex_gist_fine",
        "l1": "codex_gist_coarse",
        "l12": "codex_gist_intermediate",
        "l3": "codex_gist_intermediate",
        "l4": "codex_gist_intermediate",
    },
    "xenium_brca": {
        "l2": "xenium_brca_fine",
        "l1": "xenium_brca_coarse",
        "l12": "xenium_brca_intermediate",
        "l3": "xenium_brca_intermediate",
        "l4": "xenium_brca_intermediate",
    },
    "codex_gbm": {
        "l2": "codex_gbm_fine",
        "l1": "codex_gbm_coarse",
        "l12": "codex_gbm_intermediate",
        "l3": "codex_gbm_intermediate",
        "l4": "codex_gbm_intermediate",
    },
}


def scheme_for_pan_organ(
    pan_organ: str,
    *,
    head: str = "l2",
) -> str:
    """Return the canonical scheme for a given organ and head/tier key."""
    organ = normalize_dataset_id(pan_organ)
    key = str(head).strip().lower()
    # allow semantic aliases
    key = {
        "fine": "l2",
        "coarse": "l1",
        "lineage": "l1",
        "intermediate": "l12",
        "cniche": "l3",
        "tniche": "l4",
    }.get(key, key)
    if organ not in _PAN_ORGAN_HEAD_SCHEMES:
        raise ValueError(f"Unknown pan_organ {pan_organ!r}")
    if key not in _PAN_ORGAN_HEAD_SCHEMES[organ]:
        raise ValueError(
            f"Unknown head {head!r} for pan_organ {organ!r}; "
            f"choose from {sorted(_PAN_ORGAN_HEAD_SCHEMES[organ])}"
        )
    return _PAN_ORGAN_HEAD_SCHEMES[organ][key]


def resolve_effective_scheme(
    *,
    pan_organ: str | None = None,
    head: str = "l2",
    scheme: str | None = None,
    spatial_color_scheme: str | None = None,
    roc_color_scheme: str | None = None,
    legacy_default_scheme: str | None = None,
) -> str | None:
    """Resolve an effective scheme, honoring explicit overrides first.

    Precedence:
    1) explicit scheme parameters (``scheme`` / ``spatial_color_scheme`` /
       ``roc_color_scheme``) when set to a non-empty value;
    2) ``pan_organ`` + ``head`` mapping;
    3) the provided ``legacy_default_scheme``.
    """

    def _pick(v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    explicit = _pick(scheme) or _pick(spatial_color_scheme) or _pick(roc_color_scheme)
    if explicit is not None and explicit.lower() not in ("auto",):
        return explicit
    if pan_organ is not None:
        return scheme_for_pan_organ(pan_organ, head=head)
    return legacy_default_scheme


##########################################
# 2026.08.14 clinical group order by pan_organ
# 2026.09.01: add GIST clinical information
##########################################
CLINICAL_GROUP_ORDER_BY_ORGAN: dict[str, dict[str, tuple[str, ...]]] = {
    "xenium_lung": {
        "Status": ("Control", "Disease"),
        "Sample_Affect_Pairing": ("Unaffected", "Less_Affected", "More_Affected"),
    },
    "codex_hcc": {
        "Response": ("Responder", "Non_Responder"),
        "diagnosis": ("Pre", "Post"),
    },
    "codex_pdac": {
        "coverslip": ("c001", "c003", "c005", "c007"),
        "SAMPLE_LABEL": ("TA-237", "TA-256", "TA-257", "TA-258"),
    },
    "codex_gist": {
        "coverslip": ("c009", "c011", "c013"),
        "tma_block": ("2", "4"),
        "site_group": ("Stomach", "Small intestine", "Other"),
        "size_group": ("<5 cm", "5–10 cm", "≥10 cm"),
        "mitotic_group": ("≤5 /50 HPF", ">5 /50 HPF"),
        "mutation_group": ("KIT", "Non-KIT", "Unknown"),
        "risk_group": ("Low", "Intermediate", "High"),
        "primary_group": ("Primary", "Metastasis", "Unknown"),
    },
}

CLINICAL_COLUMNS_BY_ORGAN: dict[str, tuple[str, ...]] = {
    "xenium_lung": ("Status", "Sample_Affect_Pairing"),
    "codex_hcc": ("Response", "diagnosis", "treatment"),
    "codex_pdac": ("coverslip", "SAMPLE_LABEL"),
    "codex_gist": ("coverslip", "SAMPLE_LABEL"),
}


def clinical_group_order(
    pan_organ: str | None = "xenium_lung",
    *,
    extra: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return x-axis group order for clinical boxplots.

    Use ``pan_organ='xenium_lung'`` or ``'codex_hcc'`` (aliases via
    :func:`normalize_dataset_id`). Unknown organs return an empty mapping.
    ``extra`` overrides or extends the organ defaults.
    """
    organ = normalize_pan_organ(pan_organ) if pan_organ else "xenium_lung"
    order = dict(CLINICAL_GROUP_ORDER_BY_ORGAN.get(organ or "", {}))
    if extra:
        for key, values in extra.items():
            order[str(key)] = tuple(str(v) for v in values)
    return order


def clinical_columns_for_pan_organ(
    pan_organ: str | None = "xenium_lung",
) -> tuple[str, ...]:
    """Default clinical grouping columns for one ``pan_organ``."""
    organ = normalize_pan_organ(pan_organ) if pan_organ else "xenium_lung"
    return CLINICAL_COLUMNS_BY_ORGAN.get(organ or "", ())
##########################################

def normalize_tier(
    tier: str | None = None,
    *,
    dataset: str = "codex_escc",
    tier_hint: str | None = None,
) -> str:
    """Normalize legacy column/head names to a semantic tier."""
    dataset_id = normalize_dataset_id(dataset)
    raw = str(tier_hint or tier or "fine").strip().lower().replace("-", "_")
    semantic = {
        "fine": "fine", "ct": "fine", "final_ct": "fine", "l2": "fine",
        "intermediate": "intermediate", "sublineage": "intermediate",
        "final_sublineage": "intermediate", "l12": "intermediate",
        "lineage": "lineage", "coarse": "coarse", "lineage_bucket": "lineage_bucket",
        "cniche": "cniche", "c_niche": "cniche", "l3": "cniche",
        "tniche": "tniche", "t_niche": "tniche", "l4": "tniche",
    }
    if raw in ("auto", "none"):
        return "auto"
    if raw in semantic:
        return semantic[raw]
    if dataset_id == "codex_escc":
        return {
            "celltype": "fine", "celltype_level12": "intermediate",
            "celltype_level1": "lineage", "celltype_level0": "coarse",
            "celltype_level01": "lineage_bucket",
        }.get(raw, raw)
    if dataset_id == "xenium_lung":
        return {
            "celltype": "fine", "celltype_level2": "fine",
            "final_lineage": "lineage", "celltype_level1": "lineage",
            "final_sublineage": "intermediate", "cniche": "cniche", "tniche": "tniche",
        }.get(raw, raw)
    # HCC model/head L1 and final_lineage are the four-class coarse target.
    return {
        "celltype": "fine", "celltype_level2": "fine", "l1": "coarse",
        "celltype_level1": "coarse", "final_lineage": "coarse",
        "celltype_level0": "coarse", "celltype_level12": "intermediate",
    }.get(raw, raw)


def _infer_tier(labels: list[str], dataset: str) -> str:
    candidates = {
        "codex_escc": ("coarse", "lineage_bucket", "lineage", "intermediate", "fine"),
        "xenium_lung": ("lineage", "intermediate", "cniche", "tniche", "fine"),
        "codex_hcc": ("coarse", "intermediate", "fine"),
        "codex_pdac": ("coarse", "intermediate", "fine"),
        "codex_gist": ("coarse", "intermediate", "fine"),
        "xenium_brca": ("coarse", "intermediate", "fine"),
        "codex_gbm": ("coarse", "intermediate", "fine"),
    }[dataset]
    label_set = set(labels)
    for candidate in candidates:
        palette = get_palette(dataset, candidate, rgba=False)
        if label_set and label_set <= set(palette):
            return candidate
    return "fine"


def resolve_tier(
    labels: Iterable[Any] = (),
    *,
    dataset: str = "codex_escc",
    tier: str | None = None,
    scheme: str | None = None,
    tier_hint: str | None = None,
) -> str:
    """Resolve an explicit, legacy, scheme-derived, or inferred semantic tier."""
    names = [str(value) for value in labels]
    dataset_id = normalize_dataset_id(dataset, scheme=scheme)
    scheme_key = str(scheme).strip().lower() if scheme else ""
    scheme_tier = _SCHEME_DEFAULTS.get(scheme_key, (dataset_id, None))[1]
    requested = tier_hint or tier or scheme_tier or "fine"
    semantic_tier = normalize_tier(requested, dataset=dataset_id)
    return _infer_tier(names, dataset_id) if semantic_tier == "auto" else semantic_tier


def get_palette(
    dataset: str,
    tier: str = "fine",
    *,
    rgba: bool = True,
) -> dict[str, Any]:
    """Return a copy of a registered palette."""
    dataset_id = normalize_dataset_id(dataset)
    semantic_tier = normalize_tier(tier, dataset=dataset_id)
    if dataset_id == "codex_escc":
        palette = {
            "intermediate": _LEVEL12_SPATIAL_DISTINCT_RGBA,
            "lineage": _LEVEL1_SPATIAL_DISTINCT_RGBA,
            "coarse": _LEVEL0_SPATIAL_DISTINCT_RGBA,
            "lineage_bucket": _LEVEL01_SPATIAL_DISTINCT_RGBA,
        }.get(semantic_tier)
        if semantic_tier == "fine":
            palette = build_ncrt_spatial_color_map("fine")
    elif dataset_id == "xenium_lung":
        palette = {
            "fine": Cell_Type_COLORS, "intermediate": SUBLINEAGE_COLORS,
            "lineage": LINEAGE_COLORS, "coarse": LINEAGE_COLORS,
            "cniche": CNICHE_COLORS, "tniche": TNICHE_COLORS,
        }.get(semantic_tier)
    elif dataset_id == "codex_pdac":
        palette = {
            "fine": Cell_Type_COLORS_CODEX_pdac_level2,
            "intermediate": Cell_Type_COLORS_CODEX_pdac_level1,
            "coarse": Cell_Type_COLORS_CODEX_pdac_level0,
            "lineage": Cell_Type_COLORS_CODEX_pdac_level0,
        }.get(semantic_tier)
    elif dataset_id == "codex_gist":
        palette = {
            "fine": Cell_Type_COLORS_CODEX_gist_level2,
            "intermediate": Cell_Type_COLORS_CODEX_gist_level1,
            "coarse": Cell_Type_COLORS_CODEX_gist_level0,
            "lineage": Cell_Type_COLORS_CODEX_gist_level0,
        }.get(semantic_tier)
    elif dataset_id == "xenium_brca":
        palette = {
            "fine": Cell_Type_COLORS_Xenium_brca_level2,
            "intermediate": Cell_Type_COLORS_Xenium_brca_level1,
            "coarse": Cell_Type_COLORS_Xenium_brca_level0,
            "lineage": Cell_Type_COLORS_Xenium_brca_level0,
        }.get(semantic_tier)
    elif dataset_id == "codex_gbm":
        palette = {
            "fine": Cell_Type_COLORS_CODEX_gbm_level2,
            "intermediate": Cell_Type_COLORS_CODEX_gbm_level1,
            "coarse": Cell_Type_COLORS_CODEX_gbm_level0,
            "lineage": Cell_Type_COLORS_CODEX_gbm_level0,
        }.get(semantic_tier)
    else:
        palette = {
            "fine": Cell_Type_COLORS_CODEX_hcc_level2,
            "intermediate": Cell_Type_COLORS_CODEX_hcc_level1,
            "coarse": Cell_Type_COLORS_CODEX_hcc_level0,
            "lineage": Cell_Type_COLORS_CODEX_hcc_level0,
        }.get(semantic_tier)
    if palette is None:
        raise ValueError(f"Unsupported tier {tier!r} for dataset {dataset_id!r}")
    if rgba:
        return {str(key): normalize_color_rgba(value) for key, value in palette.items()}
    return dict(palette)


def resolve_palette(
    labels: Iterable[Any],
    dataset: str = "codex_escc",
    tier: str | None = None,
    scheme: str | None = None,
    tier_hint: str | None = None,
    canonical_labels: Iterable[Any] | None = None,
    color_overrides: Mapping[Any, Any] | None = None,
    *,
    pan_organ: str | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    """Resolve label colors with deterministic fine-tier fallbacks.

    Unknown labels at non-fine tiers are gray. Explicit overrides always win.
    """
    names = [str(value) for value in labels]
    dataset_id = normalize_dataset_id(dataset, scheme=scheme)
    scheme_key = str(scheme).strip().lower() if scheme else ""
    if pan_organ is not None and scheme_key not in _SCHEME_DEFAULTS:
        dataset_id = normalize_dataset_id(pan_organ)
    semantic_tier = resolve_tier(
        names, dataset=dataset_id, tier=tier, scheme=scheme, tier_hint=tier_hint,
    )
    palette = get_palette(dataset_id, semantic_tier)
    result = {name: palette.get(name, GRAY_RGBA) for name in names}

    if semantic_tier == "fine":
        universe = sorted({str(x) for x in (canonical_labels if canonical_labels is not None else names)})
        missing = [name for name in universe if name not in palette]
        if missing:
            pool = _tab_colors(darken_ncrt=dataset_id == "codex_escc")
            fallback = {name: pool[i % len(pool)] for i, name in enumerate(missing)}
            for name in names:
                if name in fallback:
                    result[name] = fallback[name]
    if color_overrides:
        for key, value in color_overrides.items():
            if value is not None:
                result[str(key)] = normalize_color_rgba(value)
    return result


def resolve_roc_colors(
    labels: Iterable[Any],
    dataset: str = "codex_escc",
    tier: str | None = None,
    scheme: str | None = None,
    tier_hint: str | None = None,
    canonical_labels: Iterable[Any] | None = None,
    color_overrides: Mapping[Any, Any] | None = None,
    *,
    pan_organ: str | None = None,
) -> list[tuple[float, float, float, float]]:
    """Return an RGBA list aligned with ``labels`` for ROC curves."""
    names = [str(value) for value in labels]
    resolved = resolve_palette(
        names, dataset=dataset, tier=tier, scheme=scheme, tier_hint=tier_hint,
        canonical_labels=canonical_labels, color_overrides=color_overrides,
        pan_organ=pan_organ,
    )
    return [resolved.get(name, GRAY_RGBA) for name in names]


def ncrt_roc_color_overrides(class_names, celltype_col: str = "celltype"):
    """Legacy ROC helper for the canonical ``codex_escc`` palette."""
    return resolve_palette(class_names, dataset="codex_escc", tier=celltype_col)


def xenium_lineage_rgba_overrides(labels):
    return {
        str(label): _hex_to_rgba(LINEAGE_COLORS[str(label)])
        for label in labels if str(label) in LINEAGE_COLORS
    }


def xenium_final_ct_rgba_overrides(labels, canonical_labels=None):
    universe = canonical_labels if canonical_labels is not None else labels
    return resolve_palette(
        universe, dataset="xenium_lung", tier="fine", canonical_labels=canonical_labels,
    )


def resolve_xenium_spatial_color_overrides(
    unique_labels, *, tier: str = "auto", canonical_labels=None,
):
    resolved = resolve_palette(
        unique_labels, dataset="xenium_lung", tier=tier,
        canonical_labels=canonical_labels,
    )
    return resolved or None


def niche_palette_rgba_overrides(labels, palette):
    return {
        str(label): normalize_color_rgba(palette[str(label)])
        for label in labels if str(label) in palette
    }


_STARDIST_TIER_PALETTES = {
    "l12": SUBLINEAGE_COLORS, "l3": CNICHE_COLORS, "l4": TNICHE_COLORS,
}
DEFAULT_L2_CLASS_NAMES_CSV = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial"
    / "validation_external_stardist_matched_AUROC_all_samples_class_names.csv"
)


def load_global_l2_class_names(csv_path: str | Path | None = None) -> list[str]:
    """Load the global Xenium training class order used for stable L2 colors."""
    import pandas as pd

    path = Path(csv_path or DEFAULT_L2_CLASS_NAMES_CSV).expanduser()
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    name_col = "final_CT" if "final_CT" in frame.columns else frame.columns[-1]
    if "class_index" in frame.columns:
        frame = frame.sort_values("class_index")
    return [str(value) for value in frame[name_col].tolist()]


def stardist_tier_rgba_overrides(class_names, tier: str):
    tier_key = str(tier).lower()
    if tier_key == "l1":
        overrides = xenium_lineage_rgba_overrides(class_names)
    elif tier_key == "l2":
        canonical = load_global_l2_class_names() or list(class_names)
        overrides = xenium_final_ct_rgba_overrides(class_names, canonical_labels=canonical)
    elif tier_key in _STARDIST_TIER_PALETTES:
        overrides = niche_palette_rgba_overrides(class_names, _STARDIST_TIER_PALETTES[tier_key])
    else:
        return None
    return overrides or None
