#!/usr/bin/env python3
"""v1 (HE-realistic) mapping used by ``build_pancancer_celltype.py``.

Writes the ``celltype`` sheet (training ontology) inside ``PanCancerCellType.xlsx``.
The fine synonym inventory is ``celltype_fine``.
Do not emit a separate workbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_pancancer_celltype as v0

L1_ORDER = ["Epithelial", "Immune", "Stromal", "Endothelial", "Neural"]
L12_ORDER = [
    "Tumor",
    "Epithelial",
    "T_cell",
    "B_Plasma",
    "Myeloid",
    "Stromal",
    "Endothelial",
    "Neural",
]
L12_PARENT = {
    "Tumor": "Epithelial",
    "Epithelial": "Epithelial",
    "T_cell": "Immune",
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Endothelial",
    "Neural": "Neural",
}

# Native source label → (L2, L12). L1 follows L12_PARENT.
#
# Pan-cancer L2 core = BRCA ∪ HCC ∪ PDAC ∪ GIST ∪ GBM (17 classes).
# Lung labels that match a core class are merged there (airway → Epithelial,
# homeostatic FBs → Fibroblast, immune / endothelial synonyms, etc.).
# Lung-only biology is added as 3 coarse extras — not native-fine (no AT1 vs AT2,
# no RASC vs KRT5, no adventitial vs alveolar FB).
# Profibrotic macs (M2, SPP1, IFN, Monocytes/MDMs) → Macrophage_activated.

CORE_L2 = frozenset({
    "Tumor",
    "DCIS",
    "Epithelial",
    "Myoepithelial",
    "CD4_T",
    "CD8_T",
    "T_cell",
    "B_cell",
    "Plasma",
    "DC",
    "Macrophage",
    "Macrophage_activated",
    "Neutrophil",
    "Fibroblast",
    "Stromal",
    "Endothelial",
    "Neural",
})
LUNG_UNIQUE_L2 = frozenset({"Alveolar", "Injury_epithelial", "Myofibroblast"})
NATIVE_TO_V1: dict[str, tuple[str, str]] = {
    # ---- tumor ----
    "Invasive_Tumor": ("Tumor", "Tumor"),
    "Prolif_Invasive_Tumor": ("Tumor", "Tumor"),
    "DCIS_1": ("DCIS", "Tumor"),
    "DCIS_2": ("DCIS", "Tumor"),
    "T_Cell_&_Tumor_Hybrid": ("Tumor", "Tumor"),
    "Epithelium (INOS+)": ("Tumor", "Tumor"),
    "Epithelium (INOS-)": ("Tumor", "Tumor"),
    "AC-like": ("Tumor", "Tumor"),
    "MES-like": ("Tumor", "Tumor"),
    "OC-like": ("Tumor", "Tumor"),
    "NPC-like": ("Tumor", "Tumor"),
    "G1S": ("Tumor", "Tumor"),
    "G2M": ("Tumor", "Tumor"),
    # ---- epithelium (non-malignant / mixed / airway) ----
    "Epithelial cells": ("Epithelial", "Epithelial"),
    "Basal": ("Epithelial", "Epithelial"),
    "Secretory": ("Epithelial", "Epithelial"),
    "Goblet": ("Epithelial", "Epithelial"),
    "Multiciliated": ("Epithelial", "Epithelial"),
    "PNEC": ("Epithelial", "Epithelial"),
    "Proliferating Airway": ("Epithelial", "Epithelial"),
    "Mesothelial": ("Epithelial", "Epithelial"),
    # Lung-unique extras (coarse): alveolar parenchyma vs injury / failed regeneration
    "AT1": ("Alveolar", "Epithelial"),
    "AT2": ("Alveolar", "Epithelial"),
    "Proliferating AT2": ("Alveolar", "Epithelial"),
    "KRT5-/KRT17+": ("Injury_epithelial", "Epithelial"),
    "RASC": ("Injury_epithelial", "Epithelial"),
    "Transitional AT2": ("Injury_epithelial", "Epithelial"),
    "Myoepi_ACTA2+": ("Myoepithelial", "Epithelial"),
    "Myoepi_KRT15+": ("Myoepithelial", "Epithelial"),
    # ---- T / B / DC (TLS) ----
    "CD4 T cells": ("CD4_T", "T_cell"),
    "CD4+_T_Cells": ("CD4_T", "T_cell"),
    "CD4+ T-cells": ("CD4_T", "T_cell"),
    "Helper T cells": ("CD4_T", "T_cell"),
    "Tregs": ("CD4_T", "T_cell"),
    "CD8 T cells": ("CD8_T", "T_cell"),
    "CD8+_T_Cells": ("CD8_T", "T_cell"),
    "CD8+ T-cells": ("CD8_T", "T_cell"),
    "Cytotoxic T cells": ("CD8_T", "T_cell"),
    "T cells": ("T_cell", "T_cell"),
    "INFg+": ("T_cell", "T_cell"),
    "Proliferating T-cells": ("T_cell", "T_cell"),
    "Lymphocyte": ("T_cell", "T_cell"),
    "NK/NKT": ("T_cell", "T_cell"),
    "Proliferating NK/NKT": ("T_cell", "T_cell"),
    "B cells": ("B_cell", "B_Plasma"),
    "B_Cells": ("B_cell", "B_Plasma"),
    "Proliferating B cells": ("B_cell", "B_Plasma"),
    "Plasma": ("Plasma", "B_Plasma"),
    "Plasma cells": ("Plasma", "B_Plasma"),
    "Dendritic cells": ("DC", "Myeloid"),
    "DCs": ("DC", "Myeloid"),
    "cDCs": ("DC", "Myeloid"),
    "Migratory DCs": ("DC", "Myeloid"),
    "IRF7+_DCs": ("DC", "Myeloid"),
    "LAMP3+_DCs": ("DC", "Myeloid"),
    "pDCs": ("DC", "Myeloid"),
    "Langerhans cells": ("DC", "Myeloid"),
    # ---- myeloid ----
    "Macrophages": ("Macrophage", "Myeloid"),
    "Macrophages_1": ("Macrophage", "Myeloid"),
    "Macrophages_2": ("Macrophage", "Myeloid"),
    "Alveolar Macrophages": ("Macrophage", "Myeloid"),
    "Interstitial Macrophages": ("Macrophage", "Myeloid"),
    "Mac_Tmr": ("Macrophage", "Myeloid"),
    "Mac_other": ("Macrophage", "Myeloid"),
    "Mac_SEPP1": ("Macrophage", "Myeloid"),
    "Proliferating Myeloid": ("Macrophage", "Myeloid"),
    "Monocytes": ("Macrophage", "Myeloid"),
    "Macrophages M2-like": ("Macrophage_activated", "Myeloid"),
    "SPP1+ Macrophages": ("Macrophage_activated", "Myeloid"),
    "Mac_SPP1": ("Macrophage_activated", "Myeloid"),
    "Macrophages - IFN-activated": ("Macrophage_activated", "Myeloid"),
    "Monocytes/MDMs": ("Macrophage_activated", "Myeloid"),
    "Neutrophils": ("Neutrophil", "Myeloid"),
    "Neutrophils\t": ("Neutrophil", "Myeloid"),
    "Basophils": ("Neutrophil", "Myeloid"),
    "Mast": ("Neutrophil", "Myeloid"),
    "Mast_Cells": ("Neutrophil", "Myeloid"),
    # ---- stromal ----
    "Fibroblasts": ("Fibroblast", "Stromal"),
    "Alveolar FBs": ("Fibroblast", "Stromal"),
    "Adventitial FBs": ("Fibroblast", "Stromal"),
    "Subpleural FBs": ("Fibroblast", "Stromal"),
    "Myofibroblasts": ("Myofibroblast", "Stromal"),
    "Activated Fibrotic FBs": ("Myofibroblast", "Stromal"),
    "Inflammatory FBs": ("Myofibroblast", "Stromal"),
    "Proliferating FBs": ("Myofibroblast", "Stromal"),
    "CAF": ("Stromal", "Stromal"),
    "Stromal": ("Stromal", "Stromal"),
    "Stromal cells": ("Stromal", "Stromal"),
    "Stromal_&_T_Cell_Hybrid": ("Stromal", "Stromal"),
    "Collagen_fibrils": ("Stromal", "Stromal"),
    "SMCs/Pericytes": ("Stromal", "Stromal"),
    "Perivascular-Like": ("Stromal", "Stromal"),
    # ---- endothelial ----
    "Endothelial": ("Endothelial", "Endothelial"),
    "Endothelial cells": ("Endothelial", "Endothelial"),
    "Endothelial cells\t": ("Endothelial", "Endothelial"),
    "Arteriole": ("Endothelial", "Endothelial"),
    "Capillary": ("Endothelial", "Endothelial"),
    "Venous": ("Endothelial", "Endothelial"),
    "Lymphatic": ("Endothelial", "Endothelial"),
    "Lymphatic Endothelial cells": ("Endothelial", "Endothelial"),
    "Vascular": ("Endothelial", "Endothelial"),
    "lowQ_vas": ("Endothelial", "Endothelial"),
    # ---- neural ----
    "Oligodendrocyte": ("Neural", "Neural"),
    # ---- CRC Flex / RCTD Label1 (map onto existing 20 L2; no new heads) ----
    "Tumor I": ("Tumor", "Tumor"),
    "Tumor II": ("Tumor", "Tumor"),
    "Tumor III": ("Tumor", "Tumor"),
    "Tumor IV": ("Tumor", "Tumor"),
    "Tumor V": ("Tumor", "Tumor"),
    "Enterocyte": ("Epithelial", "Epithelial"),
    "Tuft": ("Epithelial", "Epithelial"),
    "Neuroendocrine": ("Epithelial", "Epithelial"),
    "Enteric Glial": ("Neural", "Neural"),
    "Enteric glial": ("Neural", "Neural"),
    "Adipocyte": ("Stromal", "Stromal"),
    "Fibroblast": ("Fibroblast", "Stromal"),
    "Myofibroblast": ("Myofibroblast", "Stromal"),
    "Pericytes": ("Stromal", "Stromal"),
    "Proliferating Fibroblast": ("Myofibroblast", "Stromal"),
    "Proliferating fibroblast": ("Myofibroblast", "Stromal"),
    "Vascular Fibroblast": ("Fibroblast", "Stromal"),
    "Vascular fibroblast": ("Fibroblast", "Stromal"),
    "Epithelial": ("Stromal", "Stromal"),
    "SM Stress Response": ("Stromal", "Stromal"),
    "SM stress response": ("Stromal", "Stromal"),
    "Smooth Muscle": ("Stromal", "Stromal"),
    "Smooth muscle": ("Stromal", "Stromal"),
    "Unknown III (SM)": ("Stromal", "Stromal"),
    "vSM": ("Stromal", "Stromal"),
    "Macrophage": ("Macrophage", "Myeloid"),
    "Neutrophil": ("Neutrophil", "Myeloid"),
    "Proliferating Macrophages": ("Macrophage", "Myeloid"),
    "Proliferating macrophages": ("Macrophage", "Myeloid"),
    "cDC I": ("DC", "Myeloid"),
    "mRegDC": ("DC", "Myeloid"),
    "pDC": ("DC", "Myeloid"),
    "CD4 T cell": ("CD4_T", "T_cell"),
    "CD4⁺ T cell": ("CD4_T", "T_cell"),
    "CD8 T cell": ("CD8_T", "T_cell"),
    "CD8⁺ T cell": ("CD8_T", "T_cell"),
    "NK": ("T_cell", "T_cell"),
    "Mature B": ("B_cell", "B_Plasma"),
    "Memory B": ("B_cell", "B_Plasma"),
    "Proliferating Immune II": ("B_cell", "B_Plasma"),
    "Proliferating immune II": ("B_cell", "B_Plasma"),
    "Lymphatic Endothelial": ("Endothelial", "Endothelial"),
    "Lymphatic endothelial": ("Endothelial", "Endothelial"),
    "Unlabeled": ("Other", "Stromal"),
    "Unknown": ("Other", "Stromal"),
    "Stroma Uncharacterized": ("Other", "Stromal"),
    "Other": ("Other", "Stromal"),
    "LowQ": ("Other", "Stromal"),
}

L2_NOTES = {
    "Tumor": "Invasive / cycling malignant parenchyma (BRCA invasive, HCC INOS±, GBM states, CRC Tumor I–V). Reserved for Prostate tumor.",
    "DCIS": "BRCA DCIS (DCIS_1 + DCIS_2). One class: the 1/2 split is BRCA-only and not needed pan-cancer.",
    "CD4_T": "Helper / CD4 T (incl. Treg). Split from CD8 because nuclear size and helper vs cytotoxic cues are weakly visible on H&E.",
    "CD8_T": "Cytotoxic / CD8 T.",
    "Epithelial": "Residual / airway / mixed epithelium (PDAC, GIST, lung secretory/basal/goblet/mesothelial). CRC Enterocyte / Tuft / Goblet / neuroendocrine map here.",
    "Alveolar": "Lung-unique extra (coarse): AT1 + AT2 + proliferating AT2. FRI/ARI denominator. Not in the 5-cancer union.",
    "Injury_epithelial": "Lung-unique extra (coarse): KRT5-/KRT17+ + RASC + transitional AT2. FRI injury numerator. Not AT1 vs RASC vs KRT5 as separate heads.",
    "Myoepithelial": "Breast myoepithelium (WSI-visible as a layer; hard on a 16 px patch).",
    "T_cell": "Unspecified T / NK / GBM Lymphocyte / proliferating T / INFg+. Use when the source does not split CD4 vs CD8.",
    "B_cell": "TLS B compartment (incl. proliferating B).",
    "Plasma": "Clock-face plasma cells; kept because they are often H&E-visible and sit in TLS.",
    "DC": "TLS third member. All DC / pDC / Langerhans collapsed.",
    "Macrophage": "Generic macrophages + GIST monocytes. TNI myeloid = this + Macrophage_activated.",
    "Macrophage_activated": "TNI M2 / FRI profibrotic macs: M2-like, SPP1+, IFN-activated, Monocytes/MDMs.",
    "Neutrophil": "Granulocytes (neutrophil + basophil + mast). Mast granules are H&E-visible but too rare to keep a head.",
    "Fibroblast": "Quiescent / tissue fibroblasts (lung alveolar/adventitial/subpleural FBs; CODEX Fibroblasts).",
    "Myofibroblast": "Lung extra (coarse): myofibroblasts + activated/inflammatory/proliferating FBs. Also CRC Myofibroblast / proliferating fibroblast. ARI + FRI fibrotic-stromal numerator.",
    "Stromal": "CAF, collagen, GIST stromal/tumor, SMC/pericyte, CRC smooth muscle / vSM / adipocyte. GIST tumor stays here (mesenchymal).",
    "Endothelial": "Blood + lymphatic + GBM Vascular. TNI endothelium.",
    "Neural": "Oligodendrocyte + CRC enteric glia. Reserved Nerve for prostate / PDAC.",
    "Other": "Dropped at train time.",
}

# Reserved organs reuse existing L2 (do not add new classes).
RESERVED_ON_EXISTING = {
    "Tumor": "Prostate; extra HCC",
    "Epithelial": "Prostate (Luminal)",
    "Myoepithelial": "Prostate basal",
    "Stromal": "Prostate smooth muscle; GIST tumor",
    "Neural": "Prostate / PDAC nerve",
}


def _strip(value) -> str:
    return v0._strip(value)


def unified_of(native_l2: str) -> tuple[str, str, str]:
    key = _strip(native_l2)
    if key not in NATIVE_TO_V1:
        raise KeyError(f"No v1 mapping for native L2 {key!r}")
    l2, l12 = NATIVE_TO_V1[key]
    return L12_PARENT[l12], l12, l2


def attach_v1_columns(df: pd.DataFrame, native_l2_col: str) -> pd.DataFrame:
    """Add ``L1_v1`` / ``L12_v1`` / ``L2_v1`` / ``trainable_v1`` beside v0 columns."""
    out = df.copy()
    mapped = out[native_l2_col].map(
        lambda x: unified_of(x) if _strip(x) else (pd.NA, pd.NA, pd.NA)
    )
    out["L1_v1"] = [t[0] for t in mapped]
    out["L12_v1"] = [t[1] for t in mapped]
    out["L2_v1"] = [t[2] for t in mapped]
    out["trainable_v1"] = out[native_l2_col].map(
        lambda x: "N" if _strip(x) in v0.DROPPED_NATIVE else "Y"
    )
    return out


def build_celltype_sheet(
    aliases_by_l2: dict[str, list[str]],
    present_by_l2: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for native, (l2, l12) in NATIVE_TO_V1.items():
        if l2 == "Other" or l2 in seen:
            continue
        seen.add(l2)
        aliases = sorted(set(aliases_by_l2.get(l2, [])))
        rows.append(
            {
                "L1": L12_PARENT[l12],
                "L12": l12,
                "L2": l2,
                "L2_aliases": "; ".join(aliases),
                "present_in": "; ".join(present_by_l2.get(l2, [])),
                "l2_scope": "lung_unique" if l2 in LUNG_UNIQUE_L2 else "core_5union",
                "reserved_for": RESERVED_ON_EXISTING.get(l2, ""),
                "trainable": "Y",
                "index_role": {
                    "Tumor": "malignant compartment",
                    "DCIS": "in situ tumor",
                    "CD4_T": "TLS T",
                    "CD8_T": "TLS T",
                    "Epithelial": "normal / residual / airway epithelium",
                    "Alveolar": "FRI/ARI denominator (lung extra)",
                    "Injury_epithelial": "FRI numerator (lung extra)",
                    "Myoepithelial": "breast architecture",
                    "T_cell": "TLS",
                    "B_cell": "TLS",
                    "Plasma": "TLS-adjacent",
                    "DC": "TLS",
                    "Macrophage": "TNI myeloid",
                    "Macrophage_activated": "TNI M2 / FRI profibrotic mac",
                    "Neutrophil": "inflammation",
                    "Fibroblast": "SRI / baseline stroma",
                    "Myofibroblast": "ARI + FRI fibrotic stroma (lung extra)",
                    "Stromal": "SRI / GIST tumor",
                    "Endothelial": "TNI endothelium",
                    "Neural": "glia / reserved nerve",
                }.get(l2, ""),
                "notes": L2_NOTES.get(l2, ""),
            }
        )
    rows.append(
        {
            "L1": "Stromal",
            "L12": "Stromal",
            "L2": "Other",
            "L2_aliases": "Unknown; Unlabeled; Other; Stroma Uncharacterized; LowQ",
            "present_in": "Lung; BRCA; HCC; PDAC; GIST; GBM; CRC",
            "l2_scope": "excluded",
            "reserved_for": "",
            "trainable": "N",
            "index_role": "excluded",
            "notes": L2_NOTES["Other"],
        }
    )
    out = pd.DataFrame(rows)
    out["L1"] = pd.Categorical(out["L1"], L1_ORDER, ordered=True)
    out["L12"] = pd.Categorical(out["L12"], L12_ORDER, ordered=True)
    out = out.sort_values(["L1", "L12", "L2"]).reset_index(drop=True)
    dup = out["L2"][out["L2"].duplicated()].tolist()
    if dup:
        raise ValueError(f"L2 must be unique: {dup}")
    n_train = int((out["trainable"] == "Y").sum())
    train_set = set(out.loc[out["trainable"] == "Y", "L2"].astype(str))
    expected = CORE_L2 | LUNG_UNIQUE_L2
    if train_set != expected:
        raise ValueError(f"v1 L2 {sorted(train_set)} != core∪lung extras {sorted(expected)}")
    if n_train > 24:
        raise ValueError(f"v1 L2 trainable={n_train} exceeds 24")
    return out


def celltype_v1_from_native(
    native_cols: dict[str, tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build the training ``celltype`` table from frames that already have ``L2_v1``."""
    aliases: dict[str, list[str]] = {}
    present: dict[str, list[str]] = {}
    for cancer, (col, df) in native_cols.items():
        for native, l2 in zip(df[col], df["L2_v1"], strict=True):
            if pd.isna(l2):
                continue
            aliases.setdefault(str(l2), [])
            if _strip(native) and _strip(native) != str(l2):
                aliases[str(l2)].append(_strip(native))
            if cancer not in present.setdefault(str(l2), []):
                present[str(l2)].append(cancer)
    return build_celltype_sheet(aliases, present)
