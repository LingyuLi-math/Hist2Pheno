#!/usr/bin/env python3
"""Build PanCancerCellType.xlsx from the seven public-cohort hierarchy workbooks.

Run from the Hist2Pheno repo root (or anywhere; paths are absolute to this file)::

    conda activate SeededNTM
    python code/Hist2Pheno_pkg/dataset/build_pancancer_celltype.py

The ``celltype`` sheet is the HE-realistic training ontology (5 / 8 / 20 = 17 core + 3 lung extras).
``celltype_fine`` is the v0 synonym inventory (5 / 11 / 59).
Each cancer keeps its original multi-level table plus v0 and v1 mapping columns.
CODEX ESCC (in-house) is intentionally omitted. CRC maps onto existing v1 L2 (no new heads).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "PanCancerCellType.xlsx"

# ---------------------------------------------------------------------------
# Ground-truth hierarchy files (CODEX ESCC excluded)
# ---------------------------------------------------------------------------
GT = {
    "Lung": {
        "cancer": "Lung (IPF atlas; not carcinoma)",
        "pan_organ": "xenium_lung",
        "path": REPO
        / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Annotation/HE_Annotations/41588_2025_2080_MOESM5_ESM.xlsx",
        "sheet": "Celltype",
        "gt_cells_csv": REPO
        / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases/VUILD107MA/VUILD107MA_cells_partitioned_by_annotation_sample_match_with_pixel.csv",
        "rel": "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Annotation/HE_Annotations/41588_2025_2080_MOESM5_ESM.xlsx",
        "n_he": "25 Complete + 20 Incomplete",
        "gt_cells": 637738,
        "l2_train": 47,
        "l12_train": 6,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Endothelial / Mesenchymal",
        "note": "Five-head in training (also CNiche=12, TNiche=12). Hierarchy reconstructed from MOESM5 Celltype + GT CSV lineage columns.",
    },
    "BRCA": {
        "cancer": "Breast (DCIS + invasive)",
        "pan_organ": "xenium_brca",
        "path": REPO / "data/Xemium/BRCA/Annotation/GSE243275_Barcode_Cell_Type_MatricesLY.xlsx",
        "sheet": "celltype",
        "rel": "data/Xemium/BRCA/Annotation/GSE243275_Barcode_Cell_Type_MatricesLY.xlsx",
        "n_he": "2 (rep1 / rep2)",
        "gt_cells": 264518,
        "l2_train": 16,
        "l12_train": 8,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Stromal / Endothelial",
        "note": "Drop Unlabeled + 3 red L2 (hybrids, Perivascular-Like) at match.",
    },
    "HCC": {
        "cancer": "Hepatocellular carcinoma",
        "pan_organ": "codex_hcc",
        "path": REPO
        / "data/CODEX/HCC/Michael_data_transfer/s4769/HE/s4769_he_mapping_updated_Visium.xlsx",
        "sheet": "Celltype",
        "rel": "data/CODEX/HCC/Michael_data_transfer/s4769/HE/s4769_he_mapping_updated_Visium.xlsx",
        "n_he": "36 annotated (+ 8 StarDist-only)",
        "gt_cells": 1095779,
        "l2_train": 12,
        "l12_train": 6,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Stromal / Endothelial",
        "note": "Drop Unknown + Stroma Uncharacterized. Excel L1/L0 = Hist2Pheno L12/L1.",
    },
    "PDAC": {
        "cancer": "Pancreatic ductal adenocarcinoma",
        "pan_organ": "codex_pdac",
        "path": REPO
        / "data/CODEX/HCC/Michael_data_transfer/s1167/raw_metadata_updated.xlsx",
        "sheet": "Celltype",
        "rel": "data/CODEX/HCC/Michael_data_transfer/s1167/raw_metadata_updated.xlsx",
        "n_he": "278 annotated / 195 unlabeled cores",
        "gt_cells": 1504982,
        "l2_train": 11,
        "l12_train": 6,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Stromal / Endothelial",
        "note": "Filter cohort=='Pancreas TMA'. Drop Other. Copy also at s1167/s1167_celltype_hierarchy.xlsx.",
    },
    "GIST": {
        "cancer": "Gastrointestinal stromal tumor",
        "pan_organ": "codex_gist",
        "path": REPO
        / "data/CODEX/HCC/Michael_data_transfer/s1167/raw_metadata_updated.xlsx",
        "sheet": "Celltype",
        "rel": "data/CODEX/HCC/Michael_data_transfer/s1167/raw_metadata_updated.xlsx",
        "n_he": "550 annotated cores",
        "gt_cells": 1785460,
        "l2_train": 11,
        "l12_train": 6,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Stromal / Endothelial",
        "note": "Same workbook as PDAC; filter cohort=='GIST TMA'. Tumor cells are mesenchymal (Stromal cells).",
    },
    "GBM": {
        "cancer": "IDH-mutant astrocytoma",
        "pan_organ": "codex_gbm",
        "path": REPO
        / "data/CODEX/GBM/WangLab/3_Annotation_Table/GBM_sc_seg_celltypes_hierarchy.xlsx",
        "sheet": "Celltype",
        "rel": "data/CODEX/GBM/WangLab/3_Annotation_Table/GBM_sc_seg_celltypes_hierarchy.xlsx",
        "n_he": "2 (Initial / Recurrent)",
        "gt_cells": 175210,
        "l2_train": 16,
        "l12_train": 9,
        "l1_train": 5,
        "l1_native": "Tumor / Myeloid / Lymph / Oligo / Vascular",
        "note": "Native L12 is spatial_niche SN1–SN9 (not a lineage parent). Unified L12 uses biological parents. Mac_SEPP1 is Myeloid (Rec) and Vascular (Ini).",
    },
    "CRC": {
        "cancer": "Colorectal carcinoma",
        "pan_organ": "xenium_crc",
        "path": REPO / "data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx",
        "sheet": "celltype",
        "rel": "data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx",
        "n_he": "2 labeled (P1/P2); P5 HE only",
        "gt_cells": 364539,
        "l2_train": 38,
        "l12_train": 9,
        "l1_train": 4,
        "l1_native": "Epithelial / Immune / Stromal / Endothelial (Neuronal folded into Stromal)",
        "note": "Visium HD RCTD DeconvolutionLabel1 transferred onto Xenium cells (singlet only), not native Xenium GT. Drop Unlabeled. Flex L12=9 (Tumor / Intestinal Epithelial / Fibroblast / Smooth Muscle / Myeloid / T cells / B cells / Endothelial / Neuronal). P5 HE + alignment exist; annotation sheet not written yet. Oliveira et al., Nat Genet 2025.",
    },
}

# Unified L1 / L12 (stable). New organs map here; do not rename these.
L1_ORDER = ["Epithelial", "Immune", "Stromal", "Endothelial", "Neural"]
L12_PARENT = {
    "Tumor": "Epithelial",
    "Epithelial": "Epithelial",
    "Myoepithelial": "Epithelial",
    "T_cell": "Immune",
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "Lymphoid": "Immune",
    "Stromal": "Stromal",
    "Perivascular": "Stromal",
    "Endothelial": "Endothelial",
    "Neural": "Neural",
}

# Native L2 → (unified L2, L12). L1 is implied by L12.
# Organ-specific epithelial states stay as distinct L2 (different H&E morphology).
# Shared immune / stromal / endothelial names are collapsed to one canonical L2.
NATIVE_TO_UNIFIED: dict[str, tuple[str, str]] = {
    # ---- tumor / epithelial ----
    "Invasive_Tumor": ("Tumor", "Tumor"),
    "Prolif_Invasive_Tumor": ("Tumor_proliferating", "Tumor"),
    "DCIS_1": ("DCIS", "Tumor"),
    "DCIS_2": ("DCIS", "Tumor"),
    "T_Cell_&_Tumor_Hybrid": ("Tumor_T_hybrid", "Tumor"),
    "Epithelium (INOS+)": ("Hepatocyte_INOS_pos", "Tumor"),
    "Epithelium (INOS-)": ("Hepatocyte_INOS_neg", "Tumor"),
    "Epithelial cells": ("Epithelial", "Epithelial"),
    "AC-like": ("AC-like", "Tumor"),
    "MES-like": ("MES-like", "Tumor"),
    "OC-like": ("OC-like", "Tumor"),
    "NPC-like": ("NPC-like", "Tumor"),
    "G1S": ("Tumor_proliferating", "Tumor"),
    "G2M": ("Tumor_proliferating", "Tumor"),
    "AT1": ("AT1", "Epithelial"),
    "AT2": ("AT2", "Epithelial"),
    "Transitional AT2": ("Transitional_AT2", "Epithelial"),
    "Proliferating AT2": ("Proliferating_AT2", "Epithelial"),
    "KRT5-/KRT17+": ("KRT5neg_KRT17pos", "Epithelial"),
    "Basal": ("Basal", "Epithelial"),
    "Secretory": ("Secretory", "Epithelial"),
    "Goblet": ("Goblet", "Epithelial"),
    "Multiciliated": ("Multiciliated", "Epithelial"),
    "PNEC": ("PNEC", "Epithelial"),
    "RASC": ("RASC", "Epithelial"),
    "Proliferating Airway": ("Proliferating_Airway", "Epithelial"),
    "Myoepi_ACTA2+": ("Myoepithelial", "Myoepithelial"),
    "Myoepi_KRT15+": ("Myoepithelial", "Myoepithelial"),
    "Mesothelial": ("Mesothelial", "Epithelial"),
    # ---- T / B / NK ----
    "CD4 T cells": ("CD4_T", "T_cell"),
    "CD4+_T_Cells": ("CD4_T", "T_cell"),
    "CD4+ T-cells": ("CD4_T", "T_cell"),
    "Helper T cells": ("CD4_T", "T_cell"),
    "CD8 T cells": ("CD8_T", "T_cell"),
    "CD8+_T_Cells": ("CD8_T", "T_cell"),
    "CD8+ T-cells": ("CD8_T", "T_cell"),
    "Cytotoxic T cells": ("CD8_T", "T_cell"),
    "Tregs": ("Treg", "T_cell"),
    "T cells": ("T_cell", "T_cell"),
    "INFg+": ("INFg_T", "T_cell"),
    "Proliferating T-cells": ("Proliferating_T", "T_cell"),
    "Lymphocyte": ("Lymphocyte", "Lymphoid"),
    "B cells": ("B_cell", "B_Plasma"),
    "B_Cells": ("B_cell", "B_Plasma"),
    "Proliferating B cells": ("Proliferating_B", "B_Plasma"),
    "Plasma": ("Plasma", "B_Plasma"),
    "Plasma cells": ("Plasma", "B_Plasma"),
    "NK/NKT": ("NK", "Lymphoid"),
    "Proliferating NK/NKT": ("Proliferating_NK", "Lymphoid"),
    # ---- myeloid ----
    "Macrophages": ("Macrophage", "Myeloid"),
    "Macrophages_1": ("Macrophage", "Myeloid"),
    "Macrophages_2": ("Macrophage", "Myeloid"),
    "Alveolar Macrophages": ("Macrophage", "Myeloid"),
    "Interstitial Macrophages": ("Macrophage", "Myeloid"),
    "Macrophages - IFN-activated": ("Macrophage", "Myeloid"),
    "Mac_Tmr": ("Macrophage", "Myeloid"),
    "Mac_other": ("Macrophage", "Myeloid"),
    "Mac_SEPP1": ("Macrophage", "Myeloid"),
    "Macrophages M2-like": ("Macrophage_M2", "Myeloid"),
    "SPP1+ Macrophages": ("Macrophage_SPP1", "Myeloid"),
    "Mac_SPP1": ("Macrophage_SPP1", "Myeloid"),
    "Monocytes": ("Monocyte", "Myeloid"),
    "Monocytes/MDMs": ("Monocyte", "Myeloid"),
    "Neutrophils": ("Neutrophil", "Myeloid"),
    "Neutrophils\t": ("Neutrophil", "Myeloid"),
    "Dendritic cells": ("DC", "Myeloid"),
    "DCs": ("DC", "Myeloid"),
    "cDCs": ("DC", "Myeloid"),
    "Migratory DCs": ("DC", "Myeloid"),
    "IRF7+_DCs": ("DC", "Myeloid"),
    "LAMP3+_DCs": ("DC", "Myeloid"),
    "pDCs": ("pDC", "Myeloid"),
    "Langerhans cells": ("Langerhans", "Myeloid"),
    "Mast": ("Mast", "Myeloid"),
    "Mast_Cells": ("Mast", "Myeloid"),
    "Basophils": ("Basophil", "Myeloid"),
    "Proliferating Myeloid": ("Proliferating_Myeloid", "Myeloid"),
    # ---- stromal ----
    "Fibroblasts": ("Fibroblast", "Stromal"),
    "Activated Fibrotic FBs": ("Fibroblast", "Stromal"),
    "Adventitial FBs": ("Fibroblast", "Stromal"),
    "Alveolar FBs": ("Fibroblast", "Stromal"),
    "Inflammatory FBs": ("Fibroblast", "Stromal"),
    "Subpleural FBs": ("Fibroblast", "Stromal"),
    "Proliferating FBs": ("Fibroblast", "Stromal"),
    "Myofibroblasts": ("Myofibroblast", "Stromal"),
    "CAF": ("CAF", "Stromal"),
    "Stromal": ("Stromal", "Stromal"),
    "Stromal cells": ("Stromal", "Stromal"),
    "Stromal_&_T_Cell_Hybrid": ("Stromal_T_hybrid", "Stromal"),
    "Collagen_fibrils": ("Collagen", "Stromal"),
    "SMCs/Pericytes": ("SMC_Pericyte", "Perivascular"),
    "Perivascular-Like": ("SMC_Pericyte", "Perivascular"),
    # ---- endothelial / vascular ----
    "Endothelial": ("Endothelial", "Endothelial"),
    "Endothelial cells": ("Endothelial", "Endothelial"),
    "Endothelial cells\t": ("Endothelial", "Endothelial"),
    "Arteriole": ("Arteriole", "Endothelial"),
    "Capillary": ("Capillary", "Endothelial"),
    "Venous": ("Venous", "Endothelial"),
    "Lymphatic": ("Lymphatic_endothelial", "Endothelial"),
    "Lymphatic Endothelial cells": ("Lymphatic_endothelial", "Endothelial"),
    "Vascular": ("Endothelial", "Endothelial"),
    "lowQ_vas": ("Endothelial", "Endothelial"),
    # ---- neural ----
    "Oligodendrocyte": ("Oligodendrocyte", "Neural"),
    # ---- CRC Flex / RCTD Label1 (Oliveira Nat Genet 2025) ----
    "Tumor I": ("Tumor", "Tumor"),
    "Tumor II": ("Tumor", "Tumor"),
    "Tumor III": ("Tumor", "Tumor"),
    "Tumor IV": ("Tumor", "Tumor"),
    "Tumor V": ("Tumor", "Tumor"),
    "Enterocyte": ("Enterocyte", "Epithelial"),
    "Tuft": ("Tuft", "Epithelial"),
    "Neuroendocrine": ("Epithelial", "Epithelial"),
    "Enteric Glial": ("Nerve", "Neural"),
    "Enteric glial": ("Nerve", "Neural"),
    "Adipocyte": ("Stromal", "Stromal"),
    "Fibroblast": ("Fibroblast", "Stromal"),
    "Myofibroblast": ("Myofibroblast", "Stromal"),
    "Pericytes": ("SMC_Pericyte", "Perivascular"),
    "Proliferating Fibroblast": ("Fibroblast", "Stromal"),
    "Proliferating fibroblast": ("Fibroblast", "Stromal"),
    "Vascular Fibroblast": ("Fibroblast", "Stromal"),
    "Vascular fibroblast": ("Fibroblast", "Stromal"),
    # Flex cluster named Epithelial sits under Smooth Muscle — not intestinal epithelium.
    "Epithelial": ("Smooth_muscle", "Stromal"),
    "SM Stress Response": ("Smooth_muscle", "Stromal"),
    "SM stress response": ("Smooth_muscle", "Stromal"),
    "Smooth Muscle": ("Smooth_muscle", "Stromal"),
    "Smooth muscle": ("Smooth_muscle", "Stromal"),
    "Unknown III (SM)": ("Smooth_muscle", "Stromal"),
    "vSM": ("Smooth_muscle", "Stromal"),
    "Macrophage": ("Macrophage", "Myeloid"),
    "Neutrophil": ("Neutrophil", "Myeloid"),
    "Proliferating Macrophages": ("Proliferating_Myeloid", "Myeloid"),
    "Proliferating macrophages": ("Proliferating_Myeloid", "Myeloid"),
    "cDC I": ("DC", "Myeloid"),
    "mRegDC": ("DC", "Myeloid"),
    "pDC": ("pDC", "Myeloid"),
    "CD4 T cell": ("CD4_T", "T_cell"),
    "CD4⁺ T cell": ("CD4_T", "T_cell"),
    "CD8 T cell": ("CD8_T", "T_cell"),
    "CD8⁺ T cell": ("CD8_T", "T_cell"),
    "NK": ("NK", "Lymphoid"),
    "Mature B": ("B_cell", "B_Plasma"),
    "Memory B": ("B_cell", "B_Plasma"),
    "Proliferating Immune II": ("Proliferating_B", "B_Plasma"),
    "Proliferating immune II": ("Proliferating_B", "B_Plasma"),
    "Lymphatic Endothelial": ("Lymphatic_endothelial", "Endothelial"),
    "Lymphatic endothelial": ("Lymphatic_endothelial", "Endothelial"),
    # ---- dropped / other ----
    "Unlabeled": ("Other", "Stromal"),
    "Unknown": ("Other", "Stromal"),
    "Stroma Uncharacterized": ("Other", "Stromal"),
    "Other": ("Other", "Stromal"),
    "LowQ": ("Other", "Stromal"),
}

# Extra L2 reserved so Prostate / extra HCC do not force a schema change.
# CRC already fills Enterocyte / Tuft / Nerve / Smooth_muscle / Tumor from Flex natives.
RESERVED_L2 = [
    ("Epithelial", "Tumor", "Tumor", "Y", "Prostate; extra HCC", "Generic malignant epithelial (CRC Tumor I–V already mapped; leftover for prostate / extra HCC)."),
    ("Epithelial", "Epithelial", "Enterocyte", "Y", "", "Intestinal absorptive epithelium (CRC)."),
    ("Epithelial", "Epithelial", "Tuft", "Y", "", "Rare intestinal chemosensory cell (CRC)."),
    ("Epithelial", "Epithelial", "Luminal", "Y", "Prostate", "Prostate luminal epithelium."),
    ("Epithelial", "Myoepithelial", "Basal_myoepithelial", "Y", "Prostate; BRCA", "Prostate basal / breast myoepithelial if not already Myoepithelial."),
    ("Neural", "Neural", "Nerve", "Y", "Prostate; PDAC", "Nerve fascicles / Schwann. CRC enteric glia already mapped here."),
    ("Stromal", "Stromal", "Smooth_muscle", "Y", "Prostate; GIST", "Muscularis / vascular smooth muscle. CRC SM / vSM already mapped here."),
]

DROPPED_NATIVE = {
    "Unlabeled",
    "Unknown",
    "Stroma Uncharacterized",
    "Other",
    "LowQ",
    "T_Cell_&_Tumor_Hybrid",
    "Stromal_&_T_Cell_Hybrid",
    "Perivascular-Like",
}

L2_NOTES = {
    "Tumor": "Shared malignant epithelial / glioma parenchyma (BRCA invasive, CRC Tumor I–V; leftover reserved for Prostate / extra HCC).",
    "Hepatocyte_INOS_pos": "HCC epithelium INOS+. Treated as malignant parenchyma (source has no normal-hepatocyte label).",
    "Hepatocyte_INOS_neg": "HCC epithelium INOS−. Same parent as INOS+.",
    "Epithelial": "Mixed or residual epithelium (PDAC / GIST). No tumor-vs-normal split in those sources.",
    "Stromal": "GIST 'Stromal cells' includes KIT+ tumor (mesenchymal). Do not recode to Epithelial.",
    "Enterocyte": "CRC intestinal absorptive epithelium (Flex Enterocyte).",
    "Tuft": "CRC chemosensory epithelial cell (Flex parent Neuronal; unified parent Epithelial).",
    "Nerve": "CRC enteric glia (Flex Enteric Glial). Reserved also for prostate / PDAC nerve.",
    "Smooth_muscle": "CRC muscularis / vSM / SM stress / Unknown III (SM). Flex cluster 'Epithelial' under Smooth Muscle maps here.",
    "Other": "Dropped at train time. Kept only so new datasets can park Unknown / Other.",
    "Mac_SEPP1": "Ini annotates as Vascular; Rec as Myeloid. Unified parent is Myeloid.",
    "Tumor_T_hybrid": "BRCA red L2; dropped at match. Listed so the native name still maps.",
    "Stromal_T_hybrid": "BRCA red L2; dropped at match.",
}

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
TITLE_FONT = Font(bold=True, name="Calibri", size=14, color="1F4E79")
NOTE_FONT = Font(italic=True, name="Calibri", size=10, color="595959")
WRAP = Alignment(wrap_text=True, vertical="center")
THIN = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)
L1_FILL = {
    "Epithelial": PatternFill("solid", fgColor="FCE4D6"),
    "Immune": PatternFill("solid", fgColor="DDEBF7"),
    "Stromal": PatternFill("solid", fgColor="E2EFDA"),
    "Endothelial": PatternFill("solid", fgColor="FFF2CC"),
    "Neural": PatternFill("solid", fgColor="E2D5F1"),
}
DROP_FILL = PatternFill("solid", fgColor="F2F2F2")
RESERVE_FILL = PatternFill("solid", fgColor="FFF2CC")


def _strip(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\t", "").strip()


def unified_of(native_l2: str) -> tuple[str, str, str]:
    key = _strip(native_l2)
    if key not in NATIVE_TO_UNIFIED:
        raise KeyError(f"No unified mapping for native L2 {key!r}")
    l2, l12 = NATIVE_TO_UNIFIED[key]
    return L12_PARENT[l12], l12, l2


def load_lung() -> pd.DataFrame:
    meta = GT["Lung"]
    raw = pd.read_excel(meta["path"], sheet_name=meta["sheet"])
    raw["final_CT"] = raw["final_CT"].map(_strip)
    gt = pd.read_csv(meta["gt_cells_csv"], usecols=["final_CT", "final_lineage", "final_sublineage"])
    for col in ("final_CT", "final_lineage", "final_sublineage"):
        gt[col] = gt[col].map(_strip)
    hier = gt.drop_duplicates()
    # Types in MOESM5 that are absent from this one sample (Langerhans, Mesothelial).
    extra = [
        {"final_CT": "Langerhans cells", "final_lineage": "Immune", "final_sublineage": "Myeloid"},
        {"final_CT": "Mesothelial", "final_lineage": "Mesenchymal", "final_sublineage": "Mesenchymal"},
    ]
    hier = pd.concat([hier, pd.DataFrame(extra)], ignore_index=True)
    hier = hier.drop_duplicates("final_CT")
    out = raw.merge(hier, on="final_CT", how="left")
    return out


def load_brca() -> pd.DataFrame:
    df = pd.read_excel(GT["BRCA"]["path"], sheet_name=GT["BRCA"]["sheet"])
    for col in df.columns:
        df[col] = df[col].map(lambda v: v if pd.isna(v) else _strip(v))
    return df


def load_hcc() -> pd.DataFrame:
    df = pd.read_excel(GT["HCC"]["path"], sheet_name=GT["HCC"]["sheet"])
    for col in df.columns:
        df[col] = df[col].map(lambda v: v if pd.isna(v) else _strip(v))
    return df


def load_s1167(cohort: str) -> pd.DataFrame:
    df = pd.read_excel(GT["PDAC"]["path"], sheet_name=GT["PDAC"]["sheet"])
    for col in df.columns:
        df[col] = df[col].map(lambda v: v if pd.isna(v) else _strip(v))
    return df[df["cohort"] == cohort].reset_index(drop=True)


def load_gbm() -> pd.DataFrame:
    df = pd.read_excel(GT["GBM"]["path"], sheet_name=GT["GBM"]["sheet"])
    for col in ("celltype_level1", "celltype_level12", "celltype_level2", "note"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: v if pd.isna(v) else _strip(v))
    return df


def load_crc() -> pd.DataFrame:
    df = pd.read_excel(GT["CRC"]["path"], sheet_name=GT["CRC"]["sheet"])
    for col in df.columns:
        df[col] = df[col].map(lambda v: v if pd.isna(v) else _strip(v))
    return df


def attach_unified(df: pd.DataFrame, native_l2_col: str) -> pd.DataFrame:
    out = df.copy()
    mapped = out[native_l2_col].map(lambda x: unified_of(x) if _strip(x) else (pd.NA, pd.NA, pd.NA))
    out["L1"] = [t[0] for t in mapped]
    out["L12"] = [t[1] for t in mapped]
    out["L2"] = [t[2] for t in mapped]
    out["trainable"] = out[native_l2_col].map(lambda x: "N" if _strip(x) in DROPPED_NATIVE else "Y")
    return out


def build_celltype_sheet(aliases_by_l2: dict[str, list[str]], present_by_l2: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    seen = set()
    for native, (l2, l12) in NATIVE_TO_UNIFIED.items():
        if l2 == "Other":
            continue
        key = (L12_PARENT[l12], l12, l2)
        if key in seen:
            continue
        seen.add(key)
        aliases = sorted(set(aliases_by_l2.get(l2, [])))
        dropped_only = bool(aliases) and all(a in DROPPED_NATIVE for a in aliases + ([l2] if l2 in DROPPED_NATIVE else []))
        if l2 in {"Tumor_T_hybrid", "Stromal_T_hybrid", "Other"}:
            dropped_only = True
        rows.append(
            {
                "L1": key[0],
                "L12": l12,
                "L2": l2,
                "L2_aliases": "; ".join(aliases),
                "present_in": "; ".join(present_by_l2.get(l2, [])),
                "reserved_for": "",
                "trainable": "N" if dropped_only else "Y",
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
            "reserved_for": "",
            "trainable": "N",
            "notes": L2_NOTES["Other"],
        }
    )
    for l1, l12, l2, train, reserved, note in RESERVED_L2:
        if any(r["L2"] == l2 and r["L12"] == l12 for r in rows):
            # Tumor already exists from BRCA/GBM; just mark reserved_for.
            for r in rows:
                if r["L2"] == l2 and r["L12"] == l12:
                    extra = reserved
                    if r["reserved_for"]:
                        extra = "; ".join(sorted(set(r["reserved_for"].split("; ") + reserved.split("; "))))
                    r["reserved_for"] = extra
                    if note and not r["notes"]:
                        r["notes"] = note
            continue
        rows.append(
            {
                "L1": l1,
                "L12": l12,
                "L2": l2,
                "L2_aliases": "",
                "present_in": "",
                "reserved_for": reserved,
                "trainable": train,
                "notes": note,
            }
        )
    order_l12 = list(L12_PARENT)
    out = pd.DataFrame(rows)
    out["L1"] = pd.Categorical(out["L1"], L1_ORDER, ordered=True)
    out["L12"] = pd.Categorical(out["L12"], order_l12, ordered=True)
    out = out.sort_values(["L1", "L12", "L2"]).reset_index(drop=True)
    dup = out["L2"][out["L2"].duplicated()].tolist()
    if dup:
        raise ValueError(f"L2 must be unique on celltype (one parent L12): {dup}")
    return out


def _style_header(ws: Worksheet, ncol: int) -> None:
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = THIN
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 28


def _autosize(ws: Worksheet, min_w: int = 12, max_w: int = 42) -> None:
    for idx, col in enumerate(ws.columns, start=1):
        values = ["" if c.value is None else str(c.value) for c in col]
        width = min(max_w, max(min_w, max((len(v) for v in values), default=min_w) + 2))
        ws.column_dimensions[get_column_letter(idx)].width = width


def write_df(
    ws: Worksheet,
    df: pd.DataFrame,
    *,
    l1_col: str | None = "L1",
    trainable_col: str | None = "trainable",
) -> None:
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(r_idx, c_idx, value)
            cell.alignment = WRAP
            cell.border = THIN
            cell.font = Font(name="Calibri", size=10)
    _style_header(ws, df.shape[1])
    cols = list(df.columns)
    l1_i = cols.index(l1_col) + 1 if l1_col and l1_col in cols else None
    tr_i = cols.index(trainable_col) + 1 if trainable_col and trainable_col in cols else None
    reserve_i = cols.index("reserved_for") + 1 if "reserved_for" in cols else None
    for r in range(2, ws.max_row + 1):
        if l1_i:
            fill = L1_FILL.get(str(ws.cell(r, l1_i).value))
            if fill:
                ws.cell(r, l1_i).fill = fill
        if tr_i and str(ws.cell(r, tr_i).value) == "N":
            for c in range(1, ws.max_column + 1):
                if ws.cell(r, c).fill.fgColor is None or ws.cell(r, c).fill.fgColor.rgb == "00000000":
                    ws.cell(r, c).fill = DROP_FILL
        if reserve_i and str(ws.cell(r, reserve_i).value or "").strip() and not str(ws.cell(r, cols.index("present_in") + 1).value or "").strip():
            for c in range(1, ws.max_column + 1):
                ws.cell(r, c).fill = RESERVE_FILL
    _autosize(ws)
    if df.shape[0] and df.shape[1]:
        ref = f"A1:{get_column_letter(df.shape[1])}{df.shape[0] + 1}"
        name = f"T_{ws.title.replace(' ', '_')}"
        table = Table(displayName=name[:31], ref=ref)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        try:
            ws.add_table(table)
        except ValueError:
            pass


def write_notes(ws: Worksheet, lines: list[str], start_row: int) -> None:
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=min(8, ws.max_column or 8))
    cell = ws.cell(start_row, 1, " | ".join(lines))
    cell.font = NOTE_FONT
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[start_row].height = 36


def main() -> None:
    from build_pancancer_celltype_v1 import attach_v1_columns, celltype_v1_from_native

    lung = attach_v1_columns(attach_unified(load_lung(), "final_CT"), "final_CT")
    brca = attach_v1_columns(attach_unified(load_brca(), "celltype_level2"), "celltype_level2")
    hcc = attach_v1_columns(attach_unified(load_hcc(), "celltype_level2"), "celltype_level2")
    pdac = attach_v1_columns(
        attach_unified(load_s1167("Pancreas TMA"), "celltype_level2"), "celltype_level2"
    )
    gist = attach_v1_columns(
        attach_unified(load_s1167("GIST TMA"), "celltype_level2"), "celltype_level2"
    )
    gbm = attach_v1_columns(attach_unified(load_gbm(), "celltype_level2"), "celltype_level2")
    crc = attach_v1_columns(attach_unified(load_crc(), "celltype_level2"), "celltype_level2")

    native_cols = {
        "Lung": ("final_CT", lung),
        "BRCA": ("celltype_level2", brca),
        "HCC": ("celltype_level2", hcc),
        "PDAC": ("celltype_level2", pdac),
        "GIST": ("celltype_level2", gist),
        "GBM": ("celltype_level2", gbm),
        "CRC": ("celltype_level2", crc),
    }
    aliases: dict[str, list[str]] = {}
    present: dict[str, list[str]] = {}
    for cancer, (col, df) in native_cols.items():
        for native, l2 in zip(df[col], df["L2"], strict=True):
            if pd.isna(l2):
                continue
            aliases.setdefault(str(l2), [])
            if _strip(native) and _strip(native) != str(l2):
                aliases[str(l2)].append(_strip(native))
            if cancer not in present.setdefault(str(l2), []):
                present[str(l2)].append(cancer)

    celltype = build_celltype_sheet(aliases, present)
    celltype_v1 = celltype_v1_from_native(native_cols)

    summary_rows = []
    for name, meta in GT.items():
        summary_rows.append(
            {
                "cancer": name,
                "full_name": meta["cancer"],
                "pan_organ": meta["pan_organ"],
                "gt_hierarchy": meta["rel"],
                "sheet": meta["sheet"],
                "HE_sections": meta["n_he"],
                "GT_cells_on_HE": meta["gt_cells"],
                "native_L2_train": meta["l2_train"],
                "native_L12_train": meta["l12_train"],
                "native_L1_train": meta["l1_train"],
                "native_L1_classes": meta["l1_native"],
                "unified_L1": "Epithelial / Immune / Stromal / Endothelial / Neural",
                "v0_L12_n": int(celltype["L12"].nunique()),
                "v0_L2_train": int((celltype["trainable"] == "Y").sum()),
                "v1_L12_n": 8,
                "v1_L2_train": int((celltype_v1["trainable"] == "Y").sum()),
                "note": meta["note"],
            }
        )
    summary = pd.DataFrame(summary_rows)

    legend = pd.DataFrame(
        [
            ("Purpose", "Shared 3-level cell annotation for a pan-cancer Hist2Pheno (Fu/Gerstung PC-CHiP and Kather et al., Nat Cancer 2020: one H&E model across organs)."),
            ("Not included", "CODEX ESCC (in-house NCRT). Do not mix that hierarchy into this workbook."),
            ("celltype sheet", "HE-realistic training ontology: 5 L1 / 8 L12 / 20 L2 = 17-class 5-cancer union (core) + 3 coarse lung extras. Use this for pan-cancer Hist2Pheno."),
            ("celltype_fine sheet", "v0 synonym inventory: 5 L1 / 11 L12 / 65 trainable L2 (CRC fills Enterocyte / Tuft / Nerve / Smooth_muscle). Too fine for a 16 px H&E patch."),
            ("L1", "Same in v0 and v1: Epithelial | Immune | Stromal | Endothelial | Neural."),
            ("L12 v1", "Tumor / Epithelial / T_cell / B_Plasma / Myeloid / Stromal / Endothelial / Neural."),
            ("L2 v1 core", "5-cancer union (17): Tumor, DCIS, Epithelial, Myoepithelial, CD4_T, CD8_T, T_cell, B_cell, Plasma, DC, Macrophage, Macrophage_activated, Neutrophil, Fibroblast, Stromal, Endothelial, Neural."),
            ("L2 v1 lung extras", "Coarse only: Alveolar (AT1+AT2+prolif AT2), Injury_epithelial (KRT5-/KRT17++RASC+transitional AT2), Myofibroblast (myofibroblast+activated/inflammatory/prolif FBs). Do not add AT1 vs AT2 or FB subtypes."),
            ("Lung merge", "Airway / mesothelial → Epithelial; homeostatic FBs → Fibroblast; T/B/DC/mac/neutrophil/endothelial synonyms → matching core class. See l2_scope on celltype."),
            ("Dataset sheets", "Original multi-level table + v0 columns L1/L12/L2 + v1 columns L1_v1/L12_v1/L2_v1. Includes CRC."),
            ("GBM L12", "Native L12 is spatial niche SN1–SN9. Unified L12 ignores SN and uses cell_type / subcluster biology."),
            ("How to add Prostate", "New sheet with native labels. Map onto celltype L2 (Tumor / Epithelial / Neural / Stromal). Do not add new celltype L2 rows."),
            ("CRC", "Sheet CRC: Flex / RCTD Label1 (38 L2). v1 collapses Tumor I–V → Tumor, Enterocyte/Goblet/Tuft/NE → Epithelial, enteric glia → Neural, SM/vSM/CAF/pericyte → Stromal; CD4/CD8/B/Plasma/DC/mac/neutrophil/endothelial reuse core classes. v0 fills reserved Enterocyte / Tuft / Nerve / Smooth_muscle. Labels are transferred Visium HD RCTD, not Xenium GT."),
            ("Head names", "Hist2Pheno train heads: L2=fine=final_CT, L12=intermediate=final_sublineage, L1=coarse=final_lineage."),
            ("Counts", "GT_cells_on_HE from Hist2Pheno_Datasets.xlsx (2026-09-15 snapshot). Lung GT cells = Complete_Cases only. CRC = P1+P2 transferred singlets on HE."),
        ],
        columns=["item", "text"],
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Legend"
    ws["A1"] = "PanCancerCellType — Hist2Pheno shared ontology"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:B1")
    ws["A2"] = (
        "One workbook: sheet celltype = HE-realistic training heads (5/8/20 = 17 core + 3 lung); "
        "sheet celltype_fine = v0 inventory (5/11/65). Seven public cohorts (CRC mapped onto existing v1 L2). "
        "CODEX ESCC omitted."
    )
    ws["A2"].font = NOTE_FONT
    ws.merge_cells("A2:B2")
    for r_idx, row in enumerate(dataframe_to_rows(legend, index=False, header=True), start=4):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(r_idx, c_idx, value)
            cell.alignment = WRAP
            cell.border = THIN
            cell.font = Font(name="Calibri", size=10)
    for cell in ws[4]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 110
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 32
    for r in range(5, 5 + len(legend)):
        ws.row_dimensions[r].height = 36
    ws.freeze_panes = "A5"

    ws = wb.create_sheet("celltype", 1)
    write_df(ws, celltype_v1)
    ws.column_dimensions["D"].width = 70
    ws.column_dimensions["H"].width = 28
    ws.column_dimensions["I"].width = 72

    ws = wb.create_sheet("celltype_fine", 2)
    write_df(ws, celltype)
    ws.column_dimensions["D"].width = 55
    ws.column_dimensions["E"].width = 28
    ws.column_dimensions["H"].width = 70

    ws = wb.create_sheet("summary", 3)
    write_df(ws, summary, l1_col=None, trainable_col=None)
    ws.column_dimensions["D"].width = 72
    ws.column_dimensions["M"].width = 70

    def _src_note(name: str) -> str:
        m = GT[name]
        return f"Source: {m['rel']}  |  sheet '{m['sheet']}'  |  {m['note']}"

    sheets_data = [
        ("Lung", lung, "final_lineage"),
        ("BRCA", brca, "celltype_level1"),
        ("HCC", hcc, "celltype_level0"),
        ("PDAC", pdac, "celltype_level0"),
        ("GIST", gist, "celltype_level0"),
        ("GBM", gbm, "celltype_level1"),
        ("CRC", crc, "celltype_level1"),
    ]
    for name, df, native_l1 in sheets_data:
        ws = wb.create_sheet(name)
        # prepend source note as sheet-level comment row is awkward with tables;
        # put a last column instead.
        out = df.copy()
        out.insert(0, "source_file", GT[name]["rel"])
        out.insert(1, "source_sheet", GT[name]["sheet"])
        write_df(ws, out, l1_col="L1")
        ws.sheet_properties.tabColor = "1F4E79"

    # Put a compact unique-L2 view for GBM on the same book (native SN triples stay on GBM).
    gbm_unique = (
        gbm[
            [
                "celltype_level2",
                "celltype_level1",
                "L1",
                "L12",
                "L2",
                "trainable",
                "L1_v1",
                "L12_v1",
                "L2_v1",
                "trainable_v1",
            ]
        ]
        .drop_duplicates()
        .sort_values(["L1", "L12", "L2", "celltype_level1"])
        .reset_index(drop=True)
    )
    ws = wb.create_sheet("GBM_unique_L2")
    write_df(ws, gbm_unique)
    ws["A1"].comment = None

    wb.save(OUT)
    print(f"Wrote {OUT}")
    train = celltype_v1[celltype_v1["trainable"] == "Y"]
    print(
        f"celltype      L1={train['L1'].nunique()} L12={train['L12'].nunique()} "
        f"L2={train['L2'].nunique()}"
    )
    print(
        f"celltype_fine L1={celltype['L1'].nunique()} L12={celltype['L12'].nunique()} "
        f"L2={celltype['L2'].nunique()}"
    )
    native_key = {
        "Lung": "final_CT",
        "BRCA": "celltype_level2",
        "HCC": "celltype_level2",
        "PDAC": "celltype_level2",
        "GIST": "celltype_level2",
        "GBM": "celltype_level2",
        "CRC": "celltype_level2",
    }
    for name, df, _ in sheets_data:
        print(f"  {name}: {len(df)} rows, native L2={df[native_key[name]].nunique()}")


if __name__ == "__main__":
    main()
