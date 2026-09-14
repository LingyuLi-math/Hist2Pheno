"""Xenium CRC / Visium HD paths, constants, and analysis helpers.

Used by ``Data_process_HEcelltype_CRC.ipynb``. Twin of ``code/Xenium_brca/brca_paths.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Prefer files that exist in this checkout. The paper parquet clone is optional.
_REPO_MARKERS = (
    "code/Xenium_crc/crc_paths.py",
    "data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx",
    "code/Xenium_crc/HumanColonCancer_VisiumHD/MetaData/P2CRC_Metadata.parquet",
)
META_MARKER = _REPO_MARKERS[0]

PATIENTS = ["P1CRC", "P2CRC", "P5CRC"]
PATIENT_FOLDER = {"P1CRC": "P1_CRC", "P2CRC": "P2_CRC", "P5CRC": "P5_CRC"}
DEFAULT_HIERARCHY_SHEET = "celltype"
DEFAULT_ANNOTATION_XLSX_NAME = "CRC_Barcode_Cell_Type_Matrices.xlsx"
CELLTYPE_SHEET_COLS = ["celltype_level1", "celltype_level12", "celltype_level2"]
CRC_COLUMN_RENAME = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
    "final_lineage": "celltype_level1",
}

# Hist2Pheno Cases / StarDist / UNI (P2 first; twin of brca_paths).
DEFAULT_CRC_ROOT = _REPO_ROOT / "data/Xemium/CRC"
DEFAULT_CASES_ROOT = DEFAULT_CRC_ROOT / "Cases"
DEFAULT_RESULTS_DIR = DEFAULT_CRC_ROOT / "Results"
DEFAULT_STARDIST_ROOT = DEFAULT_CRC_ROOT / "StarDist_Segment"
DEFAULT_HIERARCHY_XLSX = DEFAULT_CRC_ROOT / "Annotation" / DEFAULT_ANNOTATION_XLSX_NAME
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"
CELLS_WITH_PIXEL_SUFFIX = "_cells_with_pixel.csv"
CELLS_MATCHED_STARDIST_SUFFIX = "_cells_matched_by_stardist.csv"
HE_H5AD_SUFFIX = "_matched_features.h5ad"
STARDIST_H5AD_SUFFIX = "_matched_features_stardist.h5ad"
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"
# Add-on HE OME and Visium HD fullres share ~0.274 µm/px (Oliveira alignment).
HE_UM_PER_PX = 0.2738
UNI_SCALE_TO_TARGET_05 = HE_UM_PER_PX / 0.5  # ≈ 0.548; patch_size=16 → ~8 µm

def _crc_sample(patient: str) -> dict[str, object]:
    """P1CRC / P2CRC / P5CRC share the 10x add-on FFPE HE / StarDist naming."""
    folder = PATIENT_FOLDER[patient]
    prefix = f"Xenium_V1_Human_Colon_Cancer_{folder}_Add_on_FFPE"
    return {
        "patient": patient,
        "prefix": prefix,
        "data_dir": DEFAULT_CRC_ROOT / folder,
        "he_tif": DEFAULT_CRC_ROOT / "HE_images" / f"{prefix}_he_image.ome.tif",
        "stardist_key": f"{prefix}_he_image.ome",
        "annotation_sheet": folder,
        "alignment_csv": DEFAULT_CRC_ROOT
        / folder
        / f"{prefix}_he_imagealignment.csv",
        # P2 already extracted with PNG dumps; P1/P5 skip sc_pth_16_16_image.
        "save_patch_images": patient == "P2CRC",
    }


SAMPLES: dict[str, dict[str, object]] = {p: _crc_sample(p) for p in PATIENTS}

# Flex 2025 Level1 (9 classes) = Excel ``celltype_level12``.
L1_ORDER = [
    "Tumor",
    "Intestinal Epithelial",
    "Fibroblast",
    "Smooth Muscle",
    "Myeloid",
    "T cells",
    "B cells",
    "Endothelial",
    "Neuronal",
]
L12_ORDER = L1_ORDER
L1_COLORS = {
    "Tumor": "#c44e52",
    "Intestinal Epithelial": "#dd8452",
    "Fibroblast": "#4c72b0",
    "Smooth Muscle": "#8172b3",
    "Myeloid": "#55a868",
    "T cells": "#937860",
    "B cells": "#da8bc3",
    "Endothelial": "#64b5cd",
    "Neuronal": "#8c8c8c",
}
L12_COLORS = L1_COLORS

# Oliveira et al. Nat Genet 2025 Flex L2 legend (sampled from the paper color key).
# Keys are Flex 2025 / DeconvolutionLabel1 names; paper spellings are aliased.
L2_ORDER = [
    "Adipocyte",
    "CAF",
    "CD4 T cell",
    "CD8 T cell",
    "cDC I",
    "Endothelial",
    "Enteric Glial",
    "Enterocyte",
    "Epithelial",
    "Fibroblast",
    "Goblet",
    "Lymphatic Endothelial",
    "Macrophage",
    "Mast",
    "Mature B",
    "Memory B",
    "mRegDC",
    "Myofibroblast",
    "Neuroendocrine",
    "Neutrophil",
    "NK",
    "pDC",
    "Pericytes",
    "Plasma",
    "Proliferating Fibroblast",
    "Proliferating Immune II",
    "Proliferating Macrophages",
    "SM Stress Response",
    "Smooth Muscle",
    "Tuft",
    "Tumor I",
    "Tumor II",
    "Tumor III",
    "Tumor IV",
    "Tumor V",
    "Unknown III (SM)",
    "Vascular Fibroblast",
    "vSM",
]
L2_COLORS = {
    "Adipocyte": "#415ca9",
    "CAF": "#486a85",
    "CD4 T cell": "#efe584",
    "CD8 T cell": "#d595a6",
    "cDC I": "#b6b7b9",
    "Endothelial": "#812168",
    "Enteric Glial": "#fdc314",
    "Enterocyte": "#934924",
    "Epithelial": "#50b849",
    "Fibroblast": "#cdddb6",
    "Goblet": "#612a7b",
    "Lymphatic Endothelial": "#af1f64",
    "Macrophage": "#749b5a",
    "Mast": "#42bb92",
    "Mature B": "#5db1db",
    "Memory B": "#996728",
    "mRegDC": "#546fb5",
    "Myofibroblast": "#e1b069",
    "Neuroendocrine": "#971f20",
    "Neutrophil": "#847b8c",
    "NK": "#99ca3b",
    "pDC": "#fdd147",
    "Pericytes": "#d58f5c",
    "Plasma": "#cd3d33",
    "Proliferating Fibroblast": "#99ca3b",
    "Proliferating Immune II": "#c85328",
    "Proliferating Macrophages": "#5b665e",
    "SM Stress Response": "#c9992b",
    "Smooth Muscle": "#7b67a4",
    "Tuft": "#45b64a",
    "Tumor I": "#78c269",
    "Tumor II": "#3b1d53",
    "Tumor III": "#565fac",
    "Tumor IV": "#c8992d",
    "Tumor V": "#e8c66f",
    "Unknown III (SM)": "#0099cb",
    "Vascular Fibroblast": "#991a37",
    "vSM": "#bb6437",
    # paper legend spellings
    "CD4⁺ T cell": "#efe584",
    "CD8⁺ T cell": "#d595a6",
    "Enteric glial": "#fdc314",
    "Lymphatic endothelial": "#af1f64",
    "Proliferating fibroblast": "#99ca3b",
    "Proliferating immune II": "#c85328",
    "Proliferating macrophages": "#5b665e",
    "SM stress response": "#c9992b",
    "Smooth muscle": "#7b67a4",
    "Vascular fibroblast": "#991a37",
}

# Excel ``celltype_level1``: 4 coarse parents (same names as BRCA / HCC / PDAC).
# Flex Neuronal (enteric glia / neuroendocrine / tuft) is folded into Stromal.
L1_COARSE_ORDER = [
    "Epithelial",
    "Stromal",
    "Immune",
    "Endothelial",
]
L12_TO_L1 = {
    "Tumor": "Epithelial",
    "Intestinal Epithelial": "Epithelial",
    "Fibroblast": "Stromal",
    "Smooth Muscle": "Stromal",
    "Myeloid": "Immune",
    "T cells": "Immune",
    "B cells": "Immune",
    "Endothelial": "Endothelial",
    "Neuronal": "Stromal",
}
L1_COARSE_COLORS = {
    "Epithelial": "#c44e52",
    "Stromal": "#4c72b0",
    "Immune": "#55a868",
    "Endothelial": "#64b5cd",
}
CLASS_ORDER = ["singlet", "doublet_certain", "doublet_uncertain", "reject"]
CLASS_COLORS = {
    "singlet": "#4c72b0",
    "doublet_certain": "#c44e52",
    "doublet_uncertain": "#dd8452",
    "reject": "#8c8c8c",
}
PERIPH_ORDER = ["Tumor", "50 micron", "Tissue"]
PERIPH_COLORS = {
    "Tumor": "#c44e52",
    "50 micron": "#e6a817",
    "Tissue": "#d0d0d0",
}
UNSUP_COLORS = {**L1_COLORS, "Unknown": "#bdbdbd"}
MAC_COLORS = {
    "Macrophage-SPP1+": "#d62728",
    "Macrophage-SELENOP+": "#1f77b4",
}
GOBLET_COLORS = {
    "Goblet-0": "#1b9e77",
    "Goblet-1": "#d95f02",
    "Goblet-2": "#7570b3",
    "Goblet-3": "#e7298a",
    "Goblet-4": "#66a61e",
    "Goblet-5": "#e6ab02",
    "Goblet-6": "#a6761d",
}
PATIENT_COLORS = {"P1CRC": "#4c72b0", "P2CRC": "#c44e52", "P5CRC": "#55a868"}
MAP_STATUS_ORDER = [
    "singlet",
    "doublet_certain",
    "doublet_uncertain",
    "reject",
    "mapped_no_label",
    "unmapped",
]
XENIUM_COLS = [
    "cell_id",
    "x_centroid",
    "y_centroid",
    "transcript_counts",
    "total_counts",
    "cell_area",
    "nucleus_area",
    "x_centroid_visium_scale",
    "y_centroid_visium_scale",
]


def find_repo_root(start: Path | None = None) -> Path:
    extras = [
        _REPO_ROOT,
        Path("/home/lingyu/ssd2/Python/Hist2Pheno"),
        Path("/nobackup2/users/lingyu/Python/Hist2Pheno"),
    ]
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents, *extras):
        if any((candidate / marker).is_file() for marker in _REPO_MARKERS):
            return candidate
    raise FileNotFoundError("Could not locate the Hist2Pheno repository root")


class CrcPaths:
    """Resolved data paths for one Hist2Pheno checkout."""

    def __init__(self, repo: Path | None = None):
        if repo is not None:
            self.repo = Path(repo)
        elif any((_REPO_ROOT / marker).is_file() for marker in _REPO_MARKERS):
            self.repo = _REPO_ROOT
        else:
            self.repo = find_repo_root()
        self.crc_dir = self.repo / "code" / "Xenium_crc"
        self.visium_hd_crc = self.repo / "data" / "VisiumHD" / "CRC"
        # Paper clone lives with the Space Ranger data (not under code/Xenium_crc).
        self.meta_dir = (
            self.visium_hd_crc / "HumanColonCancer_VisiumHD" / "MetaData"
        )
        _legacy_meta = self.crc_dir / "HumanColonCancer_VisiumHD" / "MetaData"
        if not (self.meta_dir / "SingleCell_MetaData_2025.csv").is_file() and (
            _legacy_meta / "SingleCell_MetaData_2025.csv"
        ).is_file():
            self.meta_dir = _legacy_meta
        self.xenium_root = self.repo / "data" / "Xemium" / "CRC"
        self.xenium_dir = self.xenium_root / "Xenium_Visium_Alignment"
        self.annotation_dir = self.xenium_root / "Annotation"
        self.annotation_xlsx = self.annotation_dir / DEFAULT_ANNOTATION_XLSX_NAME
        self.sc_path = self.meta_dir / "SingleCell_MetaData_2025.csv"
        self.sc_path_2024 = self.meta_dir / "SingleCell_MetaData_2024.csv"
        self.bin_meta_paths = {p: self.meta_dir / f"{p}_Metadata.parquet" for p in PATIENTS}

    def xenium_path(self, name: str) -> Path:
        return self.xenium_dir / f"Xenium_{name.replace('CRC', '')}_cell_info.csv"

    def tissue_pos_path(self, name: str) -> Path:
        return (
            self.visium_hd_crc
            / PATIENT_FOLDER[name]
            / "binned_outputs"
            / "square_008um"
            / "spatial"
            / "tissue_positions.parquet"
        )

    def scalefactors_path(self, name: str) -> Path:
        return self.tissue_pos_path(name).with_name("scalefactors_json.json")

    def patient_sheet_name(self, name: str) -> str:
        """Excel sample sheet, e.g. ``P2CRC`` → ``P2_CRC``."""
        return PATIENT_FOLDER[name]


def sample_ids() -> list[str]:
    return list(SAMPLES)


def sample_config(sample: str) -> dict[str, object]:
    if sample not in SAMPLES:
        raise KeyError(f"Unknown CRC sample {sample!r}; choose from {sample_ids()}")
    return SAMPLES[sample]


def sample_dir(sample: str, cases_root: Path | str | None = None) -> Path:
    root = Path(cases_root) if cases_root is not None else DEFAULT_CASES_ROOT
    return root / sample


def he_tif_path(sample: str) -> Path:
    """Working HE used by StarDist / UNI (P2: add-on OME-TIFF)."""
    return Path(sample_config(sample)["he_tif"])


def he_alignment_csv_path(sample: str) -> Path:
    return Path(sample_config(sample)["alignment_csv"])


def stardist_csv_path(
    sample: str,
    stardist_root: Path | str | None = None,
) -> Path:
    cfg = sample_config(sample)
    key = str(cfg["stardist_key"])
    root = Path(stardist_root) if stardist_root is not None else DEFAULT_STARDIST_ROOT
    return root / key / f"{key}{STARDIST_CSV_SUFFIX}"


def cells_with_pixel_path(sample: str, cases_root: Path | str | None = None) -> Path:
    return sample_dir(sample, cases_root) / f"{sample}{CELLS_WITH_PIXEL_SUFFIX}"


def cells_matched_stardist_path(
    sample: str, cases_root: Path | str | None = None
) -> Path:
    return sample_dir(sample, cases_root) / f"{sample}{CELLS_MATCHED_STARDIST_SUFFIX}"


def list_crc_samples(
    cases_root: Path | str | None = None,
    *,
    require_cells_csv: bool = False,
    require_h5ad: bool = False,
) -> list[str]:
    root = Path(cases_root) if cases_root is not None else DEFAULT_CASES_ROOT
    out = []
    for sample in sample_ids():
        if require_cells_csv and not cells_with_pixel_path(sample, root).is_file():
            continue
        if require_h5ad and not (
            sample_dir(sample, root) / f"{sample}{HE_H5AD_SUFFIX}"
        ).is_file():
            continue
        out.append(sample)
    return out


def majority_l2_to_l1(keep: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, float]]:
    """Majority-vote Flex L2 → L1 map among QCFilter == Keep cells."""
    pair_n = (
        keep.groupby(["Level2", "Level1"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["Level2", "n"], ascending=[True, False])
    )
    maj = pair_n.groupby("Level2", as_index=False).head(1).copy()
    tot = keep.groupby("Level2").size()
    maj["n_total"] = maj["Level2"].map(tot)
    maj["purity"] = maj["n"] / maj["n_total"]
    maj["n_alt_L1"] = maj["Level2"].map(keep.groupby("Level2")["Level1"].nunique())
    maj = maj.sort_values(
        ["Level1", "Level2"],
        key=lambda s: s.map({v: i for i, v in enumerate(L1_ORDER)})
        if s.name == "Level1"
        else s,
    ).reset_index(drop=True)
    l2_to_l1 = dict(zip(maj["Level2"], maj["Level1"]))
    l2_purity = dict(zip(maj["Level2"], maj["purity"]))
    return maj, l2_to_l1, l2_purity


def l12_to_l1(values: pd.Series) -> pd.Series:
    """Map Flex 9-class L12 names onto the 4 coarse ``celltype_level1`` labels."""
    return values.map(L12_TO_L1)


def _sort_celltype_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_k1"] = out["celltype_level1"].map(
        {v: i for i, v in enumerate(L1_COARSE_ORDER)}
    )
    out["_k12"] = out["celltype_level12"].map({v: i for i, v in enumerate(L12_ORDER)})
    out = out.sort_values(["_k1", "_k12", "celltype_level2"], kind="mergesort")
    return out.drop(columns=["_k1", "_k12"]).reset_index(drop=True)


def build_celltype_hierarchy(maj: pd.DataFrame) -> pd.DataFrame:
    """BRCA-style ``celltype`` sheet: coarse L1 / Flex L12 / Flex L2.

    Column order is ``celltype_level1``, ``celltype_level12``, ``celltype_level2``.
    ``celltype_level12`` is the Flex 2025 Level1 (9 classes);
    ``celltype_level1`` collapses those into 4 coarse parents
    (Epithelial / Stromal / Immune / Endothelial; Neuronal → Stromal).
    Appends ``Unlabeled`` with empty parents.
    """
    l12 = maj["Level1"].astype(str)
    labeled = _sort_celltype_sheet(
        pd.DataFrame(
            {
                "celltype_level1": l12_to_l1(l12),
                "celltype_level12": l12,
                "celltype_level2": maj["Level2"].astype(str),
            }
        )
    )
    unlabeled = pd.DataFrame(
        {
            "celltype_level1": [pd.NA],
            "celltype_level12": [pd.NA],
            "celltype_level2": ["Unlabeled"],
        }
    )
    return pd.concat([labeled, unlabeled], ignore_index=True)[CELLTYPE_SHEET_COLS]


def normalize_celltype_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute coarse L1 from L12 and restore ``celltype_level1, L12, L2`` order."""
    out = df.copy()
    needed = {"celltype_level12", "celltype_level2"}
    missing = needed - set(out.columns)
    if missing:
        raise KeyError(f"celltype sheet missing {sorted(missing)}")
    unlabeled = out["celltype_level2"].astype(str).eq("Unlabeled") | out[
        "celltype_level12"
    ].isna()
    out["celltype_level1"] = l12_to_l1(out["celltype_level12"].astype("string"))
    out.loc[unlabeled, "celltype_level1"] = pd.NA
    out.loc[unlabeled, "celltype_level12"] = pd.NA
    labeled = _sort_celltype_sheet(out.loc[~unlabeled, CELLTYPE_SHEET_COLS])
    rest = out.loc[unlabeled, CELLTYPE_SHEET_COLS]
    return pd.concat([labeled, rest], ignore_index=True)


def load_crc_celltype_hierarchy(
    xlsx_path: str | Path | None = None,
    *,
    sheet_name: str = DEFAULT_HIERARCHY_SHEET,
) -> pd.DataFrame:
    """Load L2→L12→L1 from ``CRC_Barcode_Cell_Type_Matrices.xlsx``.

    Normalized like ``load_brca_celltype_hierarchy``:
    ``celltype_level2`` / ``celltype_level1`` (intermediate) /
    ``celltype_level0`` (coarse). Drops ``Unlabeled``.
    """
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_HIERARCHY_XLSX
    df = pd.read_excel(path, sheet_name=sheet_name)
    hist2pheno_cols = ["celltype_level2", "celltype_level12", "celltype_level1"]
    hcc_excel_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    if set(hist2pheno_cols).issubset(df.columns):
        out = df[hist2pheno_cols].copy()
        out = out.rename(
            columns={
                "celltype_level12": "celltype_level1",
                "celltype_level1": "celltype_level0",
            }
        )
    elif set(hcc_excel_cols).issubset(df.columns):
        out = df[hcc_excel_cols].copy()
    else:
        raise KeyError(
            f"{sheet_name!r} needs {hist2pheno_cols} or {hcc_excel_cols}; "
            f"got {list(df.columns)}"
        )
    hierarchy_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    for col in hierarchy_cols:
        out[col] = out[col].map(
            lambda value: str(value).strip() if pd.notna(value) else pd.NA
        )
    out = out.dropna(subset=["celltype_level1", "celltype_level0"])
    out = out.drop_duplicates().reset_index(drop=True)
    duplicated = out.loc[
        out["celltype_level2"].duplicated(keep=False), "celltype_level2"
    ].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate celltype_level2: {sorted(duplicated.tolist())}")
    return out


def attach_l1(df: pd.DataFrame, l2_to_l1: dict[str, str]) -> pd.DataFrame:
    """Map DeconvolutionLabel1 (L2) to Flex majority Level1."""
    out = df.copy()
    out["L1_from_Label1"] = out["DeconvolutionLabel1"].map(l2_to_l1)
    out["unmapped_L2"] = out["DeconvolutionLabel1"].notna() & out["L1_from_Label1"].isna()
    return out


def read_bin_meta(path: Path) -> pd.DataFrame:
    """Load one patient 8 µm parquet (RCTD labels + spatial annotations)."""
    df = pd.read_parquet(path)
    df["tissue"] = pd.to_numeric(df["tissue"], errors="coerce").fillna(0).astype(int)
    return df


def load_bin_meta(name: str, paths: CrcPaths, l2_to_l1: dict[str, str]) -> pd.DataFrame:
    return attach_l1(read_bin_meta(paths.bin_meta_paths[name]), l2_to_l1)


def load_all_bin_meta(paths: CrcPaths, l2_to_l1: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {p: load_bin_meta(p, paths, l2_to_l1) for p in PATIENTS}


def print_label_types(s: pd.Series, name: str) -> pd.Series:
    assigned = s.dropna()
    types = sorted(assigned.unique().tolist())
    vc = assigned.value_counts()
    print(f"\n{name}  (cell type used in this notebook)")
    print(f"  n types (excluding NA) = {len(types)}")
    print(f"  n NA bins              = {int(s.isna().sum()):,}")
    print(f"  types: {types}")
    print(vc.to_string())
    return vc


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    na_label = "unassigned"
    cls = df["DeconvolutionClass"].astype("object")
    cls = cls.where(df["DeconvolutionClass"].notna(), na_label)
    vc = cls.value_counts()
    tab = pd.DataFrame({"n": vc, "pct_all": 100 * vc / n})
    n_asg = int(df["DeconvolutionClass"].notna().sum())
    tab["pct_assigned"] = np.where(
        tab.index == na_label, np.nan, 100 * tab["n"] / max(n_asg, 1)
    )
    order = [c for c in CLASS_ORDER if c in tab.index]
    if na_label in tab.index:
        order.append(na_label)
    return tab.reindex(order)


def patient_report(
    name: str,
    df: pd.DataFrame,
    l2_to_l1: dict[str, str],
    *,
    verbose: bool = True,
) -> dict:
    asg = df[df["DeconvolutionClass"].notna()].copy()
    l2_1 = set(asg["DeconvolutionLabel1"].dropna().unique())
    l1_1 = set(asg["L1_from_Label1"].dropna().unique())
    unmapped = sorted(l2_1 - set(l2_to_l1))
    if verbose:
        print("=" * 72)
        print(f"{name}  bins={len(df):,}  assigned={len(asg):,} ({100 * len(asg) / len(df):.1f}%)")
        print("- class")
        print(class_table(df).round(2).to_string())
        print(f"- unique L2 Label1={len(l2_1)}")
        print(
            f"- unique L1 from Label1={len(l1_1)}  "
            f"{sorted(l1_1, key=lambda x: L1_ORDER.index(x) if x in L1_ORDER else 99)}"
        )
        print(f"- L2 not in Flex mapping: {unmapped or '(none)'}")
        print("- Label1 → L1")
        l1n = asg["L1_from_Label1"].value_counts().reindex(L1_ORDER)
        print((100 * l1n / l1n.sum()).round(2).rename("pct").to_string())
        top = asg["DeconvolutionLabel1"].value_counts().head(8)
        print("- top Label1 L2:", ", ".join(f"{k} ({v:,})" for k, v in top.items()))
    return {
        "patient": name,
        "n_bins": len(df),
        "n_assigned": len(asg),
        "frac_assigned": len(asg) / len(df),
        "l2_label1": l2_1,
        "l1_label1": l1_1,
        "unmapped": unmapped,
        "class_n": asg["DeconvolutionClass"].value_counts(),
        "l2_n": asg["DeconvolutionLabel1"].value_counts(),
        "l1_n": asg["L1_from_Label1"].value_counts(),
        "assigned": asg,
    }


def on_tissue(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["tissue"] == 1]


def periphery_summary(bin_meta: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for p, df in bin_meta.items():
        on = on_tissue(df)
        mac = df["MacrophageSubtype"].value_counts(dropna=True)
        gob = df["GobletSubcluster"].value_counts(dropna=True)
        rows.append(
            {
                "patient": p,
                "on_tissue": len(on),
                "Tumor": int((on["Periphery"] == "Tumor").sum()),
                "50 micron": int((on["Periphery"] == "50 micron").sum()),
                "Tissue": int((on["Periphery"] == "Tissue").sum()),
                "SPP1+": int(mac.get("Macrophage-SPP1+", 0)),
                "SELENOP+": int(mac.get("Macrophage-SELENOP+", 0)),
                "Goblet bins": int(gob.sum()),
            }
        )
    return pd.DataFrame(rows)


def load_xenium_cells(name: str, paths: CrcPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.xenium_path(name), usecols=XENIUM_COLS)
    df["patient"] = name
    return df


def map_xenium_to_bins(
    name: str,
    vis_df: pd.DataFrame,
    paths: CrcPaths,
    xen: pd.DataFrame | None = None,
) -> dict:
    """Nearest on-tissue 8 µm bin in Visium full-res pixels (x→col, y→row)."""
    xen = load_xenium_cells(name, paths) if xen is None else xen
    tp = pd.read_parquet(paths.tissue_pos_path(name))
    sf = json.loads(paths.scalefactors_path(name).read_text())
    mpp = float(sf["microns_per_pixel"])
    bin_px = float(sf["spot_diameter_fullres"])
    on = tp.loc[tp["in_tissue"] == 1]
    tree = cKDTree(np.c_[on["pxl_col_in_fullres"], on["pxl_row_in_fullres"]])
    pts = np.c_[xen["x_centroid_visium_scale"], xen["y_centroid_visium_scale"]]
    dist_px, nn = tree.query(pts, k=1, workers=-1)
    col_ok = xen["x_centroid_visium_scale"].between(
        tp["pxl_col_in_fullres"].min(), tp["pxl_col_in_fullres"].max()
    )
    row_ok = xen["y_centroid_visium_scale"].between(
        tp["pxl_row_in_fullres"].min(), tp["pxl_row_in_fullres"].max()
    )
    in_bbox = col_ok & row_ok
    mapped = dist_px < bin_px
    out = xen.copy()
    out["nearest_barcode"] = on["barcode"].to_numpy()[nn]
    out["dist_px"] = dist_px
    out["dist_um"] = dist_px * mpp
    out["in_visium_bbox"] = in_bbox.to_numpy()
    out["mapped_to_bin"] = mapped
    hit = out.loc[mapped].merge(
        vis_df[
            [
                "barcode",
                "DeconvolutionClass",
                "DeconvolutionLabel1",
                "L1_from_Label1",
                "Periphery",
            ]
        ],
        left_on="nearest_barcode",
        right_on="barcode",
        how="left",
    )
    return {
        "patient": name,
        "xen": out,
        "hit": hit,
        "n_cells": len(out),
        "n_bbox": int(in_bbox.sum()),
        "n_mapped": int(mapped.sum()),
        "median_um_bbox": float(np.median(out.loc[in_bbox, "dist_um"]))
        if in_bbox.any()
        else np.nan,
        "median_um_mapped": float(np.median(out.loc[mapped, "dist_um"]))
        if mapped.any()
        else np.nan,
        "n_bins_hit": int(hit["nearest_barcode"].nunique()) if len(hit) else 0,
        "mpp": mpp,
        "bin_px": bin_px,
        "tp": tp,
    }


def label_xenium_from_visium(xen_mapped: pd.DataFrame, vis_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``map_status`` and ``Label1_from_VisiumHD`` to mapped Xenium cells."""
    out = xen_mapped.merge(
        vis_df[["barcode", "DeconvolutionClass", "DeconvolutionLabel1", "L1_from_Label1"]],
        left_on="nearest_barcode",
        right_on="barcode",
        how="left",
    ).drop(columns=["barcode"])
    mapped = out["mapped_to_bin"]
    cls = out["DeconvolutionClass"]
    status = pd.Series("unmapped", index=out.index, dtype="string")
    status = status.mask(mapped & cls.eq("singlet"), "singlet")
    status = status.mask(mapped & cls.eq("doublet_certain"), "doublet_certain")
    status = status.mask(mapped & cls.eq("doublet_uncertain"), "doublet_uncertain")
    status = status.mask(mapped & cls.eq("reject"), "reject")
    status = status.mask(mapped & cls.isna(), "mapped_no_label")
    out["map_status"] = status
    out["Label1_from_VisiumHD"] = out["DeconvolutionLabel1"].where(
        mapped & out["DeconvolutionLabel1"].notna()
    )
    return out


def map_status_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    vc = df["map_status"].value_counts()
    return (
        pd.DataFrame({"n": vc, "pct": 100 * vc / n})
        .reindex(MAP_STATUS_ORDER)
        .dropna(how="all")
    )


def compare_mapped_vs_visium(hit: pd.DataFrame, vis: pd.DataFrame) -> dict:
    """Cell-weighted L1 / Periphery of mapped Xenium vs Visium HD bins."""
    vis_asg = vis.loc[vis["DeconvolutionClass"].notna()]
    xen_l1 = hit["L1_from_Label1"].value_counts(dropna=True).reindex(L1_ORDER).fillna(0)
    vis_l1 = vis_asg["L1_from_Label1"].value_counts(dropna=True).reindex(L1_ORDER).fillna(0)
    l1_comp = pd.DataFrame(
        {
            "Xenium cells %": 100 * xen_l1 / xen_l1.sum(),
            "Visium bins %": 100 * vis_l1 / vis_l1.sum(),
        }
    )
    obs = np.vstack([xen_l1.to_numpy(), vis_l1.to_numpy()])
    keep = obs.sum(axis=0) > 0
    chi2, pval, dof, _ = chi2_contingency(obs[:, keep])
    xen_per = hit["Periphery"].value_counts(dropna=True).reindex(PERIPH_ORDER).fillna(0)
    vis_per = vis.loc[vis["tissue"] == 1, "Periphery"].value_counts(dropna=True).reindex(
        PERIPH_ORDER
    ).fillna(0)
    per_comp = pd.DataFrame(
        {
            "Xenium cells %": 100 * xen_per / xen_per.sum(),
            "Visium on-tissue bins %": 100 * vis_per / vis_per.sum(),
        }
    )
    chi2p, pp, dofp, _ = chi2_contingency(np.vstack([xen_per.to_numpy(), vis_per.to_numpy()]))
    return {
        "l1": l1_comp,
        "l1_chi2": (chi2, dof, pval),
        "periphery": per_comp,
        "periphery_chi2": (chi2p, dofp, pp),
    }


def overlap_row(m: dict, vis: pd.DataFrame) -> dict:
    n_on = int((vis["tissue"] == 1).sum())
    n_asg = int(vis["DeconvolutionClass"].notna().sum())
    return {
        "patient": m["patient"],
        "Xenium cells": m["n_cells"],
        "in Visium bbox": m["n_bbox"],
        "bbox %": 100 * m["n_bbox"] / m["n_cells"],
        "mapped <1 bin": m["n_mapped"],
        "mapped %": 100 * m["n_mapped"] / m["n_cells"],
        "median µm": m["median_um_mapped"],
        "bins hit": m["n_bins_hit"],
        "Visium on-tissue": n_on,
        "bins hit %": 100 * m["n_bins_hit"] / max(n_on, 1),
        "Visium assigned": n_asg,
    }
