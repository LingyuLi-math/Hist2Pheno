"""CODEX GBM (WangLab Visium HD) sample registry, hierarchy, and case paths."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBM_ROOT = _REPO_ROOT / "data/CODEX/GBM"
DEFAULT_WANGLAB_ROOT = DEFAULT_GBM_ROOT / "WangLab"
DEFAULT_ANNOTATION_ROOT = DEFAULT_WANGLAB_ROOT / "3_Annotation_Table"
DEFAULT_ANNOTATION_DIR = DEFAULT_ANNOTATION_ROOT / "2_Single_Nuclei"
DEFAULT_RESULTS_DIR = DEFAULT_GBM_ROOT / "Results"
DEFAULT_CASES_ROOT = DEFAULT_GBM_ROOT / "Cases"
DEFAULT_STARDIST_ROOT = DEFAULT_WANGLAB_ROOT / "StarDist_Segment"
DEFAULT_MICROSCOPE_DIR = DEFAULT_WANGLAB_ROOT / "4_Images" / "Microscope_Image"
DEFAULT_LOC_DIR = DEFAULT_WANGLAB_ROOT / "2_Single_Nuclei_Matrix"
DEFAULT_HIERARCHY_XLSX = DEFAULT_ANNOTATION_ROOT / "GBM_sc_seg_celltypes_hierarchy.xlsx"
DEFAULT_HIERARCHY_SHEET = "Celltype"
DEFAULT_GD_LABEL_TXT = DEFAULT_ANNOTATION_ROOT / "1104_Rec_HD_GD_Label.txt"
DEFAULT_NUCLEI_SN_GD_XLSX = (
    DEFAULT_ANNOTATION_ROOT / "GBM_nuclei_primarySN_GD_annotation.xlsx"
)

###############################################
# 2026.09.08 LLY:
###############################################
# Training drops these labels on cell_type (L1) and subcluster (L2).
# Unknown/LowQ rows are omitted from the hierarchy workbook and from GT CSVs.
GBM_EXCLUDED_CELLTYPES = ("Unknown", "LowQ")
# spatial_niche LowQ and missing/unlabeled are dropped (L12 / final_sublineage).
GBM_UNLABELED_SN = "unlabeled"
GBM_EXCLUDED_SN = ("LowQ", GBM_UNLABELED_SN)
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"
CELLS_WITH_PIXEL_SUFFIX = "_cells_with_pixel.csv"
CELLS_MATCHED_STARDIST_SUFFIX = "_cells_matched_by_stardist.csv"
HE_H5AD_SUFFIX = "_matched_features.h5ad"
STARDIST_H5AD_SUFFIX = "_matched_features_stardist.h5ad"
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"

GBM_COLUMN_RENAME = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
    "final_lineage": "celltype_level1",
}

# Rec mapping uses numbered prefixes (0Tumor, 1Myeloid, ...); Ini does not.
_CELL_TYPE_PREFIX_RE = re.compile(r"^\d+")

SAMPLES: dict[str, dict[str, object]] = {
    "P174511_Initial": {
        "short": "Ini",
        "stardist_key": "174511-2-3",
        "he_tif": DEFAULT_MICROSCOPE_DIR / "174511-2-3.tif",
        "loc_csv": DEFAULT_LOC_DIR / "P174511_2_3_loc.csv",
        "annotation_xlsx": DEFAULT_ANNOTATION_DIR
        / "0917_Ini_sc_seg_celltypes.mapping.xlsx",
        "bin_nuclei_mapping_xlsx": DEFAULT_ANNOTATION_ROOT
        / "3_Mapping_bin_nuclei"
        / "0917_200G_celltype_spatialNiches_mapping_Ini.xlsx",
        "spatial_niches_xlsx": DEFAULT_ANNOTATION_ROOT
        / "1_Bin"
        / "0917_ST_HD_200G_Ini_SpatialNiches.xlsx",
        "cell_info_csv": DEFAULT_RESULTS_DIR
        / "gbm_P174511_Initial_cell_info_HE_by_annotation.csv",
        "bin_dir": DEFAULT_WANGLAB_ROOT / "1_Bin_Matrix" / "P174511_Initial",
    },
    "P179161_Recurrent": {
        "short": "Rec",
        "stardist_key": "179161-4-3",
        "he_tif": DEFAULT_MICROSCOPE_DIR / "179161-4-3.tif",
        "loc_csv": DEFAULT_LOC_DIR / "P179161_4_3_loc.csv",
        "annotation_xlsx": DEFAULT_ANNOTATION_DIR
        / "0917_Rec_sc_seg_celltypes.mapping.xlsx",
        "bin_nuclei_mapping_xlsx": DEFAULT_ANNOTATION_ROOT
        / "3_Mapping_bin_nuclei"
        / "0917_200G_celltype_spatialNiches_mapping_Rec.xlsx",
        "spatial_niches_xlsx": DEFAULT_ANNOTATION_ROOT
        / "1_Bin"
        / "0917_ST_HD_200G_Rec_SpatialNiches.xlsx",
        "cell_info_csv": DEFAULT_RESULTS_DIR
        / "gbm_P179161_Recurrent_cell_info_HE_by_annotation.csv",
        "bin_dir": DEFAULT_WANGLAB_ROOT / "1_Bin_Matrix" / "P179161_Recurrent",
    },
}


def sample_ids() -> list[str]:
    return list(SAMPLES)


def sample_config(sample: str) -> dict[str, object]:
    if sample not in SAMPLES:
        raise KeyError(f"Unknown GBM sample {sample!r}; choose from {sample_ids()}")
    return SAMPLES[sample]


def sample_dir(sample: str, cases_root: Path | str | None = None) -> Path:
    root = Path(cases_root) if cases_root is not None else DEFAULT_CASES_ROOT
    return root / sample


def he_tif_path(sample: str) -> Path:
    return Path(sample_config(sample)["he_tif"])


def loc_csv_path(sample: str) -> Path:
    return Path(sample_config(sample)["loc_csv"])


def annotation_xlsx_path(sample: str) -> Path:
    return Path(sample_config(sample)["annotation_xlsx"])


def bin_nuclei_mapping_xlsx_path(sample: str) -> Path:
    return Path(sample_config(sample)["bin_nuclei_mapping_xlsx"])


def spatial_niches_xlsx_path(sample: str) -> Path:
    return Path(sample_config(sample)["spatial_niches_xlsx"])


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


def normalize_gbm_cell_type(value: object) -> str:
    """Strip Rec numeric prefixes (``0Tumor`` → ``Tumor``)."""
    text = str(value).strip() if pd.notna(value) else ""
    return _CELL_TYPE_PREFIX_RE.sub("", text)


##################################################
# 2026.09.08,revise the hierarchy xlsx
# 2026.09.09 LLY: levels follow cell-type meaning (not SN as coarse):
#   subcluster     → L2  fine         (celltype_level2 / final_CT)
#   spatial_niche  → L12 intermediate (celltype_level12 / final_sublineage)
#   cell_type      → L1  coarse       (celltype_level1 / final_lineage)
##################################################
GBM_HIERARCHY_SOURCE_COLS = {
    "subcluster": "celltype_level2",
    "spatial_niche": "celltype_level12",
    "cell_type": "celltype_level1",
}


def _strip_label(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text in ("", "nan", "None") else text


def keep_gbm_spatial_niche(value: object) -> bool:
    """True for usable L12 SN labels (SN1–SN9). Drop LowQ and missing."""
    key = _strip_label(value)
    if pd.isna(key):
        return False
    return str(key) not in GBM_EXCLUDED_SN


def fill_gbm_spatial_niche(value: object) -> object:
    """Return a cleaned SN label, or NA if LowQ / missing (those rows are dropped)."""
    key = _strip_label(value)
    if pd.isna(key) or str(key) in GBM_EXCLUDED_SN:
        return pd.NA
    return str(key)


def is_excluded_gbm_cell_label(value: object) -> bool:
    """True for Unknown / LowQ / missing cell_type or subcluster labels."""
    key = _strip_label(value)
    if pd.isna(key):
        return True
    return str(key) in GBM_EXCLUDED_CELLTYPES


def filter_usable_gbm_cell_labels(
    df: pd.DataFrame,
    *,
    cols: tuple[str, ...] = ("cell_type", "subcluster"),
) -> pd.DataFrame:
    """Drop rows whose cell_type or subcluster is Unknown / LowQ / missing."""
    out = df
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Missing {col!r} for cell-label filter")
        out = out.loc[~out[col].map(is_excluded_gbm_cell_label)].copy()
    return out.reset_index(drop=True)


def filter_usable_gbm_sn(
    df: pd.DataFrame,
    *,
    sn_col: str = "spatial_niche",
    barcode_col: str = "bin_barcode",
    require_barcode: bool = True,
) -> pd.DataFrame:
    """Drop LowQ / missing ``spatial_niche`` so every remaining row has a 16 µm bin.

    After this filter ``bin_barcode`` is expected to be non-null (primary SN
    always comes with the WangLab mapping ``index``).
    """
    if sn_col not in df.columns:
        raise KeyError(f"Missing {sn_col!r} for SN filter")
    out = df.loc[df[sn_col].map(keep_gbm_spatial_niche)].copy()
    if require_barcode:
        if barcode_col not in out.columns:
            raise KeyError(f"Missing {barcode_col!r} after SN filter")
        out = out.loc[out[barcode_col].map(_strip_label).notna()].copy()
    return out.reset_index(drop=True)
##################################################


##################################################
# 2026.09.08, revise the hierarchy xlsx and update cell level labels
##################################################
def load_gbm_celltype_hierarchy(
    xlsx_path: str | Path | None = None,
    *,
    sheet_name: str = DEFAULT_HIERARCHY_SHEET,
    drop_excluded: bool = True,
) -> pd.DataFrame:
    """Load GBM hierarchy as ``celltype_level2`` / ``level1`` / ``level0``.

    Current workbook (BRCA-style ``Celltype`` sheet) maps::

        subcluster     → celltype_level2  (fine / ``final_CT``)
        spatial_niche  → celltype_level12 (intermediate / ``final_sublineage``)
        cell_type      → celltype_level1  (coarse / ``final_lineage``)

    One row is an observed triple. ``Mac_SEPP1`` maps to both Myeloid and
    Vascular (Ini vs Rec), so duplicate ``celltype_level2`` is expected.
    ``spatial_niche`` LowQ / missing and Unknown / LowQ cell labels are dropped.
    Returned names follow the HCC matcher: ``level12``→``level1``,
    ``level1``→``level0``.
    """
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_HIERARCHY_XLSX
    df = pd.read_excel(path, sheet_name=sheet_name)
    two_level_cols = ["celltype_level2", "celltype_level1"]
    hist2pheno_cols = ["celltype_level2", "celltype_level12", "celltype_level1"]
    hcc_excel_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    has_l12 = "celltype_level12" in df.columns
    has_l0 = "celltype_level0" in df.columns
    if set(hist2pheno_cols).issubset(df.columns):
        out = df[hist2pheno_cols].copy()
        out = out.rename(
            columns={
                "celltype_level12": "celltype_level1",
                "celltype_level1": "celltype_level0",
            }
        )
        unique_l2_required = False
    elif set(hcc_excel_cols).issubset(df.columns):
        out = df[hcc_excel_cols].copy()
        unique_l2_required = True
    elif set(two_level_cols).issubset(df.columns) and not has_l12 and not has_l0:
        out = df[two_level_cols].copy()
        out["celltype_level0"] = out["celltype_level1"]
        unique_l2_required = True
    else:
        raise KeyError(
            f"{sheet_name!r} needs {hist2pheno_cols} "
            f"(or {hcc_excel_cols} / {two_level_cols}); got {list(df.columns)}"
        )
    hierarchy_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    for col in hierarchy_cols:
        out[col] = out[col].map(_strip_label)
    # SN is the intermediate head (level1 after the L12→L1 rename).
    out = out[out["celltype_level1"].map(keep_gbm_spatial_niche)].copy()
    if drop_excluded:
        for col in hierarchy_cols:
            out = out[~out[col].map(is_excluded_gbm_cell_label)].copy()
    out = out.dropna(subset=hierarchy_cols)
    out = out.drop_duplicates(subset=hierarchy_cols).reset_index(drop=True)
    if unique_l2_required:
        duplicated = out.loc[
            out["celltype_level2"].duplicated(keep=False), "celltype_level2"
        ].unique()
        if len(duplicated):
            raise ValueError(
                f"Duplicate celltype_level2: {sorted(duplicated.tolist())}"
            )
    return out
##################################################

def list_gbm_samples(
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
