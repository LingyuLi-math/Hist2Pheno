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
# Training drops these L2 labels. Unknown is still a real annotated class in the xlsx.
GBM_EXCLUDED_CELLTYPES = ("Unknown", "LowQ")
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


def load_gbm_celltype_hierarchy(
    xlsx_path: str | Path | None = None,
    *,
    sheet_name: str = DEFAULT_HIERARCHY_SHEET,
    drop_excluded: bool = True,
) -> pd.DataFrame:
    """Load WangLab L2 (``subcluster``) → L1 (``cell_type``) mapping.

    GBM annotations are two-level. The Excel sheet has
    ``celltype_level2`` / ``celltype_level1`` only (no invented HCC coarse
    layer). The returned frame copies L1 into ``celltype_level0`` so the
    three-head matcher can set both ``final_sublineage`` and
    ``final_lineage`` to ``cell_type``.
    """
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_HIERARCHY_XLSX
    df = pd.read_excel(path, sheet_name=sheet_name)
    two_level_cols = ["celltype_level2", "celltype_level1"]
    hist2pheno_cols = ["celltype_level2", "celltype_level12", "celltype_level1"]
    hcc_excel_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    has_l12 = "celltype_level12" in df.columns
    has_l0 = "celltype_level0" in df.columns
    if set(two_level_cols).issubset(df.columns) and not has_l12 and not has_l0:
        out = df[two_level_cols].copy()
        out["celltype_level0"] = out["celltype_level1"]
    elif set(hist2pheno_cols).issubset(df.columns):
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
            f"{sheet_name!r} needs {two_level_cols} "
            f"(or {hist2pheno_cols} / {hcc_excel_cols}); got {list(df.columns)}"
        )
    hierarchy_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    for col in hierarchy_cols:
        out[col] = out[col].map(
            lambda value: str(value).strip() if pd.notna(value) else pd.NA
        )
    if drop_excluded:
        out = out[~out["celltype_level2"].isin(GBM_EXCLUDED_CELLTYPES)].copy()
    out = out.dropna(subset=["celltype_level1", "celltype_level0"])
    out = out.drop_duplicates().reset_index(drop=True)
    duplicated = out.loc[
        out["celltype_level2"].duplicated(keep=False), "celltype_level2"
    ].unique()
    if len(duplicated):
        raise ValueError(f"Duplicate celltype_level2: {sorted(duplicated.tolist())}")
    return out


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
