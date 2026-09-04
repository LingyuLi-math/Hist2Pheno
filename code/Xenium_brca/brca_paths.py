"""Xenium BRCA sample registry, hierarchy, and Hist2Pheno case paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRCA_ROOT = _REPO_ROOT / "data/Xemium/BRCA"
DEFAULT_ANNOTATION_DIR = DEFAULT_BRCA_ROOT / "Annotation"
DEFAULT_RESULTS_DIR = DEFAULT_BRCA_ROOT / "Results"
DEFAULT_CASES_ROOT = DEFAULT_BRCA_ROOT / "Cases"
DEFAULT_STARDIST_ROOT = DEFAULT_BRCA_ROOT / "StarDist_Segment"
DEFAULT_HIERARCHY_XLSX = (
    DEFAULT_ANNOTATION_DIR / "GSE243275_Barcode_Cell_Type_MatricesLY.xlsx"
)
DEFAULT_HIERARCHY_SHEET = "celltype"
STARDIST_CSV_SUFFIX = "_Float_prob0.01_nms_0.3.csv"
CELLS_WITH_PIXEL_SUFFIX = "_cells_with_pixel.csv"
CELLS_MATCHED_STARDIST_SUFFIX = "_cells_matched_by_stardist.csv"
HE_H5AD_SUFFIX = "_matched_features.h5ad"
STARDIST_H5AD_SUFFIX = "_matched_features_stardist.h5ad"
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"

BRCA_COLUMN_RENAME = {
    "final_CT": "celltype",
    "final_sublineage": "celltype_level12",
    "final_lineage": "celltype_level1",
}

SAMPLES: dict[str, dict[str, object]] = {
    "rep1": {
        "replicate": 1,
        "prefix": "Xenium_FFPE_Human_Breast_Cancer_Rep1",
        "data_dir": DEFAULT_BRCA_ROOT / "human_breast_Xenium_rep1",
        "stardist_key": "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image",
        "cell_info_csv": DEFAULT_RESULTS_DIR
        / "xenium_brca_rep1_cell_info_HE_by_annotation.csv",
        "annotation_sheet": "Xenium R1 Fig1-5 (supervised)",
    },
    "rep2": {
        "replicate": 2,
        "prefix": "Xenium_FFPE_Human_Breast_Cancer_Rep2",
        "data_dir": DEFAULT_BRCA_ROOT / "human_breast_Xenium_rep2",
        "stardist_key": "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image",
        "cell_info_csv": DEFAULT_RESULTS_DIR
        / "xenium_brca_rep2_cell_info_HE_by_annotation.csv",
        "annotation_sheet": "Xenium R2 Fig1-5 (supervised)LY",
    },
}


def sample_ids() -> list[str]:
    return list(SAMPLES)


def sample_config(sample: str) -> dict[str, object]:
    if sample not in SAMPLES:
        raise KeyError(f"Unknown BRCA sample {sample!r}; choose from {sample_ids()}")
    return SAMPLES[sample]


def sample_dir(sample: str, cases_root: Path | str | None = None) -> Path:
    root = Path(cases_root) if cases_root is not None else DEFAULT_CASES_ROOT
    return root / sample


def he_tif_path(sample: str) -> Path:
    """Working HE ``.tif`` used by StarDist / UNI (may differ from OME canvas)."""
    cfg = sample_config(sample)
    prefix = str(cfg["prefix"])
    return Path(cfg["data_dir"]) / f"{prefix}_he_image.tif"

###########################################################
# 2026.09.04, for brca
###########################################################
def he_ome_tif_path(sample: str) -> Path:
    """Official post-Xenium HE OME-TIFF that ``*_he_imagealignment.csv`` registers."""
    cfg = sample_config(sample)
    prefix = str(cfg["prefix"])
    return Path(cfg["data_dir"]) / f"{prefix}_he_image.ome.tif"


###########################################################
# 2026.09.03, Xenium alignment matrix (microns → H&E pixels)
# 2026.09.04, for brca, add function to scale HE OME-TIFF pixels onto a separately exported working .tif
# 10x Explorer HE↔morphology affine (NOT Visium).
# spatial_HE = scale_ome_to_tif( inv(M) @ (µm / 0.2125) )
###########################################################
def he_alignment_csv_path(sample: str) -> Path:
    """Path to ``*_he_imagealignment.csv`` (post-Xenium HE ↔ Xenium morphology)."""
    cfg = sample_config(sample)
    prefix = str(cfg["prefix"])
    return Path(cfg["data_dir"]) / f"{prefix}_he_imagealignment.csv"


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


def load_brca_celltype_hierarchy(
    xlsx_path: str | Path | None = None,
    *,
    sheet_name: str = DEFAULT_HIERARCHY_SHEET,
) -> pd.DataFrame:
    """Load L2→L12→L1 mapping. Drops Unlabeled (empty parents).

    The LY ``celltype`` sheet uses Hist2Pheno-style names
    (``celltype_level2`` / ``celltype_level12`` / ``celltype_level1``).
    The returned frame is normalized to the HCC-style names used by matching
    (``celltype_level2`` / ``celltype_level1`` / ``celltype_level0``), so
    ``final_sublineage`` = intermediate and ``final_lineage`` = coarse.
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


def list_brca_samples(
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
