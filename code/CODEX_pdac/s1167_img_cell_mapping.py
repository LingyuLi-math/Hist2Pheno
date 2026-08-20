## 2026.08.14 load s1167 CODEX cell annotation (.csv), transform HE (.zarr) to .tif,
##            and save cell-type annotation on the HE image (.jpg).
##            Layout differs from s4769: each ACQUISITION_ID folder contains he_img/,
##            {acq}.cell_data.csv, and optionally {acq}.{id}.cell_types.csv.
## 2026.08.18 metadata source is raw_metadata_updated.xlsx
##            sheets: Clinical_info (cores), Celltype (hierarchy). Do not use raw_metadata.csv
##            or the incomplete s1167_metadata sheet.

"""s1167 CODEX HE (Zarr) loading and spatial visualization (PDAC / GIST TMA)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.patches import Rectangle

DEFAULT_CODEX_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer"
)
S1167_SAMPLE = "s1167"
DEFAULT_METADATA_FILE = "raw_metadata_updated.xlsx"
DEFAULT_METADATA_SHEET = "Clinical_info"
DEFAULT_METADATA_CSV = DEFAULT_METADATA_FILE  # backward alias; do not read .csv
DEFAULT_COHORT = "Pancreas TMA"
DEFAULT_STARDIST_HE_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/StarDist_Segment_pdac/HE_images"
)
DEFAULT_STARDIST_RESULT_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/StarDist_Segment_pdac/pdac_result"
)
DEFAULT_STARDIST_GIST_HE_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/StarDist_Segment_gist/HE_images"
)
DEFAULT_STARDIST_GIST_RESULT_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/StarDist_Segment_gist/gist_result"
)
DEFAULT_CELLTYPE_HIERARCHY_XLSX = DEFAULT_METADATA_FILE
DEFAULT_CELLTYPE_HIERARCHY_SHEET = "Celltype"
DEFAULT_SAVED_CELLTYPE_HIERARCHY_XLSX = "s1167_celltype_hierarchy.xlsx"
DEFAULT_S1167_CELLTYPE_FILTER = ("Other",)

# Coverslip → typical tissue_type in the current xlsx (source of truth is the Excel column).
PREFIX_COHORT = {
    "Charvill-94_c001": "Pancreas TMA",
    "Charvill-94_c003": "Pancreas TMA",
    "Charvill-94_c005": "Pancreas TMA",
    "Charvill-94_c007": "Pancreas TMA",
    "Charvill-94_c009": "GIST TMA",
    "Charvill-94_c011": "GIST TMA",
    "Charvill-94_c013": "GIST TMA",
}
COHORT_ALIASES = {
    "pdac": "Pancreas TMA",
    "pancreas": "Pancreas TMA",
    "pancreas tma": "Pancreas TMA",
    "gist": "GIST TMA",
    "gist tma": "GIST TMA",
}

S1167_CELLTYPE_COLORS = {
    "Tregs": "#ffbb78",
    "Helper T cells": "#ff7f0e",
    "Cytotoxic T cells": "#d62728",
    "T cells": "#c44e52",
    "B cells": "#f7b6d2",
    "Plasma cells": "#e377c2",
    "DCs": "#8c564b",
    "Dendritic cells": "#8c564b",
    "Monocytes": "#c49c94",
    "Macrophages": "#9467bd",
    "Neutrophils": "#bcbd22",
    "Fibroblasts": "#98df8a",
    "Stromal cells": "#2ca02c",
    "Endothelial cells": "#1f77b4",
    "Lymphatic Endothelial cells": "#aec7e8",
    "Epithelial cells": "#17becf",
    "Other": "#7f7f7f",
    "Unannotated": "#bdbdbd",
}

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
from plotting_palettes import resolve_palette  # noqa: E402

_he_rgb_cache: dict[str, np.ndarray] = {}


def clear_he_rgb_cache() -> None:
    _he_rgb_cache.clear()


def sample_dir(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    return Path(base_dir) / sample


def metadata_path(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    csv_name: str = DEFAULT_METADATA_FILE,
) -> Path:
    return sample_dir(sample, base_dir) / csv_name


def normalize_cohort(cohort: str | None) -> str | None:
    if cohort is None:
        return None
    key = str(cohort).strip()
    return COHORT_ALIASES.get(key.lower(), key)


def acq_dir(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    return sample_dir(sample, base_dir) / str(acq_id).strip()


def he_img_path(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    """HE Zarr lives inside the acquisition folder (not HE/{MATCHED_HE}/)."""
    return acq_dir(acq_id, sample, base_dir) / "he_img"


def he_tif_path(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    return acq_dir(acq_id, sample, base_dir) / "figures" / f"{acq_id}.tif"


def he_celltype_jpg_path(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    return acq_dir(acq_id, sample, base_dir) / "figures" / f"{acq_id}_celltype_on_HE.jpg"


def acquisition_prefix(acq_id: str) -> str:
    text = str(acq_id)
    for prefix in PREFIX_COHORT:
        if text.startswith(prefix):
            return prefix
    return text.split("_v", 1)[0]


def cohort_for_acq(acq_id: str) -> str:
    return PREFIX_COHORT.get(acquisition_prefix(acq_id), "Unknown")


def _annotate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach prefix. ``cohort`` follows xlsx ``tissue_type`` (not coverslip prefix)."""
    out = df.copy()
    out["ACQUISITION_ID"] = out["ACQUISITION_ID"].astype(str).str.strip()
    out["prefix"] = out["ACQUISITION_ID"].map(acquisition_prefix)
    if "tissue_type" in out.columns:
        tt = out["tissue_type"].astype(str).str.strip()
        missing = out["tissue_type"].isna() | tt.isin(["", "nan", "None"])
        out["cohort"] = out["tissue_type"].where(~missing, other="Unknown")
    else:
        out["cohort"] = "Unknown"
    out["tissue_type_filled"] = out["cohort"]
    return out


def _read_s1167_metadata_table(path: Path, sheet_name: str = DEFAULT_METADATA_SHEET) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(
        f"Expected raw_metadata_updated.xlsx, got {path.name}. "
        "Do not use raw_metadata.csv or the s1167_metadata sheet."
    )


def load_s1167_metadata(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    csv_name: str = DEFAULT_METADATA_FILE,
    *,
    cohort: str | None = DEFAULT_COHORT,
    annotated_only: bool = False,
    with_he_only: bool = True,
    sheet_name: str = DEFAULT_METADATA_SHEET,
) -> pd.DataFrame:
    """Load ``raw_metadata_updated.xlsx`` sheet ``Clinical_info`` and attach ``prefix`` / ``cohort``.

    ``cohort`` follows Excel ``tissue_type``. Default ``Pancreas TMA`` (PDAC).
    Aliases: ``PDAC``, ``Pancreas``. ``annotated_only=True`` keeps cores that
    have a cell-type CSV (PDAC: 278 / 473; c005/c007 have none).
    Do not use sheet ``s1167_metadata`` (incomplete tissue_type, same as the old CSV).
    """
    path = metadata_path(sample, base_dir, csv_name)
    df = _annotate_metadata(_read_s1167_metadata_table(path, sheet_name=sheet_name))
    if with_he_only:
        keep = [
            aid for aid in df["ACQUISITION_ID"]
            if he_img_path(aid, sample, base_dir).is_dir()
        ]
        df = df[df["ACQUISITION_ID"].isin(keep)].copy()
    cohort_key = normalize_cohort(cohort)
    if cohort_key:
        df = df[df["cohort"].astype(str) == cohort_key].copy()
    if annotated_only:
        df = filter_with_celltype_annotation(df, sample=sample, base_dir=base_dir)
    return df.reset_index(drop=True)


def summarize_s1167_metadata(
    meta_df: pd.DataFrame | None = None,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> pd.DataFrame:
    """Print cohort / HE / cell-type coverage. Returns the unfiltered table."""
    if meta_df is None:
        meta_df = load_s1167_metadata(
            sample=sample, base_dir=base_dir, cohort=None, annotated_only=False,
        )
    df = meta_df.copy()
    if "has_he" not in df.columns:
        df["has_he"] = [
            he_img_path(aid, sample, base_dir).is_dir()
            for aid in df["ACQUISITION_ID"]
        ]
    if "has_cell_types" not in df.columns:
        df["has_cell_types"] = [
            has_codex_celltype_annotation(aid, sample, base_dir)
            for aid in df["ACQUISITION_ID"]
        ]
    print(f"s1167 metadata: {len(df)} rows  ({metadata_path(sample, base_dir)})")
    print("\ncohort × has_cell_types:")
    print(pd.crosstab(df["cohort"], df["has_cell_types"], margins=True).to_string())
    if "prefix" in df.columns:
        print("\nprefix × tissue_type:")
        print(pd.crosstab(df["prefix"], df["tissue_type"].fillna("NA"), margins=True).to_string())
    if "tissue_type" in df.columns:
        print("\ntissue_type (Clinical_info):")
        print(df["tissue_type"].fillna("NA").value_counts(dropna=False).to_string())
    n_he = int(df["has_he"].sum())
    n_ann = int(df["has_cell_types"].sum())
    print(f"\nHE zarr present: {n_he}/{len(df)}   cell_types.csv present: {n_ann}/{len(df)}")
    return df


def filter_with_celltype_annotation(
    meta_df: pd.DataFrame,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> pd.DataFrame:
    keep = [
        has_codex_celltype_annotation(str(aid), sample, base_dir)
        for aid in meta_df["ACQUISITION_ID"]
    ]
    return meta_df.loc[keep].copy().reset_index(drop=True)


def metadata_row_for_acq(acq_id: str, meta_df: pd.DataFrame) -> pd.Series | None:
    hit = meta_df[meta_df["ACQUISITION_ID"].astype(str) == str(acq_id).strip()]
    if hit.empty:
        return None
    return hit.iloc[0]


def celltype_hierarchy_path(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    xlsx_name: str = DEFAULT_CELLTYPE_HIERARCHY_XLSX,
) -> Path:
    return sample_dir(sample, base_dir) / xlsx_name


def save_s1167_celltype_hierarchy(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    src_xlsx: str = DEFAULT_CELLTYPE_HIERARCHY_XLSX,
    sheet_name: str = DEFAULT_CELLTYPE_HIERARCHY_SHEET,
    dest_xlsx: str = DEFAULT_SAVED_CELLTYPE_HIERARCHY_XLSX,
    skip_if_exists: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """Copy the ``Celltype`` sheet to ``s1167_celltype_hierarchy.xlsx`` (both cohorts, including Other)."""
    src = celltype_hierarchy_path(sample, base_dir, src_xlsx)
    df = pd.read_excel(src, sheet_name=sheet_name)
    dest = sample_dir(sample, base_dir) / dest_xlsx
    if skip_if_exists and dest.is_file():
        return dest, df
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(dest, sheet_name=sheet_name, index=False)
    return dest, df


def load_codex_he_alignment(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    cohort: str | None = DEFAULT_COHORT,
    annotated_only: bool = False,
) -> pd.DataFrame:
    """s1167 analog of HCC ``ALIGNED=='Y'``.

    Every acquisition has ``he_img/`` (no MATCHED_HE Excel).
    ``annotated_only=True`` is the analog of ``Annotation=='Y'`` (cell-type CSV).
    """
    return load_s1167_metadata(
        sample=sample,
        base_dir=base_dir,
        cohort=cohort,
        annotated_only=annotated_only,
        with_he_only=True,
    )


def load_s1167_celltype_hierarchy(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    xlsx_name: str = DEFAULT_CELLTYPE_HIERARCHY_XLSX,
    sheet_name: str = DEFAULT_CELLTYPE_HIERARCHY_SHEET,
    *,
    cohort: str | None = DEFAULT_COHORT,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
) -> pd.DataFrame:
    """Load Level2→Level1→Level0 hierarchy from ``raw_metadata_updated.xlsx`` sheet ``Celltype``."""
    path = celltype_hierarchy_path(sample, base_dir, xlsx_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    hierarchy_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    missing = [col for col in hierarchy_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{sheet_name!r} sheet missing hierarchy columns: {missing}")

    out = df.copy()
    if cohort and "cohort" in out.columns:
        cohort_key = normalize_cohort(cohort)
        out = out[out["cohort"].astype(str).str.strip() == cohort_key].copy()

    for col in hierarchy_cols:
        out[col] = out[col].map(
            lambda value: str(value).strip() if pd.notna(value) else pd.NA
        )

    excluded = {str(value).strip() for value in (celltype_filter or ())}
    out = out[~out["celltype_level2"].isin(excluded)].copy()
    keep_cols = (["cohort"] if "cohort" in out.columns else []) + hierarchy_cols
    out = out[keep_cols].dropna(subset=hierarchy_cols).drop_duplicates().reset_index(drop=True)

    duplicated_l2 = out.loc[
        out["celltype_level2"].duplicated(keep=False), "celltype_level2"
    ].unique()
    if len(duplicated_l2):
        raise ValueError(
            "Each celltype_level2 must have one hierarchy mapping; duplicates: "
            f"{sorted(duplicated_l2.tolist())}"
        )
    return out


def load_s1167_celltypes(
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    cohort: str | None = DEFAULT_COHORT,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
) -> list[str]:
    """Retained Level2 labels for a cohort (excludes ``Other`` by default)."""
    hier = load_s1167_celltype_hierarchy(
        sample=sample, base_dir=base_dir, cohort=cohort, celltype_filter=celltype_filter,
    )
    return hier["celltype_level2"].astype(str).tolist()


def collect_celltypes_from_mapping(
    meta_df: pd.DataFrame,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
) -> list[str]:
    """Unique ANNOTATION_LABEL values on disk for the given cores."""
    excluded = {str(value).strip() for value in (celltype_filter or ())}
    labels: set[str] = set()
    for aid in meta_df["ACQUISITION_ID"].astype(str):
        path = cell_types_path(aid, sample, base_dir)
        if path is None:
            continue
        series = pd.read_csv(path, usecols=["ANNOTATION_LABEL"])["ANNOTATION_LABEL"]
        labels.update(str(v).strip() for v in series.dropna())
    return sorted(name for name in labels if name and name not in excluded)


def _normalize_he_rgb_hwc(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype.kind == "f":
        return np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.dtype != np.uint8:
        return rgb.astype(np.uint8)
    return rgb


def load_he_rgb(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Load Zarr RGB HE (CHW) → HWC uint8 numpy array."""
    p = he_img_path(acq_id, sample, base_dir)
    z_he = zarr.open(str(p), mode="r")
    if verbose:
        print(f"HE {acq_id}: shape={z_he.shape}, dtype={z_he.dtype}")
    rgb = _normalize_he_rgb_hwc(np.transpose(z_he[:], (1, 2, 0)))
    if verbose and z_he.dtype.kind == "f":
        print("  → converted float HE to uint8 for display")
    return rgb


def get_he_rgb(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> np.ndarray:
    cache_key = f"{base_dir}|{sample}|{acq_id}"
    if cache_key not in _he_rgb_cache:
        _he_rgb_cache[cache_key] = load_he_rgb(acq_id, sample, base_dir, verbose=False)
    return _he_rgb_cache[cache_key]


def plot_he_overview(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    crop_half: int = 800,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> tuple[np.ndarray, Path | None]:
    """Full HE + center crop with red box; optionally save overview JPEG."""
    he_rgb = load_he_rgb(acq_id, sample, base_dir)
    print(f"HE RGB array: {he_rgb.shape}, value range [{he_rgb.min()}, {he_rgb.max()}]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h, w = he_rgb.shape[:2]
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - crop_half), min(h, cy + crop_half)
    x0, x1 = max(0, cx - crop_half), min(w, cx + crop_half)

    axes[0].imshow(he_rgb)
    axes[0].add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.5, edgecolor="red", facecolor="none", linestyle="-",
        )
    )
    axes[0].set_title(f"Full HE (H×W={he_rgb.shape[0]}×{he_rgb.shape[1]})")
    axes[0].axis("off")

    axes[1].imshow(he_rgb[y0:y1, x0:x1])
    axes[1].set_title(f"Center crop [{y0}:{y1}, {x0}:{x1}]")
    axes[1].axis("off")

    fig.suptitle(f"s1167 HE preview ({acq_id})", y=1.02)
    fig.tight_layout()

    save_path = None
    if save:
        fig_dir = acq_dir(acq_id, sample, base_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / "he_img_overview.jpg"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return he_rgb, save_path


def export_he_zarr_to_tif(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> Path:
    """Convert Zarr he_img (CHW uint8 RGB) → {acq_id}/figures/{acq_id}.tif."""
    import tifffile

    src = he_img_path(acq_id, sample, base_dir)
    if not src.is_dir():
        raise FileNotFoundError(src)

    out_path = he_tif_path(acq_id, sample, base_dir)
    if skip_if_exists and out_path.is_file() and not overwrite:
        if verbose:
            print(f"SKIP (exists): {out_path}")
        return out_path

    z_he = zarr.open(str(src), mode="r")
    rgb = _normalize_he_rgb_hwc(np.transpose(z_he[:], (1, 2, 0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, rgb, photometric="rgb")
    if verbose:
        print(f"Saved: {out_path}  shape={rgb.shape} dtype={rgb.dtype}")
    return out_path


def export_he_zarrs_to_tif(
    meta_df: pd.DataFrame,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Export HE Zarr → TIFF for each ACQUISITION_ID in ``meta_df``."""
    acq_ids = meta_df["ACQUISITION_ID"].astype(str).str.strip().unique().tolist()
    if verbose:
        print(f"Exporting {len(acq_ids)} HE Zarr → TIFF")
    saved: list[Path] = []
    for aid in acq_ids:
        saved.append(
            export_he_zarr_to_tif(
                aid, sample, base_dir,
                overwrite=overwrite, skip_if_exists=skip_if_exists, verbose=verbose,
            )
        )
    return saved


def copy_he_tifs(
    tif_paths: list[str | Path],
    dest_dir: str | Path = DEFAULT_STARDIST_HE_DIR,
    *,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> dict[str, list[Path]]:
    """Copy exported HE TIFFs into a flat StarDist HE_images folder."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied, skipped, missing = [], [], []
    for src in tif_paths:
        src = Path(src)
        out = dest / src.name
        if not src.is_file():
            missing.append(src)
            if verbose:
                print(f"MISSING: {src}")
            continue
        if skip_if_exists and out.exists():
            skipped.append(out)
            if verbose:
                print(f"SKIP (dest exists): {out}")
            continue
        shutil.copy2(str(src), str(out))
        copied.append(out)
        if verbose:
            print(f"COPIED: {src.name}")
    if verbose:
        print(
            f"Done: copied={len(copied)} skipped={len(skipped)} "
            f"missing={len(missing)}\nDest: {dest}"
        )
    return {"copied": copied, "skipped": skipped, "missing": missing}


def cell_data_path(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path:
    return acq_dir(acq_id, sample, base_dir) / f"{acq_id}.cell_data.csv"


def cell_types_path(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> Path | None:
    """Return the cell-type CSV if present.

    c013 / GIST use ``{acq}.{numeric_id}.cell_types.csv``;
    c001 / c003 use ``{acq}.cell_types.csv``.
    """
    folder = acq_dir(acq_id, sample, base_dir)
    matches = sorted(folder.glob(f"{acq_id}.*.cell_types.csv"))
    if matches:
        return matches[0]
    fallback = folder / f"{acq_id}.cell_types.csv"
    return fallback if fallback.is_file() else None


def has_codex_celltype_annotation(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> bool:
    return cell_data_path(acq_id, sample, base_dir).is_file() and cell_types_path(
        acq_id, sample, base_dir
    ) is not None


def _ensure_celltype_column(df: pd.DataFrame, acq_id: str = "") -> pd.DataFrame:
    if "celltype" in df.columns:
        return df
    for col in ("ANNOTATION_LABEL", "ANNOTATION_LABEL_y", "annotation_label", "cell_type", "CellType"):
        if col in df.columns:
            return df.rename(columns={col: "celltype"})
    raise KeyError(
        f"No celltype column for {acq_id!r}; available columns: {list(df.columns)}"
    )


UNANNOTATED_LABEL = "Unannotated"


def load_codex_cells(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    allow_missing_annotation: bool = False,
) -> pd.DataFrame:
    data_fp = cell_data_path(acq_id, sample, base_dir)
    types_fp = cell_types_path(acq_id, sample, base_dir)
    if not data_fp.is_file():
        raise FileNotFoundError(data_fp)
    data = pd.read_csv(data_fp).rename(columns={"X": "x_pix", "Y": "y_pix"})
    if types_fp is None:
        if not allow_missing_annotation:
            raise FileNotFoundError(
                acq_dir(acq_id, sample, base_dir) / f"{acq_id}.*.cell_types.csv"
            )
        data["celltype"] = UNANNOTATED_LABEL
        return data
    merged = data.merge(pd.read_csv(types_fp), on="CELL_ID", how="inner").rename(
        columns={"ANNOTATION_LABEL": "celltype"}
    )
    return _ensure_celltype_column(merged, acq_id)


def load_cells_by_acq(
    primary_acq_id: str,
    also_view: list[str] | None = None,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
) -> dict[str, pd.DataFrame]:
    cells_by_acq = {primary_acq_id: load_codex_cells(primary_acq_id, sample, base_dir)}
    for extra_id in also_view or []:
        extra_id = str(extra_id).strip()
        if extra_id and extra_id not in cells_by_acq:
            cells_by_acq[extra_id] = load_codex_cells(extra_id, sample, base_dir)
    return cells_by_acq

########################################################
# 2026.08.18 LLY, add the coverslip_from_acq function
########################################################
def coverslip_from_acq(acq_id: str) -> str:
    prefix = acquisition_prefix(acq_id)
    if "_c" in prefix:
        return "c" + prefix.rsplit("_c", 1)[-1]
    return prefix


def attach_coverslip(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``coverslip`` (c001, c003, …) for stacked-bar grouping."""
    out = meta_df.copy()
    out["ACQUISITION_ID"] = out["ACQUISITION_ID"].astype(str).str.strip()
    out["coverslip"] = out["ACQUISITION_ID"].map(coverslip_from_acq)
    if "prefix" not in out.columns:
        out["prefix"] = out["ACQUISITION_ID"].map(acquisition_prefix)
    return out


def load_cells_from_mapping(
    mapping_df: pd.DataFrame,
    acq_ids: list[str] | None = None,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    skip_missing_annotation: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load cell tables for every core in ``mapping_df`` (HCC ``load_cells_from_mapping`` analog)."""
    ids = acq_ids or mapping_df["ACQUISITION_ID"].astype(str).tolist()
    cells_by_acq: dict[str, pd.DataFrame] = {}
    skipped = 0
    for i, aid in enumerate(ids, start=1):
        aid = str(aid).strip()
        if skip_missing_annotation and not has_codex_celltype_annotation(aid, sample, base_dir):
            skipped += 1
            continue
        cells_by_acq[aid] = load_codex_cells(
            aid, sample, base_dir,
            allow_missing_annotation=not skip_missing_annotation,
        )
        if verbose and (i % 50 == 0 or i == len(ids)):
            print(f"  loaded {len(cells_by_acq)}/{len(ids)} (skipped {skipped})")
    if verbose:
        print(f"Loaded {len(cells_by_acq)} dataset(s), skipped {skipped}")
    return cells_by_acq


def pool_cells_by_coverslip(
    cells_by_acq: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Concatenate cells within each coverslip (c001, c003, …)."""
    groups: dict[str, list[pd.DataFrame]] = {}
    for aid, df in cells_by_acq.items():
        groups.setdefault(coverslip_from_acq(aid), []).append(df)
    return {
        cs: pd.concat(parts, ignore_index=True)
        for cs, parts in sorted(groups.items())
    }


def celltype_count_table(
    cells_by_acq: dict[str, pd.DataFrame],
    *,
    celltype_filter: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Wide cell-count table, excluding the specified cell types."""
    parts = []
    for acq_id, df in cells_by_acq.items():
        df = _ensure_celltype_column(df, acq_id)
        parts.append(df["celltype"].value_counts().rename(acq_id))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, axis=1).fillna(0).astype(int)
    excluded = set(celltype_filter or ())
    if excluded:
        table = table.drop(index=list(excluded), errors="ignore")
    table.loc["_TOTAL_"] = table.sum(axis=0)
    return table


def summarize_celltype_distributions(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame | None = None,
    *,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_S1167_CELLTYPE_FILTER,
    verbose: bool = False,
) -> pd.DataFrame:
    """Summarize counts after excluding ``celltype_filter`` (default: ``Other``)."""
    if mapping_df is None:
        mapping_df = pd.DataFrame()
    excluded = set(celltype_filter or ())
    if verbose:
        for aid, df in cells_by_acq.items():
            row = metadata_row_for_acq(aid, mapping_df) if not mapping_df.empty else None
            label = None if row is None else row.get("SAMPLE_LABEL")
            df = _ensure_celltype_column(df, aid)
            counts = df.loc[~df["celltype"].isin(excluded), "celltype"].value_counts()
            print(f"\n=== {aid} (sample {label}) — {int(counts.sum())} retained cells ===")
            print(counts.to_string())
    table = celltype_count_table(cells_by_acq, celltype_filter=celltype_filter)
    if table.empty:
        print("\n=== Summary: 0 datasets loaded ===")
        return table
    filter_note = f" (excluded: {', '.join(sorted(excluded))})" if excluded else ""
    print(
        f"\n=== Summary: {len(cells_by_acq)} datasets, "
        f"{int(table.loc['_TOTAL_'].sum())} retained cells{filter_note} ==="
    )
    return table
########################################################

def summarize_acquisitions(
    cells_by_acq: dict[str, pd.DataFrame],
    meta_df: pd.DataFrame | None = None,
) -> None:
    for aid, df in cells_by_acq.items():
        row = None if meta_df is None else metadata_row_for_acq(aid, meta_df)
        cohort = None if row is None else row.get("cohort")
        print(f"\n=== {aid} (cohort={cohort}) — {len(df)} cells ===")
        print(
            f"  x_pix [{df['x_pix'].min()}, {df['x_pix'].max()}], "
            f"y_pix [{df['y_pix'].min()}, {df['y_pix'].max()}]"
        )
        print(df["celltype"].value_counts().to_string())


def plot_codex_spatial(
    cells_df: pd.DataFrame,
    title: str,
    save_path: str | Path,
    he_rgb_bg: np.ndarray | None = None,
    *,
    show: bool = True,
    save: bool = True,
    dpi: int = 200,
    point_size: float = 3,
) -> None:
    color_overrides = resolve_palette(
        cells_df["celltype"].unique(),
        dataset="codex_hcc",
        tier="fine",
        color_overrides=S1167_CELLTYPE_COLORS,
    )
    fig, ax = plt.subplots(figsize=(12, 10))

    if he_rgb_bg is not None:
        he_h, he_w = he_rgb_bg.shape[:2]
        ax.imshow(he_rgb_bg, extent=[0, he_w, he_h, 0], aspect="equal", zorder=0)
        ax.set_xlim(0, he_w)
        ax.set_ylim(he_h, 0)
    else:
        ax.set_aspect("equal", adjustable="box")

    for ct in sorted(cells_df["celltype"].unique(), key=str):
        sub = cells_df[cells_df["celltype"] == ct]
        rgba = color_overrides.get(str(ct), (0.5, 0.5, 0.5, 1.0))
        ax.scatter(
            sub["x_pix"], sub["y_pix"],
            s=point_size, alpha=0.75, c=[rgba], label=f"{ct} (n={len(sub)})",
            linewidths=0, zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=3, frameon=False)
    fig.tight_layout()
    if save:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def export_codex_celltypes_on_he_jpg(
    acq_id: str,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    show: bool = False,
    dpi: int = 200,
    verbose: bool = True,
) -> Path | None:
    """Plot CODEX cell types on the acquisition HE and save JPG."""
    if not has_codex_celltype_annotation(acq_id, sample, base_dir):
        if verbose:
            print(f"SKIP (no cell type annotation): {acq_id}")
        return None

    out_path = he_celltype_jpg_path(acq_id, sample, base_dir)
    if skip_if_exists and out_path.is_file() and not overwrite:
        if verbose:
            print(f"SKIP (exists): {out_path}")
        return out_path

    cells = load_codex_cells(acq_id, sample, base_dir)
    he_bg = get_he_rgb(acq_id, sample, base_dir)
    title = f"{acq_id} cell types on HE"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_codex_spatial(
        cells, title, out_path, he_rgb_bg=he_bg, show=show, dpi=dpi,
    )
    return out_path


def export_codex_celltypes_to_jpg(
    meta_df: pd.DataFrame,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    show: bool = False,
    dpi: int = 200,
    verbose: bool = True,
) -> list[Path]:
    if verbose:
        print(f"Exporting {len(meta_df)} CODEX cell-type maps → JPG")
    saved: list[Path] = []
    skipped = 0
    for aid in meta_df["ACQUISITION_ID"].astype(str).str.strip():
        out = export_codex_celltypes_on_he_jpg(
            aid, sample, base_dir,
            overwrite=overwrite,
            skip_if_exists=skip_if_exists,
            show=show,
            dpi=dpi,
            verbose=verbose,
        )
        if out is None:
            skipped += 1
        else:
            saved.append(out)
    if verbose:
        print(f"Done: {len(saved)} JPG saved, {skipped} skipped (no annotation CSV)")
    return saved


def show_saved_jpg(
    jpg_path: str | Path,
    *,
    title: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 10),
) -> Path:
    import matplotlib.image as mpimg

    path = Path(jpg_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    img = mpimg.imread(str(path))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    print(f"Displayed (existing): {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def preview_celltype_jpgs(
    meta_df: pd.DataFrame,
    sample: str = S1167_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_DIR,
    *,
    acq_ids: list[str] | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 10),
) -> list[Path]:
    ids = acq_ids or meta_df["ACQUISITION_ID"].astype(str).str.strip().tolist()
    shown: list[Path] = []
    for aid in ids:
        path = he_celltype_jpg_path(aid, sample, base_dir)
        if not path.is_file():
            print(f"SKIP (no JPG): {path}")
            continue
        show_saved_jpg(path, title=aid, show=show, figsize=figsize)
        shown.append(path)
    return shown
