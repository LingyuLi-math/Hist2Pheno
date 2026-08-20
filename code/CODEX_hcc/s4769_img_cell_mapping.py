## 2026.06.19 load the codex cell annotation (.csv), transform the HE image(.zarr) to .tif, and save the celltype annotation on the HE image(.jpg)
## 2026.06.20 add the function to load the celltype annotation from the CODEX_celltype_annotation.csv


"""s4769 CODEX / Visium ↔ aligned HE (Zarr) loading and spatial visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.patches import Rectangle

DEFAULT_CODEX_HCC_DIR = (
    Path(__file__).resolve().parents[2] / "data/CODEX/HCC/Michael_data_transfer"
)
S4769_SAMPLE = "s4769"
DEFAULT_HE_MAPPING_XLSX = "HE/s4769_he_mapping_updated_Visium.xlsx"
DEFAULT_HE_MAPPING_SHEET = "Clinical_info"
DEFAULT_CELLTYPE_SHEET = "Celltype"

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
from plotting_palettes import DEFAULT_CELLTYPE_FILTER, resolve_palette  # noqa: E402

_he_rgb_cache: dict[str, np.ndarray] = {}


def clear_he_rgb_cache() -> None:
    _he_rgb_cache.clear()


def he_mapping_path(
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    xlsx_name: str = DEFAULT_HE_MAPPING_XLSX,
) -> Path:
    return Path(base_dir) / sample / xlsx_name


def visium_dir(
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    return Path(base_dir) / sample / "Visium"


def he_img_path(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    return Path(base_dir) / sample / "HE" / he_key / "he_img"


def display_label_to_visium_csv_name(display_label: str) -> str:
    """Map CODEX_REGION_DISPLAY_LABEL (e.g. 71_2) → Visium CSV filename."""
    patient, region = str(display_label).split("_", 1)
    return f"{int(patient):03d}_{region}_tissue_positions_list.csv"


def visium_csv_path(
    display_label: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    return visium_dir(sample, base_dir) / display_label_to_visium_csv_name(display_label)


def load_visium_he_mapping(
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    xlsx_name: str = DEFAULT_HE_MAPPING_XLSX,
    sheet_name: str = DEFAULT_HE_MAPPING_SHEET,
    *,
    visium_only: bool = True,
) -> pd.DataFrame:
    """Load HE↔CODEX↔Visium mapping table; default keeps Visium=='Y' rows only."""
    path = he_mapping_path(sample, base_dir, xlsx_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    if visium_only:
        df = df[df["Visium"].astype(str).str.upper() == "Y"].copy()
    df = df.reset_index(drop=True)
    return df


## 2026.06.18 LLY — CODEX↔HE alignment (ALIGNED=='Y', 38 regions)
## 2026.08.14 LLY — optional Annotation=='Y' (36 annotated regions)
def load_codex_he_alignment(
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    xlsx_name: str = DEFAULT_HE_MAPPING_XLSX,
    sheet_name: str = DEFAULT_HE_MAPPING_SHEET,
    *,
    aligned_only: bool = True,
    annotated_only: bool = False,
) -> pd.DataFrame:
    """Load HE↔CODEX mapping table.

    ``aligned_only=True`` keeps ``ALIGNED=='Y'`` (38 regions).
    ``annotated_only=True`` further keeps ``Annotation=='Y'`` (36 regions).
    """
    path = he_mapping_path(sample, base_dir, xlsx_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    if aligned_only:
        df = df[df["ALIGNED"].astype(str).str.upper() == "Y"].copy()
    if annotated_only:
        df = filter_alignment_with_annotation(df)
    return df.reset_index(drop=True)


def load_codex_celltype_hierarchy(
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    xlsx_name: str = DEFAULT_HE_MAPPING_XLSX,
    sheet_name: str = DEFAULT_CELLTYPE_SHEET,
    *,
    celltype_filter: list[str] | tuple[str, ...] | None = DEFAULT_CELLTYPE_FILTER,
) -> pd.DataFrame:
    """Load the HCC Level2→Level1→Level0 cell-type hierarchy from Excel."""
    path = he_mapping_path(sample, base_dir, xlsx_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    hierarchy_cols = ["celltype_level2", "celltype_level1", "celltype_level0"]
    missing = [col for col in hierarchy_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{sheet_name!r} sheet missing hierarchy columns: {missing}")

    out = df[hierarchy_cols].copy()
    for col in hierarchy_cols:
        out[col] = out[col].map(
            lambda value: str(value).strip() if pd.notna(value) else pd.NA
        )

    excluded = {str(value).strip() for value in (celltype_filter or ())}
    out = out[~out["celltype_level2"].isin(excluded)].copy()
    out = out.dropna(subset=hierarchy_cols).drop_duplicates().reset_index(drop=True)

    duplicated_l2 = out.loc[
        out["celltype_level2"].duplicated(keep=False), "celltype_level2"
    ].unique()
    if len(duplicated_l2):
        raise ValueError(
            "Each celltype_level2 must have one hierarchy mapping; duplicates: "
            f"{sorted(duplicated_l2.tolist())}"
        )
    return out


def mapping_row_for_acq(acq_id: str, mapping_df: pd.DataFrame) -> pd.Series | None:
    rows = mapping_df[mapping_df["CODEX_ACQUISITION_ID"].astype(str) == str(acq_id)]
    if rows.empty:
        return None
    return rows.iloc[0]


def he_key_for_acq(acq_id: str, mapping_df: pd.DataFrame) -> str | None:
    row = mapping_row_for_acq(acq_id, mapping_df)
    if row is None:
        return None
    val = row.get("MATCHED_HE")
    if pd.isna(val) or not str(val).strip():
        return None
    return str(val).strip()


def display_label_for_acq(acq_id: str, mapping_df: pd.DataFrame) -> str | None:
    row = mapping_row_for_acq(acq_id, mapping_df)
    if row is None:
        return None
    val = row.get("CODEX_REGION_DISPLAY_LABEL")
    if pd.isna(val) or not str(val).strip():
        return None
    return str(val).strip()





###################################################################
# 2026.06.19 LLY, transform the HE image(.zarr) to .tif
###################################################################
def he_tif_path(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    """Output TIFF path: HE/{he_key}/figures/{he_key}.tif"""
    return he_img_path(he_key, sample, base_dir).parent / "figures" / f"{he_key}.tif"


def export_he_zarr_to_tif(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> Path:
    """Convert Zarr he_img (CHW uint8 RGB) → {he_key}.tif under figures/."""
    import tifffile

    src = he_img_path(he_key, sample, base_dir)
    if not src.is_dir():
        raise FileNotFoundError(src)

    out_path = he_tif_path(he_key, sample, base_dir)
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


def export_aligned_he_zarrs_to_tif(
    alignment_df: pd.DataFrame | None = None,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Export all unique MATCHED_HE Zarr images from ALIGNED=='Y' mapping to TIFF."""
    if alignment_df is None:
        alignment_df = load_codex_he_alignment(sample=sample, base_dir=base_dir)
    he_keys = alignment_df["MATCHED_HE"].astype(str).str.strip().unique().tolist()
    if verbose:
        print(f"Exporting {len(he_keys)} HE Zarr → TIFF")
    saved: list[Path] = []
    for he_key in he_keys:
        saved.append(
            export_he_zarr_to_tif(
                he_key, sample, base_dir,
                overwrite=overwrite, skip_if_exists=skip_if_exists, verbose=verbose,
            )
        )
    return saved
###################################################################


###################################################################
# 2026.06.19 LLY, save the aligned codex celltypes on HE image(.jpg)
# Note: s4769\FinalLiv-27_c001_v001_r001_reg012\, there is no celltype annotation
###################################################################
def he_celltype_jpg_path(
    acq_id: str,
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> Path:
    """Output cell-type spatial JPG: HE/{he_key}/figures/{acq_id}_celltype_on_HE.jpg"""
    ## use the HE image name
    # return he_img_path(he_key, sample, base_dir).parent / "figures" / f"{he_key}.jpg"   
    ## use the acq_id name
    return he_img_path(he_key, sample, base_dir).parent / "figures" / f"{acq_id}_celltype_on_HE.jpg"  


def export_codex_celltypes_on_he_jpg(
    acq_id: str,
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    region_label: str | None = None,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    show: bool = False,
    dpi: int = 200,
    verbose: bool = True,
) -> Path | None:
    """Plot CODEX cell types on MATCHED_HE and save as {acq_id}_celltype_on_HE.jpg."""
    cell_data_path, cell_types_path = codex_cell_annotation_paths(acq_id, sample, base_dir)
    if not cell_types_path.is_file():
        if verbose:
            print(
                f"SKIP (no cell type annotation): {acq_id}\n"
                f"  missing: {cell_types_path}"
            )
        return None
    if not cell_data_path.is_file():
        if verbose:
            print(
                f"SKIP (no cell data): {acq_id}\n"
                f"  missing: {cell_data_path}"
            )
        return None

    out_path = he_celltype_jpg_path(acq_id, he_key, sample, base_dir)
    if skip_if_exists and out_path.is_file() and not overwrite:
        if verbose:
            print(f"SKIP (exists): {out_path}")
        return out_path

    cells = load_codex_cells(acq_id, sample, base_dir)
    he_bg = get_he_rgb(he_key, sample, base_dir)
    label_txt = f" ({region_label})" if region_label else ""
    title = f"{acq_id}{label_txt} cell types on HE\nMATCHED_HE={he_key}"
    plot_codex_spatial(
        cells, title, out_path, he_rgb_bg=he_bg, show=show, dpi=dpi,
    )
    return out_path


def export_aligned_codex_celltypes_to_jpg(
    alignment_df: pd.DataFrame | None = None,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    overwrite: bool = False,
    skip_if_exists: bool = True,
    show: bool = False,
    dpi: int = 200,
    verbose: bool = True,
) -> list[Path]:
    """Export CODEX cell-type spatial plots for all ALIGNED=='Y' rows (38 regions)."""
    if alignment_df is None:
        alignment_df = load_codex_he_alignment(sample=sample, base_dir=base_dir)
    if verbose:
        print(f"Exporting {len(alignment_df)} CODEX cell-type maps → JPG")
    saved: list[Path] = []
    skipped = 0
    for _, row in alignment_df.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"]).strip()
        he_key = str(row["MATCHED_HE"]).strip()
        region_label = str(row.get("CODEX_REGION_DISPLAY_LABEL", "")).strip() or None
        out = export_codex_celltypes_on_he_jpg(
            acq_id, he_key, sample, base_dir,
            region_label=region_label,
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
        print(f"Done: {len(saved)} JPG saved, {skipped} skipped (no annotation CSV or missing data)")
    return saved

###################################################################


###################################################################
# 2026.06.19 LLY, normalize the HE image(.zarr) to .tif
# Note: some Zarr stores float 0–255, so we need to normalize it to uint8
# s4769/HE/awy-98938_aligned_8ded610e
# s4769/HE/awy-98938_aligned_a5afeef9
# s4769/HE/awy-98938_aligned_59f27598
###################################################################
def _normalize_he_rgb_hwc(rgb: np.ndarray) -> np.ndarray:
    """Ensure HWC RGB is uint8 for imshow / TIFF export (some Zarr stores float 0–255)."""
    rgb = np.asarray(rgb)
    if rgb.dtype.kind == "f":
        return np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.dtype != np.uint8:
        return rgb.astype(np.uint8)
    return rgb
###################################################################

def load_he_rgb(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Load Zarr RGB HE (CHW) → HWC uint8 numpy array."""
    p = he_img_path(he_key, sample, base_dir)
    z_he = zarr.open(str(p), mode="r")
    if verbose:
        print(f"HE {he_key}: shape={z_he.shape}, dtype={z_he.dtype}")
    rgb = _normalize_he_rgb_hwc(np.transpose(z_he[:], (1, 2, 0)))
    if verbose and z_he.dtype.kind == "f":
        print(f"  → converted float HE to uint8 for display")
    return rgb


def get_he_rgb(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> np.ndarray:
    cache_key = f"{base_dir}|{sample}|{he_key}"
    if cache_key not in _he_rgb_cache:
        _he_rgb_cache[cache_key] = load_he_rgb(he_key, sample, base_dir, verbose=False)
    return _he_rgb_cache[cache_key]


def plot_he_overview(
    he_key: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    crop_half: int = 800,
    save: bool = True,
    show: bool = True,
    dpi: int = 150,
) -> tuple[np.ndarray, Path | None]:
    """Full HE + center crop with red box; optionally save overview JPEG."""
    he_rgb = load_he_rgb(he_key, sample, base_dir)
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

    fig.suptitle(f"s4769 aligned HE preview ({he_key})", y=1.02)
    fig.tight_layout()

    save_path = None
    if save:
        fig_dir = he_img_path(he_key, sample, base_dir).parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / "he_img_overview.jpg"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return he_rgb, save_path


###################################################################
# 2026.06.19 LLY, load the codex cell annotation (.csv)
###################################################################
def codex_cell_annotation_paths(
    acq_id: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> tuple[Path, Path]:
    acq_dir = Path(base_dir) / sample / acq_id
    return (
        acq_dir / f"{acq_id}.cell_data.csv",
        acq_dir / f"{acq_id}.cell_types.csv",
    )


def has_codex_celltype_annotation(
    acq_id: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> bool:
    cell_data_path, cell_types_path = codex_cell_annotation_paths(acq_id, sample, base_dir)
    return cell_data_path.is_file() and cell_types_path.is_file()


## 2026.06.19 LLY, filter the alignment_df with Annotation=='Y'
def filter_alignment_with_annotation(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with Annotation=='Y', or fallback to existing cell_types.csv on disk."""
    if "Annotation" in alignment_df.columns:
        return alignment_df[
            alignment_df["Annotation"].astype(str).str.upper() == "Y"
        ].copy().reset_index(drop=True)
    rows = []
    for _, row in alignment_df.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"])
        if has_codex_celltype_annotation(acq_id):
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _ensure_celltype_column(df: pd.DataFrame, acq_id: str = "") -> pd.DataFrame:
    """Normalize merged CODEX table to always include a ``celltype`` column."""
    if "celltype" in df.columns:
        return df
    for col in ("ANNOTATION_LABEL", "ANNOTATION_LABEL_y", "annotation_label", "cell_type", "CellType"):
        if col in df.columns:
            return df.rename(columns={col: "celltype"})
    raise KeyError(
        f"No celltype column for {acq_id!r}; available columns: {list(df.columns)}"
    )


def load_codex_cells(
    acq_id: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> pd.DataFrame:
    cell_data_path, cell_types_path = codex_cell_annotation_paths(acq_id, sample, base_dir)
    for fp in (cell_data_path, cell_types_path):
        if not fp.is_file():
            raise FileNotFoundError(fp)
    merged = (
        pd.read_csv(cell_data_path)
        .merge(pd.read_csv(cell_types_path), on="CELL_ID", how="inner")
        .rename(columns={"X": "x_pix", "Y": "y_pix", "ANNOTATION_LABEL": "celltype"})
    )
    return _ensure_celltype_column(merged, acq_id)

def load_visium_tissue_positions(
    display_label: str,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    in_tissue_only: bool = True,
) -> pd.DataFrame:
    """Load Visium spot list for CODEX_REGION_DISPLAY_LABEL (HE full-res pixel coords)."""
    csv_path = visium_csv_path(display_label, sample, base_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if in_tissue_only and "in_tissue" in df.columns:
        df = df[df["in_tissue"] == 1].copy()
    return df.rename(
        columns={
            "pxl_col_in_fullres": "x_pix",
            "pxl_row_in_fullres": "y_pix",
        }
    )

def load_visium_for_acq(
    acq_id: str,
    mapping_df: pd.DataFrame,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    **kwargs,
) -> pd.DataFrame:
    label = display_label_for_acq(acq_id, mapping_df)
    if label is None:
        raise KeyError(f"No CODEX_REGION_DISPLAY_LABEL for {acq_id!r} in mapping table")
    return load_visium_tissue_positions(label, sample, base_dir, **kwargs)


def load_cells_by_acq(
    primary_acq_id: str,
    also_view: list[str] | None = None,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> dict[str, pd.DataFrame]:
    cells_by_acq = {primary_acq_id: load_codex_cells(primary_acq_id, sample, base_dir)}
    for extra_id in also_view or []:
        extra_id = str(extra_id).strip()
        if extra_id and extra_id not in cells_by_acq:
            cells_by_acq[extra_id] = load_codex_cells(extra_id, sample, base_dir)
    return cells_by_acq


## 2026.06.19 LLY, Adjust, load the cells from the mapping table
def load_cells_from_mapping(
    mapping_df: pd.DataFrame,
    acq_ids: list[str] | None = None,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    *,
    skip_missing_annotation: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    ids = acq_ids or mapping_df["CODEX_ACQUISITION_ID"].astype(str).tolist()
    cells_by_acq: dict[str, pd.DataFrame] = {}
    skipped = 0
    for aid in ids:
        aid = str(aid).strip()
        if skip_missing_annotation and not has_codex_celltype_annotation(aid, sample, base_dir):
            if verbose:
                _, types_path = codex_cell_annotation_paths(aid, sample, base_dir)
                print(f"SKIP (no annotation): {aid} — missing {types_path.name}")
            skipped += 1
            continue
        cells_by_acq[aid] = load_codex_cells(aid, sample, base_dir)
    if verbose:
        print(f"Loaded {len(cells_by_acq)} dataset(s), skipped {skipped}")
    return cells_by_acq


#####################################################
# 2026.06.19 LLY, count the celltype distribution
# 2026.08.10 LLY, add the celltype_filter parameter to exclude the unknown and stroma uncharacterized cells
#####################################################
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

#####################################################
## 2026.06.19 LLY, summarize the celltype distribution
# 2026.08.10 LLY, add the celltype_filter parameter to exclude the unknown and stroma uncharacterized cells
#####################################################
def summarize_celltype_distributions(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame | None = None,
    *,
    celltype_filter: list[str] | tuple[str, ...] | None = (
        "Unknown",
        "Stroma Uncharacterized",
    ),
) -> pd.DataFrame:
    """Print and summarize counts after excluding ``celltype_filter``."""
    if mapping_df is None:
        mapping_df = pd.DataFrame()
    excluded = set(celltype_filter or ())
    for aid, df in cells_by_acq.items():
        row = mapping_row_for_acq(aid, mapping_df) if not mapping_df.empty else None
        label = None if row is None else row.get("CODEX_REGION_DISPLAY_LABEL")
        df = _ensure_celltype_column(df, aid)
        counts = df.loc[~df["celltype"].isin(excluded), "celltype"].value_counts()
        print(f"\n=== {aid} (region {label}) — {int(counts.sum())} retained cells ===")
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


## 2026.08.14 LLY, summarize the CODEX↔HE alignment
def summarize_codex_he_alignment(
    mapping_df: pd.DataFrame,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> None:
    """Print CODEX↔HE alignment rows with HE Zarr / Visium CSV / Annotation checks."""
    print(f"CODEX↔HE alignment rows: {len(mapping_df)}")
    for _, row in mapping_df.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"])
        label = str(row["CODEX_REGION_DISPLAY_LABEL"])
        he_key = str(row["MATCHED_HE"])
        he_path = he_img_path(he_key, sample, base_dir)
        vis_path = visium_csv_path(label, sample, base_dir)
        has_visium = str(row.get("Visium", "")).upper() == "Y"
        has_anno = str(row.get("Annotation", "")).upper() == "Y"
        print(
            f"\n=== {acq_id} | {label} ===\n"
            f"  MATCHED_HE={he_key!r}  exists={he_path.is_dir()}\n"
            f"  ALIGNED={str(row.get('ALIGNED', '')).upper()}  "
            f"Annotation={has_anno}  Visium={has_visium}  "
            f"CSV={vis_path.name!r}  exists={vis_path.is_file()}"
        )


def summarize_visium_he_mapping(
    mapping_df: pd.DataFrame,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
) -> None:
    """Print mapping table with Visium CSV / HE Zarr existence checks."""
    print(f"Visium mapping rows: {len(mapping_df)}")
    for _, row in mapping_df.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"])
        label = str(row["CODEX_REGION_DISPLAY_LABEL"])
        he_key = str(row["MATCHED_HE"])
        vis_path = visium_csv_path(label, sample, base_dir)
        he_path = he_img_path(he_key, sample, base_dir)
        print(
            f"\n=== {acq_id} | {label} ===\n"
            f"  MATCHED_HE={he_key!r}  exists={he_path.is_dir()}\n"
            f"  Visium CSV={vis_path.name!r}  exists={vis_path.is_file()}"
        )


def is_he_aligned(acq_id: str, mapping_df: pd.DataFrame) -> bool:
    return he_key_for_acq(acq_id, mapping_df) is not None

def summarize_acquisitions(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame,
) -> None:
    for aid, df in cells_by_acq.items():
        row = mapping_row_for_acq(aid, mapping_df)
        label = None if row is None else row.get("CODEX_REGION_DISPLAY_LABEL")
        he_key = he_key_for_acq(aid, mapping_df)
        aligned = is_he_aligned(aid, mapping_df)
        print(f"\n=== {aid} (region {label}) — {len(df)} cells ===")
        print(f"  MATCHED_HE={he_key!r}  →  has aligned HE: {aligned}")
        if not aligned:
            print("  ⚠ No MATCHED_HE in mapping table.")
        print(
            f"  x_pix [{df['x_pix'].min()}, {df['x_pix'].max()}], "
            f"y_pix [{df['y_pix'].min()}, {df['y_pix'].max()}]"
        )
        print(df["celltype"].value_counts().to_string())


###################################################################
# 2026.06.19 LLY, show the saved celltype on HE image(.jpg)
###################################################################
def show_saved_jpg(
    jpg_path: str | Path,
    *,
    title: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 10),
) -> Path:
    """Display an existing JPG without re-rendering or saving."""
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


def plot_celltype_jpg(
    filename: str | Path,
    *,
    he_key: str | None = None,
    sample: str = S4769_SAMPLE,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    title: str | None = None,
    show: bool = True,
) -> Path:
    """Show a saved cell-type-on-HE JPG by filename (optionally under HE/{he_key}/figures/)."""
    path = Path(filename)
    if not path.is_file():
        if he_key is None:
            raise FileNotFoundError(path)
        path = he_img_path(he_key, sample, base_dir).parent / "figures" / path.name
    # return show_saved_jpg(path, title=title or path.name, show=show)
    return show_saved_jpg(path, title=None, show=show)

###################################################################
# 2026.06.19 LLY, preview the aligned celltype JPG
###################################################################
def preview_aligned_celltype_jpgs(
    alignment_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    sample: str = S4769_SAMPLE,
    acq_ids: list[str] | None = None,
    *,
    show: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Display saved {acq_id}_celltype_on_HE.jpg for rows in alignment_df (no cells CSV needed)."""
    rows = alignment_df
    if acq_ids is not None:
        acq_set = set(map(str, acq_ids))
        rows = alignment_df[alignment_df["CODEX_ACQUISITION_ID"].astype(str).isin(acq_set)]

    shown: list[Path] = []
    for _, row in rows.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"]).strip()
        he_key = str(row["MATCHED_HE"]).strip()
        label = str(row.get("CODEX_REGION_DISPLAY_LABEL", "")).strip()
        jpg_path = he_celltype_jpg_path(acq_id, he_key, sample, base_dir)
        title = f"{acq_id} ({label}) cell types on HE\nMATCHED_HE={he_key}"
        if not jpg_path.is_file():
            if verbose:
                print(f"SKIP (no JPG): {jpg_path}")
            continue
        # shown.append(show_saved_jpg(jpg_path, title=title, show=show))
        shown.append(show_saved_jpg(jpg_path, title=None, show=show))
    if verbose:
        print(f"Displayed {len(shown)} JPG(s)")
    return shown

###################################################################

###################################################################
# 2026.06.19 LLY, plot the codex cell spatial distribution on HE image(.jpg)
###################################################################
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


def plot_visium_spatial(
    spots_df: pd.DataFrame,
    title: str,
    save_path: str | Path,
    he_rgb_bg: np.ndarray | None = None,
    *,
    show: bool = True,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    if he_rgb_bg is not None:
        he_h, he_w = he_rgb_bg.shape[:2]
        ax.imshow(he_rgb_bg, extent=[0, he_w, he_h, 0], aspect="equal", zorder=0)
        ax.set_xlim(0, he_w)
        ax.set_ylim(he_h, 0)
    else:
        ax.set_aspect("equal", adjustable="box")

    ax.scatter(
        spots_df["x_pix"], spots_df["y_pix"],
        s=12, alpha=0.85, c="#e41a1c", linewidths=0, zorder=2,
        label=f"Visium spots (n={len(spots_df)})",
    )
    ax.set_title(title)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_acq_spatial(
    cells_by_acq: dict[str, pd.DataFrame],
    mapping_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    sample: str = S4769_SAMPLE,
    *,
    show: bool = True,
    reuse_existing: bool = True,
    save: bool = True,
) -> None:
    """Plot each CODEX acquisition on its MATCHED_HE from the alignment table.

    If reuse_existing=True and the JPG already exists, only display it (no re-save).
    """
    for plot_acq_id, plot_cells in cells_by_acq.items():
        he_key = he_key_for_acq(plot_acq_id, mapping_df)
        label = display_label_for_acq(plot_acq_id, mapping_df)
        if he_key:
            out_path = he_celltype_jpg_path(plot_acq_id, he_key, sample, base_dir)
            title = f"{plot_acq_id} ({label}) cell types on HE\nMATCHED_HE={he_key}"
            if reuse_existing and out_path.is_file():
                show_saved_jpg(out_path, title=title, show=show)
                continue
            he_bg = get_he_rgb(he_key, sample, base_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plot_codex_spatial(
                plot_cells, title, out_path, he_rgb_bg=he_bg, show=show, save=save,
            )
        else:
            fig_dir = Path(base_dir) / sample / plot_acq_id / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            title = f"{plot_acq_id} cell types (CODEX pixel space)\nno MATCHED_HE in mapping"
            out_path = fig_dir / f"{plot_acq_id}_celltype_codex_only.jpg"
            if reuse_existing and out_path.is_file():
                show_saved_jpg(out_path, title=title, show=show)
                continue
            plot_codex_spatial(plot_cells, title, out_path, he_rgb_bg=None, show=show, save=save)


def plot_all_visium_on_he(
    mapping_df: pd.DataFrame,
    base_dir: str | Path = DEFAULT_CODEX_HCC_DIR,
    sample: str = S4769_SAMPLE,
    acq_ids: list[str] | None = None,
    *,
    show: bool = True,
) -> None:
    """Plot Visium tissue spots on MATCHED_HE for each mapped acquisition."""
    rows = mapping_df
    if acq_ids is not None:
        acq_set = set(map(str, acq_ids))
        rows = mapping_df[mapping_df["CODEX_ACQUISITION_ID"].astype(str).isin(acq_set)]

    for _, row in rows.iterrows():
        acq_id = str(row["CODEX_ACQUISITION_ID"])
        label = str(row["CODEX_REGION_DISPLAY_LABEL"])
        he_key = str(row["MATCHED_HE"])
        spots = load_visium_tissue_positions(label, sample, base_dir)
        he_bg = get_he_rgb(he_key, sample, base_dir)
        fig_dir = he_img_path(he_key, sample, base_dir).parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        title = f"{acq_id} ({label}) Visium spots on HE\nMATCHED_HE={he_key}"
        out_path = fig_dir / f"{label.replace('_', '-')}_visium_on_HE.jpg"
        plot_visium_spatial(spots, title, out_path, he_rgb_bg=he_bg, show=show)
