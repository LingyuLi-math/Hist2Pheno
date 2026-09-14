"""Transfer Visium HD RCTD labels onto Xenium CRC cells.

Analog of ``code/Xenium_brca/match_xenium_cells_with_pixel.py``, but CRC has
no per-cell Xenium GT. Each Xenium cell is assigned the nearest on-tissue
8 µm Visium HD bin's ``DeconvolutionLabel1`` (Flex Level2); Level1 comes
from the Flex 2025 majority map in ``SingleCell_MetaData_2025.csv``.

Writes ``data/Xemium/CRC/Annotation/CRC_Barcode_Cell_Type_Matrices.xlsx``
(BRCA-style ``celltype`` sheet + ``P{1,2,5}_CRC`` sample sheets).

  python code/Xenium_crc/match_xenium_cells_with_visiumhd.py
  python code/Xenium_crc/match_xenium_cells_with_visiumhd.py --patient P2CRC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from crc_paths import (  # noqa: E402
    CELLTYPE_SHEET_COLS,
    CrcPaths,
    PATIENTS,
    build_celltype_hierarchy,
    l12_to_l1,
    label_xenium_from_visium,
    load_bin_meta,
    load_xenium_cells,
    majority_l2_to_l1,
    map_status_table,
    map_xenium_to_bins,
    normalize_celltype_sheet,
)

PATIENT_SHEET_COLS = [
    "Barcode",
    "Cluster",
    "celltype_level1",
    "celltype_level12",
    "celltype_level2",
    "map_status",
    "dist_um",
    "nearest_barcode",
    "x_centroid",
    "y_centroid",
    "x_centroid_visium_scale",
    "y_centroid_visium_scale",
    "transcript_counts",
]


def labeled_xenium_cells(
    name: str,
    vis_df: pd.DataFrame,
    paths: CrcPaths,
    *,
    xen: pd.DataFrame | None = None,
    map_result: dict | None = None,
    labeled: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map Xenium cells to bins and attach Visium HD Label1 / L1 / map_status."""
    if labeled is not None:
        return labeled, map_result or {}
    xen = load_xenium_cells(name, paths) if xen is None else xen
    if map_result is None:
        map_result = map_xenium_to_bins(name, vis_df, paths, xen=xen)
    return label_xenium_from_visium(map_result["xen"], vis_df), map_result


def format_patient_sheet(
    labeled: pd.DataFrame,
    l2_to_l1: dict[str, str],
    *,
    keep_status: tuple[str, ...] | None = ("singlet",),
) -> pd.DataFrame:
    """BRCA-style sample sheet: ``Barcode`` + ``Cluster`` (L2) plus L1 / mapping QC.

    Default keeps only cells whose nearest bin is RCTD ``singlet`` with a Flex
    L2 name. Pass ``keep_status=None`` to keep every mapped cell that has Label1.
    """
    out = labeled.copy()
    if keep_status is not None:
        out = out.loc[out["map_status"].isin(keep_status)].copy()
    out = out.loc[out["Label1_from_VisiumHD"].notna()].copy()
    out = out.loc[out["Label1_from_VisiumHD"].isin(l2_to_l1)].copy()
    out["Barcode"] = out["cell_id"]
    out["Cluster"] = out["Label1_from_VisiumHD"]
    out["celltype_level2"] = out["Cluster"]
    out["celltype_level12"] = out["Cluster"].map(l2_to_l1)
    out["celltype_level1"] = l12_to_l1(out["celltype_level12"])
    missing = [c for c in PATIENT_SHEET_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"patient sheet missing {missing}")
    return out[PATIENT_SHEET_COLS].reset_index(drop=True)


def remap_patient_sheet_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Fill coarse L1 from Flex L12; keep extra QC columns."""
    out = df.copy()
    if "celltype_level12" not in out.columns:
        raise KeyError("patient sheet needs celltype_level12")
    out["celltype_level1"] = l12_to_l1(out["celltype_level12"].astype("string"))
    extra = [c for c in out.columns if c not in PATIENT_SHEET_COLS]
    cols = [c for c in PATIENT_SHEET_COLS if c in out.columns] + extra
    return out[cols]


def refresh_annotation_hierarchy(
    xlsx_path: Path | None = None,
    *,
    maj: pd.DataFrame | None = None,
) -> Path:
    """Rewrite ``celltype`` + sample sheets with 4-class L1; do not remap cells."""
    paths = CrcPaths()
    path = Path(xlsx_path) if xlsx_path is not None else paths.annotation_xlsx
    xf = pd.ExcelFile(path)
    celltype = (
        build_celltype_hierarchy(maj)
        if maj is not None
        else normalize_celltype_sheet(pd.read_excel(path, sheet_name="celltype"))
    )
    sample_sheets = {}
    for sheet in xf.sheet_names:
        if sheet == "celltype":
            continue
        sample_sheets[sheet] = remap_patient_sheet_hierarchy(
            pd.read_excel(path, sheet_name=sheet)
        )
    write_crc_annotation_xlsx(path, celltype, sample_sheets, preserve_other=False)
    print(
        f"refreshed {path}  celltype L1={celltype['celltype_level1'].nunique(dropna=True)}  "
        f"L12={celltype['celltype_level12'].nunique(dropna=True)}  "
        f"sheets={['celltype', *sample_sheets]}"
    )
    return path


def write_crc_annotation_xlsx(
    path: Path,
    celltype: pd.DataFrame,
    sample_sheets: dict[str, pd.DataFrame],
    *,
    preserve_other: bool = True,
) -> Path:
    """Write ``celltype`` + named sample sheets; keep any other existing sheets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, pd.DataFrame] = {}
    if preserve_other and path.is_file():
        xf = pd.ExcelFile(path)
        skip = {"celltype", *sample_sheets}
        for sheet in xf.sheet_names:
            if sheet not in skip:
                existing[sheet] = pd.read_excel(path, sheet_name=sheet)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        celltype[CELLTYPE_SHEET_COLS].to_excel(writer, sheet_name="celltype", index=False)
        for sheet_name, df in sample_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        for sheet_name, df in existing.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def transfer_visiumhd_annotation(
    name: str,
    vis_df: pd.DataFrame,
    paths: CrcPaths,
    l2_to_l1: dict[str, str],
    *,
    xen: pd.DataFrame | None = None,
    map_result: dict | None = None,
    labeled: pd.DataFrame | None = None,
    keep_status: tuple[str, ...] | None = ("singlet",),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(labeled_all, patient_sheet, map_result)`` for one patient."""
    labeled_all, map_result = labeled_xenium_cells(
        name,
        vis_df,
        paths,
        xen=xen,
        map_result=map_result,
        labeled=labeled,
    )
    sheet = format_patient_sheet(labeled_all, l2_to_l1, keep_status=keep_status)
    return labeled_all, sheet, map_result


def process_patient(
    name: str,
    *,
    paths: CrcPaths | None = None,
    maj: pd.DataFrame | None = None,
    l2_to_l1: dict[str, str] | None = None,
    vis_df: pd.DataFrame | None = None,
    xen: pd.DataFrame | None = None,
    map_result: dict | None = None,
    labeled: pd.DataFrame | None = None,
    keep_status: tuple[str, ...] | None = ("singlet",),
    xlsx_path: Path | None = None,
    write: bool = True,
) -> dict:
    """Build hierarchy + P*_CRC sheet; optionally write / update the workbook."""
    paths = paths or CrcPaths()
    if maj is None or l2_to_l1 is None:
        sc = pd.read_csv(paths.sc_path)
        keep = sc.loc[sc["QCFilter"] == "Keep"].copy()
        maj, l2_to_l1, _ = majority_l2_to_l1(keep)
    assert l2_to_l1 is not None
    if vis_df is None:
        vis_df = load_bin_meta(name, paths, l2_to_l1)
    labeled_all, sheet, map_result = transfer_visiumhd_annotation(
        name,
        vis_df,
        paths,
        l2_to_l1,
        xen=xen,
        map_result=map_result,
        labeled=labeled,
        keep_status=keep_status,
    )
    celltype = build_celltype_hierarchy(maj)
    out: dict = {
        "patient": name,
        "sheet_name": paths.patient_sheet_name(name),
        "celltype": celltype,
        "sheet": sheet,
        "labeled": labeled_all,
        "map_result": map_result,
        "xlsx": None,
    }
    print(f"{name} map_status (all Xenium cells)")
    print(map_status_table(labeled_all).round(2).to_string())
    print(
        f"{name} → sheet {out['sheet_name']}: {len(sheet):,} cells  "
        f"L2={sheet['Cluster'].nunique()}  "
        f"L12={sheet['celltype_level12'].nunique()}  "
        f"L1={sheet['celltype_level1'].nunique()}"
    )
    if write:
        xlsx_path = Path(xlsx_path) if xlsx_path is not None else paths.annotation_xlsx
        write_crc_annotation_xlsx(
            xlsx_path,
            celltype,
            {out["sheet_name"]: sheet},
        )
        out["xlsx"] = xlsx_path
        print(f"  wrote {xlsx_path}  sheets=celltype,{out['sheet_name']}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transfer Visium HD RCTD Label1 onto Xenium CRC cells."
    )
    p.add_argument(
        "--patient",
        choices=PATIENTS,
        default=None,
        help="One patient (default: all P1CRC / P2CRC / P5CRC).",
    )
    p.add_argument("--xlsx", type=Path, default=None)
    p.add_argument(
        "--keep-status",
        nargs="*",
        default=["singlet"],
        help="RCTD classes to keep (default: singlet). Pass none to keep all labeled.",
    )
    p.add_argument("--no-write", action="store_true")
    p.add_argument(
        "--refresh-hierarchy",
        action="store_true",
        help="Only recompute celltype_level1 on an existing workbook (no remapping).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_hierarchy:
        refresh_annotation_hierarchy(args.xlsx)
        return
    keep = tuple(args.keep_status) if args.keep_status else None
    patients = [args.patient] if args.patient else list(PATIENTS)
    paths = CrcPaths()
    sc = pd.read_csv(paths.sc_path)
    keep_sc = sc.loc[sc["QCFilter"] == "Keep"].copy()
    maj, l2_to_l1, _ = majority_l2_to_l1(keep_sc)
    for name in patients:
        process_patient(
            name,
            paths=paths,
            maj=maj,
            l2_to_l1=l2_to_l1,
            keep_status=keep,
            xlsx_path=args.xlsx,
            write=not args.no_write,
        )


if __name__ == "__main__":
    main()
