"""Xenium micron coordinates -> registered HE / morphology pixel coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

# 10x Xenium File Format Documentation: morphology pixel size
XENIUM_UM_PER_PX = 0.2125

CoordUnits = Literal["um", "he_pixel"]

DEFAULT_SELECT4_DIR = Path(
    "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/Xemiun/weiqin/"
    "SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4"
)
# prob 0.01 / nms 0.3 encoded without '.' for simpler paths
STARDIST_CSV_BASENAME = "_Float_prob001_nms_03.csv"
LEGACY_STARDIST_DIR = Path(
    "/home/lingyu/ssd2/Python/Collaborate/esccAI/data/Xemiun/lung/StarDist_Segment"
)


def _processed_data_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
    )


def default_cases_roots() -> tuple[Path, ...]:
    data_root = _processed_data_root() / "Data"
    return (data_root / "Complete_Cases", data_root / "Incomplete_Cases")


def select4_stardist_csv_path(
    sample: str,
    select4_dir: str | Path | None = None,
) -> Path:
    """StarDist nuclei CSV per sample under Complete_Cases_Select4/{sample}/."""
    root = Path(select4_dir) if select4_dir is not None else DEFAULT_SELECT4_DIR
    return root / sample / f"{sample}{STARDIST_CSV_BASENAME}"


def select4_stardist_matched_csv_path(
    sample: str,
    select4_dir: str | Path | None = None,
) -> Path:
    """Xenium labels + StarDist centroid_x/y after ``match_celltype2stardist``."""
    root = Path(select4_dir) if select4_dir is not None else DEFAULT_SELECT4_DIR
    # return root / sample / f"{sample}_stardist_cells_matched.csv"    # 2026.06.24 old
    return root / sample / f"{sample}_cells_matched_by_stardist.csv"    # 2026.06.24 new


def sample_coord_units_map(
    cases_roots: Sequence[Path | str] | None = None,
) -> dict[str, CoordUnits]:
    """Map paper sample folder name -> centroid coordinate units.

    Complete_Cases centroids are in µm; Incomplete_Cases centroids are already
    full-resolution HE pixel coordinates in Weiqin's annotation table.
    """
    roots = tuple(Path(p) for p in (cases_roots or default_cases_roots()))
    out: dict[str, CoordUnits] = {}
    for root in roots:
        if not root.is_dir():
            continue
        units: CoordUnits = "he_pixel" if root.name == "Incomplete_Cases" else "um"
        for sample_dir in sorted(root.iterdir()):
            if sample_dir.is_dir():
                out[sample_dir.name] = units
    return out


def centroid_to_he_pixel(
    x_centroid: pd.Series,
    y_centroid: pd.Series,
    *,
    units: CoordUnits = "um",
    um_per_px: float = XENIUM_UM_PER_PX,
) -> tuple[pd.Series, pd.Series]:
    """Convert centroid columns to full-resolution HE pixel coordinates."""
    if units == "he_pixel":
        return x_centroid, y_centroid
    return x_centroid / um_per_px, y_centroid / um_per_px


def _resolve_coord_units(
    df: pd.DataFrame,
    coord_units: CoordUnits | dict[str, CoordUnits] | None,
    cases_roots: Sequence[Path | str] | None,
) -> CoordUnits | dict[str, CoordUnits]:
    if coord_units is not None:
        return coord_units
    if "sample" not in df.columns:
        return "um"
    return sample_coord_units_map(cases_roots)


def add_he_pixel_columns(
    df: pd.DataFrame,
    um_per_px: float = XENIUM_UM_PER_PX,
    *,
    coord_units: CoordUnits | dict[str, CoordUnits] | None = None,
    cases_roots: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Add X_pix_HE, Y_pix_HE immediately after x_centroid, y_centroid.

  Parameters
  ----------
  coord_units
      Per-table coordinate interpretation. ``"um"`` divides centroids by
      ``um_per_px``; ``"he_pixel"`` copies centroids as HE full-res pixels.
      When omitted and a ``sample`` column is present, units are inferred from
      whether each sample lives under Complete_Cases (um) or Incomplete_Cases
      (he_pixel). Unknown samples default to ``"um"``.
  cases_roots
      Case roots used for auto-detection (default: Complete_Cases +
      Incomplete_Cases under the processed data tree).
    """
    if "x_centroid" not in df.columns or "y_centroid" not in df.columns:
        raise KeyError("DataFrame must contain x_centroid and y_centroid")

    out = df.copy()
    for col in ("X_pix_HE", "Y_pix_HE"):
        if col in out.columns:
            out = out.drop(columns=[col])

    units_spec = _resolve_coord_units(out, coord_units, cases_roots)

    if isinstance(units_spec, str):
        x_pix, y_pix = centroid_to_he_pixel(
            out["x_centroid"], out["y_centroid"], units=units_spec, um_per_px=um_per_px
        )
    else:
        sample_units = out["sample"].map(units_spec).fillna("um")
        x_pix = pd.Series(index=out.index, dtype=float)
        y_pix = pd.Series(index=out.index, dtype=float)
        for units in ("um", "he_pixel"):
            mask = sample_units == units
            if not mask.any():
                continue
            x_conv, y_conv = centroid_to_he_pixel(
                out.loc[mask, "x_centroid"],
                out.loc[mask, "y_centroid"],
                units=units,
                um_per_px=um_per_px,
            )
            x_pix.loc[mask] = x_conv
            y_pix.loc[mask] = y_conv

    y_idx = int(out.columns.get_loc("y_centroid")) + 1
    out.insert(y_idx, "X_pix_HE", x_pix)
    out.insert(y_idx + 1, "Y_pix_HE", y_pix)
    return out
