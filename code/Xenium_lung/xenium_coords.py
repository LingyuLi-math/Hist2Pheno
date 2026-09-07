"""Xenium micron coordinates -> registered HE / morphology pixel coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

# 10x Xenium File Format Documentation: morphology pixel size
XENIUM_UM_PER_PX = 0.2125

CoordUnits = Literal["um", "he_pixel"]

DEFAULT_SELECT4_DIR = Path(
    "/home/lingyu/ssd2/Python/Hist2Pheno/data/Xemium/weiqin/"
    "SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4"
)
# prob 0.01 / nms 0.3 encoded without '.' for simpler paths
STARDIST_CSV_BASENAME = "_Float_prob001_nms_03.csv"
LEGACY_STARDIST_DIR = Path(
    "/home/lingyu/ssd2/Python/Hist2Pheno/data/Xemium/LUNG/StarDist_Segment"
)


def _processed_data_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
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


def load_xenium_imagealignment_matrix(path: str | Path) -> np.ndarray:
    """Load a 10x Explorer ``*_imagealignment.csv`` 3×3 affine matrix."""
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 alignment matrix in {path}, got {matrix.shape}")
    return matrix.astype(float)


def alignment_isotropic_scale(matrix: np.ndarray) -> float:
    """Similarity scale of the 2×2 linear part (``sqrt(|det A|)``)."""
    linear = np.asarray(matrix, dtype=float)[:2, :2]
    return float(np.sqrt(abs(np.linalg.det(linear))))


def microns_to_he_pixels_via_alignment(
    x_um,
    y_um,
    matrix: np.ndarray,
    *,
    um_per_px: float = XENIUM_UM_PER_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Xenium micron centroids to H&E pixels using ``he_imagealignment.csv``.

    10x documents ``C' = M @ C`` with **C** = H&E pixels and **C'** = Xenium
    morphology pixels (microns / 0.2125). Therefore::

        [x_he, y_he, 1] = M^{-1} @ [x_um / 0.2125, y_um / 0.2125, 1]

    The linear part of ``M`` often includes a reflection (negative Y), which is
    the Explorer registration that makes post-Xenium HE match morphology
    chirality — do **not** apply an extra manual Y flip on top of this.
    """
    x_arr = np.asarray(x_um, dtype=float)
    y_arr = np.asarray(y_um, dtype=float)
    inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
    homogeneous = np.column_stack(
        [x_arr / um_per_px, y_arr / um_per_px, np.ones(x_arr.shape, dtype=float)]
    )
    he_xy = homogeneous @ inverse.T
    return he_xy[:, 0], he_xy[:, 1]

###########################################################
# 2026.09.04, for brca, add function to scale HE OME-TIFF pixels onto a separately exported working .tif
# 2026.09.04 / 2026.09.07: map official HE OME-TIFF → working ``*_he_image.tif``
#
# BRCA preview working TIFs are **same-resolution crops** of the OME (not
# anisotropic full-frame resizes). Using width/height ratios alone left a
# systematic ~15–20 px structural offset vs StarDist (visible on tissue
# landmarks). Prefer ``mode="crop"`` with an estimated top-left origin.
###########################################################
def estimate_he_ome_to_working_tif_crop_origin(
    ome_path: str | Path,
    tif_path: str | Path,
    *,
    downsample: int = 8,
) -> tuple[int, int]:
    """Estimate top-left crop origin mapping OME → working TIF (same pixel size).

    Uses FFT cross-correlation on downsampled grayscale, then a small full-res
    NCC refine (±4 px). Returns ``(x0, y0)`` such that
    ``working ≈ ome[y0:y0+H, x0:x0+W]``.
    """
    import tifffile
    from PIL import Image
    from scipy.signal import fftconvolve

    with tifffile.TiffFile(ome_path) as tiff:
        ome = tiff.pages[0].asarray()
    with tifffile.TiffFile(tif_path) as tiff:
        tif = tiff.pages[0].asarray()
    ome_h, ome_w = ome.shape[:2]
    tif_h, tif_w = tif.shape[:2]
    ds = max(int(downsample), 1)
    ome_ds = np.asarray(
        Image.fromarray(ome).resize((max(1, ome_w // ds), max(1, ome_h // ds)), Image.BILINEAR)
    )
    tif_ds = np.asarray(
        Image.fromarray(tif).resize((max(1, tif_w // ds), max(1, tif_h // ds)), Image.BILINEAR)
    )
    ome_g = ome_ds.mean(-1).astype(np.float64) if ome_ds.ndim == 3 else ome_ds.astype(np.float64)
    tif_g = tif_ds.mean(-1).astype(np.float64) if tif_ds.ndim == 3 else tif_ds.astype(np.float64)
    corr = fftconvolve(ome_g - ome_g.mean(), (tif_g - tif_g.mean())[::-1, ::-1], mode="valid")
    jy, jx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    ox0, oy0 = int(jx * ds), int(jy * ds)

    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        a0 = a.astype(np.float64) - float(a.mean())
        b0 = b.astype(np.float64) - float(b.mean())
        return float((a0 * b0).mean() / (a0.std() * b0.std() + 1e-8))

    best = (-1.0, (ox0, oy0))
    probes = (
        (tif_h // 2, tif_w // 2),
        (min(900, tif_h // 4), max(tif_w - 900, 0)),
        (max(tif_h - 900, 0), min(900, tif_w // 4)),
    )
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            x0, y0 = ox0 + dx, oy0 + dy
            scores: list[float] = []
            for ty, tx in probes:
                size = 256
                if ty + size > tif_h or tx + size > tif_w:
                    continue
                if y0 + ty + size > ome_h or x0 + tx + size > ome_w:
                    continue
                if y0 + ty < 0 or x0 + tx < 0:
                    continue
                tp = tif[ty : ty + size, tx : tx + size]
                op = ome[y0 + ty : y0 + ty + size, x0 + tx : x0 + tx + size]
                tp = tp.mean(-1) if tp.ndim == 3 else tp
                op = op.mean(-1) if op.ndim == 3 else op
                scores.append(_ncc(tp, op))
            if scores:
                score = float(np.mean(scores))
                if score > best[0]:
                    best = (score, (x0, y0))
    return int(best[1][0]), int(best[1][1])
###########################################################

def scale_he_ome_pixels_to_working_tif(
    x_ome,
    y_ome,
    *,
    ome_width: int,
    ome_height: int,
    tif_width: int,
    tif_height: int,
    mode: Literal["crop", "resize"] = "crop",
    crop_origin: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map HE OME-TIFF pixels onto a separately exported working ``.tif``.

    Official ``*_he_imagealignment.csv`` is defined on ``*_he_image.ome.tif``.
    StarDist / UNI use ``*_he_image.tif``.

    Parameters
    ----------
    mode
        ``"crop"`` (default): same-resolution window; ``tif = ome - origin``.
        ``"resize"``: independent X/Y full-frame scale (legacy; usually wrong
        for the BRCA preview exports).
    crop_origin
        ``(x0, y0)`` top-left of the working TIF inside the OME. Required for
        ``mode="crop"`` (use :func:`estimate_he_ome_to_working_tif_crop_origin`).
    """
    x = np.asarray(x_ome, dtype=float)
    y = np.asarray(y_ome, dtype=float)
    if mode == "resize":
        return (
            x * (float(tif_width) / float(ome_width)),
            y * (float(tif_height) / float(ome_height)),
        )
    if mode != "crop":
        raise ValueError(f"Unknown mode={mode!r}; expected 'crop' or 'resize'")
    if crop_origin is None:
        raise ValueError("mode='crop' requires crop_origin=(x0, y0)")
    x0, y0 = float(crop_origin[0]), float(crop_origin[1])
    return x - x0, y - y0


def microns_to_he_working_tif_pixels(
    x_um,
    y_um,
    matrix: np.ndarray,
    *,
    ome_width: int,
    ome_height: int,
    tif_width: int,
    tif_height: int,
    um_per_px: float = XENIUM_UM_PER_PX,
    mode: Literal["crop", "resize"] = "crop",
    crop_origin: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Xenium µm → working HE ``.tif`` pixels via Explorer alignment + OME→tif map."""
    x_ome, y_ome = microns_to_he_pixels_via_alignment(
        x_um, y_um, matrix, um_per_px=um_per_px
    )
    return scale_he_ome_pixels_to_working_tif(
        x_ome,
        y_ome,
        ome_width=ome_width,
        ome_height=ome_height,
        tif_width=tif_width,
        tif_height=tif_height,
        mode=mode,
        crop_origin=crop_origin,
    )


def he_working_tif_um_per_px(
    matrix: np.ndarray,
    *,
    ome_width: int,
    ome_height: int,
    tif_width: int,
    tif_height: int,
    um_per_px: float = XENIUM_UM_PER_PX,
    mode: Literal["crop", "resize"] = "crop",
) -> float:
    """Approximate µm/px on the working HE ``.tif`` after alignment + OME→tif map."""
    ome_um = um_per_px * alignment_isotropic_scale(matrix)
    if mode == "crop":
        # Same pixel size as OME HE after Explorer registration.
        return float(ome_um)
    sx = float(ome_width) / float(tif_width)
    sy = float(ome_height) / float(tif_height)
    return float(ome_um * np.sqrt(sx * sy))
###########################################################