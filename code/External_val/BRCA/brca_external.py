#!/usr/bin/env python3
"""BRCA external validation. Pipeline: Hist2Pheno_pkg.external_validation.

HE + StarDist slides under ``data/Visium/BRCA``:

1. ``GSM7782699_tissue_image`` — OME TIFF (~0.548 µm/px), no Visium JSON.
2. ``200715_BC_SA2_C_D_V19F24-113.B1-Spot000001`` — pathology JPEG from
   SpatialVDJ Zenodo raw images; Visium ID ``V10F24-113_B1``.
3. ``200219_BC_SA3_V19T26-031_CE_V4-Spot000001`` — same cohort;
   Visium ID ``V19T26-031_D1``.
4. ``V1_Breast_Cancer_Block_A_Section_1_image`` — 10x public fullres TIFF.
   It has no OME ``PhysicalSize``. Space Ranger outs are
   ``10x_data/spatial``, and this TIFF is that fullres image.

The two Zenodo JPEGs are **not** Space Ranger fullres. They are ~3.333× larger
than ``outs/spatial`` fullres (inferred from ``tissue_hires`` /
``tissue_hires_scalef``). µm/px on the StarDist HE is therefore::

    (65 / spot_diameter_fullres) / (HE_width / SR_fullres_width)

Do **not** use raw ``65 / spot_diameter_fullres`` on these JPEGs (that is the
SR-fullres scale ≈ 0.57 µm/px). When the HE is the Space Ranger fullres image
the ratio is 1 and the same formula is the fullres scale. GSM keeps OME
``PhysicalSize``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import (  # noqa: E402
    ExternalCohort,
    ome_um_per_px,
    uni2_overrides,
    scale_for_mpp,
)

_VISIUM_SPOT_DIAMETER_UM = 65.0

DATA_ROOT = _REPO / "data" / "Visium" / "BRCA"
SPACERANGER_ROOT = (
    DATA_ROOT
    / "npj_data"
    / "SpatialVDJ_forZenodo"
    / "data"
    / "breast_cancer"
    / "5_spatial_GEX"
    / "spaceranger_output"
)
SAMPLES_XLSX = SPACERANGER_ROOT / "Breast_Cancer_Visium_samples.xlsx"

# Fallback if the Excel map is missing. Image basename → Visium capture area.
_STARDIST_TO_VISIUM_FALLBACK = {
    "200715_BC_SA2_C_D_V19F24-113.B1-Spot000001": "V10F24-113_B1",
    "200219_BC_SA3_V19T26-031_CE_V4-Spot000001": "V19T26-031_D1",
}


def load_stardist_to_visium(
    xlsx_path: Path | None = None,
) -> dict[str, str]:
    """Map StarDist / HE stem → Visium ID from the Zenodo sample sheet."""
    path = Path(xlsx_path) if xlsx_path is not None else SAMPLES_XLSX
    if not path.is_file():
        return dict(_STARDIST_TO_VISIUM_FALLBACK)
    try:
        import pandas as pd
    except ImportError:
        return dict(_STARDIST_TO_VISIUM_FALLBACK)

    frame = pd.read_excel(path)
    need = {"Visium ID", "Raw image"}
    if not need.issubset(frame.columns):
        return dict(_STARDIST_TO_VISIUM_FALLBACK)
    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        raw = str(row["Raw image"]).strip().replace("\\", "/")
        stem = Path(raw).stem
        vid = str(row["Visium ID"]).strip()
        if stem and vid and stem != "nan" and vid != "nan":
            out[stem] = vid
    # Keep the pathology-annotation pair even if the sheet is incomplete.
    for stem, vid in _STARDIST_TO_VISIUM_FALLBACK.items():
        out.setdefault(stem, vid)
    return out


STARDIST_TO_VISIUM = load_stardist_to_visium()

# 10x public slide: the StarDist TIFF is Space Ranger fullres, not an OME scan.
_PUBLIC_10X_SPATIAL = {
    "V1_Breast_Cancer_Block_A_Section_1_image": DATA_ROOT / "10x_data" / "spatial",
}


def visium_spatial_dir(visium_id: str) -> Path:
    """``…/spaceranger_output/{VisiumID}/outs/spatial`` (or legacy ``spatial``)."""
    base = SPACERANGER_ROOT / visium_id
    for cand in (base / "outs" / "spatial", base / "spatial"):
        if cand.is_dir():
            return cand
    return base / "outs" / "spatial"


def resolve_he_path(he_dir: Path, sample: str) -> Path:
    """Prefer ``.tif`` (GSM OME), then ``.jpg`` / ``.jpeg`` (Zenodo raw)."""
    for suffix in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        path = he_dir / f"{sample}{suffix}"
        if path.is_file():
            return path
    return he_dir / f"{sample}.tif"


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def he_over_sr_fullres_ratio(
    he_path: Path,
    scalefactors_path: Path,
    hires_path: Path,
) -> tuple[float, float, float]:
    """Return ``(ratio_w, sr_fullres_w, sr_fullres_h)`` for HE vs SR fullres."""
    sf = json.loads(scalefactors_path.read_text())
    hires_scale = float(sf["tissue_hires_scalef"])
    if hires_scale <= 0:
        raise RuntimeError(f"{scalefactors_path} has non-positive tissue_hires_scalef")
    he_w, he_h = _image_size(he_path)
    hr_w, hr_h = _image_size(hires_path)
    sr_w = hr_w / hires_scale
    sr_h = hr_h / hires_scale
    ratio_w = he_w / sr_w
    ratio_h = he_h / sr_h
    if abs(ratio_w - ratio_h) > 0.02:
        raise RuntimeError(
            f"{he_path.name}: HE/SR scale differs in x/y "
            f"({ratio_w:.4f} vs {ratio_h:.4f})"
        )
    if ratio_w < 0.98:
        raise RuntimeError(
            f"{he_path.name}: HE is smaller than SR fullres "
            f"(ratio={ratio_w:.4f}); check image pairing"
        )
    return float(ratio_w), float(sr_w), float(sr_h)

## 2026.11.01, check for public 10x spatial, add local json check for Zenodo raw images,
def spatial_for_sample(sample: str) -> tuple[str, Path] | None:
    """``(label, spatial dir)`` with ``scalefactors_json.json``, or None."""
    local = _PUBLIC_10X_SPATIAL.get(sample)
    if local is not None and (local / "scalefactors_json.json").is_file():
        return sample, local
    visium_id = STARDIST_TO_VISIUM.get(sample)
    if visium_id is None:
        return None
    spatial = visium_spatial_dir(visium_id)
    if (spatial / "scalefactors_json.json").is_file():
        return visium_id, spatial
    return None


## 2026.11.01, adjust the pathway
def mpp_from_spatial(
    he_path: Path,
    spatial: Path,
    label: str,
) -> tuple[float, str, dict]:
    """µm/px on the StarDist HE. Divide by HE/SR when the scan is larger than fullres."""
    sf_path = spatial / "scalefactors_json.json"
    hires_path = spatial / "tissue_hires_image.png"
    if not sf_path.is_file():
        raise FileNotFoundError(sf_path)
    if not hires_path.is_file():
        raise FileNotFoundError(hires_path)

    sf = json.loads(sf_path.read_text())
    diam_px = float(sf["spot_diameter_fullres"])
    if diam_px <= 0:
        raise RuntimeError(f"{sf_path} has non-positive spot_diameter_fullres={diam_px}")
    mpp_sr = _VISIUM_SPOT_DIAMETER_UM / diam_px
    ratio, sr_w, sr_h = he_over_sr_fullres_ratio(he_path, sf_path, hires_path)
    mpp_he = mpp_sr / ratio
    meta = {
        "visium_id": label,
        "mpp_sr_fullres": mpp_sr,
        "he_over_sr_fullres": ratio,
        "sr_fullres_wh": [sr_w, sr_h],
        "spot_diameter_fullres": diam_px,
        "scalefactors": str(sf_path),
    }
    source = (
        f"Visium {label}: "
        f"({_VISIUM_SPOT_DIAMETER_UM:.0f}/spot_diameter_fullres={diam_px:.4f}) "
        f"/ HE÷SR={ratio:.4f}"
    )
    return float(mpp_he), source, meta

## 2026.11.01, add local json check for Zenodo raw images
def mpp_from_visium_he(
    he_path: Path,
    visium_id: str,
) -> tuple[float, str, dict]:
    """µm/px on a Zenodo pathology JPEG, corrected for HE vs SR fullres scale."""
    return mpp_from_spatial(he_path, visium_spatial_dir(visium_id), visium_id)


def mpp_for(sample: str, he_path: Path) -> tuple[float, str, dict]:
    """Return ``(µm/px, source label, extra meta)`` for one BRCA sample."""
    found = spatial_for_sample(sample)
    if found is not None:
        label, spatial = found
        return mpp_from_spatial(he_path, spatial, label)
    if he_path.suffix.lower() in {".tif", ".tiff"}:
        mpp_x, mpp_y = ome_um_per_px(he_path)
        if abs(mpp_x - mpp_y) > 1e-3:
            raise RuntimeError(f"{sample} anisotropic pixel size {mpp_x} vs {mpp_y}")
        return float(mpp_x), "OME PhysicalSize", {}
    raise RuntimeError(
        f"{sample}: no Visium scalefactors and HE is not an OME TIFF ({he_path})"
    )


class BrcaCohort(ExternalCohort):
    """Flat ``StarDist_Segment/`` discovery with per-sample HE suffix and µm/px."""

    def sample_paths(self, sample: str) -> dict[str, Path]:
        emb = (
            self.uni_root
            / sample
            / "ImgEmbeddings_all_stardist"
            / f"sc_pth_{self.patch_size}_{self.patch_size}"
        )
        he = resolve_he_path(self.he_dir, sample)
        out: dict[str, Path] = {
            "he": he,
            "csv": self.stardist_root / sample / self.csv_template.format(sample=sample),
            "emb": emb,
            "feat": self.pred_root / sample / f"{sample}_stardist_features.h5ad",
            "pred": self.pred_root / sample / f"{sample}_stardist_pred.h5ad",
            "log": self.uni_root / sample / "logs",
        }
        found = spatial_for_sample(sample)
        if found is not None:
            _label, spatial = found
            out["visium"] = spatial
            out["scalefactors"] = spatial / "scalefactors_json.json"
            out["tissue_positions"] = spatial / "tissue_positions_list.csv"
        return out

    def geometry(self, sample: str) -> dict:
        paths = self.sample_paths(sample)
        if not paths["he"].is_file():
            raise FileNotFoundError(paths["he"])
        mpp, source, meta = mpp_for(sample, paths["he"])
        scale = scale_for_mpp(mpp, self.target_um_per_px)
        geo = {
            "sample": sample,
            "mpp_um": mpp,
            "mpp_source": source,
            "scale": scale,
            "scale_image": abs(scale - 1.0) > 1e-3,
            "patch_size_px": self.patch_size,
            "patch_um_after_rescale": self.patch_um,
            "patch_um_if_native_16px": self.patch_size * mpp,
            "he": str(paths["he"]),
            "csv": str(paths["csv"]),
            "embeddings": str(paths["emb"]),
        }
        if meta:
            geo.update(meta)
        return geo


from Hist2Pheno_pkg.external_validation import virchow2_overrides, uni_overrides

def cohort_for(embedder: str = "uni") -> BrcaCohort:
    extra = {}
    tag = str(embedder).lower()
    if tag == "uni":
        extra = uni_overrides(_REPO)
    elif tag == "uni2":
        extra = uni2_overrides(_REPO)
    elif tag == "virchow2":
        extra = virchow2_overrides(_REPO)
    elif tag != "uni":
        raise SystemExit(f"unknown embedder {embedder!r}. Choose uni, uni2 or virchow2")
    return BrcaCohort(
        name="BRCA",
        data_root=DATA_ROOT,
        repo=_REPO,
        he_suffix=".tif",  # overridden per sample in sample_paths / resolve_he_path
        **extra,
    )


def use_embedder(embedder: str = "uni") -> BrcaCohort:
    """Rebind module-level helpers. Default import stays UNI."""
    global COHORT, PRED_ROOT, discover_samples, sample_paths, geometry
    global composition_table, ensure_l2_probs, plot_spatial, plot_composition, cmd_plot
    COHORT = cohort_for(embedder)
    PRED_ROOT = COHORT.pred_root
    discover_samples = COHORT.discover_samples
    sample_paths = COHORT.sample_paths
    geometry = COHORT.geometry
    composition_table = COHORT.composition_table
    ensure_l2_probs = COHORT.ensure_l2_probs
    plot_spatial = COHORT.plot_spatial
    plot_composition = COHORT.plot_composition
    cmd_plot = COHORT.cmd_plot
    return COHORT


COHORT = cohort_for("uni")

PRED_ROOT = COHORT.pred_root
discover_samples = COHORT.discover_samples
sample_paths = COHORT.sample_paths
geometry = COHORT.geometry
composition_table = COHORT.composition_table
ensure_l2_probs = COHORT.ensure_l2_probs
plot_spatial = COHORT.plot_spatial
plot_composition = COHORT.plot_composition
cmd_plot = COHORT.cmd_plot


def plot_roi_spatial(
    sample: str,
    roi_tsv: str | Path,
    *,
    marker_px: float = 5.0,
    dpi: float = 200,
    figsize: tuple[float, float] | None = None,
    save: bool = False,
    save_dpi: int = 300,
    pred=None,
    in_roi=None,
    **kwargs,
):
    """ROI L1/L12/L2 spatial (HE + points). Parameters stay in the notebook call."""
    import anndata as ad

    from Hist2Pheno_pkg.panel_figures import (
        load_rectangle_roi,
        nuclei_in_rectangle,
        plot_roi_pred_spatial,
    )

    roi = load_rectangle_roi(roi_tsv)
    x0, y0 = roi["x_min_px"], roi["y_min_px"]
    x1, y1 = roi["x_max_px"], roi["y_max_px"]
    paths = sample_paths(sample)
    COHORT.ensure_l2_probs(sample,allow_cpu=True)  # 2026.10.28, ROI must use current prediction
    current=ad.read_h5ad(paths["pred"])
    if pred is not None:
        if pred.uns.get("pipeline_run_id")!=current.uns.get("pipeline_run_id"):
            raise ValueError("Supplied ROI prediction is stale; reload current predictions")
    else:
        pred=current
    if in_roi is None:
        xy = __import__("numpy").asarray(pred.obsm["spatial_HE"], dtype="float64")
        in_roi = nuclei_in_rectangle(xy, x0, y0, x1, y1)
    if figsize is None:
        figsize = (20.4, 14.82)
    save_path = None
    if save:
        fig_dir = COHORT.pred_root / "figures"  # 2026.10.28, live cutoff folder
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / f"{sample}_ROI_spatial_L1_L12_L2.png"
    title = kwargs.pop(
        "title",
        f"BRCA {sample}  QuPath ROI  "
        f"[{x0:.0f}:{x1:.0f}]×[{y0:.0f}:{y1:.0f}]  (no GT)",
    )
    fig, in_roi = plot_roi_pred_spatial(
        pred,
        paths["he"],
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        palettes=COHORT._palette(),
        in_roi=in_roi,
        marker_px=marker_px,
        dpi=dpi,
        figsize=figsize,
        title=title,
        save_path=save_path,
        save_dpi=save_dpi,
        **kwargs,
    )
    if save_path is not None:
        print(f"saved {save_path}")
    return fig, pred, in_roi, roi


def plot_roi_composition(
    sample: str,
    *,
    pred=None,
    in_roi=None,
    roi_tsv: str | Path | None = None,
    dpi: float = 200,
    figsize: tuple[float, float] = (12.0, 4.2),
    save: bool = False,
    save_dpi: int = 300,
    **kwargs,
):
    """ROI predicted cell-type 100% stacked composition (L1 / L12 / L2)."""
    import anndata as ad
    import numpy as np

    from Hist2Pheno_pkg.panel_figures import (
        load_rectangle_roi,
        nuclei_in_rectangle,
        plot_pred_composition_stacked,
    )

    COHORT.ensure_l2_probs(sample,allow_cpu=True)
    current=ad.read_h5ad(sample_paths(sample)["pred"])
    if pred is not None and pred.uns.get("pipeline_run_id")!=current.uns.get("pipeline_run_id"):
        raise ValueError("Supplied ROI prediction is stale")
    if pred is None:pred=current
    if in_roi is None:
        if roi_tsv is None:
            raise ValueError("Need in_roi from plot_roi_spatial, or pass roi_tsv.")
        roi = load_rectangle_roi(roi_tsv)
        xy = np.asarray(pred.obsm["spatial_HE"], dtype=np.float64)
        in_roi = nuclei_in_rectangle(
            xy, roi["x_min_px"], roi["y_min_px"], roi["x_max_px"], roi["y_max_px"]
        )
    obs = pred.obs.iloc[np.flatnonzero(in_roi)].copy()
    save_path = csv_path = None
    if save:
        fig_dir = COHORT.pred_root / "figures"  # 2026.10.28, live cutoff folder
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / f"{sample}_ROI_composition_L1_L12_L2.png"
        csv_path = fig_dir / f"{sample}_ROI_composition.csv"
    title = kwargs.pop("title", f"BRCA {sample}  QuPath ROI composition (pred)")
    fig, comp_df = plot_pred_composition_stacked(
        obs,
        COHORT._palette(),
        figsize=figsize,
        dpi=dpi,
        title=title,
        save_path=save_path,
        save_dpi=save_dpi,
        csv_path=csv_path,
        **kwargs,
    )
    if save_path is not None:
        print(f"saved {save_path}")
    return fig, comp_df


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower() in {"uni", "uni2", "virchow2"}:
        use_embedder(argv.pop(0).lower())
    return COHORT.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
