#!/usr/bin/env python3
"""Prostate external validation. The pipeline is Hist2Pheno_pkg.external_validation.

Erickson et al., Nature 2022 (Mendeley svw96g68dv). HE files are JPEG under
Patient_1/ and Patient_2/, not one flat OME TIFF folder. The same section
name (H1_2, V1_2, ...) exists in both patients, so a sample id is
``Patient_1__H1_2``.

JPEG EXIF is 72 dpi and is not a pixel size. This is Visium, not HER2ST:
each section with ``Visium_with_annotation/{section}/scalefactors_json.json``
gets its own µm/px, like NPC reads OME PhysicalSize per TIFF.

Space Ranger ``spot_diameter_fullres`` is pixels across a theoretical **65 µm**
spot (10x spatial outputs). µm/px = 65 / spot_diameter_fullres. That is the
same physical scale as NPC OME (~0.465 = 65 / 139.45). Do not use BRCA_ST's
200 µm array-pitch rule, and do not use 55 / diameter (that disagrees with
NPC OME). Sections without scalefactors fall back to HEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import ExternalCohort, scale_for_mpp, uni2_overrides

PATIENTS = ("Patient_1", "Patient_2")
# 10x Space Ranger: spot_diameter_fullres spans a theoretical 65 µm spot.
# NPC OME 0.465 µm/px = 65 / patient1 spot_diameter_fullres (139.45).
_VISIUM_SPOT_DIAMETER_UM = 65.0

# subseries patient_{1,2}_{section} -> HEST pixel_size_um_estimated
_HEST_MPP = {
    "Patient_1__H1_2": 0.6875667793808392,  # MEND161, 13184×13184, 20x
    "Patient_1__H1_4": 0.5729649301384623,  # MEND160, 15821×15821
    "Patient_1__H1_5": 0.6875645406332241,  # MEND159, 13184×13184
    "Patient_1__H2_1": 0.5736952860766773,  # MEND158, 15821×15821
    "Patient_1__H2_2": 0.5729649301384618,  # MEND157, 15821×15667
    "Patient_1__H2_5": 0.5729653938552056,  # MEND156, 15821×15821
    "Patient_1__V1_2": 0.5729653938552056,  # MEND154, 15821×15821
    "Patient_2__H2_1": 0.3485874483569101,  # MEND153, 24000×21221, 40x
    "Patient_2__H2_2": 0.34924724958603126,  # MEND152, 24000×23242
    "Patient_2__H3_1": 0.34864158883039537,  # MEND151
    "Patient_2__H3_2": 0.3483775640553357,  # MEND150
    "Patient_2__H3_4": 0.3486117565183791,  # MEND149
    "Patient_2__H3_5": 0.3486920435618712,  # MEND148
    "Patient_2__H3_6": 0.34839736147236977,  # MEND147
    "Patient_2__V1_1": 0.3413129631225956,  # MEND146, 24000×24258
    "Patient_2__V1_2": 0.34141205245499373,  # MEND145
    "Patient_2__V1_3": 0.3413461313911852,  # MEND144
    "Patient_2__V1_4": 0.341624049,  # MEND143
    "Patient_2__V1_5": 0.3412951940850326,  # MEND142
    "Patient_2__V1_6": 0.3412434442674936,  # MEND141
    "Patient_2__V2_1": 0.3416810637413744,  # MEND140
    "Patient_2__V2_2": 0.3417704356235018,  # MEND139
}

# Median of the HEST estimates that share this JPEG size. Used for Patient_2
# sections that are on disk but not in the HEST table (H1_*, H2_3–H2_6, H3_3, V2_3–V2_6).
_SIZE_MPP = {
    (13184, 13184): 0.6875656600070317,
    (15821, 15667): 0.5729649301384618,
    (15821, 15821): 0.5729653938552056,
    (24000, 21221): 0.3485874483569101,
    (24000, 23242): 0.3486266726743872,
    (24000, 24258): 0.34137909192308946,
}


def _split_sample(sample: str) -> tuple[str, str]:
    patient, _, section = sample.partition("__")
    if patient not in PATIENTS or not section or "__" in section:
        raise SystemExit(
            f"sample id must look like Patient_1__H1_2, got {sample!r}"
        )
    return patient, section


def _visium_dir(patient: str, section: str) -> Path:
    return _REPO / "data" / "Visium" / "Prostate" / patient / "Visium_with_annotation" / section


def mpp_from_scalefactors(patient: str, section: str) -> tuple[float, str] | None:
    """Per-section Visium factor, analogous to NPC ``ome_um_per_px`` per TIFF."""
    path = _visium_dir(patient, section) / "scalefactors_json.json"
    if not path.is_file():
        return None
    sf = json.loads(path.read_text())
    diam_px = float(sf["spot_diameter_fullres"])
    if diam_px <= 0:
        raise RuntimeError(f"{path} has non-positive spot_diameter_fullres={diam_px}")
    mpp = _VISIUM_SPOT_DIAMETER_UM / diam_px
    return mpp, (
        f"Visium scalefactors {section} "
        f"({_VISIUM_SPOT_DIAMETER_UM:.0f} µm / spot_diameter_fullres={diam_px:.4f} px)"
    )


def mpp_for(sample: str, he_path: Path) -> tuple[float, str]:
    """Return (µm/px, source label). JPEG 72 dpi is not used."""
    patient, section = _split_sample(sample)
    vis = mpp_from_scalefactors(patient, section)
    if vis is not None:
        return vis
    if sample in _HEST_MPP:
        return _HEST_MPP[sample], "HEST spot-spacing estimate"
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(he_path) as im:
        size = im.size
    if size not in _SIZE_MPP:
        raise RuntimeError(
            f"{sample} JPEG is {size[0]}×{size[1]}, which is not in the HEST "
            "prostate size table. Do not guess a pixel size."
        )
    return _SIZE_MPP[size], f"HEST median for {size[0]}×{size[1]}"


class ProstateCohort(ExternalCohort):
    def discover_samples(self, only: str | None = None) -> list[str]:
        if isinstance(only, (list, tuple)):
            if len(only) <= 1:
                only = only[0] if only else None
            else:
                out: list[str] = []
                for item in only:
                    for name in self.discover_samples(item):
                        if name not in out:
                            out.append(name)
                return out
        names: list[str] = []
        for patient in PATIENTS:
            root = self.data_root / patient / "StarDist_Segment"
            if not root.is_dir():
                continue
            names.extend(
                f"{patient}__{p.name}"
                for p in sorted(root.iterdir())
                if p.is_dir()
            )
        if not names:
            raise SystemExit(f"no StarDist folders under {self.data_root}/Patient_*/StarDist_Segment")
        if not only:
            return names
        if only in PATIENTS:
            hit = [s for s in names if s.startswith(only + "__")]
        elif only in names:
            hit = [only]
        else:
            # series prefix: Patient_2__H1 → Patient_2__H1_1, Patient_2__H1_2, ...
            hit = [s for s in names if s.startswith(only + "_")]
        if not hit:
            raise SystemExit(
                f"sample {only!r} not found. Use Patient_1, Patient_2, "
                "Patient_2__H1, or Patient_1__H1_2."
            )
        return hit

    def sample_paths(self, sample: str) -> dict[str, Path]:
        patient, section = _split_sample(sample)
        patient_root = self.data_root / patient
        emb = (
            self.uni_root
            / sample
            / "ImgEmbeddings_all_stardist"
            / f"sc_pth_{self.patch_size}_{self.patch_size}"
        )
        vis = _visium_dir(patient, section)
        return {
            "he": patient_root / "HE_images" / f"{section}{self.he_suffix}",
            "csv": (
                patient_root
                / "StarDist_Segment"
                / section
                / self.csv_template.format(sample=section)
            ),
            "visium": vis,
            "scalefactors": vis / "scalefactors_json.json",
            "emb": emb,
            "feat": self.pred_root / sample / f"{sample}_stardist_features.h5ad",
            "pred": self.pred_root / sample / f"{sample}_stardist_pred.h5ad",
            "log": self.uni_root / sample / "logs",
        }

    def geometry(self, sample: str) -> dict:
        paths = self.sample_paths(sample)
        if not paths["he"].is_file():
            raise FileNotFoundError(paths["he"])
        mpp, source = mpp_for(sample, paths["he"])
        scale = scale_for_mpp(mpp, self.target_um_per_px)
        return {
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


from Hist2Pheno_pkg.external_validation import virchow2_overrides, uni_overrides

def cohort_for(embedder: str = "uni") -> ProstateCohort:
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
    return ProstateCohort(
        name="Prostate",
        data_root=_REPO / "data" / "Visium" / "Prostate",
        repo=_REPO,
        he_suffix=".jpg",
        **extra,
    )


def use_embedder(embedder: str = "uni") -> ProstateCohort:
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower() in {"uni", "uni2", "virchow2"}:
        use_embedder(argv.pop(0).lower())
    return COHORT.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
