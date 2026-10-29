#!/usr/bin/env python3
"""HER2ST BRCA external validation. The pipeline is Hist2Pheno_pkg.external_validation.

Andersson et al., Nat Commun 2021. Thirty-six JPEG HE slides and StarDist
nuclei under data/ST/BRCA. No cell-type ground truth.

The JPEGs have no OME PhysicalSize. The 72 dpi tag is not a pixel size.
µm/px is the median pixel distance between array-adjacent ST spots
(centre-to-centre 200 µm) on that same image, from
data/ST/BRCA/her2st/data/ST-spotfiles/{section}_selection.tsv.
Patient A (9769_*_small) is coarser (~0.92 µm/px). The other seven patients
are ~0.69 µm/px. Spot coordinates fall inside the HE image.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "code") not in sys.path:
    sys.path.insert(0, str(_REPO / "code"))

from Hist2Pheno_pkg.external_validation import ExternalCohort, scale_for_mpp, uni2_overrides

# HER2ST section id -> HE / StarDist sample name, and µm/px from the 200 µm pitch.
_SLIDES: dict[str, tuple[str, float]] = {
    "A1": ("9769_C1_HE_small", 0.916525),
    "A2": ("9769_C2_HE_small", 0.918573),
    "A3": ("9769_D1_HE_small", 0.921836),
    "A4": ("9769_D2_HE_small", 0.914118),
    "A5": ("9769_E1_HE_small", 0.920926),
    "A6": ("9769_E2_HE_small", 0.917772),
    "B1": ("BT10737_C1_HE", 0.691349),
    "B2": ("BT10737_C2_HE", 0.689757),
    "B3": ("BT10737_D1_HE", 0.689419),
    "B4": ("BT10737_D2_HE", 0.688671),
    "B5": ("BT10737_E1_HE", 0.689440),
    "B6": ("BT10737_E2_HE", 0.690188),
    "C1": ("5714_HE_BT_C1", 0.688073),
    "C2": ("5714_HE_BT_C2", 0.689897),
    "C3": ("5714_HE_BT_D1", 0.687792),
    "C4": ("5714_HE_BT_D2", 0.685502),
    "C5": ("5714_HE_BT_E1", 0.683437),
    "C6": ("5714_HE_BT_E2", 0.687308),
    "D1": ("10426_HE_BT_C1", 0.686300),
    "D2": ("10426_HE_BT_C2", 0.686715),
    "D3": ("10426_HE_BT_D1", 0.688329),
    "D4": ("10426_HE_BT_D2", 0.686012),
    "D5": ("10426_HE_BT_E1", 0.687616),
    "D6": ("10426_HE_BT_E2", 0.687900),
    "E1": ("HE_BT23567_D2", 0.687381),
    "E2": ("HE_BT23567_E1", 0.687433),
    "E3": ("HE_BT23567_E2", 0.687757),
    "F1": ("HE_BT23810_D2", 0.687637),
    "F2": ("HE_BT23810_E1", 0.689176),
    "F3": ("HE_BT23810_E2", 0.688735),
    "G1": ("HE_BT23903_C1a", 0.687496),
    "G2": ("HE_BT23903_C2a", 0.688763),
    "G3": ("HE_BT23903_D1a", 0.689472),
    "H1": ("HE_BT24044_D2", 0.686670),
    "H2": ("HE_BT24044_E1", 0.686655),
    "H3": ("HE_BT24044_E2", 0.686582),
}

_BY_NAME = {name: (section, mpp) for section, (name, mpp) in _SLIDES.items()}


def mpp_for(sample: str) -> tuple[float, str]:
    if sample not in _BY_NAME:
        raise RuntimeError(
            f"{sample} is not in the HER2ST pixel-size table. "
            "Do not guess a pixel size from 72 dpi."
        )
    section, mpp = _BY_NAME[sample]
    return mpp, f"ST 200 µm spot pitch ({section})"


class BrcaStCohort(ExternalCohort):
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
        names = super().discover_samples(None)
        known = [name for name in names if name in _BY_NAME]
        missing = [name for name in names if name not in _BY_NAME]
        if missing:
            raise SystemExit(
                "StarDist folders with no pixel size: " + ", ".join(missing)
            )
        if not only:
            return known
        if only in known:
            return [only]
        if only in _SLIDES:
            return [_SLIDES[only][0]]
        if len(only) == 1 and only.isalpha():
            letter = only.upper()
            hit = [
                _SLIDES[sec][0]
                for sec in _SLIDES
                if sec.startswith(letter) and _SLIDES[sec][0] in known
            ]
            if hit:
                return hit
        raise SystemExit(
            f"sample {only!r} not found. Use a HER2ST id (A1), a patient "
            "letter (A–H), or the HE stem (9769_C1_HE_small)."
        )

    def geometry(self, sample: str) -> dict:
        paths = self.sample_paths(sample)
        if not paths["he"].is_file():
            raise FileNotFoundError(paths["he"])
        mpp, source = mpp_for(sample)
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

def cohort_for(embedder: str = "uni") -> BrcaStCohort:
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
    return BrcaStCohort(
        name="BRCA_ST",
        data_root=_REPO / "data" / "ST" / "BRCA",
        repo=_REPO,
        he_suffix=".jpg",
        **extra,
    )


def use_embedder(embedder: str = "uni") -> BrcaStCohort:
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
