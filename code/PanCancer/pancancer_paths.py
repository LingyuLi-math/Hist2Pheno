#!/usr/bin/env python3
"""Locate matched HE h5ads for the seven public cohorts (ESCC omitted)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

PDAC_COVERSLIPS = ("c001", "c003")
GIST_COVERSLIPS = ("c009", "c011", "c013")


@dataclass(frozen=True)
class CohortSpec:
    name: str
    tech: str
    pan_organ: str
    cases_root: Path
    # glob relative to cases_root
    h5ad_glob: str
    native_col: str = "final_CT"


def _coverslip_ok(path: Path, coverslips: tuple[str, ...]) -> bool:
    name = path.parent.name
    return any(f"_{c}_" in name for c in coverslips)


COHORTS: dict[str, CohortSpec] = {
    "Lung": CohortSpec(
        name="Lung",
        tech="Xenium",
        pan_organ="xenium_lung",
        cases_root=_REPO_ROOT
        / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "BRCA": CohortSpec(
        name="BRCA",
        tech="Xenium",
        pan_organ="xenium_brca",
        cases_root=_REPO_ROOT / "data/Xemium/BRCA/Cases",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "CRC": CohortSpec(
        name="CRC",
        tech="Xenium",
        pan_organ="xenium_crc",
        cases_root=_REPO_ROOT / "data/Xemium/CRC/Cases",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "HCC": CohortSpec(
        name="HCC",
        tech="CODEX",
        pan_organ="codex_hcc",
        cases_root=_REPO_ROOT / "data/CODEX/HCC/Michael_data_transfer/s4769/HE",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "PDAC": CohortSpec(
        name="PDAC",
        tech="CODEX",
        pan_organ="codex_pdac",
        cases_root=_REPO_ROOT / "data/CODEX/HCC/Michael_data_transfer/s1167",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "GIST": CohortSpec(
        name="GIST",
        tech="CODEX",
        pan_organ="codex_gist",
        cases_root=_REPO_ROOT / "data/CODEX/HCC/Michael_data_transfer/s1167",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
    "GBM": CohortSpec(
        name="GBM",
        tech="CODEX",
        pan_organ="codex_gbm",
        cases_root=_REPO_ROOT / "data/CODEX/GBM/Cases",
        h5ad_glob="*/*_matched_features.h5ad",
    ),
}

# Display / CLI order: 3 Xenium then 4 CODEX.
COHORT_ORDER = ["Lung", "BRCA", "CRC", "HCC", "PDAC", "GIST", "GBM"]

DEFAULT_DATA_ROOT = _REPO_ROOT / "data/PanCancer"

LUNG_INCOMPLETE_ROOT = (
    _REPO_ROOT
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases"
)

DEFAULT_CKPT = (
    DEFAULT_DATA_ROOT
    / "result_all_spatial/cross_dataset_cv/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt"
)
DEFAULT_PRED_ROOT = DEFAULT_DATA_ROOT / "pred_all_spatial"


def section_id(dataset: str, h5ad_path: Path) -> str:
    return f"{dataset}::{h5ad_path.parent.name}"


def discover_h5ads(
    datasets: list[str] | None = None,
    *,
    skip_crc_unlabeled: bool = True,
) -> list[dict]:
    """Return one dict per matched HE h5ad (does not open the file)."""
    names = list(datasets) if datasets else list(COHORT_ORDER)
    rows: list[dict] = []
    for name in names:
        if name not in COHORTS:
            raise KeyError(f"Unknown dataset {name!r}; choose from {COHORT_ORDER}")
        spec = COHORTS[name]
        root = spec.cases_root
        if not root.is_dir():
            print(f"  skip {name}: missing cases root {root}", flush=True)
            continue
        paths = sorted(root.glob(spec.h5ad_glob))
        # ignore StarDist twins
        paths = [p for p in paths if not p.name.endswith("_matched_features_stardist.h5ad")]
        paths = [p for p in paths if p.name.endswith("_matched_features.h5ad")]
        if name == "PDAC":
            paths = [p for p in paths if _coverslip_ok(p, PDAC_COVERSLIPS)]
        elif name == "GIST":
            paths = [p for p in paths if _coverslip_ok(p, GIST_COVERSLIPS)]
        elif name == "CRC" and skip_crc_unlabeled:
            # Training CV used P1+P2 only. P5CRC has RCTD labels and UNI h5ads;
            # include it at predict time via discover_crc_predict_extras().
            paths = [p for p in paths if p.parent.name in {"P1CRC", "P2CRC"}]
        for p in paths:
            rows.append(
                {
                    "dataset": name,
                    "tech": spec.tech,
                    "pan_organ": spec.pan_organ,
                    "sample": p.parent.name,
                    "section_id": section_id(name, p),
                    "h5ad": str(p.resolve()),
                    "native_col": spec.native_col,
                }
            )
    return rows


def stardist_all_h5ad_for_matched(matched_h5ad: Path) -> Path:
    """``{sample}_all_features_stardist.h5ad`` next to a matched HE h5ad."""
    sample = matched_h5ad.parent.name
    return matched_h5ad.parent / f"{sample}_all_features_stardist.h5ad"


def discover_crc_predict_extras(existing_samples: set[str] | None = None) -> list[dict]:
    """CRC matched sections omitted from pan-cancer CV (currently P5CRC).

    P5 has transferred RCTD GT + UNI embeddings; it was left out of the 5-fold
    train index so fold_plan stays unchanged.
    """
    have = {str(s) for s in (existing_samples or set())}
    extra = []
    for rec in discover_h5ads(["CRC"], skip_crc_unlabeled=False):
        if rec["sample"] in have:
            continue
        if rec["sample"] in {"P1CRC", "P2CRC"}:
            continue
        extra.append(rec)
    return extra


def discover_lung_incomplete() -> list[dict]:
    """Xenium lung Incomplete_Cases with all-nuclei StarDist h5ads (no GT)."""
    root = LUNG_INCOMPLETE_ROOT
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for p in sorted(root.glob("*/*_all_features_stardist.h5ad")):
        if p.name.endswith("_matched_features_stardist.h5ad"):
            continue
        sample = p.parent.name
        rows.append(
            {
                "dataset": "LungIncomplete",
                "tech": "Xenium",
                "pan_organ": "xenium_lung",
                "sample": sample,
                "section_id": f"LungIncomplete::{sample}",
                "h5ad": "",
                "stardist_all": str(p.resolve()),
                "native_col": "final_CT",
            }
        )
    return rows
