## 2026.08.27 Shared histology-derived niche indices (TLS / SRI / TNI)
"""Dataset-neutral TLS / SRI / TNI engine for Hist2Pheno.

New organs (GBM, BRCA, …) should:

1. Define an :class:`OrganNicheConfig` with that organ's Level-2 ``final_CT`` names.
2. :func:`register_organ_config` (optional if you pass ``organ=`` everywhere).
3. Call :func:`load_sample_with_spatial_indices` / :func:`add_spatial_indices`.

Organ-specific paths, clinical tables, and hierarchy Excel stay in the dataset
folder. Xenium lung ARI / FRI / pathologist overlays remain in
``Xenium_lung/histology_derived_niche_index.py``.

Example (GBM)::

    from histology_niche_index import OrganNicheConfig, register_organ_config
    from histology_niche_index import load_sample_with_spatial_indices

    GBM = OrganNicheConfig(
        pan_organ="codex_gbm",
        tls_b=("B cells",),
        tls_t=("CD4 T cells", "CD8 T cells"),
        tls_dc=("Dendritic cells",),
        epithelium=("Tumor",),
        fibroblast=("Fibroblasts",),
        endothelial=("Endothelial cells",),
        macrophage=("Macrophages",),
        demo_sample="...",
        clinical_key="ACQUISITION_ID",
    )
    register_organ_config(GBM)
    df, names, probs, coords, paths, um_px = load_sample_with_spatial_indices(
        sample, organ=GBM, stardist_root=..., cases_root=...
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

AUROC_CSV_NAME = "validation_external_stardist_matched_AUROC.csv"
AUROC_CLASS_NAMES_CSV_NAME = "validation_external_stardist_matched_AUROC_class_names.csv"
H5AD_NAME_TEMPLATE = "{sample}_matched_features_stardist.h5ad"

DEFAULT_UM_PER_HE_PIXEL = 0.5
DEFAULT_SPATIAL_RADIUS_UM = 55.0
DEFAULT_HOTSPOT_PERCENTILE = 95.0
DEFAULT_PLOT_VMAX_PERCENTILE = 99.0

COMPLETE_COHORT_BOX_FACE = "#f4a582"
COMPLETE_COHORT_BOX_EDGE = "#d6604b"

DEFAULT_SRI_TNI_QUADRANT_LABELS: dict[str, str] = {
    "Q1": "Normal / epithelial-dominant",
    "Q2": "Stromal remodeling (epithelial replacement by fibroblasts)",
    "Q3": "Immune / angiogenic activation without extensive remodeling",
    "Q4": "Active TME niche (fibroblasts + myeloid + endothelium)",
}
DEFAULT_SRI_TNI_QUADRANT_PLOT_LABELS: dict[str, str] = {
    "Q1": "Q1: Normal / epi-dominant",
    "Q2": "Q2: Stromal remodeling (SRI\u2191)",
    "Q3": "Q3: Activation w/o remodeling",
    "Q4": "Q4: Active TME niche (SRI\u2191+TNI\u2191)",
}

SOURCE_METRIC_LABELS: dict[str, str] = {
    "SRI_mean": "SRI  (Fib / Epi)",
    "TNI_mean": "TNI  (geomean / Epi)",
    "Epi_local_mean": "Epi_local",
    "TNI_numerator_mean": "TNI numerator  (Fib × myeloid × Endo)",
    "Fib_local_mean": "Fib_local",
    "M2_local_mean": "Myeloid_local",
    "Endo_local_mean": "Endo_local",
    "active_niche_burden": "Active TME niche burden",
}
SOURCE_METRICS: tuple[str, ...] = (
    "SRI_mean",
    "TNI_mean",
    "Epi_local_mean",
    "TNI_numerator_mean",
)


################################################################################
# Organ config
################################################################################
@dataclass(frozen=True)
class OrganNicheConfig:
    """Level-2 compartment names and path/clinical keys for one organ.

    ``macrophage`` is the TNI myeloid arm (M2-like in HCC; Macrophages in PDAC).
    """

    pan_organ: str
    tls_b: tuple[str, ...]
    tls_t: tuple[str, ...]
    tls_dc: tuple[str, ...]
    epithelium: tuple[str, ...]
    fibroblast: tuple[str, ...]
    endothelial: tuple[str, ...]
    macrophage: tuple[str, ...]
    demo_sample: str = ""
    clinical_key: str = "ACQUISITION_ID"
    cohort: str = ""
    auroc_csv_name: str = AUROC_CSV_NAME
    class_names_csv_name: str = AUROC_CLASS_NAMES_CSV_NAME
    h5ad_name_template: str = H5AD_NAME_TEMPLATE
    um_per_he_pixel: float = DEFAULT_UM_PER_HE_PIXEL
    spatial_radius_um: float = DEFAULT_SPATIAL_RADIUS_UM
    hotspot_percentile: float = DEFAULT_HOTSPOT_PERCENTILE
    quadrant_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SRI_TNI_QUADRANT_LABELS)
    )
    quadrant_plot_labels: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SRI_TNI_QUADRANT_PLOT_LABELS)
    )

    @property
    def tls_types(self) -> tuple[str, ...]:
        return self.tls_b + self.tls_t + self.tls_dc


def _hcc_quadrant_labels() -> dict[str, str]:
    labels = dict(DEFAULT_SRI_TNI_QUADRANT_LABELS)
    labels["Q4"] = "Active TME niche (fibroblasts + M2-like + endothelium)"
    return labels


ORGAN_CONFIGS: dict[str, OrganNicheConfig] = {
    "codex_hcc": OrganNicheConfig(
        pan_organ="codex_hcc",
        tls_b=("B cells",),
        tls_t=("CD4 T cells", "CD8 T cells"),
        tls_dc=("Dendritic cells",),
        epithelium=("Epithelium (INOS+)", "Epithelium (INOS-)"),
        fibroblast=("Fibroblasts",),
        endothelial=("Endothelial cells",),
        macrophage=("Macrophages M2-like",),
        demo_sample="awy-98938_aligned_0d535a74",
        clinical_key="matched_he",
        cohort="HCC",
        quadrant_labels=_hcc_quadrant_labels(),
    ),
    "codex_pdac": OrganNicheConfig(
        pan_organ="codex_pdac",
        tls_b=("B cells",),
        tls_t=("Cytotoxic T cells", "Helper T cells", "Tregs"),
        tls_dc=("Dendritic cells",),
        epithelium=("Epithelial cells",),
        fibroblast=("Fibroblasts",),
        endothelial=("Lymphatic Endothelial cells",),
        macrophage=("Macrophages",),
        demo_sample="Charvill-94_c001_v001_r001_reg001",
        clinical_key="ACQUISITION_ID",
        cohort="PDAC",
    ),
    "codex_gist": OrganNicheConfig(
        pan_organ="codex_gist",
        tls_b=("B cells",),
        tls_t=("T cells", "Cytotoxic T cells", "Helper T cells", "Tregs"),
        tls_dc=("Dendritic cells", "DCs"),
        epithelium=("Epithelial cells",),
        fibroblast=("Fibroblasts", "Stromal cells"),
        endothelial=("Endothelial cells", "Lymphatic Endothelial cells"),
        macrophage=("Macrophages", "Monocytes"),
        demo_sample="Charvill-94_c013_v001_r001_reg002",
        clinical_key="ACQUISITION_ID",
        cohort="GIST TMA",
    ),
}

_ORGAN_ALIASES = {
    "hcc": "codex_hcc",
    "codex_hcc": "codex_hcc",
    "pdac": "codex_pdac",
    "pancreas": "codex_pdac",
    "codex_pdac": "codex_pdac",
    "gist": "codex_gist",
    "gist_tma": "codex_gist",
    "codex_gist": "codex_gist",
}


def normalize_organ_key(pan_organ: str) -> str:
    key = str(pan_organ).strip().lower().replace("-", "_")
    if key in ORGAN_CONFIGS:
        return key
    if key in _ORGAN_ALIASES:
        return _ORGAN_ALIASES[key]
    raise KeyError(
        f"Unknown pan_organ={pan_organ!r}. Register with register_organ_config() "
        f"or pass an OrganNicheConfig. Known: {sorted(ORGAN_CONFIGS)}"
    )


def register_organ_config(config: OrganNicheConfig) -> OrganNicheConfig:
    """Add or replace a config so ``get_organ_config(config.pan_organ)`` works."""
    ORGAN_CONFIGS[config.pan_organ] = config
    _ORGAN_ALIASES[str(config.pan_organ).strip().lower()] = config.pan_organ
    return config


def get_organ_config(pan_organ: str | OrganNicheConfig | None) -> OrganNicheConfig:
    if isinstance(pan_organ, OrganNicheConfig):
        return pan_organ
    if pan_organ is None:
        raise TypeError("organ / pan_organ is required")
    return ORGAN_CONFIGS[normalize_organ_key(pan_organ)]


################################################################################
# Probability / neighborhood primitives
################################################################################
def load_auroc_class_names(names_csv: str | Path) -> list[str]:
    path = Path(names_csv).expanduser().resolve()
    df = pd.read_csv(path)
    name_col = "final_CT" if "final_CT" in df.columns else df.columns[-1]
    if "class_index" in df.columns:
        df = df.sort_values("class_index")
    return df[name_col].astype(str).tolist()


def load_stardist_auroc_csv(
    auroc_csv: str | Path,
    class_names_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load per-cell StarDist AUROC table. Returns ``(df, class_names, prob_matrix)``."""
    path = Path(auroc_csv).expanduser().resolve()
    df = pd.read_csv(path)
    if class_names_csv is None:
        sibling = path.with_name(path.stem + "_class_names.csv")
        if sibling.is_file():
            class_names_csv = sibling
        else:
            raise FileNotFoundError(
                f"class_names CSV not found next to {path.name}; pass class_names_csv."
            )
    class_names = load_auroc_class_names(class_names_csv)
    prob_cols = [f"prob_{j}" for j in range(len(class_names))]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"AUROC CSV missing columns: {missing[:5]} ...")
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    return df, class_names, probs


def class_name_to_prob_index(class_names: Sequence[str]) -> dict[str, int]:
    return {str(n): i for i, n in enumerate(class_names)}


def prob_columns_for_cell_types(
    class_names: Sequence[str],
    cell_types: Iterable[str],
    *,
    skip_missing: bool = True,
) -> list[int]:
    idx = class_name_to_prob_index(class_names)
    cols = []
    for ct in cell_types:
        ct = str(ct).strip()
        if ct not in idx:
            if skip_missing:
                continue
            raise KeyError(f"Cell type {ct!r} not in class_names.")
        cols.append(idx[ct])
    return cols


def sum_prob_columns(probs: np.ndarray, col_indices: Sequence[int]) -> np.ndarray:
    if not col_indices:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return probs[:, list(col_indices)].sum(axis=1)


def safe_ratio(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return numer / np.maximum(denom, eps)


def geometric_mean3(a, b, c, *, eps: float = 1e-8) -> np.ndarray:
    return np.power(
        np.maximum(a, eps) * np.maximum(b, eps) * np.maximum(c, eps),
        1.0 / 3.0,
    )


def um_to_he_pixel(radius_um: float, um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL) -> float:
    if um_per_pixel <= 0:
        raise ValueError(f"um_per_pixel must be > 0, got {um_per_pixel}")
    return float(radius_um) / float(um_per_pixel)


def he_pixel_to_um(radius_px: float, um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL) -> float:
    return float(radius_px) * float(um_per_pixel)


def resolve_spatial_radius(
    radius_um: float = DEFAULT_SPATIAL_RADIUS_UM,
    *,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
) -> tuple[float, float, float]:
    """Return ``(radius_um, radius_px, um_per_pixel)``."""
    return float(radius_um), um_to_he_pixel(radius_um, um_per_pixel), float(um_per_pixel)


def infer_um_per_he_pixel(
    sample_dir: str | Path,
    *,
    h5ad_path: str | Path | None = None,
    csv_name_template: str = "{sample}_spatial_coords_um_pix_from_zarr.csv",
    fallback: float = DEFAULT_UM_PER_HE_PIXEL,
) -> float:
    """Infer μm/pixel from ``{sample}_spatial_coords_um_pix_from_zarr.csv``."""
    sample_dir = Path(sample_dir).expanduser().resolve()
    sample = sample_dir.name
    csv_path = sample_dir / csv_name_template.format(sample=sample)
    if not csv_path.is_file():
        return fallback
    coord_csv = pd.read_csv(csv_path, usecols=["cell_id", "x_centroid", "X_pix_HE"])
    if h5ad_path is not None:
        import anndata as ad

        adata = ad.read_h5ad(str(Path(h5ad_path).expanduser().resolve()))
        he_x = pd.DataFrame(
            {"cell_id": adata.obs_names.astype(str), "xh": adata.obsm["spatial_HE"][:, 0]}
        )
        merged = coord_csv.merge(he_x, on="cell_id", how="inner")
        if len(merged) >= 10:
            ratios = merged["x_centroid"] / merged["xh"]
            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratios):
                return float(ratios.median())
    ratios = (coord_csv["x_centroid"] / coord_csv["X_pix_HE"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(ratios.median()) if len(ratios) else fallback


def percentile_vlim(
    values: np.ndarray,
    *,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = DEFAULT_PLOT_VMAX_PERCENTILE,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(values, vmin_percentile))
    hi = float(np.percentile(values, vmax_percentile))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def compartment_prob_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    cell_types: Sequence[str],
) -> np.ndarray:
    """Per-cell softmax mass summed over ``cell_types``."""
    return sum_prob_columns(probs, prob_columns_for_cell_types(class_names, cell_types))


def load_spatial_coords_from_h5ad(
    h5ad_path: str | Path,
    obsm_key: str = "spatial_HE",
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(coords (N,2), cell_ids)`` from a matched StarDist h5ad."""
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    adata = ad.read_h5ad(str(path))
    if obsm_key not in adata.obsm:
        raise KeyError(f"{path.name}: obsm[{obsm_key!r}] not found.")
    coords = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coords (N, 2+), got {coords.shape}")
    return coords[:, :2], adata.obs_names.to_numpy(dtype=str)


def align_auroc_df_with_h5ad(
    df: pd.DataFrame,
    coords: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join AUROC rows to ``spatial_HE`` via ``cell_id``, else require equal length."""
    if "cell_id" in df.columns:
        coord_df = pd.DataFrame(
            {"cell_id": cell_ids.astype(str), "coord_x": coords[:, 0], "coord_y": coords[:, 1]}
        )
        merged = df.copy()
        merged["cell_id"] = merged["cell_id"].astype(str)
        merged = merged.merge(coord_df, on="cell_id", how="left", validate="one_to_one")
        if merged["coord_x"].isna().any():
            n_bad = int(merged["coord_x"].isna().sum())
            raise ValueError(f"{n_bad} AUROC rows have no matching spatial coordinate.")
        return merged, merged[["coord_x", "coord_y"]].to_numpy(dtype=np.float64)
    if len(df) != len(coords):
        raise ValueError(
            f"AUROC rows ({len(df)}) != h5ad cells ({len(coords)}); "
            "re-save AUROC CSV from h5ad or include cell_id column."
        )
    out = df.copy()
    out["coord_x"] = coords[:, 0]
    out["coord_y"] = coords[:, 1]
    return out, coords


def load_auroc_with_spatial_coords(
    auroc_csv: str | Path,
    stardist_h5ad: str | Path,
    class_names_csv: str | Path | None = None,
    *,
    obsm_key: str = "spatial_HE",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df, class_names, probs = load_stardist_auroc_csv(auroc_csv, class_names_csv)
    coords, cell_ids = load_spatial_coords_from_h5ad(stardist_h5ad, obsm_key=obsm_key)
    df, coords = align_auroc_df_with_h5ad(df, coords, cell_ids)
    return df, class_names, probs, coords


def local_neighborhood_mean(
    values: np.ndarray,
    coords: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Radius-neighborhood mean of ``values`` (same units as ``coords``)."""
    from sklearn.neighbors import BallTree

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values and coords length mismatch.")
    tree = BallTree(coords, metric="euclidean")
    neighbor_idx = tree.query_radius(coords, r=float(radius))
    out = np.zeros(values.shape[0], dtype=np.float64)
    for i, idx in enumerate(neighbor_idx):
        out[i] = float(values[idx].mean()) if len(idx) else values[i]
    return out


def detect_tls_hotspot_cells(
    spatial_tls: np.ndarray,
    coords: np.ndarray,
    *,
    percentile: float = DEFAULT_HOTSPOT_PERCENTILE,
    min_score: float | None = None,
) -> tuple[np.ndarray, float]:
    """Flag cells at/above the given percentile of a spatial score."""
    spatial_tls = np.asarray(spatial_tls, dtype=np.float64)
    thresh = float(np.percentile(spatial_tls, percentile))
    if min_score is not None:
        thresh = max(thresh, float(min_score))
    return spatial_tls >= thresh, thresh


def summarize_tls_hotspots(
    df: pd.DataFrame,
    spatial_tls_col: str = "idx_TLS_spatial",
    candidate_col: str = "tls_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
) -> dict:
    if candidate_col not in df.columns:
        raise KeyError(f"Missing column {candidate_col!r}")
    sub = df.loc[df[candidate_col]].copy()
    return {
        "n_cells_total": int(len(df)),
        "n_candidate_cells": int(len(sub)),
        "fraction_candidate": float(len(sub) / max(len(df), 1)),
        "max_tls_spatial": float(df[spatial_tls_col].max()),
        "mean_tls_spatial_candidate": float(sub[spatial_tls_col].mean()) if len(sub) else float("nan"),
        "centroid_x": float(sub[coord_x].mean()) if len(sub) else float("nan"),
        "centroid_y": float(sub[coord_y].mean()) if len(sub) else float("nan"),
    }


def pvalue_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_sample_color_map(sample_ids: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    """Stable sample → RGBA colors (tab20 + tab20b; turbo if more than 40)."""
    import matplotlib.pyplot as plt

    unique = sorted({str(s) for s in sample_ids})
    if not unique:
        return {}
    palette = [plt.get_cmap("tab20")(i) for i in range(20)] + [
        plt.get_cmap("tab20b")(i) for i in range(20)
    ]
    n = len(unique)
    if n > len(palette):
        extra = plt.get_cmap("turbo", n - len(palette))
        palette.extend(extra(i) for i in range(n - len(palette)))
    return {sid: palette[i] for i, sid in enumerate(unique)}


################################################################################
# TLS / SRI / TNI
################################################################################
def missing_index_cell_types(
    class_names: Sequence[str],
    organ: str | OrganNicheConfig,
    *,
    require_all: bool = False,
) -> list[str]:
    """Compartment names missing from ``class_names``.

    Default: a compartment is missing only if **none** of its synonyms match
    (PDAC/GIST style). ``require_all=True`` flags every absent synonym (HCC).
    """
    c = get_organ_config(organ)
    present = {str(name) for name in class_names}
    groups = {
        "TLS_B": c.tls_b,
        "TLS_T": c.tls_t,
        "TLS_DC": c.tls_dc,
        "epithelium": c.epithelium,
        "fibroblast": c.fibroblast,
        "endothelial": c.endothelial,
        "macrophage": c.macrophage,
    }
    if require_all:
        needed = set(c.tls_types + c.epithelium + c.fibroblast + c.endothelial + c.macrophage)
        return sorted(needed - present)
    return [name for name, types in groups.items() if not any(t in present for t in types)]


def index_formula_table(organ: str | OrganNicheConfig) -> pd.DataFrame:
    c = get_organ_config(organ)
    return pd.DataFrame(
        [
            {"index": "TLS", "formula": f"sum({'+ '.join(c.tls_types)})"},
            {
                "index": "TLS_spatial",
                "formula": "(B_local × T_local × DC_local)^(1/3) within radius R μm",
            },
            {
                "index": "SRI_ratio",
                "formula": f"{'+'.join(c.fibroblast)} / {'+'.join(c.epithelium)}",
            },
            {"index": "SRI_spatial", "formula": "Fib_local / Epi_local within radius R μm"},
            {
                "index": "TNI_ratio",
                "formula": (
                    f"({'+'.join(c.fibroblast)} + {'+'.join(c.macrophage)} + "
                    f"{'+'.join(c.endothelial)}) / epithelium"
                ),
            },
            {
                "index": "TNI_spatial",
                "formula": "(Fib_local × myeloid_local × Endo_local)^(1/3) / Epi_local",
            },
        ]
    )


def compute_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    organ: str | OrganNicheConfig,
) -> np.ndarray:
    c = get_organ_config(organ)
    return sum_prob_columns(probs, prob_columns_for_cell_types(class_names, c.tls_types))


def compute_sri_ratio_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    organ: str | OrganNicheConfig,
) -> np.ndarray:
    c = get_organ_config(organ)
    return safe_ratio(
        compartment_prob_per_cell(probs, class_names, c.fibroblast),
        compartment_prob_per_cell(probs, class_names, c.epithelium),
    )


def compute_tni_ratio_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    organ: str | OrganNicheConfig,
) -> np.ndarray:
    c = get_organ_config(organ)
    numer = (
        compartment_prob_per_cell(probs, class_names, c.fibroblast)
        + compartment_prob_per_cell(probs, class_names, c.macrophage)
        + compartment_prob_per_cell(probs, class_names, c.endothelial)
    )
    return safe_ratio(numer, compartment_prob_per_cell(probs, class_names, c.epithelium))


def compute_spatial_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    organ: str | OrganNicheConfig,
    *,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = get_organ_config(organ)
    radius_px = um_to_he_pixel(
        radius_um if radius_um is not None else c.spatial_radius_um,
        um_per_pixel if um_per_pixel is not None else c.um_per_he_pixel,
    )
    b_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.tls_b), coords, radius_px
    )
    t_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.tls_t), coords, radius_px
    )
    dc_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.tls_dc), coords, radius_px
    )
    return geometric_mean3(b_loc, t_loc, dc_loc, eps=eps), b_loc, t_loc, dc_loc


def compute_spatial_sri_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    organ: str | OrganNicheConfig,
    *,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = get_organ_config(organ)
    radius_px = um_to_he_pixel(
        radius_um if radius_um is not None else c.spatial_radius_um,
        um_per_pixel if um_per_pixel is not None else c.um_per_he_pixel,
    )
    fib_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.fibroblast), coords, radius_px
    )
    epi_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.epithelium), coords, radius_px
    )
    return safe_ratio(fib_loc, epi_loc), fib_loc, epi_loc


def compute_spatial_tni_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    organ: str | OrganNicheConfig,
    *,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = get_organ_config(organ)
    radius_px = um_to_he_pixel(
        radius_um if radius_um is not None else c.spatial_radius_um,
        um_per_pixel if um_per_pixel is not None else c.um_per_he_pixel,
    )
    fib_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.fibroblast), coords, radius_px
    )
    mac_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.macrophage), coords, radius_px
    )
    endo_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.endothelial), coords, radius_px
    )
    epi_loc = local_neighborhood_mean(
        compartment_prob_per_cell(probs, class_names, c.epithelium), coords, radius_px
    )
    return safe_ratio(geometric_mean3(fib_loc, mac_loc, endo_loc, eps=eps), epi_loc), fib_loc, mac_loc, endo_loc, epi_loc


def add_abundance_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    organ: str | OrganNicheConfig,
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}TLS"] = compute_tls_per_cell(probs, class_names, organ)
    out[f"{prefix}SRI_ratio"] = compute_sri_ratio_per_cell(probs, class_names, organ)
    out[f"{prefix}TNI_ratio"] = compute_tni_ratio_per_cell(probs, class_names, organ)
    return out


def add_spatial_indices(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    organ: str | OrganNicheConfig,
    *,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
    hotspot_percentile: float | None = None,
    prefix: str = "idx_",
) -> pd.DataFrame:
    c = get_organ_config(organ)
    radius_um = float(c.spatial_radius_um if radius_um is None else radius_um)
    um_per_pixel = float(c.um_per_he_pixel if um_per_pixel is None else um_per_pixel)
    hotspot_percentile = float(
        c.hotspot_percentile if hotspot_percentile is None else hotspot_percentile
    )
    out = df.copy()
    tls_s, b_loc, t_loc, dc_loc = compute_spatial_tls_per_cell(
        probs, class_names, coords, c, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    sri_s, fib_sri, epi_sri = compute_spatial_sri_per_cell(
        probs, class_names, coords, c, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    tni_s, fib_tni, mac_loc, endo_loc, epi_tni = compute_spatial_tni_per_cell(
        probs, class_names, coords, c, radius_um=radius_um, um_per_pixel=um_per_pixel
    )
    tni_num = geometric_mean3(fib_tni, mac_loc, endo_loc)
    out[f"{prefix}TLS_spatial"] = tls_s
    out[f"{prefix}TLS_B_local"] = b_loc
    out[f"{prefix}TLS_T_local"] = t_loc
    out[f"{prefix}TLS_DC_local"] = dc_loc
    out[f"{prefix}SRI_spatial"] = sri_s
    out[f"{prefix}TNI_spatial"] = tni_s
    out[f"{prefix}Epi_local"] = epi_sri
    out[f"{prefix}Fib_local"] = fib_sri
    out[f"{prefix}M2_local"] = mac_loc
    out[f"{prefix}Endo_local"] = endo_loc
    out[f"{prefix}TNI_numerator_local"] = tni_num
    out[f"{prefix}TNI_Fib_local"] = fib_tni
    out[f"{prefix}TNI_M2_local"] = mac_loc
    out[f"{prefix}TNI_Endo_local"] = endo_loc
    out["tls_spatial_radius_um"] = radius_um
    out["tls_um_per_he_pixel"] = um_per_pixel
    tls_flag, tls_thr = detect_tls_hotspot_cells(tls_s, coords, percentile=hotspot_percentile)
    sri_flag, sri_thr = detect_tls_hotspot_cells(sri_s, coords, percentile=hotspot_percentile)
    tni_flag, tni_thr = detect_tls_hotspot_cells(tni_s, coords, percentile=hotspot_percentile)
    out["tls_candidate"] = tls_flag
    out["sri_candidate"] = sri_flag
    out["tni_candidate"] = tni_flag
    out["tls_spatial_threshold"] = tls_thr
    out["sri_spatial_threshold"] = sri_thr
    out["tni_spatial_threshold"] = tni_thr
    return out


def summarize_indices_by_sample(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    cols = [c for c in ("idx_TLS", "idx_SRI_ratio", "idx_TNI_ratio") if c in df.columns]
    row = {"Sample": sample, "n_cells": int(len(df))}
    for col in cols:
        row[col] = float(df[col].mean())
    return pd.DataFrame([row])


################################################################################
# Paths / load
################################################################################
@dataclass(frozen=True)
class SpatialPaths:
    sample: str
    auroc_csv: Path
    class_names_csv: Path
    stardist_h5ad: Path


def spatial_paths(
    sample: str,
    *,
    organ: str | OrganNicheConfig | None = None,
    stardist_root: str | Path,
    cases_root: str | Path,
) -> SpatialPaths:
    """AUROC under ``stardist_root/{sample}/``; h5ad under ``cases_root/{sample}/``."""
    c = get_organ_config(organ) if organ is not None else None
    sample = str(sample)
    star = Path(stardist_root) / sample
    auroc_name = c.auroc_csv_name if c is not None else AUROC_CSV_NAME
    names_name = c.class_names_csv_name if c is not None else AUROC_CLASS_NAMES_CSV_NAME
    h5ad_name = (c.h5ad_name_template if c is not None else H5AD_NAME_TEMPLATE).format(
        sample=sample
    )
    return SpatialPaths(
        sample=sample,
        auroc_csv=star / auroc_name,
        class_names_csv=star / names_name,
        stardist_h5ad=Path(cases_root) / sample / h5ad_name,
    )


def discover_spatial_samples(
    clinical: pd.DataFrame,
    *,
    stardist_root: str | Path,
    cases_root: str | Path,
    sample_col: str | None = None,
    organ: str | OrganNicheConfig | None = None,
    require_h5ad: bool = True,
) -> list[str]:
    if sample_col is None:
        sample_col = get_organ_config(organ).clinical_key if organ is not None else "ACQUISITION_ID"
    samples = []
    for sample in clinical[sample_col].dropna().astype(str).unique():
        paths = spatial_paths(
            sample, organ=organ, stardist_root=stardist_root, cases_root=cases_root
        )
        if not paths.auroc_csv.is_file() or not paths.class_names_csv.is_file():
            continue
        if require_h5ad and not paths.stardist_h5ad.is_file():
            continue
        samples.append(sample)
    return sorted(samples)


def load_sample_with_spatial_indices(
    sample: str,
    *,
    organ: str | OrganNicheConfig,
    stardist_root: str | Path,
    cases_root: str | Path,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
    require_all_cell_types: bool = False,
):
    """Load matched AUROC, attach HE coords, compute TLS / SRI / TNI."""
    c = get_organ_config(organ)
    paths = spatial_paths(
        sample, organ=c, stardist_root=stardist_root, cases_root=cases_root
    )
    if not paths.auroc_csv.is_file():
        raise FileNotFoundError(paths.auroc_csv)
    if not paths.stardist_h5ad.is_file():
        raise FileNotFoundError(paths.stardist_h5ad)
    df, class_names, probs = load_stardist_auroc_csv(paths.auroc_csv, paths.class_names_csv)
    missing = missing_index_cell_types(
        class_names, c, require_all=require_all_cell_types
    )
    if missing:
        raise KeyError(
            f"{sample}: AUROC class_names missing index compartments {missing}."
        )
    coords_raw, cell_ids = load_spatial_coords_from_h5ad(paths.stardist_h5ad)
    df, coords = align_auroc_df_with_h5ad(df, coords_raw, cell_ids)
    prob_cols = [f"prob_{j}" for j in range(len(class_names))]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    df = add_abundance_indices(df, probs, class_names, c)
    df = add_spatial_indices(
        df, probs, class_names, coords, c,
        radius_um=radius_um, um_per_pixel=um_per_pixel,
    )
    um_px = float(c.um_per_he_pixel if um_per_pixel is None else um_per_pixel)
    return df, class_names, probs, coords, paths, um_px


def build_sample_dict(
    samples: Sequence[str],
    *,
    organ: str | OrganNicheConfig,
    stardist_root: str | Path,
    cases_root: str | Path,
    radius_um: float | None = None,
    um_per_pixel: float | None = None,
    require_all_cell_types: bool = False,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        df, *_ = load_sample_with_spatial_indices(
            sample,
            organ=organ,
            stardist_root=stardist_root,
            cases_root=cases_root,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
            require_all_cell_types=require_all_cell_types,
        )
        out[str(sample)] = df
        print(f"  loaded {sample}: {len(df):,} cells", flush=True)
    return out


################################################################################
# Cohort SRI / TNI (column names, no lung ARI/FRI)
################################################################################
def compute_active_niche_burden(
    df: pd.DataFrame,
    *,
    sri_col: str = "idx_SRI_spatial",
    tni_col: str = "idx_TNI_spatial",
) -> float:
    """``mean(sqrt(SRI_i * TNI_i))`` over cells (no percentile cutoff)."""
    sub = df[[sri_col, tni_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return float("nan")
    sri = sub[sri_col].astype(np.float64).clip(lower=0.0)
    tni = sub[tni_col].astype(np.float64).clip(lower=0.0)
    return float(np.sqrt(sri * tni).mean())


def compute_global_q4_percent(
    df: pd.DataFrame,
    *,
    sri_threshold: float,
    tni_threshold: float,
    sri_col: str = "idx_SRI_spatial",
    tni_col: str = "idx_TNI_spatial",
) -> float:
    sub = df[[sri_col, tni_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return float("nan")
    sri = sub[sri_col].to_numpy(dtype=np.float64)
    tni = sub[tni_col].to_numpy(dtype=np.float64)
    return float(((sri >= sri_threshold) & (tni >= tni_threshold)).mean() * 100.0)


def spearman_sri_tni(
    df: pd.DataFrame,
    *,
    sri_col: str = "idx_SRI_spatial",
    tni_col: str = "idx_TNI_spatial",
) -> dict[str, float]:
    from scipy.stats import spearmanr

    sub = df[[sri_col, tni_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n_cells": int(len(sub))}
    rho, p = spearmanr(sub[sri_col], sub[tni_col])
    return {"rho": float(rho), "p_value": float(p), "n_cells": int(len(sub))}


def batch_analyze_sri_tni_cohort(
    sample_dict: Mapping[str, pd.DataFrame],
    *,
    sri_col: str = "idx_SRI_spatial",
    tni_col: str = "idx_TNI_spatial",
    percentile: float = DEFAULT_HOTSPOT_PERCENTILE,
) -> tuple[pd.DataFrame, dict]:
    """Per-sample Global Q4 % and active niche burden on SRI/TNI."""
    if not sample_dict:
        raise ValueError("sample_dict is empty.")
    sri_all = np.concatenate(
        [df[sri_col].to_numpy(dtype=np.float64) for df in sample_dict.values()]
    )
    tni_all = np.concatenate(
        [df[tni_col].to_numpy(dtype=np.float64) for df in sample_dict.values()]
    )
    sri_t = float(np.nanpercentile(sri_all, percentile))
    tni_t = float(np.nanpercentile(tni_all, percentile))
    meta = {
        "sri_threshold_global": sri_t,
        "tni_threshold_global": tni_t,
        "ari_threshold_global": sri_t,
        "fri_threshold_global": tni_t,
        "percentile": float(percentile),
        "n_cells_pooled": int(np.isfinite(sri_all).sum()),
        "n_samples": len(sample_dict),
    }
    rows = []
    for sample_name, df in sample_dict.items():
        corr = spearman_sri_tni(df, sri_col=sri_col, tni_col=tni_col)
        rows.append(
            {
                "Sample": str(sample_name),
                "n_cells": int(len(df)),
                "SRI_median": float(df[sri_col].median()),
                "TNI_median": float(df[tni_col].median()),
                "ARI_median": float(df[sri_col].median()),
                "FRI_median": float(df[tni_col].median()),
                "Q4_global_percent": compute_global_q4_percent(
                    df, sri_threshold=sri_t, tni_threshold=tni_t,
                    sri_col=sri_col, tni_col=tni_col,
                ),
                "active_niche_burden": compute_active_niche_burden(
                    df, sri_col=sri_col, tni_col=tni_col
                ),
                "sri_threshold_global": sri_t,
                "tni_threshold_global": tni_t,
                "cohort_percentile": percentile,
                "Spearman_rho": corr["rho"],
                "Spearman_p": corr["p_value"],
            }
        )
    return pd.DataFrame(rows), meta


def merge_batch_with_clinical(
    batch_df: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    sample_col: str = "Sample",
    clinical_key: str,
    clinical_columns: Sequence[str] = (),
) -> pd.DataFrame:
    out = batch_df.copy()
    out[sample_col] = out[sample_col].astype(str)
    clin = clinical.copy()
    clin[clinical_key] = clin[clinical_key].astype(str)
    clin = clin.drop_duplicates(clinical_key)
    merged = out.merge(clin, left_on=sample_col, right_on=clinical_key, how="left")
    for col in clinical_columns:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged


def test_clinical_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    pan_organ: str,
    clinical_columns: Sequence[str],
    alternative: str = "two-sided",
    method: str = "rank",
    pairwise: bool = True,
    pairwise_correction: str = "holm",
) -> pd.DataFrame:
    from uni_label_cv_helpers import test_metric_by_clinical_groups

    return test_metric_by_clinical_groups(
        df,
        metric_col=metric_col,
        clinical_columns=tuple(clinical_columns),
        pan_organ=pan_organ,
        method=method,
        alternative=alternative,
        pairwise=pairwise,
        pairwise_correction=pairwise_correction,
    )


################################################################################
# Source-metric decomposition (SRI / TNI / Epi vs ratio inflation)
################################################################################
def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _column_or_none(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def epi_local_values(df: pd.DataFrame, *, prefix: str = "idx_") -> np.ndarray:
    epi_col = _column_or_none(df, f"{prefix}Epi_local")
    if epi_col is not None:
        return pd.to_numeric(df[epi_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    sri_col = _column_or_none(df, f"{prefix}SRI_spatial")
    if fib_col is None or sri_col is None:
        raise KeyError("Need idx_Epi_local, or idx_SRI_spatial plus Fib_local.")
    fib = pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64)
    sri = pd.to_numeric(df[sri_col], errors="coerce").to_numpy(dtype=np.float64)
    return fib / np.maximum(sri, 1e-8)


def tni_numerator_values(df: pd.DataFrame, *, prefix: str = "idx_") -> np.ndarray:
    num_col = _column_or_none(df, f"{prefix}TNI_numerator_local")
    if num_col is not None:
        return pd.to_numeric(df[num_col], errors="coerce").to_numpy(dtype=np.float64)
    fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
    m2_col = _column_or_none(df, f"{prefix}M2_local", f"{prefix}TNI_M2_local")
    endo_col = _column_or_none(df, f"{prefix}Endo_local", f"{prefix}TNI_Endo_local")
    if fib_col is None or m2_col is None or endo_col is None:
        raise KeyError("Need Fib/myeloid/Endo local columns.")
    fib = np.maximum(pd.to_numeric(df[fib_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    m2 = np.maximum(pd.to_numeric(df[m2_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    endo = np.maximum(pd.to_numeric(df[endo_col], errors="coerce").to_numpy(dtype=np.float64), 1e-8)
    return np.power(fib * m2 * endo, 1.0 / 3.0)


def summarize_source_metrics(
    sample_dict: Mapping[str, pd.DataFrame],
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    rows = []
    for sample, df in sample_dict.items():
        sri_col = _column_or_none(df, f"{prefix}SRI_spatial")
        tni_col = _column_or_none(df, f"{prefix}TNI_spatial")
        if sri_col is None or tni_col is None:
            raise KeyError(f"{sample}: missing SRI/TNI spatial columns.")
        fib_col = _column_or_none(df, f"{prefix}Fib_local", f"{prefix}TNI_Fib_local")
        m2_col = _column_or_none(df, f"{prefix}M2_local", f"{prefix}TNI_M2_local")
        endo_col = _column_or_none(df, f"{prefix}Endo_local", f"{prefix}TNI_Endo_local")
        row = {
            "Sample": str(sample),
            "n_cells": int(len(df)),
            "SRI_mean": _finite_mean(df[sri_col]),
            "TNI_mean": _finite_mean(df[tni_col]),
            "Epi_local_mean": _finite_mean(epi_local_values(df, prefix=prefix)),
            "TNI_numerator_mean": _finite_mean(tni_numerator_values(df, prefix=prefix)),
        }
        if fib_col is not None:
            row["Fib_local_mean"] = _finite_mean(df[fib_col])
        if m2_col is not None:
            row["M2_local_mean"] = _finite_mean(df[m2_col])
        if endo_col is not None:
            row["Endo_local_mean"] = _finite_mean(df[endo_col])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str,
    metric_cols: Sequence[str] | None = None,
    clinical_columns: Sequence[str] = (),
) -> pd.DataFrame:
    if patient_col not in df.columns:
        raise KeyError(f"Missing {patient_col!r}; join clinical metadata first.")
    work = df.copy()
    work[patient_col] = work[patient_col].astype(str)
    if metric_cols is None:
        metric_cols = [
            c
            for c in (
                "active_niche_burden",
                "Q4_global_percent",
                "SRI_mean",
                "TNI_mean",
                "SRI_median",
                "TNI_median",
                "Epi_local_mean",
                "TNI_numerator_mean",
                "Fib_local_mean",
                "M2_local_mean",
                "Endo_local_mean",
            )
            if c in work.columns
        ]
    rows = []
    for pid, sub in work.groupby(patient_col, sort=True):
        row: dict = {
            patient_col: str(pid),
            "n_regions": int(len(sub)),
            "n_cells": int(pd.to_numeric(sub["n_cells"], errors="coerce").sum())
            if "n_cells" in sub.columns
            else int(len(sub)),
        }
        for col in metric_cols:
            row[col] = _finite_mean(sub[col])
        for col in clinical_columns:
            if col not in sub.columns or col == patient_col:
                continue
            vals = sub[col].astype(str).unique().tolist()
            row[col] = vals[0]
            if len(vals) > 1:
                print(f"  warn: {patient_col}={pid} has mixed {col}={vals}; using {vals[0]}")
        rows.append(row)
    return pd.DataFrame(rows)


def test_source_metrics(
    df: pd.DataFrame,
    *,
    pan_organ: str,
    clinical_columns: Sequence[str],
    metric_cols: Sequence[str] = SOURCE_METRICS,
    alternative: str = "two-sided",
    method: str = "rank",
    pairwise: bool = True,
    pairwise_correction: str = "holm",
) -> pd.DataFrame:
    frames = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        frames.append(
            test_clinical_groups(
                df,
                metric_col=col,
                pan_organ=pan_organ,
                clinical_columns=clinical_columns,
                alternative=alternative,
                method=method,
                pairwise=pairwise,
                pairwise_correction=pairwise_correction,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _tab10_group_colors(order: Sequence[str]) -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {str(lab): cmap(i % 10) for i, lab in enumerate(order)}


def _pairwise_dicts_from_stat_row(row) -> list[dict]:
    if row is None:
        return []
    pw_raw = row.get("pairwise") if hasattr(row, "get") else None
    if isinstance(pw_raw, np.ndarray):
        return [x for x in pw_raw.tolist() if isinstance(x, dict)]
    if isinstance(pw_raw, (list, tuple)):
        return [x for x in pw_raw if isinstance(x, dict)]
    return []


def plot_source_metric_grid(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    pan_organ: str,
    group_col: str,
    metric_cols: Sequence[str] = SOURCE_METRICS,
    sample_col: str | None = None,
    metric_labels: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    point_size: float = 36,
    color_by: Literal["sample", "group"] = "sample",
    show_pairwise: bool = False,
    pairwise_correction: str = "holm",
):
    """Side-by-side boxplots for SRI / TNI / Epi / TNI numerator.

    ``color_by='group'`` colors boxes and points by the x-axis group (with a
    shared legend). ``show_pairwise=True`` draws Holm-adjusted two-group
    brackets, matching :func:`plot_q4_clinical_comparison`.
    """
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plotting_palettes import clinical_group_order

    xenium_dir = Path(__file__).resolve().parent.parent / "Xenium_lung"
    if str(xenium_dir) not in sys.path:
        sys.path.insert(0, str(xenium_dir))
    from histology_derived_niche_index import (
        _annotate_pairwise_brackets,
        pairwise_group_metric_tests,
    )

    labels = dict(SOURCE_METRIC_LABELS)
    if metric_labels:
        labels.update(metric_labels)
    cols = [c for c in metric_cols if c in df.columns]
    if not cols:
        raise KeyError(f"None of {list(metric_cols)} found in dataframe.")
    color_by_key = str(color_by).strip().lower()
    if color_by_key not in ("sample", "group"):
        raise ValueError(f"color_by must be 'sample' or 'group', got {color_by!r}")
    color_by_sample = (
        color_by_key == "sample" and bool(sample_col) and sample_col in df.columns
    )
    color_by_group = color_by_key == "group"
    n = len(cols)
    legend_w = 1.8 if (color_by_sample or color_by_group) else 0.0
    if figsize is None:
        figsize = (4.2 * n + legend_w, 5.6 if show_pairwise else 5.2)

    order_map = clinical_group_order(pan_organ)
    sample_colors: dict[str, tuple] = {}
    sample_order: list[str] = []
    if color_by_sample:
        sample_order = sorted(df[sample_col].astype(str).unique().tolist())
        sample_colors = build_sample_color_map(sample_order)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, layout="constrained")
    rng = np.random.default_rng(0)
    group_colors: dict[str, tuple] = {}
    for j, col in enumerate(cols):
        ax = axes[0, j]
        sub = df.dropna(subset=[group_col, col]).copy()
        present = sub[group_col].astype(str).unique().tolist()
        preset = [g for g in order_map.get(group_col, ()) if g in present]
        rest = sorted(g for g in present if g not in preset)
        order = preset + rest
        if color_by_group:
            group_colors = _tab10_group_colors(order)
        data = [sub.loc[sub[group_col].astype(str) == g, col].to_numpy(float) for g in order]
        positions = np.arange(1, len(order) + 1)
        bp = ax.boxplot(
            data, positions=positions, widths=0.55, patch_artist=True,
            showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, g in zip(bp["boxes"], order):
            if color_by_group:
                patch.set(facecolor=group_colors[str(g)], alpha=0.45, edgecolor="0.35")
            else:
                patch.set(
                    facecolor=COMPLETE_COHORT_BOX_FACE, alpha=0.85,
                    edgecolor=COMPLETE_COHORT_BOX_EDGE,
                )
        for pos, g in zip(positions, order):
            gdf = sub.loc[sub[group_col].astype(str) == g]
            n_pts = len(gdf)
            if n_pts == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=n_pts)
            y_vals = gdf[col].astype(float).to_numpy()
            x_vals = np.full(n_pts, pos) + jitter
            if color_by_sample:
                pt_colors = [sample_colors[str(sid)] for sid in gdf[sample_col].astype(str)]
                ax.scatter(
                    x_vals, y_vals, s=point_size, c=pt_colors,
                    edgecolors="0.35", linewidths=0.4, alpha=0.95, zorder=3,
                )
            elif color_by_group:
                ax.scatter(
                    x_vals, y_vals, s=point_size,
                    c=[group_colors[str(g)]] * n_pts,
                    edgecolors="0.35", linewidths=0.4, alpha=0.95, zorder=3,
                )
            else:
                ax.scatter(
                    x_vals, y_vals, s=point_size, c="#b2182b",
                    edgecolors="0.35", linewidths=0.4, alpha=0.95, zorder=3,
                )
        flat = np.concatenate([d for d in data if len(d)]) if data else np.array([0.0])
        y_min, y_max = float(np.min(flat)), float(np.max(flat))
        y_span = max(y_max - y_min, 1e-9)
        y_hi = y_max + y_span * 0.12

        hits = stats_summary
        if stats_summary is not None and len(stats_summary) and "metric_col" in stats_summary.columns:
            hits = stats_summary.loc[stats_summary["metric_col"] == col]
        if len(hits) and "clinical_variable" in hits.columns:
            hits = hits.loc[hits["clinical_variable"] == group_col]
        stat_row = hits.iloc[0] if len(hits) else None

        pairwise_rows: list[dict] = []
        if show_pairwise:
            pairwise_rows = _pairwise_dicts_from_stat_row(stat_row)
            if not pairwise_rows:
                pw_groups, pw_labels = [], []
                for lab, arr in zip(order, data):
                    vals = np.asarray(arr, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) >= 2:
                        pw_groups.append(vals)
                        pw_labels.append(str(lab))
                pairwise_rows = pairwise_group_metric_tests(
                    pw_groups, pw_labels, method="rank",
                    correction=pairwise_correction,
                )
            if pairwise_rows:
                y_hi = max(
                    y_hi,
                    _annotate_pairwise_brackets(
                        ax, pairwise_rows, order, y_max=y_max, y_span=y_span,
                    ),
                )

        show_overall = (
            stat_row is not None
            and np.isfinite(float(stat_row.get("p_value", np.nan)))
            and not (show_pairwise and len(pairwise_rows) == 1)
        )
        if show_overall:
            p = float(stat_row["p_value"])
            test = str(stat_row.get("test", ""))
            y_text = y_hi + y_span * 0.06
            ax.text(
                float(np.mean(positions)),
                y_text,
                f"{test}\np={p:.3g} ({pvalue_to_stars(p)})",
                ha="center", va="bottom", fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
                clip_on=False, zorder=4,
            )
            y_hi = y_text + y_span * 0.16
        ax.set_ylim(y_min - y_span * 0.06, y_hi + y_span * 0.04)
        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
        ax.set_title(labels.get(col, col), fontsize=11, pad=6)
        ax.set_ylabel("mean per region or patient" if j == 0 else "")
        ax.grid(axis="y", alpha=0.25)

    if color_by_group and group_colors:
        handles = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=group_colors[lab], markeredgecolor="0.35",
                markersize=8, label=lab,
            )
            for lab in group_colors
        ]
        fig.legend(
            handles, list(group_colors), loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=9, title=group_col, frameon=True, borderaxespad=0.0,
        )
    elif color_by_sample and sample_order:
        handles = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=sample_colors[sid], markeredgecolor="0.35",
                markersize=5, label=sid,
            )
            for sid in sample_order
        ]
        fig.legend(
            handles, sample_order, loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=7, title=sample_col, frameon=False, borderaxespad=0.0,
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig, axes
