## 2026.06.27 try the TLS and TIL defination, for downstream analysis
##            add StarDist all nuclei for downstream analysis
## 2026.07.05 add the StarDist all nuclei for Incomplete_Cases
##            using command: conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/transer_embedding_label_h5ad.py --cases-set incomplete --steps stardist_all_h5ad
##            see the clinical difference of Active Proliferative Niche Burden (APNB) on Incomplete_Cases
##            add the function to plot the predicted cell type spatial distribution from the label h5ad [Incomplete_Cases]

"""
Histology-derived niche indices from StarDist Level2 softmax probabilities.

Input tables (per sample or pooled):
  **Per-sample model** (``Complete_Cases/{sample}/{sample}_project_all_UNI/result/``):
    dataset-specific checkpoint; one AUROC CSV per Complete_Cases folder.
  **Cross-dataset model** (``result_all_spatial/stardist/{sample}/``) is the default
  prediction source (``PREDICTION_SOURCE_DEFAULT``). Per-sample loaders remain
  available via ``prediction_source="per_sample"``.

  ``validation_external_stardist_matched_AUROC.csv``
    - one row per cell; ``prob_0`` … ``prob_{C-1}`` are predicted class probabilities
      (soft abundance per cell), **not** pre-aggregated tissue fractions.
  ``validation_external_stardist_matched_AUROC_class_names.csv``
    - maps ``prob_j`` → ``final_CT`` name.

Cell-type metadata (niche / function / index definitions):
  ``41588_2025_2080_MOESM5_ESM.xlsx``, sheet ``Celltype``.

Prior knowledge summary (from annotation sheet):
  - **TLS** (Tertiary Lymphoid Structure): adaptive immune aggregate —
    B cells + T cells + dendritic cells (abundance sum over softmax probs).
  - **TLS_spatial**: co-localization score in a local H&E neighborhood —
    ``(B_local × T_local × DC_local) ^ (1/3)`` using nuclei coordinates
    (``spatial_HE`` in StarDist ``matched_features_stardist.h5ad``; **pixel units**,
    converted internally from μm via ``um_per_he_pixel``).
  - **TLS_spatial + entropy** (optional): Shannon entropy of normalized local
    compartment proportions ``p(B), p(T), p(DC)`` rewards balanced B/T/DC mix;
    combined score ``TLS_spatial + w · H / log(3)`` (``use_entropy=True``).
  - **TLS_spatial_weighted** (cross-dataset biomarker): same co-localization geometry
    but compartment soft mass uses **fixed Level-2 head weight magnitudes**
    ``||W_c||_2`` from the shared cross-dataset checkpoint; geometric mean mapped
    to ``(0, 1)`` via logistic transform for dataset-agnostic clinical scoring.
  - **FRI** (Fibrotic Remodeling Index): ratio of three explicit niche compartments
    (Fibrotic stromal, Injury epithelium, Profibrotic macrophages) vs AT1+AT2.
  - **FRI_spatial**: local co-localization
    ``(FS_local × IE_local × PM_local)^(1/3) / AT1+AT2_local`` within radius R μm.
  - **FRI score**: weighted soft-abundance sum over FRI member cell types.
  - **ARI** (Alveolar Remodeling Index): (Activated Fibrotic FBs + Myofibroblasts)
    / (AT1 + AT2).
  - **ARI_spatial**: ``FB_local / AT1+AT2_local`` within radius R μm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_ANNOTATION_XLSX = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed"
    / "Annotation/HE_Annotations/41588_2025_2080_MOESM5_ESM.xlsx"
)

##################################################################
# 2026.06.27use the per-sample's output
##################################################################
DEFAULT_COMPLETE_CASES_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases"
)
DEFAULT_PROJECT_SUFFIX = "_project_all_UNI"

##################################################################
# Cross-dataset spatial model (shared checkpoint, per-sample preds)
##################################################################
DEFAULT_CROSS_DATASET_STARDIST_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial/stardist"
)
DEFAULT_INCOMPLETE_CASES_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases"
)
DEFAULT_INCOMPLETE_STARDIST_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial/stardist_Incomplete_Cases"
)
STARDIST_INCOMPLETE_RESULT_SUBDIR = "stardist_Incomplete_Cases"
DEFAULT_CROSS_DATASET_MODEL_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial"
    / "cross_dataset_cv/D_emph_L2_spatial_bs4096/best_mlp_gpu.pt"
)
DEFAULT_CROSS_DATASET_CLASS_NAMES_CSV = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial"
    / "validation_external_stardist_matched_AUROC_all_samples_class_names.csv"
)
DEFAULT_TLS_WEIGHTS_CSV = (
    Path(__file__).resolve().parents[2]
    / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial"
    / "cross_dataset_cv/D_emph_L2_spatial_bs4096/tls_level2_compartment_weights.csv"
)

PREDICTION_SOURCE_PER_SAMPLE = "per_sample"
PREDICTION_SOURCE_CROSS_DATASET = "cross_dataset"
PREDICTION_SOURCE_STARDIST_ALL = "stardist_all"
PREDICTION_SOURCE_DEFAULT = PREDICTION_SOURCE_CROSS_DATASET
PredictionSource = str  # "per_sample" | "cross_dataset" | "stardist_all"

DEFAULT_POOLED_SAVE_RESULT = "result_all_spatial"
STARDIST_ALL_LABEL_H5AD_SUFFIX = "_all_features_stardist_label.h5ad"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
##################################################################


# Index member sets (model ``final_CT`` names).
TLS_B_CELLS = ("B cells", 
    "Proliferating B cells",
    # "Plasma",
)
TLS_T_CELLS = (
    "CD4+ T-cells",
    "CD8+ T-cells",
    "Tregs",
    "Proliferating T-cells",
    # "NK/NKT",
    # "Proliferating NK/NKT",
)
TLS_DC_CELLS = (
    "cDCs",
    "pDCs",
    "Migratory DCs",
    # "Langerhans cells",
)
TLS_CELL_TYPES = TLS_B_CELLS + TLS_T_CELLS + TLS_DC_CELLS


# 2026.06.27 LLY: add the ARI index member sets
## Alveolar Remodeling Index (ARI) = (Activated Fibrotic FBs+Myofibroblasts)/(AT1+AT2)
ARI_FB_CELLS = ("Activated Fibrotic FBs", 
    "Myofibroblasts",
)

Alveolar_CELLS = ("AT1", "AT2")
ALVEOLAR_EPITHELIUM = Alveolar_CELLS

# 2026.06.27 LLY: add the FRI index member sets
## Spatial Fibrotic Remodeling Index (sFRI) = (Fibrotic stromal+Injury epithelium+Profibrotic macrophages)/(AT1+AT2)
# sFRI quantifies the local co-occurrence of fibrotic stromal cells, injury-associated epithelial cells, and profibrotic macrophages relative to preserved alveolar epithelium.
FRI_Fibrotic_stromal_CELLS = (
    "Activated Fibrotic FBs",
    "Inflammatory FBs",
    "Myofibroblasts",
    "Proliferating FBs",
)

FRI_Injury_epithelium_CELLS = (
    "KRT5-/KRT17+",
    "RASC",
    "Transitional AT2",
)

FRI_Profibrotic_macrophages_CELLS = (
    "Macrophages - IFN-activated",
    "SPP1+ Macrophages",
    # "Monocytes/MDMs",
)

FRI_NICHE_CELL_TYPES: dict[str, tuple[str, ...]] = {
    "fibrotic_stromal": FRI_Fibrotic_stromal_CELLS,
    "injury_epithelium": FRI_Injury_epithelium_CELLS,
    "profibrotic_macrophages": FRI_Profibrotic_macrophages_CELLS,
}

FRI_NICHE_ORDER: tuple[str, ...] = tuple(FRI_NICHE_CELL_TYPES.keys())
FRI_ALL_NUMERATOR_CELLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        FRI_Fibrotic_stromal_CELLS
        + FRI_Injury_epithelium_CELLS
        + FRI_Profibrotic_macrophages_CELLS
    )
)


# ``obsm['spatial_HE']`` is in HE image pixels; SpatialPF Xenium ≈ 0.2125 μm/px.
DEFAULT_UM_PER_HE_PIXEL = 0.2125
DEFAULT_TLS_SPATIAL_RADIUS_UM = 75.0
DEFAULT_TLS_HOTSPOT_PERCENTILE = 95.0
DEFAULT_TLS_PLOT_VMAX_PERCENTILE = 99.0
DEFAULT_FRI_SPATIAL_RADIUS_UM = DEFAULT_TLS_SPATIAL_RADIUS_UM
DEFAULT_ARI_SPATIAL_RADIUS_UM = DEFAULT_TLS_SPATIAL_RADIUS_UM


@dataclass(frozen=True)
class IndexSpec:
    name: str
    formula: str
    member_cell_types: tuple[str, ...] = ()
    numerator_cell_types: tuple[str, ...] = ()
    denominator_cell_types: tuple[str, ...] = ()
    niche_groups: tuple[str, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict)


INDEX_SPECS: dict[str, IndexSpec] = {
    "TLS": IndexSpec(
        name="TLS",
        formula="sum(B cells + T cells + dendritic cells)",
        member_cell_types=TLS_CELL_TYPES,
    ),
    "TLS_spatial": IndexSpec(
        name="TLS_spatial",
        formula="(B_local * T_local * DC_local)^(1/3) within radius R (μm)",
        member_cell_types=TLS_CELL_TYPES,
    ),
    "TLS_spatial_entropy": IndexSpec(
        name="TLS_spatial_entropy",
        formula="TLS_spatial + w * H / log(3); H = -sum p_k log p_k over k in {B,T,DC}",
        member_cell_types=TLS_CELL_TYPES,
    ),
    "TLS_spatial_weighted": IndexSpec(
        name="TLS_spatial_weighted",
        formula=(
            "sigmoid(geomean(B_w_local,T_w_local,DC_w_local)/scale); "
            "w_c = ||W_c||_2; B_w_local = mean_j sum_{c in B} w_c * p_j(c)"
        ),
        member_cell_types=TLS_CELL_TYPES,
    ),
    "FRI_ratio": IndexSpec(
        name="FRI_ratio",
        formula="(Fibrotic stromal + Injury epithelium + Profibrotic macrophages) / (AT1 + AT2)",
        member_cell_types=FRI_ALL_NUMERATOR_CELLS,
        denominator_cell_types=ALVEOLAR_EPITHELIUM,
    ),
    "FRI_score": IndexSpec(
        name="FRI_score",
        formula="sum_c (w_c * p_i(c)) over FRI niche member cell types",
        member_cell_types=FRI_ALL_NUMERATOR_CELLS,
    ),
    "FRI_spatial": IndexSpec(
        name="FRI_spatial",
        formula=(
            "(FS_local × IE_local × PM_local)^(1/3) / max(AT1+AT2_local, eps) "
            "within radius R (μm)"
        ),
        member_cell_types=FRI_ALL_NUMERATOR_CELLS,
        denominator_cell_types=ALVEOLAR_EPITHELIUM,
    ),
    "ARI_ratio": IndexSpec(
        name="ARI_ratio",
        formula="(Activated Fibrotic FBs + Myofibroblasts) / (AT1 + AT2)",
        numerator_cell_types=ARI_FB_CELLS,
        denominator_cell_types=ALVEOLAR_EPITHELIUM,
    ),
    "ARI_spatial": IndexSpec(
        name="ARI_spatial",
        formula="FB_local / max(AT1+AT2_local, eps) within radius R (μm)",
        numerator_cell_types=ARI_FB_CELLS,
        denominator_cell_types=ALVEOLAR_EPITHELIUM,
    ),
}

INDEX_HIGHLIGHT_OTHER_LABEL = "other"
DEFAULT_INDEX_OTHER_RGBA = (0.82, 0.82, 0.82, 0.45)


def index_highlight_cell_types() -> tuple[str, ...]:
    """
    Level-2 ``final_CT`` names used by TLS / FRI / ARI index definitions.

    Union of TLS members, FRI numerator niches, ARI fibrotic FBs, and alveolar AT1+AT2.
    """
    members = dict.fromkeys(TLS_CELL_TYPES)
    for ct in FRI_ALL_NUMERATOR_CELLS:
        members.setdefault(ct, None)
    for ct in ARI_FB_CELLS:
        members.setdefault(ct, None)
    for ct in ALVEOLAR_EPITHELIUM:
        members.setdefault(ct, None)
    return tuple(members.keys())


def load_celltype_annotation(
    xlsx_path: str | Path = DEFAULT_ANNOTATION_XLSX,
    sheet_name: str = "Celltype",
) -> pd.DataFrame:
    """Load ``Celltype`` sheet (``final_CT``, ``Niches``, ``Functions``, ``Index``)."""
    path = Path(xlsx_path).expanduser().resolve()
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "final_CT" not in df.columns:
        raise ValueError(f"Sheet {sheet_name!r} must contain column 'final_CT'.")
    return df.copy()


##################################################################
# 2026.06.27load clinical information for each sample
##################################################################
def load_clinical_info(
    xlsx_path: str | Path = DEFAULT_ANNOTATION_XLSX,
    sheet_name: str = "Clinical_info",
) -> pd.DataFrame:
    """
    Load sample / demographic metadata from MOESM5 ``Supplementary Table 1``.

    Columns include ``Donor_ID``, ``Status``, ``Sex``, ``Age``, ``Clinical_Diagnosis``, etc.
    """
    path = Path(xlsx_path).expanduser().resolve()
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    if "Donor_ID" not in df.columns:
        raise ValueError(f"Sheet {sheet_name!r} must contain column 'Donor_ID'.")
    return df.copy()


TLS_CLINICAL_METRICS: tuple[str, str] = ("mean_idx_TLS", "mean_idx_TLS_spatial")
DEFAULT_CLINICAL_GROUP_COLUMNS: tuple[str, ...] = (
    "Status",
    "Sample_Affect_Pairing",
    "Clinical_Diagnosis",
    "TMA",
)


def donor_id_from_sample(sample: str) -> str:
    """
    Strip MA/LA suffix to obtain donor prefix (used by pathologist annotation paths).

    Note: ``clinical_info['Donor_ID']`` equals the Complete_Cases **folder name**
    (e.g. ``VUILD107MA``). Merge clinical tables on ``sample`` directly.
    """
    sample = str(sample).strip()
    m = re.match(r"^(THD\d+)$", sample)
    if m:
        return m.group(1)
    m = re.match(r"^(VUHD\d+)[AB]?$", sample)
    if m:
        return m.group(1)
    m = re.match(r"^(TILD\d+)(?:MA\d?|LA)$", sample)
    if m:
        return m.group(1)
    m = re.match(r"^(VUILD\d+)(?:MA\d?|LA\d?)$", sample)
    if m:
        return m.group(1)
    return sample


def merge_sample_tls_with_clinical(
    tls_summary: pd.DataFrame,
    clinical_info: pd.DataFrame,
    *,
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Attach ``clinical_info`` to a per-sample TLS summary table."""
    out = tls_summary.copy()
    if sample_col not in out.columns:
        raise KeyError(f"Missing column {sample_col!r}")
    out[sample_col] = out[sample_col].astype(str)
    clinical = clinical_info.copy()
    clinical["Donor_ID"] = clinical["Donor_ID"].astype(str)
    merged = out.merge(clinical, left_on=sample_col, right_on="Donor_ID", how="left")
    for col in DEFAULT_CLINICAL_GROUP_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged


def pvalue_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_cohort_tls_clinical_table(
    clinical_info: pd.DataFrame,
    cases_root: str | Path | None = None,
    annotation: pd.DataFrame | None = None,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
) -> pd.DataFrame:
    """Per-sample abundance + spatial TLS merged with ``clinical_info``."""
    if annotation is None:
        annotation = load_celltype_annotation()
    spatial = summarize_complete_cases_spatial_tls(
        cases_root,
        project_suffix=project_suffix,
        radius_um=radius_um,
        annotation=annotation,
    )
    if "mean_idx_TLS" not in spatial.columns:
        idx = summarize_complete_cases_indices(
            annotation, cases_root, project_suffix=project_suffix
        )
        spatial = spatial.merge(
            idx[[ "sample", "idx_TLS"]].rename(columns={"idx_TLS": "mean_idx_TLS"}),
            on="sample",
            how="left",
        )
    return merge_sample_tls_with_clinical(spatial, clinical_info)


def _group_samples_for_testing(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    *,
    min_group_size: int = 2,
) -> list[tuple[str, np.ndarray]]:
    if group_col not in df.columns:
        raise KeyError(f"Missing column {group_col!r}")
    if metric not in df.columns:
        raise KeyError(f"Missing column {metric!r}")
    sub = df.dropna(subset=[group_col, metric])
    groups: list[tuple[str, np.ndarray]] = []
    for label, gdf in sub.groupby(group_col, sort=True):
        vals = gdf[metric].to_numpy(dtype=np.float64)
        if len(vals) >= min_group_size:
            groups.append((str(label), vals))
    return groups


def test_tls_metric_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    *,
    min_group_size: int = 2,
) -> dict:
    """Run Mann–Whitney (2 groups) or Kruskal–Wallis (>2 groups) on one metric."""
    from scipy import stats

    groups = _group_samples_for_testing(
        df, group_col, metric, min_group_size=min_group_size
    )
    group_names = [g[0] for g in groups]
    samples = [len(g[1]) for g in groups]
    means = [float(np.mean(g[1])) for g in groups]
    medians = [float(np.median(g[1])) for g in groups]

    if len(groups) < 2:
        return {
            "clinical_variable": group_col,
            "metric": metric,
            "test": "insufficient_groups",
            "p_value": np.nan,
            "statistic": np.nan,
            "n_groups": len(groups),
            "group_labels": group_names,
            "group_n": samples,
            "group_mean": means,
            "group_median": medians,
            "significance": "NA",
        }

    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(groups[0][1], groups[1][1], alternative="two-sided")
        test_name = "Mann-Whitney U"
    else:
        stat, p = stats.kruskal(*[g[1] for g in groups])
        test_name = "Kruskal-Wallis"

    return {
        "clinical_variable": group_col,
        "metric": metric,
        "test": test_name,
        "p_value": float(p),
        "statistic": float(stat),
        "n_groups": len(groups),
        "group_labels": group_names,
        "group_n": samples,
        "group_mean": means,
        "group_median": medians,
        "significance": pvalue_to_stars(float(p)),
    }


def test_tls_by_clinical_groups(
    df: pd.DataFrame,
    *,
    clinical_columns: Sequence[str] = DEFAULT_CLINICAL_GROUP_COLUMNS,
    metrics: Sequence[str] = TLS_CLINICAL_METRICS,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Test sample-level TLS differences across clinical metadata columns.

    Uses Mann–Whitney U for two groups and Kruskal–Wallis for three or more.
    Groups with fewer than ``min_group_size`` samples are excluded from testing.
    """
    rows = []
    for group_col in clinical_columns:
        for metric in metrics:
            result = test_tls_metric_by_group(
                df, group_col, metric, min_group_size=min_group_size
            )
            rows.append(result)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["neg_log10_p"] = -np.log10(summary["p_value"].clip(lower=1e-300))
    return summary


def clinical_group_summary_table(
    df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str] = TLS_CLINICAL_METRICS,
) -> pd.DataFrame:
    """Descriptive mean / median / n for each clinical group and TLS metric."""
    rows = []
    for label, gdf in df.groupby(group_col, sort=True):
        row = {"group": str(label), "n_samples": len(gdf)}
        for metric in metrics:
            if metric in gdf.columns:
                vals = gdf[metric].astype(float)
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_median"] = float(vals.median())
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_tls_clinical_metrics(
    metrics: Sequence[str] | None,
    tls_variant: str,
) -> tuple[str, ...]:
    """Map ``tls_variant`` to sample-level TLS metric column names."""
    variant_map = {
        "abundance": ("mean_idx_TLS",),
        "spatial": ("mean_idx_TLS_spatial",),
        "both": TLS_CLINICAL_METRICS,
    }
    if metrics is not None:
        return tuple(metrics)
    if tls_variant not in variant_map:
        raise ValueError(
            f"Unknown tls_variant={tls_variant!r}; expected one of {tuple(variant_map)}."
        )
    return variant_map[tls_variant]


def plot_tls_clinical_comparison(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    clinical_columns: Sequence[str] = DEFAULT_CLINICAL_GROUP_COLUMNS,
    tls_variant: Literal["abundance", "spatial", "both"] = "both",
    metrics: Sequence[str] | None = None,
    metric_labels: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    y_annotation_pad: float = 0.28,
):
    """
    Box + strip plots of sample-level TLS by clinical metadata, with p-values.

    Parameters
    ----------
    tls_variant
        Convenience selector when ``metrics`` is omitted:
        ``"abundance"`` → ``mean_idx_TLS`` only;
        ``"spatial"`` → ``mean_idx_TLS_spatial`` only;
        ``"both"`` → both rows (default).
    metrics
        Explicit metric column names; overrides ``tls_variant`` when provided.
    y_annotation_pad
        Fraction of the data y-span reserved **above** the highest point for the
        p-value annotation (keeps text off the boxplots).

    Returns ``(fig, axes_array)``.
    """
    import matplotlib.pyplot as plt

    metrics = _resolve_tls_clinical_metrics(metrics, tls_variant)

    if metric_labels is None:
        metric_labels = {
            "mean_idx_TLS": "Abundance TLS",
            "mean_idx_TLS_spatial": "Spatial TLS",
        }

    variant_titles = {
        "abundance": "Abundance TLS",
        "spatial": "Spatial TLS",
        "both": "Abundance vs Spatial TLS",
    }
    suptitle_metric = variant_titles.get(tls_variant, "TLS")

    n_rows = len(metrics)
    n_cols = len(clinical_columns)
    if figsize is None:
        row_h = 3.6 if n_rows == 1 else 4.0
        figsize = (max(3.2 * n_cols, 10.0), max(row_h * n_rows, 4.5))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    stats_lookup = {
        (row["clinical_variable"], row["metric"]): row
        for _, row in stats_summary.iterrows()
    }

    for i, metric in enumerate(metrics):
        for j, group_col in enumerate(clinical_columns):
            ax = axes[i, j]
            sub = df.dropna(subset=[group_col, metric]).copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            order = (
                sub.groupby(group_col)[metric]
                .median()
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
            data = [sub.loc[sub[group_col].astype(str) == g, metric].to_numpy(float) for g in order]
            positions = np.arange(1, len(order) + 1)
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
            )
            for patch in bp["boxes"]:
                patch.set(facecolor="#aec7e8", alpha=0.7, edgecolor="#1f77b4")

            rng = np.random.default_rng(0)
            for pos, vals in zip(positions, data):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full(len(vals), pos) + jitter,
                    vals,
                    s=22,
                    c="#1f77b4",
                    alpha=0.85,
                    zorder=3,
                )

            flat_vals = np.concatenate([d for d in data if len(d)]) if data else np.array([0.0])
            y_min = float(np.min(flat_vals)) if flat_vals.size else 0.0
            y_max = float(np.max(flat_vals)) if flat_vals.size else 1.0
            y_span = max(y_max - y_min, abs(y_max) * 0.05, 1e-9)
            pad_top = y_span * float(y_annotation_pad)
            pad_bottom = y_span * 0.06
            y_upper = y_max + pad_top
            y_lower = y_min - pad_bottom
            if y_lower >= y_upper:
                y_upper = y_max + 1.0
                y_lower = y_min - 0.1
            ax.set_ylim(y_lower, y_upper)

            stat_row = stats_lookup.get((group_col, metric))
            if stat_row is not None and np.isfinite(stat_row.get("p_value", np.nan)):
                p = float(stat_row["p_value"])
                stars = stat_row.get("significance", pvalue_to_stars(p))
                stat_y = y_max + pad_top * 0.55
                ax.text(
                    float(np.mean(positions)),
                    stat_y,
                    f"{stat_row['test']}\np={p:.3g} ({stars})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
                    clip_on=False,
                    zorder=5,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
            metric_label = metric_labels.get(metric, metric)
            ax.set_title(f"{group_col}", fontsize=10)
            if j == 0:
                ax.annotate(
                    metric_label,
                    xy=(-0.42, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Sample-level predicted {suptitle_metric} vs clinical metadata",
        y=1.01,
        fontsize=12,
    )
    fig.subplots_adjust(left=0.10, top=0.92 if n_rows > 1 else 0.88)
    fig.tight_layout()
    return fig, axes


def plot_tls_clinical_pvalue_summary(
    stats_summary: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (12.0, 4.5),
):
    """Side-by-side ``-log10(p)`` bar plots: Abundance TLS (left) vs Spatial TLS (right)."""
    import matplotlib.pyplot as plt

    metric_specs = (
        ("mean_idx_TLS", "Abundance TLS", "#2ca02c"),
        ("mean_idx_TLS_spatial", "Spatial TLS", "#ff7f0e"),
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, (metric, title, color) in zip(axes, metric_specs):
        sub = stats_summary.loc[stats_summary["metric"] == metric].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["neg_log10_p"], color=color, alpha=0.85)
        ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["clinical_variable"], rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("-log10(p)")
        for bar, (_, row) in zip(bars, sub.iterrows()):
            if np.isfinite(row["p_value"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{row['significance']}\np={row['p_value']:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Statistical significance across clinical groups", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig, axes
##################################################################

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
    """
    Load per-cell StarDist AUROC table.

    Returns
    -------
    df, class_names, prob_matrix (N, C)
    """
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
    """
    Map ``final_CT`` names to ``prob_*`` column indices.

    Labels absent from ``class_names`` are skipped (e.g. cell type not present in a
    given dataset's AUROC table).
    """
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


def cell_types_for_niches(
    annotation: pd.DataFrame,
    niches: Iterable[str],
    class_names: Sequence[str] | None = None,
) -> list[str]:
    niche_set = {str(n).strip() for n in niches}
    mask = annotation["Niches"].astype(str).isin(niche_set)
    types = sorted(annotation.loc[mask, "final_CT"].dropna().astype(str).unique())
    if class_names is None:
        return types
    present = set(class_names)
    return [t for t in types if t in present]


def sum_prob_columns(probs: np.ndarray, col_indices: Sequence[int]) -> np.ndarray:
    if not col_indices:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return probs[:, col_indices].sum(axis=1)


def safe_ratio(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return numer / np.maximum(denom, eps)


def um_to_he_pixel(radius_um: float, um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL) -> float:
    """Convert a radius in μm to ``spatial_HE`` pixel units."""
    if um_per_pixel <= 0:
        raise ValueError(f"um_per_pixel must be > 0, got {um_per_pixel}")
    return float(radius_um) / float(um_per_pixel)


def he_pixel_to_um(radius_px: float, um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL) -> float:
    """Convert a ``spatial_HE`` pixel radius to μm."""
    return float(radius_px) * float(um_per_pixel)


def infer_um_per_he_pixel(
    sample_dir: str | Path,
    *,
    h5ad_path: str | Path | None = None,
    csv_name_template: str = "{sample}_spatial_coords_um_pix_from_zarr.csv",
) -> float:
    """
    Infer μm/pixel for ``spatial_HE`` from ``{sample}_spatial_coords_um_pix_from_zarr.csv``.

    Falls back to ``DEFAULT_UM_PER_HE_PIXEL`` when the CSV is missing.
    """
    sample_dir = Path(sample_dir).expanduser().resolve()
    sample = sample_dir.name
    csv_path = sample_dir / csv_name_template.format(sample=sample)
    if not csv_path.is_file():
        return DEFAULT_UM_PER_HE_PIXEL

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
    if len(ratios):
        return float(ratios.median())
    return DEFAULT_UM_PER_HE_PIXEL


def resolve_spatial_radius(
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    *,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
) -> tuple[float, float, float]:
    """Return ``(radius_um, radius_px, um_per_pixel)``."""
    radius_px = um_to_he_pixel(radius_um, um_per_pixel)
    return float(radius_um), radius_px, float(um_per_pixel)


def percentile_vlim(
    values: np.ndarray,
    *,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
) -> tuple[float, float]:
    """Robust color limits for skewed per-cell score maps."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(values, vmin_percentile))
    hi = float(np.percentile(values, vmax_percentile))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _resolve_equal_aspect_grid_figsize(
    figsize: tuple[float, float],
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    *,
    nrows: int = 2,
    ncols: int = 2,
    title_pad: float = 0.10,
) -> tuple[float, float]:
    """
    Match figure height to H&E coordinate aspect for an equal-aspect subplot grid.

    With ``sharex/sharey`` and ``set_aspect("equal")``, passing a wide ``figsize``
    without enough height only adds horizontal whitespace; keeping width from
    ``figsize[0]`` and deriving height avoids that.
    """
    x = np.asarray(coords_x, dtype=np.float64)
    y = np.asarray(coords_y, dtype=np.float64)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-9)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-9)
    fw = float(figsize[0])
    fh = fw * (nrows / ncols) * (y_span / x_span) * (1.0 + title_pad)
    return (fw, fh)


##################################################################
# 2026.06.27try the TLS in spatial context 
##################################################################
def compute_tls_per_cell(probs: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Per-cell TLS abundance score = sum of predicted probs for TLS member classes."""
    cols = prob_columns_for_cell_types(class_names, TLS_CELL_TYPES)
    return sum_prob_columns(probs, cols)


def compartment_prob_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    cell_types: Sequence[str],
) -> np.ndarray:
    """Per-cell soft probability mass for one TLS compartment (B / T / DC)."""
    cols = prob_columns_for_cell_types(class_names, cell_types)
    return sum_prob_columns(probs, cols)


def load_spatial_coords_from_h5ad(
    h5ad_path: str | Path,
    obsm_key: str = "spatial_HE",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load nuclei coordinates from a StarDist / HE matched ``.h5ad``.

    Returns ``(coords (N,2), cell_ids)``.
    """
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    adata = ad.read_h5ad(str(path))
    if obsm_key not in adata.obsm:
        raise KeyError(f"{path.name}: obsm[{obsm_key!r}] not found.")
    coords = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coords (N, 2+), got {coords.shape}")
    coords = coords[:, :2]
    cell_ids = adata.obs_names.to_numpy(dtype=str)
    return coords, cell_ids


def align_auroc_df_with_h5ad(
    df: pd.DataFrame,
    coords: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Align AUROC table rows with ``spatial_HE`` coordinates.

    If ``cell_id`` is present, join on it; otherwise require identical row count
    and assume the same cell order as the h5ad.
    """
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
        out_coords = merged[["coord_x", "coord_y"]].to_numpy(dtype=np.float64)
        return merged, out_coords

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
    """
    Load AUROC probabilities and matched ``spatial_HE`` coordinates.

    Returns ``(df, class_names, probs, coords)``.
    """
    df, class_names, probs = load_stardist_auroc_csv(auroc_csv, class_names_csv)
    coords, cell_ids = load_spatial_coords_from_h5ad(stardist_h5ad, obsm_key=obsm_key)
    df, coords = align_auroc_df_with_h5ad(df, coords, cell_ids)
    return df, class_names, probs, coords


def local_neighborhood_mean(
    values: np.ndarray,
    coords: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Radius-neighborhood mean of ``values`` at each cell (same units as ``coords``)."""
    from sklearn.neighbors import BallTree

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values and coords length mismatch.")
    tree = BallTree(coords, metric="euclidean")
    neighbor_idx = tree.query_radius(coords, r=float(radius))
    out = np.zeros(values.shape[0], dtype=np.float64)
    for i, idx in enumerate(neighbor_idx):
        if len(idx) == 0:
            out[i] = values[i]
        else:
            out[i] = float(values[idx].mean())
    return out


TLS_COMPARTMENT_ENTROPY_MAX = float(np.log(3.0))


def compartment_entropy(
    b_local: np.ndarray,
    t_local: np.ndarray,
    dc_local: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Shannon entropy of normalized local B / T / DC proportions.

    High entropy indicates a balanced mix of the three TLS compartments within
    the neighborhood. Maximum ``log(3)`` when ``p(B) = p(T) = p(DC) = 1/3``.
    """
    b_local = np.asarray(b_local, dtype=np.float64)
    t_local = np.asarray(t_local, dtype=np.float64)
    dc_local = np.asarray(dc_local, dtype=np.float64)
    total = b_local + t_local + dc_local
    probs = np.stack([b_local, t_local, dc_local], axis=1) / np.maximum(total[:, None], eps)
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def combine_spatial_tls_with_entropy(
    spatial_tls: np.ndarray,
    entropy: np.ndarray,
    *,
    entropy_weight: float = 1.0,
) -> np.ndarray:
    """Combine co-localization score with normalized compartment entropy."""
    spatial_tls = np.asarray(spatial_tls, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    entropy_norm = entropy / TLS_COMPARTMENT_ENTROPY_MAX
    return spatial_tls + float(entropy_weight) * entropy_norm


def compute_spatial_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    radius_px: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatial TLS score per cell (co-localization).

    ``coords`` must be ``spatial_HE`` **pixel** coordinates. Neighborhood radius is
    specified in μm (``radius_um``) and converted internally unless ``radius_px`` is
    given explicitly.

    For each cell ``i``, within radius ``R``:
      B_local = mean_j Σ_{c∈B} p_j(c)
      T_local = mean_j Σ_{c∈T} p_j(c)
      DC_local = mean_j Σ_{c∈DC} p_j(c)
      TLS_spatial(i) = (B_local · T_local · DC_local) ^ (1/3)

    The cube root (geometric mean) is high only when **all three** compartments
    co-occur locally — unlike the abundance sum ``idx_TLS``.
    """
    if radius_px is None:
        radius_px = um_to_he_pixel(radius_um, um_per_pixel)
    b_prob = compartment_prob_per_cell(probs, class_names, TLS_B_CELLS)
    t_prob = compartment_prob_per_cell(probs, class_names, TLS_T_CELLS)
    dc_prob = compartment_prob_per_cell(probs, class_names, TLS_DC_CELLS)

    b_local = local_neighborhood_mean(b_prob, coords, radius_px)
    t_local = local_neighborhood_mean(t_prob, coords, radius_px)
    dc_local = local_neighborhood_mean(dc_prob, coords, radius_px)

    spatial_tls = np.power(
        np.maximum(b_local, eps) * np.maximum(t_local, eps) * np.maximum(dc_local, eps),
        1.0 / 3.0,
    )
    return spatial_tls, b_local, t_local, dc_local


def compute_cell_type_abundance_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    cell_types: Sequence[str],
) -> np.ndarray:
    """Per-cell sum of softmax probabilities over ``cell_types``."""
    cols = prob_columns_for_cell_types(class_names, cell_types)
    return sum_prob_columns(probs, cols)


def compute_fri_niche_abundances_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Per-cell abundance for each FRI niche compartment."""
    return {
        niche: compute_cell_type_abundance_per_cell(probs, class_names, types)
        for niche, types in FRI_NICHE_CELL_TYPES.items()
    }


def compute_spatial_fri_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    radius_px: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatial FRI score per cell (fibrosis niche co-localization / alveolar).

    Within radius ``R``:
      FS_local, IE_local, PM_local = local means of the three FRI niche abundances
      alv_local = local mean of AT1+AT2
      FRI_spatial(i) = (FS_local · IE_local · PM_local)^(1/3) / max(alv_local, eps)
    """
    if radius_px is None:
        radius_px = um_to_he_pixel(radius_um, um_per_pixel)
    niche_ab = compute_fri_niche_abundances_per_cell(probs, class_names)
    fs_local = local_neighborhood_mean(
        niche_ab["fibrotic_stromal"], coords, radius_px
    )
    ie_local = local_neighborhood_mean(
        niche_ab["injury_epithelium"], coords, radius_px
    )
    pm_local = local_neighborhood_mean(
        niche_ab["profibrotic_macrophages"], coords, radius_px
    )
    alv_prob = compute_cell_type_abundance_per_cell(probs, class_names, ALVEOLAR_EPITHELIUM)
    alv_local = local_neighborhood_mean(alv_prob, coords, radius_px)
    fri_spatial = np.power(
        np.maximum(fs_local, eps)
        * np.maximum(ie_local, eps)
        * np.maximum(pm_local, eps),
        1.0 / 3.0,
    )
    fri_spatial = fri_spatial / np.maximum(alv_local, eps)
    return fri_spatial, fs_local, ie_local, pm_local, alv_local


def compute_spatial_ari_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_ARI_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    radius_px: float | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatial ARI score per cell.

    Within radius ``R``:
      FB_local = local mean of Activated Fibrotic FBs + Myofibroblasts
      alv_local = local mean of AT1+AT2
      ARI_spatial(i) = FB_local / max(alv_local, eps)
    """
    if radius_px is None:
        radius_px = um_to_he_pixel(radius_um, um_per_pixel)
    fb_prob = compute_cell_type_abundance_per_cell(probs, class_names, ARI_FB_CELLS)
    alv_prob = compute_cell_type_abundance_per_cell(probs, class_names, ALVEOLAR_EPITHELIUM)
    fb_local = local_neighborhood_mean(fb_prob, coords, radius_px)
    alv_local = local_neighborhood_mean(alv_prob, coords, radius_px)
    ari_spatial = fb_local / np.maximum(alv_local, eps)
    return ari_spatial, fb_local, alv_local


def add_spatial_fri_ari_to_auroc_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """
    Add spatial FRI / ARI columns and hotspot flags to an AUROC dataframe.

    Hotspot detection uses the 95th percentile of ``idx_FRI_spatial`` and
    ``idx_ARI_spatial`` (same convention as ``tls_candidate``).
    """
    radius_um, radius_px, um_per_pixel = resolve_spatial_radius(
        radius_um, um_per_pixel=um_per_pixel
    )
    out = df.copy()
    fri_s, fs_loc, ie_loc, pm_loc, alv_loc_fri = compute_spatial_fri_per_cell(
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
    )
    ari_s, fb_loc, alv_loc_ari = compute_spatial_ari_per_cell(
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
    )
    out[f"{prefix}FRI_spatial"] = fri_s
    out[f"{prefix}FRI_FS_local"] = fs_loc
    out[f"{prefix}FRI_IE_local"] = ie_loc
    out[f"{prefix}FRI_PM_local"] = pm_loc
    out[f"{prefix}FRI_alveolar_local"] = alv_loc_fri
    out[f"{prefix}ARI_spatial"] = ari_s
    out[f"{prefix}ARI_FB_local"] = fb_loc
    out[f"{prefix}ARI_alveolar_local"] = alv_loc_ari
    out["fri_ari_spatial_radius_um"] = radius_um
    out["fri_ari_spatial_radius_px"] = radius_px
    out["fri_ari_um_per_he_pixel"] = um_per_pixel

    fri_cand, fri_thresh = detect_tls_hotspot_cells(
        fri_s, coords, percentile=hotspot_percentile
    )
    ari_cand, ari_thresh = detect_tls_hotspot_cells(
        ari_s, coords, percentile=hotspot_percentile
    )
    out["fri_candidate"] = fri_cand
    out["fri_spatial_threshold"] = fri_thresh
    out["ari_candidate"] = ari_cand
    out["ari_spatial_threshold"] = ari_thresh
    return out


def add_all_spatial_niche_indices_to_auroc_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    annotation: pd.DataFrame,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """
    Add abundance + spatial TLS / FRI / ARI columns in one call.

    Intended for cross-dataset (or per-sample) AUROC tables that already include
    ``cell_id`` and will receive ``coord_x`` / ``coord_y`` from ``spatial_HE``.
    """
    out = add_histology_indices_to_auroc_df(
        df, probs, class_names, annotation, prefix=prefix
    )
    out = add_spatial_tls_to_auroc_df(
        out,
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
        hotspot_percentile=hotspot_percentile,
        use_entropy=use_entropy,
        entropy_weight=entropy_weight,
        prefix=prefix,
    )
    out = add_spatial_fri_ari_to_auroc_df(
        out,
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
        hotspot_percentile=hotspot_percentile,
        prefix=prefix,
    )
    return out


def summarize_fri_ari_hotspots(
    df: pd.DataFrame,
    *,
    spatial_fri_col: str = "idx_FRI_spatial",
    spatial_ari_col: str = "idx_ARI_spatial",
    fri_candidate_col: str = "fri_candidate",
    ari_candidate_col: str = "ari_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
) -> dict[str, dict]:
    """Basic summary of spatial FRI / ARI hotspot cells."""
    out: dict[str, dict] = {}
    for index_name, spatial_col, candidate_col in (
        ("FRI", spatial_fri_col, fri_candidate_col),
        ("ARI", spatial_ari_col, ari_candidate_col),
    ):
        if candidate_col not in df.columns:
            raise KeyError(f"Missing column {candidate_col!r}")
        sub = df.loc[df[candidate_col]].copy()
        out[index_name] = {
            "n_cells_total": int(len(df)),
            "n_candidate_cells": int(len(sub)),
            "fraction_candidate": float(len(sub) / max(len(df), 1)),
            f"max_{spatial_col}": float(df[spatial_col].max()),
            f"mean_{spatial_col}_candidate": float(sub[spatial_col].mean())
            if len(sub)
            else float("nan"),
            "centroid_x": float(sub[coord_x].mean()) if len(sub) else float("nan"),
            "centroid_y": float(sub[coord_y].mean()) if len(sub) else float("nan"),
        }
    return out


def plot_fri_ari_abundance_vs_spatial(
    df: pd.DataFrame,
    sample: str,
    *,
    radius_um: float | None = None,
    fri_abundance_col: str = "idx_FRI_ratio",
    fri_spatial_col: str = "idx_FRI_spatial",
    ari_abundance_col: str = "idx_ARI_ratio",
    ari_spatial_col: str = "idx_ARI_spatial",
    fri_candidate_col: str = "fri_candidate",
    ari_candidate_col: str = "ari_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    vmin_percentile: float = 1.0,
    point_size: float = 0.3,
    candidate_size: float = 2.0,
    candidate_color: str = "cyan",
    figsize: tuple[float, float] = (12.0, 10.0),
    fit_figsize_to_coords: bool = True,
    dpi: int = 150,
    save_path: str | Path | None = None,
    axes: Sequence | None = None,
):
    """
    2×2 spatial maps: **ARI first** (row 1), then **FRI** (row 2).

    Biological read order (increasing specificity):
      1. **ARI** — structural remodeling: fibrotic stroma vs preserved alveolar epithelium.
      2. **FRI** — active profibrotic niche: co-localized stromal + injury + macrophage signal.
      3. **TLS** (separate plot) — adaptive immune niche remodeling.

    Interpretation: high ARI alone → alveolar structure replaced by fibrosis; FRI may
    still be low. High ARI **and** high FRI → active fibrosis microenvironment.

    Cyan dots mark ``ari_candidate`` / ``fri_candidate`` hotspots on spatial panels.

    ``figsize[0]`` sets the figure width (inches). When ``fit_figsize_to_coords=True``
    (default), height is derived from ``coord_x/coord_y`` span so wider figures
    enlarge the tissue panels instead of adding empty margins. Set
    ``fit_figsize_to_coords=False`` to use the ``figsize`` tuple literally.

    If ``save_path`` is set, saves the figure after layout (only when
    this function creates the figure, i.e. ``axes is None``).
    """
    import matplotlib.pyplot as plt

    required = (fri_abundance_col, fri_spatial_col, ari_abundance_col, ari_spatial_col)
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}")

    if radius_um is None and "fri_ari_spatial_radius_um" in df.columns:
        radius_um = float(df["fri_ari_spatial_radius_um"].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_FRI_SPATIAL_RADIUS_UM

    x = df[coord_x].to_numpy(dtype=np.float64)
    y = df[coord_y].to_numpy(dtype=np.float64)
    plot_figsize = (
        _resolve_equal_aspect_grid_figsize(figsize, x, y, nrows=2, ncols=2)
        if fit_figsize_to_coords
        else figsize
    )

    created_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(
            2, 2, figsize=plot_figsize, sharex=True, sharey=True, layout="constrained"
        )
    else:
        axes = np.asarray(axes)
        if axes.shape != (2, 2):
            raise ValueError("Expected a 2×2 axes array.")
        fig = axes[0, 0].figure
        if fit_figsize_to_coords:
            fig.set_size_inches(*plot_figsize)

    panels = [
        (
            df[ari_abundance_col].to_numpy(dtype=np.float64),
            f"{sample}: ARI abundance — structural remodeling ({ari_abundance_col})",
            "viridis",
            None,
        ),
        (
            df[ari_spatial_col].to_numpy(dtype=np.float64),
            f"{sample}: ARI spatial (R={radius_um:g} μm)",
            "magma",
            ari_candidate_col,
        ),
        (
            df[fri_abundance_col].to_numpy(dtype=np.float64),
            f"{sample}: FRI abundance — profibrotic niche ({fri_abundance_col})",
            "viridis",
            None,
        ),
        (
            df[fri_spatial_col].to_numpy(dtype=np.float64),
            f"{sample}: FRI spatial (R={radius_um:g} μm)",
            "magma",
            fri_candidate_col,
        ),
    ]
    flat_axes = axes.ravel()
    for ax, (values, title, cmap, candidate_col) in zip(flat_axes, panels):
        vmin, vmax = percentile_vlim(
            values, vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile
        )
        sc = ax.scatter(x, y, c=values, s=point_size, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    axes[0, 0].invert_yaxis()
    for ax, candidate_col in zip(flat_axes[1::2], (ari_candidate_col, fri_candidate_col)):
        _plot_index_candidate_markers(
            ax,
            df,
            candidate_col=candidate_col,
            coord_x=coord_x,
            coord_y=coord_y,
            candidate_size=candidate_size,
            candidate_color=candidate_color,
            label=f"{candidate_col.replace('_candidate', '')} candidate",
        )

    if created_fig:
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


def plot_ari_fri_abundance_vs_spatial(
    df: pd.DataFrame,
    sample: str,
    **kwargs,
):
    """Alias: ARI (structural) row first, FRI (active profibrotic) row second."""
    return plot_fri_ari_abundance_vs_spatial(df, sample, **kwargs)


##################################################################
# 2026.06.27 LLY: add the ARI–FRI hierarchical remodeling analysis
# ARI–FRI hierarchical remodeling analysis
# ARI = broad structural alveolar remodeling; FRI = active profibrotic niche
# embedded within remodeled tissue (not competing biomarkers).
##################################################################
ARI_FRI_QUADRANT_ORDER: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

ARI_FRI_QUADRANT_LABELS: dict[str, str] = {
    "Q1": "Normal alveolar region",
    "Q2": "Broad fibrotic remodeling (alveolar replacement)",
    "Q3": "Immune / epithelial activation without extensive remodeling",
    "Q4": "Active fibrotic niche (stromal + injury + profibrotic macrophages)",
}

ARI_FRI_QUADRANT_PLOT_LABELS: dict[str, str] = {
    "Q1": "Q1: Normal alveolar",
    "Q2": "Q2: Broad remodeling (ARI\u2191)",
    "Q3": "Q3: Activation w/o remodeling",
    "Q4": "Q4: Active fibrotic niche (ARI\u2191+FRI\u2191)",
}

ARI_FRI_QUADRANT_COLORS: dict[str, str] = {
    "Q1": "#d9d9d9",
    "Q2": "#ff8c00",
    "Q3": "#4169e1",
    "Q4": "#dc143c",
}

DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE = 95.0
DEFAULT_Q4_CLINICAL_COLUMNS: tuple[str, ...] = ("Status", "Sample_Affect_Pairing")

# Boxplot styling for cohort clinical comparisons (Complete vs Incomplete).
COMPLETE_COHORT_BOX_FACE = "#f4a582"
COMPLETE_COHORT_BOX_EDGE = "#d6604b"
INCOMPLETE_COHORT_BOX_FACE = "#92c5de"
INCOMPLETE_COHORT_BOX_EDGE = "#2166ac"


def otsu_threshold_1d(values: np.ndarray) -> float:
    """Otsu threshold for a 1-D array (fallback when percentile is too conservative)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin
    hist, bin_edges = np.histogram(values, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    cum_mean = np.cumsum(hist * bin_centers)
    total_mean = cum_mean[-1] / max(weight1[-1], 1.0)
    mean1 = cum_mean / np.maximum(weight1, 1.0)
    mean2 = (total_mean * weight1[-1] - cum_mean) / np.maximum(weight2, 1.0)
    between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    if between.size == 0:
        return float(np.median(values))
    return float(bin_centers[int(np.argmax(between))])


def compute_ari_fri_thresholds(
    ari: np.ndarray,
    fri: np.ndarray,
    *,
    method: Literal["percentile", "otsu"] = "percentile",
    percentile: float = DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE,
) -> tuple[float, float]:
    """Return ``(ari_threshold, fri_threshold)`` for quadrant classification."""
    ari = np.asarray(ari, dtype=np.float64)
    fri = np.asarray(fri, dtype=np.float64)
    if method == "percentile":
        return float(np.percentile(ari[np.isfinite(ari)], percentile)), float(
            np.percentile(fri[np.isfinite(fri)], percentile)
        )
    if method == "otsu":
        return otsu_threshold_1d(ari), otsu_threshold_1d(fri)
    raise ValueError(f"Unknown threshold method={method!r}; expected 'percentile' or 'otsu'.")


def assign_ari_fri_quadrant_codes(
    ari: np.ndarray,
    fri: np.ndarray,
    *,
    ari_threshold: float,
    fri_threshold: float,
) -> np.ndarray:
    """
    Classify cells into Q1–Q4.

    Q1 low/low, Q2 high ARI / low FRI, Q3 low ARI / high FRI, Q4 high/high.
    """
    ari = np.asarray(ari, dtype=np.float64)
    fri = np.asarray(fri, dtype=np.float64)
    high_ari = ari >= ari_threshold
    high_fri = fri >= fri_threshold
    codes = np.full(ari.shape[0], "Q1", dtype=object)
    codes[high_ari & ~high_fri] = "Q2"
    codes[~high_ari & high_fri] = "Q3"
    codes[high_ari & high_fri] = "Q4"
    return codes


def classify_ari_fri_quadrants(
    df: pd.DataFrame,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    threshold_method: Literal["percentile", "otsu"] = "percentile",
    percentile: float = DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE,
    ari_threshold: float | None = None,
    fri_threshold: float | None = None,
    quadrant_col: str = "ari_fri_quadrant",
) -> tuple[pd.DataFrame, dict]:
    """Add ``quadrant_col`` and return threshold metadata."""
    if ari_col not in df.columns or fri_col not in df.columns:
        raise KeyError(f"Dataframe must contain {ari_col!r} and {fri_col!r}.")
    out = df.copy()
    ari = out[ari_col].to_numpy(dtype=np.float64)
    fri = out[fri_col].to_numpy(dtype=np.float64)
    if ari_threshold is None or fri_threshold is None:
        ari_t, fri_t = compute_ari_fri_thresholds(
            ari, fri, method=threshold_method, percentile=percentile
        )
        ari_threshold = ari_t if ari_threshold is None else ari_threshold
        fri_threshold = fri_t if fri_threshold is None else fri_threshold
    out[quadrant_col] = assign_ari_fri_quadrant_codes(
        ari, fri, ari_threshold=ari_threshold, fri_threshold=fri_threshold
    )
    meta = {
        "ari_threshold": float(ari_threshold),
        "fri_threshold": float(fri_threshold),
        "threshold_method": threshold_method,
        "percentile": float(percentile) if threshold_method == "percentile" else None,
    }
    return out, meta


def spearman_ari_fri_correlation(
    df: pd.DataFrame,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
) -> dict[str, float]:
    """Spearman correlation between spatial ARI and FRI (complementary, not competing)."""
    from scipy.stats import spearmanr

    sub = df[[ari_col, fri_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n_cells": int(len(sub))}
    rho, p = spearmanr(sub[ari_col], sub[fri_col])
    return {"rho": float(rho), "p_value": float(p), "n_cells": int(len(sub))}


def summarize_ari_fri_quadrants(
    df: pd.DataFrame,
    *,
    quadrant_col: str = "ari_fri_quadrant",
) -> pd.DataFrame:
    """Per-quadrant cell counts and percentages with biological labels."""
    counts = df[quadrant_col].value_counts()
    n = max(len(df), 1)
    rows = []
    for code in ARI_FRI_QUADRANT_ORDER:
        count = int(counts.get(code, 0))
        rows.append(
            {
                "quadrant": code,
                "label": ARI_FRI_QUADRANT_LABELS[code],
                "n_cells": count,
                "percent": 100.0 * count / n,
            }
        )
    return pd.DataFrame(rows)


def _fit_lowess_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.2,
    max_points: int = 50_000,
    n_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smooth for ARI→FRI trend (subsamples very large inputs)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        return x, y
    if x.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x, y = x[idx], y[idx]
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        order = np.argsort(x)
        x, y = x[order], y[order]
        n_bins = min(60, max(12, x.size // 200))
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        cx, cy = [], []
        for i in range(n_bins):
            m = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
            if np.any(m):
                cx.append(float(np.median(x[m])))
                cy.append(float(np.median(y[m])))
        return np.asarray(cx), np.asarray(cy)


def plot_ari_fri_correlation_scatter(
    df: pd.DataFrame,
    sample_name: str,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    ari_threshold: float | None = None,
    fri_threshold: float | None = None,
    quadrant_col: str = "ari_fri_quadrant",
    spearman_rho: float | None = None,
    spearman_p: float | None = None,
    point_alpha: float = 0.15,
    point_size: float = 1.5,
    max_plot_points: int = 80_000,
    lowess_frac: float = 0.2,
    figsize: tuple[float, float] = (7.5, 6.5),
    ax=None,
    panel_label: str = "",
):
    """
    Scatter of spatial ARI vs FRI with LOWESS trend and quadrant thresholds.

    Emphasizes hierarchical embedding (FRI rises within high-ARI regions).
    """
    import matplotlib.pyplot as plt

    plot_df = df[[ari_col, fri_col, quadrant_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        raise ValueError("No finite ARI/FRI values to plot.")
    if spearman_rho is None or spearman_p is None:
        corr = spearman_ari_fri_correlation(plot_df, ari_col=ari_col, fri_col=fri_col)
        spearman_rho = corr["rho"]
        spearman_p = corr["p_value"]

    if len(plot_df) > max_plot_points:
        plot_df = plot_df.sample(n=max_plot_points, random_state=0)

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for code in ARI_FRI_QUADRANT_ORDER:
        sub = plot_df.loc[plot_df[quadrant_col] == code]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[ari_col],
            sub[fri_col],
            s=point_size,
            c=ARI_FRI_QUADRANT_COLORS[code],
            alpha=point_alpha,
            label=ARI_FRI_QUADRANT_PLOT_LABELS[code],
            rasterized=True,
        )

    x_all = plot_df[ari_col].to_numpy(dtype=np.float64)
    y_all = plot_df[fri_col].to_numpy(dtype=np.float64)
    lx, ly = _fit_lowess_curve(x_all, y_all, frac=lowess_frac)
    ax.plot(lx, ly, color="black", linewidth=2.0, label="LOWESS trend", zorder=5)

    if ari_threshold is not None:
        ax.axvline(ari_threshold, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    if fri_threshold is not None:
        ax.axhline(fri_threshold, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Spatial ARI (structural remodeling)")
    ax.set_ylabel("Spatial FRI (active profibrotic niche)")
    title_prefix = "A  Feature space" if panel_label else ""
    ax.set_title(
        f"{title_prefix + ' — ' if title_prefix else ''}{sample_name}\n"
        f"Each point = one cell (not tissue layout) | "
        f"Spearman \u03c1={spearman_rho:.3f}, p={spearman_p:.2g}"
    )
    _annotate_ari_fri_quadrant_corners(
        ax,
        ari_threshold=ari_threshold,
        fri_threshold=fri_threshold,
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )
    ax.legend(loc="upper left", fontsize=7, markerscale=3, framealpha=0.9)
    ax.grid(alpha=0.25)
    if created:
        fig.tight_layout()
    return fig, ax


def _annotate_ari_fri_quadrant_corners(
    ax,
    *,
    ari_threshold: float | None,
    fri_threshold: float | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Label Q1–Q4 regions on the ARI–FRI scatter (feature space, not spatial map)."""
    if ari_threshold is None or fri_threshold is None:
        return
    xlo, xhi = xlim
    ylo, yhi = ylim
    pad_x = 0.03 * max(xhi - xlo, 1e-9)
    pad_y = 0.04 * max(yhi - ylo, 1e-9)
    corner_specs = (
        ("Q1", xlo + pad_x, ylo + pad_y, "left", "bottom"),
        ("Q2", xhi - pad_x, ylo + pad_y, "right", "bottom"),
        ("Q3", xlo + pad_x, yhi - pad_y, "left", "top"),
        ("Q4", xhi - pad_x, yhi - pad_y, "right", "top"),
    )
    for code, x, y, ha, va in corner_specs:
        ax.text(
            x,
            y,
            ARI_FRI_QUADRANT_PLOT_LABELS[code],
            ha=ha,
            va=va,
            fontsize=7,
            color=ARI_FRI_QUADRANT_COLORS[code],
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
            zorder=6,
        )


def plot_ari_fri_quadrant_spatial(
    df: pd.DataFrame,
    sample_name: str,
    *,
    quadrant_col: str = "ari_fri_quadrant",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    spearman_rho: float | None = None,
    q4_percent: float | None = None,
    point_size: float = 0.35,
    figsize: tuple[float, float] = (8.0, 8.0),
    ax=None,
    panel_label: str = "",
):
    """Spatial map colored by ARI–FRI remodeling quadrant."""
    import matplotlib.pyplot as plt

    plot_df = _dedupe_cells_for_spatial_plot(df)
    if quadrant_col not in plot_df.columns:
        raise KeyError(f"Missing column {quadrant_col!r}")
    if q4_percent is None:
        q4_percent = float((plot_df[quadrant_col] == "Q4").mean() * 100.0)

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for code in ARI_FRI_QUADRANT_ORDER:
        sub = plot_df.loc[plot_df[quadrant_col] == code]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[coord_x],
            sub[coord_y],
            s=point_size,
            c=ARI_FRI_QUADRANT_COLORS[code],
            alpha=0.75 if code == "Q4" else 0.55,
            label=ARI_FRI_QUADRANT_PLOT_LABELS[code],
            rasterized=True,
            zorder=2 if code == "Q4" else 1,
        )

    rho_txt = f"Spearman \u03c1={spearman_rho:.3f}" if spearman_rho is not None else ""
    title_prefix = "B  Tissue space" if panel_label else ""
    ax.set_title(
        f"{title_prefix + ' — ' if title_prefix else ''}{sample_name}\n"
        f"Each point = nucleus on H&E | {rho_txt} | Q4 = {q4_percent:.2f}%"
    )
    ax.set_xlabel("spatial_HE x (px)")
    ax.set_ylabel("spatial_HE y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.92)
    if created:
        fig.tight_layout()
    return fig, ax


def plot_ari_fri_hierarchical_panels(
    df: pd.DataFrame,
    sample_name: str,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    quadrant_col: str = "ari_fri_quadrant",
    ari_threshold: float,
    fri_threshold: float,
    spearman_rho: float,
    spearman_p: float,
    q4_percent: float,
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    figsize: tuple[float, float] = (14.0, 6.0),
):
    """
    One figure, two **distinct** panels:

    A) Feature space — ARI vs FRI per cell (with LOWESS + quadrant thresholds)
    B) Tissue space — physical ``coord_x/coord_y`` map colored by remodeling quadrant

    These are not duplicates: panel A is index co-variation; panel B is where
    quadrants fall on the slide.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_ari_fri_correlation_scatter(
        df,
        sample_name,
        ari_col=ari_col,
        fri_col=fri_col,
        ari_threshold=ari_threshold,
        fri_threshold=fri_threshold,
        quadrant_col=quadrant_col,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        ax=axes[0],
        panel_label="A",
    )
    plot_ari_fri_quadrant_spatial(
        df,
        sample_name,
        quadrant_col=quadrant_col,
        coord_x=coord_x,
        coord_y=coord_y,
        spearman_rho=spearman_rho,
        q4_percent=q4_percent,
        ax=axes[1],
        panel_label="B",
    )
    fig.suptitle(
        f"{sample_name} — Hierarchical remodeling: structural ARI\u2192 active FRI niche\n"
        f"Normal (Q1) \u2192 broad remodeling (Q2, ARI\u2191) \u2192 active niche (Q4, ARI\u2191+FRI\u2191)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes


def analyze_ari_fri_relationship(
    df: pd.DataFrame,
    sample_name: str,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    threshold_method: Literal["percentile", "otsu"] = "percentile",
    percentile: float = DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE,
    quadrant_col: str = "ari_fri_quadrant",
    plot: bool = True,
    plot_layout: Literal["combined", "separate"] = "combined",
    figsize: tuple[float, float] | None = None,
    figsize_scatter: tuple[float, float] = (7.5, 6.5),
    figsize_spatial: tuple[float, float] = (8.0, 8.0),
    dpi: int = 150,
    save_path: str | Path | None = None,
) -> dict:
    """
    Single-sample hierarchical ARI–FRI analysis.

    Demonstrates that FRI identifies active profibrotic niches embedded within
    ARI-defined structurally remodeled tissue (Normal \u2192 ARI\u2191 \u2192 ARI\u2191+FRI\u2191).

    If ``save_path`` is set, saves the combined hierarchical figure, or (when
    ``plot_layout="separate"``) ``{stem}_scatter{suffix}`` and
    ``{stem}_spatial{suffix}`` derived from ``save_path``.

    In Jupyter, do **not** also put ``result["figures"]["hierarchical"]`` as the
    last line when ``plot=True`` — the open figure is already rendered once at
    cell end (``%matplotlib inline``), and returning it again shows a duplicate.
    """
    df_q, thresh_meta = classify_ari_fri_quadrants(
        df,
        ari_col=ari_col,
        fri_col=fri_col,
        threshold_method=threshold_method,
        percentile=percentile,
        quadrant_col=quadrant_col,
    )
    corr = spearman_ari_fri_correlation(df_q, ari_col=ari_col, fri_col=fri_col)
    quad = summarize_ari_fri_quadrants(df_q, quadrant_col=quadrant_col)
    pct = {f"{row.quadrant}_percent": float(row.percent) for row in quad.itertuples()}
    count = {f"{row.quadrant}_n_cells": int(row.n_cells) for row in quad.itertuples()}

    summary = {
        "sample": str(sample_name),
        "n_cells": int(len(df_q)),
        "ARI_median": float(df_q[ari_col].median()),
        "FRI_median": float(df_q[fri_col].median()),
        "Spearman_rho": corr["rho"],
        "Spearman_p": corr["p_value"],
        "Q4_percent": pct.get("Q4_percent", float("nan")),
        **pct,
        **count,
        **thresh_meta,
        "quadrant_table": quad,
        "df_with_quadrants": df_q,
    }

    figures: dict[str, object] = {}
    if plot:
        if figsize is None:
            figsize = (14.0, 6.0) if plot_layout == "combined" else figsize_scatter
        if plot_layout == "combined":
            fig, _ = plot_ari_fri_hierarchical_panels(
                df_q,
                sample_name,
                ari_col=ari_col,
                fri_col=fri_col,
                quadrant_col=quadrant_col,
                ari_threshold=thresh_meta["ari_threshold"],
                fri_threshold=thresh_meta["fri_threshold"],
                spearman_rho=corr["rho"],
                spearman_p=corr["p_value"],
                q4_percent=summary["Q4_percent"],
                figsize=figsize,
            )
            figures["hierarchical"] = fig
        else:
            fig_sc, _ = plot_ari_fri_correlation_scatter(
                df_q,
                sample_name,
                ari_col=ari_col,
                fri_col=fri_col,
                ari_threshold=thresh_meta["ari_threshold"],
                fri_threshold=thresh_meta["fri_threshold"],
                quadrant_col=quadrant_col,
                spearman_rho=corr["rho"],
                spearman_p=corr["p_value"],
                figsize=figsize_scatter,
            )
            fig_map, _ = plot_ari_fri_quadrant_spatial(
                df_q,
                sample_name,
                quadrant_col=quadrant_col,
                spearman_rho=corr["rho"],
                q4_percent=summary["Q4_percent"],
                figsize=figsize_spatial,
            )
            figures["correlation"] = fig_sc
            figures["spatial_quadrants"] = fig_map
        if save_path is not None:
            save_path = Path(save_path)
            if plot_layout == "combined":
                figures["hierarchical"].savefig(save_path, dpi=dpi, bbox_inches="tight")
            else:
                suffix = save_path.suffix or ".jpg"
                stem = save_path.stem
                parent = save_path.parent
                figures["correlation"].savefig(
                    parent / f"{stem}_scatter{suffix}", dpi=dpi, bbox_inches="tight"
                )
                figures["spatial_quadrants"].savefig(
                    parent / f"{stem}_spatial{suffix}", dpi=dpi, bbox_inches="tight"
                )
        summary["figures"] = figures
    return summary


def batch_analyze_ari_fri(
    sample_dict: Mapping[str, pd.DataFrame],
    *,
    plot: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Batch hierarchical ARI–FRI analysis across samples.

    ``sample_dict`` maps sample name \u2192 cell-level dataframe with spatial indices.
    """
    rows = []
    for sample_name, df in sample_dict.items():
        result = analyze_ari_fri_relationship(df, sample_name, plot=plot, **kwargs)
        rows.append(
            {
                "Sample": result["sample"],
                "ARI_median": result["ARI_median"],
                "FRI_median": result["FRI_median"],
                "Spearman_rho": result["Spearman_rho"],
                "Spearman_p": result["Spearman_p"],
                "Q1_percent": result.get("Q1_percent", float("nan")),
                "Q2_percent": result.get("Q2_percent", float("nan")),
                "Q3_percent": result.get("Q3_percent", float("nan")),
                "Q4_percent": result.get("Q4_percent", float("nan")),
                "n_cells": result["n_cells"],
                "ari_threshold": result["ari_threshold"],
                "fri_threshold": result["fri_threshold"],
                "threshold_method": result["threshold_method"],
            }
        )
    return pd.DataFrame(rows)

##############################################
## 2026-07-03: Cross-dataset cohort analysis
##############################################
# Xenium-lung compatibility alias. Canonical source is
# ``plotting_palettes.clinical_group_order(pan_organ=...)``.
DEFAULT_CLINICAL_GROUP_ORDER: dict[str, tuple[str, ...]] = {
    "Status": ("Control", "Disease"),
    "Sample_Affect_Pairing": ("Unaffected", "Less_Affected", "More_Affected"),
}


def _resolve_clinical_group_order(
    pan_organ: str | None = None,
    group_order: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Organ-specific clinical x-axis order (``xenium_lung`` / ``codex_hcc``)."""
    import sys

    pkg_dir = Path(__file__).resolve().parents[1] / "Hist2Pheno_pkg"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    from plotting_palettes import clinical_group_order

    return clinical_group_order(pan_organ or "xenium_lung", extra=group_order)


def _build_sample_color_map(sample_ids: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    """Stable sample → RGBA colors (tab20 + tab20b; up to 40 distinct samples)."""
    import matplotlib.pyplot as plt

    unique = sorted({str(s) for s in sample_ids})
    n = len(unique)
    if n == 0:
        return {}
    palette = [plt.get_cmap("tab20")(i) for i in range(20)] + [
        plt.get_cmap("tab20b")(i) for i in range(20)
    ]
    if n > len(palette):
        extra = plt.get_cmap("turbo", n - len(palette))
        palette.extend(extra(i) for i in range(n - len(palette)))
    return {sid: palette[i] for i, sid in enumerate(unique)}


def compute_cohort_ari_fri_thresholds(
    sample_dict: Mapping[str, pd.DataFrame],
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    percentile: float = DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE,
) -> dict:
    """
    Pool all cells across a cohort and return global ARI/FRI cutoffs.

    Used for **Global Q4**: cells with ``ARI >= ari_threshold_global`` and
    ``FRI >= fri_threshold_global`` (same thresholds applied to every sample).
    """
    if not sample_dict:
        raise ValueError("sample_dict is empty.")
    ari_parts: list[np.ndarray] = []
    fri_parts: list[np.ndarray] = []
    for df in sample_dict.values():
        if ari_col not in df.columns or fri_col not in df.columns:
            raise KeyError(f"Each dataframe must contain {ari_col!r} and {fri_col!r}.")
        ari_parts.append(df[ari_col].to_numpy(dtype=np.float64))
        fri_parts.append(df[fri_col].to_numpy(dtype=np.float64))
    ari_all = np.concatenate(ari_parts)
    fri_all = np.concatenate(fri_parts)
    ari_t, fri_t = compute_ari_fri_thresholds(
        ari_all, fri_all, method="percentile", percentile=percentile
    )
    valid = np.isfinite(ari_all) & np.isfinite(fri_all)
    return {
        "ari_threshold_global": float(ari_t),
        "fri_threshold_global": float(fri_t),
        "percentile": float(percentile),
        "n_cells_pooled": int(valid.sum()),
        "n_samples": len(sample_dict),
    }


def compute_active_niche_burden(
    df: pd.DataFrame,
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
) -> float:
    """
    Continuous active niche burden per sample: ``mean(sqrt(ARI_i * FRI_i))`` over cells.

    Equivalent to the mean per-cell geometric mean of ARI and FRI; compresses scale
    vs ``mean(ARI * FRI)`` and is less driven by extreme co-elevation outliers.
    No percentile threshold.
    """
    sub = df[[ari_col, fri_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return float("nan")
    ari = sub[ari_col].astype(np.float64).clip(lower=0.0)
    fri = sub[fri_col].astype(np.float64).clip(lower=0.0)
    return float(np.sqrt(ari * fri).mean())


def compute_global_q4_percent(
    df: pd.DataFrame,
    *,
    ari_threshold: float,
    fri_threshold: float,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
) -> float:
    """Fraction of cells in Global Q4 (%) using cohort-wide ARI/FRI cutoffs."""
    sub = df[[ari_col, fri_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return float("nan")
    ari = sub[ari_col].to_numpy(dtype=np.float64)
    fri = sub[fri_col].to_numpy(dtype=np.float64)
    q4 = (ari >= ari_threshold) & (fri >= fri_threshold)
    return float(q4.mean() * 100.0)


def batch_analyze_ari_fri_cohort(
    sample_dict: Mapping[str, pd.DataFrame],
    *,
    ari_col: str = "idx_ARI_spatial",
    fri_col: str = "idx_FRI_spatial",
    percentile: float = DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE,
    include_per_sample_q4: bool = False,
    plot: bool = False,
    **analyze_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Cohort fibrotic-niche summary for cross-dataset comparison.

    Adds per sample:

    - ``Q4_global_percent``: ARI/FRI both >= cohort pooled ``percentile`` cutoffs
    - ``active_niche_burden``: ``mean(sqrt(ARI * FRI))`` (continuous, no threshold)

    Optionally includes per-sample ``Q4_percent`` (local 95th-percentile thresholds)
    when ``include_per_sample_q4=True``.

    Returns ``(batch_df, cohort_meta)`` where ``cohort_meta`` holds global thresholds.
    """
    cohort_meta = compute_cohort_ari_fri_thresholds(
        sample_dict,
        ari_col=ari_col,
        fri_col=fri_col,
        percentile=percentile,
    )
    ari_g = cohort_meta["ari_threshold_global"]
    fri_g = cohort_meta["fri_threshold_global"]

    rows = []
    for sample_name, df in sample_dict.items():
        row: dict = {
            "Sample": str(sample_name),
            "n_cells": int(len(df)),
            "ARI_median": float(df[ari_col].median()),
            "FRI_median": float(df[fri_col].median()),
            "Q4_global_percent": compute_global_q4_percent(
                df,
                ari_threshold=ari_g,
                fri_threshold=fri_g,
                ari_col=ari_col,
                fri_col=fri_col,
            ),
            "active_niche_burden": compute_active_niche_burden(
                df, ari_col=ari_col, fri_col=fri_col
            ),
            "ari_threshold_global": ari_g,
            "fri_threshold_global": fri_g,
            "cohort_percentile": percentile,
        }
        corr = spearman_ari_fri_correlation(df, ari_col=ari_col, fri_col=fri_col)
        row["Spearman_rho"] = corr["rho"]
        row["Spearman_p"] = corr["p_value"]
        if include_per_sample_q4:
            result = analyze_ari_fri_relationship(
                df, sample_name, plot=plot, ari_col=ari_col, fri_col=fri_col, **analyze_kwargs
            )
            row["Q4_percent"] = result.get("Q4_percent", float("nan"))
            row["ari_threshold_local"] = result.get("ari_threshold", float("nan"))
            row["fri_threshold_local"] = result.get("fri_threshold", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows), cohort_meta
##############################################

def merge_ari_fri_batch_with_clinical(
    batch_summary: pd.DataFrame,
    clinical_info: pd.DataFrame,
    *,
    sample_col: str = "Sample",
) -> pd.DataFrame:
    """Attach clinical metadata to batch ARI–FRI summary (merge on sample / Donor_ID)."""
    out = batch_summary.copy()
    out[sample_col] = out[sample_col].astype(str)
    clinical = clinical_info.copy()
    clinical["Donor_ID"] = clinical["Donor_ID"].astype(str)
    merged = out.merge(clinical, left_on=sample_col, right_on="Donor_ID", how="left")
    for col in DEFAULT_Q4_CLINICAL_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)
    return merged


def test_q4_by_clinical_groups(
    df: pd.DataFrame,
    *,
    q4_col: str = "Q4_percent",
    sample_col: str = "Sample",
    clinical_columns: Sequence[str] = DEFAULT_Q4_CLINICAL_COLUMNS,
    pan_organ: str | None = "xenium_lung",
    min_group_size: int = 2,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> pd.DataFrame:
    """
    Compare a sample-level niche metric across clinical groups.

    Default ``q4_col`` is legacy per-sample ``Q4_percent``; for cohort analysis use
    ``Q4_global_percent`` or ``active_niche_burden``.

    Uses two-sided Mann–Whitney U (Wilcoxon rank-sum) for exactly two groups;
    Kruskal–Wallis for three or more.
    """
    from scipy.stats import kruskal, mannwhitneyu

    order_map = _resolve_clinical_group_order(pan_organ)
    rows = []
    for group_col in clinical_columns:
        if group_col not in df.columns or q4_col not in df.columns:
            continue
        sub = df.dropna(subset=[group_col, q4_col]).copy()
        sub[group_col] = sub[group_col].astype(str)
        present = sub[group_col].unique().tolist()
        preset = [g for g in order_map.get(group_col, ()) if g in present]
        rest = sorted(g for g in present if g not in preset)
        label_order = preset + rest
        groups = []
        labels = []
        for label in label_order:
            vals = sub.loc[sub[group_col] == label, q4_col].astype(float).to_numpy()
            if len(vals) >= min_group_size:
                groups.append(vals)
                labels.append(str(label))
        if len(groups) < 2:
            rows.append(
                {
                    "clinical_variable": group_col,
                    "metric_col": q4_col,
                    "test": "insufficient_groups",
                    "p_value": float("nan"),
                    "statistic": float("nan"),
                    "n_groups": len(groups),
                    "group_labels": labels,
                }
            )
            continue
        if len(groups) == 2:
            if alternative == "two-sided":
                stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            elif alternative == "greater":
                stat, p = mannwhitneyu(groups[1], groups[0], alternative="greater")
            else:
                stat, p = mannwhitneyu(groups[1], groups[0], alternative="less")
            test_name = "Mann-Whitney U"
        else:
            stat, p = kruskal(*groups)
            test_name = "Kruskal-Wallis"
        medians = [float(np.median(g)) for g in groups]
        rows.append(
            {
                "clinical_variable": group_col,
                "metric_col": q4_col,
                "test": test_name,
                "p_value": float(p),
                "statistic": float(stat),
                "n_groups": len(groups),
                "group_labels": labels,
                "group_median_q4": medians,
                "group_median_values": medians,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["neg_log10_p"] = -np.log10(out["p_value"].clip(lower=1e-300))
    return out


def plot_q4_clinical_comparison(
    df: pd.DataFrame,
    stats_summary: pd.DataFrame,
    *,
    q4_col: str = "Q4_percent",
    sample_col: str = "Sample",
    clinical_columns: Sequence[str] = DEFAULT_Q4_CLINICAL_COLUMNS,
    pan_organ: str | None = "xenium_lung",
    group_order: Mapping[str, Sequence[str]] | None = None,
    figsize: tuple[float, float] | None = None,
    y_annotation_pad: float = 0.22,
    ylabel: str | None = None,
    suptitle: str | None = None,
    show_sample_legend: bool = True,
    legend_ncol: int = 1,
    legend_fontsize: float = 6.5,
    point_size: float = 28,
    dpi: int = 150,
    save_path: str | Path | None = None,
    box_facecolor: str = COMPLETE_COHORT_BOX_FACE,
    box_edgecolor: str = COMPLETE_COHORT_BOX_EDGE,
):
    """
    Boxplots of a sample-level niche metric by clinical group.

    Each scatter point is one sample. When ``sample_col`` is present and
    ``show_sample_legend=True``, points are colored by sample (consistent across
    panels) with a shared sample legend on the right.

    Use ``q4_col='Q4_global_percent'`` or ``'active_niche_burden'`` for cohort analysis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cols = [c for c in clinical_columns if c in df.columns]
    if not cols:
        raise KeyError(f"No clinical columns found among {clinical_columns!r}.")
    if figsize is None:
        base_w = max(3.5 * len(cols), 8.0)
        figsize = (base_w + (2.2 if show_sample_legend and sample_col in df.columns else 0.0), 5.0)

    sample_colors: dict[str, tuple[float, float, float, float]] = {}
    sample_order: list[str] = []
    color_by_sample = show_sample_legend and sample_col in df.columns
    if color_by_sample:
        sub_all = df.dropna(subset=[q4_col, sample_col])
        sample_order = sorted(sub_all[sample_col].astype(str).unique().tolist())
        sample_colors = _build_sample_color_map(sample_order)

    fig, axes = plt.subplots(1, len(cols), figsize=figsize, squeeze=False)
    stats_lookup = {row["clinical_variable"]: row for _, row in stats_summary.iterrows()}
    order_map = _resolve_clinical_group_order(pan_organ, group_order)

    if ylabel is None:
        if q4_col == "active_niche_burden":
            ylabel = "Active Profibrotic Niche Burden (APNB)"
        elif q4_col == "Q4_global_percent":
            ylabel = "Global Q4 % (cohort ARI/FRI cutoffs)"
        else:
            ylabel = "Q4 % (active fibrotic niche)"

    rng = np.random.default_rng(0)
    for j, group_col in enumerate(cols):
        ax = axes[0, j]
        sub = df.dropna(subset=[group_col, q4_col]).copy()
        if color_by_sample:
            sub = sub.dropna(subset=[sample_col])
        if sub.empty:
            ax.set_axis_off()
            continue
        present = sub[group_col].astype(str).unique().tolist()
        preset = [g for g in order_map.get(group_col, ()) if g in present]
        rest = sorted(g for g in present if g not in preset)
        order = preset + rest
        data = [sub.loc[sub[group_col].astype(str) == g, q4_col].to_numpy(float) for g in order]
        positions = np.arange(1, len(order) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=box_facecolor, alpha=0.85, edgecolor=box_edgecolor)

        for pos, g in zip(positions, order):
            gdf = sub.loc[sub[group_col].astype(str) == g]
            n_pts = len(gdf)
            if n_pts == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=n_pts)
            y_vals = gdf[q4_col].astype(float).to_numpy()
            x_vals = np.full(n_pts, pos) + jitter
            if color_by_sample:
                pt_colors = [sample_colors[str(sid)] for sid in gdf[sample_col].astype(str)]
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=point_size,
                    c=pt_colors,
                    edgecolors="0.35",
                    linewidths=0.4,
                    alpha=0.95,
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=point_size,
                    c="#b2182b",
                    alpha=0.9,
                    zorder=3,
                )

        flat = np.concatenate([d for d in data if len(d)]) if data else np.array([0.0])
        y_min, y_max = float(np.min(flat)), float(np.max(flat))
        y_span = max(y_max - y_min, 1e-9)
        pad_top = y_span * y_annotation_pad
        ax.set_ylim(y_min - y_span * 0.06, y_max + pad_top)

        stat_row = stats_lookup.get(group_col)
        if stat_row is not None and np.isfinite(stat_row.get("p_value", np.nan)):
            p = float(stat_row["p_value"])
            stars = pvalue_to_stars(p)
            ax.text(
                float(np.mean(positions)),
                y_max + pad_top * 0.55,
                f"{stat_row['test']}\np={p:.3g} ({stars})",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
                clip_on=False,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
        ax.set_title(group_col, fontsize=10)
        ax.set_ylabel(ylabel if j == 0 else "")
        ax.grid(axis="y", alpha=0.25)

    if suptitle is None:
        if q4_col == "active_niche_burden":
            suptitle = (
                "Active profibrotic niche burden by clinical group\n"
                "Sample-level mean(\u221aARI\u00d7FRI); geometric co-elevation (no threshold)"
            )
        elif q4_col == "Q4_global_percent":
            suptitle = (
                "Global fibrotic niche burden (Q4%) by clinical group\n"
                "ARI↑+FRI↑ cells above cohort pooled 95th-percentile cutoffs"
            )
        else:
            suptitle = (
                "Active profibrotic niche burden (Q4) by clinical group\n"
                "Hierarchical model: Normal \u2192 ARI\u2191 (Q2) \u2192 ARI\u2191+FRI\u2191 (Q4)"
            )
    fig.suptitle(suptitle, y=1.02, fontsize=11)

    if color_by_sample and sample_order:
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=sample_colors[sid],
                markeredgecolor="0.35",
                markeredgewidth=0.4,
                markersize=5,
                label=sid,
            )
            for sid in sample_order
        ]
        n_legend = len(sample_order)
        legend_rows = int(np.ceil(n_legend / max(legend_ncol, 1)))
        right_frac = min(0.72, max(0.58, 0.82 - 0.012 * legend_rows))
        fig.legend(
            handles=legend_handles,
            title="Sample",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 0.5,
            ncol=max(legend_ncol, 1),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0.0, 0.0, right_frac, 0.94])
    else:
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


def format_cohort_niche_report(
    q4_global_stats: pd.DataFrame,
    burden_stats: pd.DataFrame,
    cohort_meta: Mapping[str, float | int],
    *,
    clinical_columns: Sequence[str] = DEFAULT_Q4_CLINICAL_COLUMNS,
) -> list[str]:
    """
    Manuscript-ready bullets and figure captions for cohort fibrotic-niche results.

    Uses ``Q4_global_percent`` and ``active_niche_burden`` statistics (not local Q4%).
    """
    pct = float(cohort_meta.get("percentile", DEFAULT_ARI_FRI_THRESHOLD_PERCENTILE))
    ari_g = float(cohort_meta["ari_threshold_global"])
    fri_g = float(cohort_meta["fri_threshold_global"])
    n_cells = int(cohort_meta.get("n_cells_pooled", 0))
    n_samples = int(cohort_meta.get("n_samples", 0))

    def _p_line(stats_df: pd.DataFrame, col: str) -> str:
        row = stats_df.loc[stats_df["clinical_variable"] == col]
        if row.empty:
            return f"{col}: n/a"
        r = row.iloc[0]
        p = float(r["p_value"])
        test = str(r["test"])
        stars = pvalue_to_stars(p) if np.isfinite(p) else "n/a"
        medians = r.get("group_median_values", r.get("group_median_q4", []))
        labels = r.get("group_labels", [])
        med_txt = ", ".join(
            f"{lab}={float(m):.3g}" for lab, m in zip(labels, medians)
        )
        return f"{col}: {test}, p={p:.4g} ({stars}); group medians — {med_txt}"

    lines = [
        "=== Cohort fibrotic niche burden (main results) ===",
        (
            f"Pooled cohort (n={n_samples} samples, {n_cells:,} cells): "
            f"Global Q4 cutoffs = {pct:.0f}th percentile "
            f"(ARI≥{ari_g:.4f}, FRI≥{fri_g:.4f}), applied to every sample."
        ),
        "Primary metrics: (1) Global Q4 % — co-elevated ARI+FRI cells; "
        "(2) active niche burden = mean(\u221aARI\u00d7FRI). Per-sample local Q4% is not used for cohort comparison.",
        "",
        "Global Q4 %:",
    ]
    for col in clinical_columns:
        lines.append(f"  • {_p_line(q4_global_stats, col)}")
    lines.extend(["", "Active niche burden (mean \u221aARI\u00d7FRI):"])
    for col in clinical_columns:
        lines.append(f"  • {_p_line(burden_stats, col)}")
    lines.extend(
        [
            "",
            "Figure caption — Global Q4 burden by clinical group:",
            (
                "Sample-level percentage of nuclei with both structural remodeling index "
                "(idx_ARI_spatial) and profibrotic niche index (idx_FRI_spatial) above "
                f"fixed cohort-wide {pct:.0f}th-percentile thresholds "
                f"(ARI≥{ari_g:.3f}, FRI≥{fri_g:.3f}; pooled across all datasets). "
                "Disease-enriched co-elevation of ARI and FRI indicates active fibrotic niche burden."
            ),
            "",
            "Figure caption — Active profibrotic niche burden by clinical group:",
            (
                "Sample-level mean(\u221aARI\u00d7FRI) over all nuclei—the per-cell geometric "
                "mean of structural remodeling and profibrotic niche signal. "
                "Complements Global Q4 without percentile thresholding."
            ),
        ]
    )
    return lines


def build_cross_dataset_sample_dict(
    samples: Sequence[str],
    annotation: pd.DataFrame,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    use_entropy: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load cross-dataset cell tables with spatial ARI/FRI for batch hierarchical analysis.
    """
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        df, class_names, probs, coords, paths, _ = load_sample_with_spatial_indices(
            sample,
            annotation,
            prediction_source=PREDICTION_SOURCE_CROSS_DATASET,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
            radius_um=radius_um,
            use_entropy=use_entropy,
        )
        out[str(sample)] = df
    return out

########################################################
## 2026.07.05 LLY: Add the function to build the StarDist all nuclei label h5ad dictionary for Incomplete_Cases
########################################################
def build_stardist_all_sample_dict(
    samples: Sequence[str],
    annotation: pd.DataFrame,
    *,
    data_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    use_entropy: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load all StarDist nuclei label h5ad tables with spatial ARI/FRI for cohort analysis.

    Works for Complete_Cases (``stardist/{sample}/``) or Incomplete_Cases
    (``stardist_Incomplete_Cases/{sample}/``) when *stardist_root* / *cases_root* are set.
    """
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        df, _, _, _, _, _ = load_stardist_all_label_with_spatial_indices(
            sample,
            annotation,
            data_root=data_root,
            cases_root=cases_root,
            stardist_root=stardist_root,
            save_result=save_result,
            radius_um=radius_um,
            use_entropy=use_entropy,
        )
        out[str(sample)] = df
    return out


def discover_stardist_incomplete_all_label_samples(
    stardist_root: str | Path | None = None,
    *,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    data_root: str | Path | None = None,
    require_valid_h5ad: bool = True,
) -> list[str]:
    """List Incomplete_Cases samples with valid ``*_all_features_stardist_label.h5ad``."""
    if stardist_root is None:
        data_root = Path(data_root or DEFAULT_COMPLETE_CASES_ROOT.parent).expanduser().resolve()
        st_root = data_root / save_result / STARDIST_INCOMPLETE_RESULT_SUBDIR
    else:
        st_root = Path(stardist_root).expanduser().resolve()
    return discover_stardist_all_label_samples(
        stardist_root=st_root,
        require_valid_h5ad=require_valid_h5ad,
    )
##################################################################

def _plot_index_candidate_markers(
    ax,
    df: pd.DataFrame,
    *,
    candidate_col: str,
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    candidate_size: float = 2.0,
    candidate_color: str = "cyan",
    label: str | None = None,
    show_legend: bool = False,
    zorder: int = 3,
) -> None:
    """Overlay index hotspot candidate nuclei (call after ``invert_yaxis``)."""
    if candidate_col not in df.columns:
        return
    plot_df = _dedupe_cells_for_spatial_plot(df)
    mask = plot_df[candidate_col].fillna(False).astype(bool)
    cand = plot_df.loc[mask]
    if len(cand) == 0:
        return
    ax.scatter(
        cand[coord_x],
        cand[coord_y],
        s=candidate_size,
        c=candidate_color,
        alpha=0.85,
        label=label or candidate_col,
        zorder=zorder,
    )
    if show_legend:
        ax.legend(loc="upper right", markerscale=2.0)


##################################################################
# Weighted spatial TLS (cross-dataset L2-head weights; biomarker scale)
##################################################################
TLS_LEVEL2_WEIGHT_METHODS = (
    "l2_norm",
    "signed_mean",
    "bias",
    "signed_mean_plus_bias",
)
DEFAULT_TLS_LEVEL2_WEIGHT_METHOD = "l2_norm"


@dataclass(frozen=True)
class TLSModelWeights:
    """
    Fixed per-``final_CT`` weights for universal TLS scoring.

    Derived once from the cross-dataset Level-2 head; reused on any new dataset
    that provides Level-2 softmax probabilities in the same label space.
    """

    class_weights: Mapping[str, float]
    method: str
    model_checkpoint: Path
    class_names_csv: Path
    b_cell_types: tuple[str, ...] = TLS_B_CELLS
    t_cell_types: tuple[str, ...] = TLS_T_CELLS
    dc_cell_types: tuple[str, ...] = TLS_DC_CELLS

    def compartment_weights(self, compartment: str) -> dict[str, float]:
        if compartment == "B":
            types = self.b_cell_types
        elif compartment == "T":
            types = self.t_cell_types
        elif compartment == "DC":
            types = self.dc_cell_types
        else:
            raise ValueError(f"Unknown compartment {compartment!r}")
        return {ct: float(self.class_weights.get(ct, 0.0)) for ct in types}

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for compartment, types in (
            ("B", self.b_cell_types),
            ("T", self.t_cell_types),
            ("DC", self.dc_cell_types),
        ):
            for ct in types:
                rows.append(
                    {
                        "final_CT": ct,
                        "tls_compartment": compartment,
                        "level2_weight": float(self.class_weights.get(ct, 0.0)),
                        "weight_method": self.method,
                        "model_checkpoint": str(self.model_checkpoint),
                    }
                )
        return pd.DataFrame(rows)


def extract_level2_head_class_weights(
    checkpoint_path: str | Path,
    class_names: Sequence[str],
    *,
    method: str = DEFAULT_TLS_LEVEL2_WEIGHT_METHOD,
    weight_key: str = "output_layer.weight",
    bias_key: str = "output_layer.bias",
) -> dict[str, float]:
    """
    Scalarize each Level-2 class row in ``output_layer.weight`` (shape ``C × H``).

    Default ``l2_norm``: per-class **weight magnitude** ``||W_c||_2`` (non-negative).

    Other methods
    -------------
    signed_mean
        Mean of the weight row (can be negative; not used by default).
    bias
        ``output_layer.bias[c]``.
    signed_mean_plus_bias
        ``mean(W_c) + bias[c]``.
    """
    import torch

    if method not in TLS_LEVEL2_WEIGHT_METHODS:
        raise ValueError(
            f"Unknown method={method!r}; expected one of {TLS_LEVEL2_WEIGHT_METHODS}."
        )
    path = Path(checkpoint_path).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if weight_key not in state:
        raise KeyError(f"Checkpoint missing {weight_key!r}: {path}")
    weight = state[weight_key].detach().cpu().numpy()
    bias = state.get(bias_key)
    bias_np = bias.detach().cpu().numpy() if bias is not None else None
    if weight.shape[0] != len(class_names):
        raise ValueError(
            f"Checkpoint has {weight.shape[0]} L2 classes but class_names has {len(class_names)}."
        )
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        row = weight[i]
        if method == "signed_mean":
            val = float(row.mean())
        elif method == "l2_norm":
            val = float(np.linalg.norm(row))
        elif method == "bias":
            if bias_np is None:
                raise KeyError(f"Checkpoint missing {bias_key!r} for method 'bias'.")
            val = float(bias_np[i])
        else:  # signed_mean_plus_bias
            if bias_np is None:
                raise KeyError(
                    f"Checkpoint missing {bias_key!r} for method 'signed_mean_plus_bias'."
                )
            val = float(row.mean() + bias_np[i])
        out[str(name)] = val
    return out


def build_tls_model_weights(
    *,
    checkpoint_path: str | Path | None = None,
    class_names_csv: str | Path | None = None,
    method: str = DEFAULT_TLS_LEVEL2_WEIGHT_METHOD,
) -> TLSModelWeights:
    """Build fixed TLS compartment weights from the cross-dataset checkpoint."""
    ckpt = Path(checkpoint_path or DEFAULT_CROSS_DATASET_MODEL_CHECKPOINT).expanduser().resolve()
    names_csv = Path(class_names_csv or DEFAULT_CROSS_DATASET_CLASS_NAMES_CSV).expanduser().resolve()
    class_names = load_auroc_class_names(names_csv)
    weights = extract_level2_head_class_weights(ckpt, class_names, method=method)
    return TLSModelWeights(
        class_weights=weights,
        method=method,
        model_checkpoint=ckpt,
        class_names_csv=names_csv,
    )


def save_tls_model_weights(
    weights: TLSModelWeights,
    out_csv: str | Path,
) -> Path:
    """Persist compartment weight table for reuse on new datasets."""
    path = Path(out_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_frame().to_csv(path, index=False)
    return path


def load_tls_model_weights(
    weights_csv: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    class_names_csv: str | Path | None = None,
) -> TLSModelWeights:
    """Load a previously exported TLS weight table."""
    path = Path(weights_csv).expanduser().resolve()
    df = pd.read_csv(path)
    required = {"final_CT", "tls_compartment", "level2_weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"TLS weights CSV must contain columns {sorted(required)}.")
    class_weights = {
        str(row.final_CT): float(row.level2_weight)
        for row in df.itertuples(index=False)
    }
    method = str(df["weight_method"].iloc[0]) if "weight_method" in df.columns else "loaded_csv"
    ckpt = Path(
        checkpoint_path
        or (df["model_checkpoint"].iloc[0] if "model_checkpoint" in df.columns else DEFAULT_CROSS_DATASET_MODEL_CHECKPOINT)
    ).expanduser().resolve()
    names_csv = Path(class_names_csv or DEFAULT_CROSS_DATASET_CLASS_NAMES_CSV).expanduser().resolve()
    return TLSModelWeights(
        class_weights=class_weights,
        method=method,
        model_checkpoint=ckpt,
        class_names_csv=names_csv,
    )


def load_or_build_tls_model_weights(
    *,
    weights_csv: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    class_names_csv: str | Path | None = None,
    method: str = DEFAULT_TLS_LEVEL2_WEIGHT_METHOD,
    save_if_built: bool = True,
) -> TLSModelWeights:
    """
    Load frozen TLS weights from CSV, or extract from checkpoint and optionally save.

    Rebuilds from checkpoint when the cached CSV was built with a different ``method``.
    """
    csv_path = Path(weights_csv or DEFAULT_TLS_WEIGHTS_CSV).expanduser().resolve()
    if csv_path.is_file():
        cached = pd.read_csv(csv_path)
        cached_method = (
            str(cached["weight_method"].iloc[0])
            if "weight_method" in cached.columns
            else None
        )
        if cached_method == method:
            return load_tls_model_weights(
                csv_path,
                checkpoint_path=checkpoint_path,
                class_names_csv=class_names_csv,
            )
    weights = build_tls_model_weights(
        checkpoint_path=checkpoint_path,
        class_names_csv=class_names_csv,
        method=method,
    )
    if save_if_built:
        save_tls_model_weights(weights, csv_path)
    return weights


def weighted_sum_prob_columns(
    probs: np.ndarray,
    col_indices: Sequence[int],
    col_weights: Sequence[float],
) -> np.ndarray:
    """Per-cell weighted sum ``sum_k w_k * prob_k`` (weights may be signed)."""
    if not col_indices:
        return np.zeros(probs.shape[0], dtype=np.float64)
    w = np.asarray(col_weights, dtype=np.float64).reshape(1, -1)
    return (probs[:, col_indices] * w).sum(axis=1)


def _compartment_cols_and_weights(
    class_names: Sequence[str],
    cell_types: Sequence[str],
    class_weights: Mapping[str, float],
    *,
    default_weight: float = 0.0,
) -> tuple[list[int], np.ndarray]:
    idx = class_name_to_prob_index(class_names)
    cols: list[int] = []
    wts: list[float] = []
    for ct in cell_types:
        ct = str(ct)
        if ct not in idx:
            continue
        cols.append(idx[ct])
        wts.append(float(class_weights.get(ct, default_weight)))
    return cols, np.asarray(wts, dtype=np.float64)


def weighted_compartment_prob_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    cell_types: Sequence[str],
    class_weights: Mapping[str, float],
) -> np.ndarray:
    """Per-cell weighted soft mass for one TLS compartment."""
    cols, wts = _compartment_cols_and_weights(class_names, cell_types, class_weights)
    return weighted_sum_prob_columns(probs, cols, wts)


def signed_geometric_mean3(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Signed cube-root geometric mean of three compartment scores.

    Magnitude: ``(|a| |b| |c|)^(1/3)``; sign flips when an odd number of inputs are negative.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    mag = np.power(np.abs(a) * np.abs(b) * np.abs(c) + eps, 1.0 / 3.0)
    n_neg = (a < 0).astype(np.int8) + (b < 0).astype(np.int8) + (c < 0).astype(np.int8)
    sign = np.where(n_neg % 2 == 1, -1.0, 1.0)
    return sign * mag


def logistic01_from_raw(
    raw: np.ndarray,
    *,
    scale: float | None = None,
    percentile: float = 75.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """
    Map signed raw scores to ``(0, 1)`` via logistic / inverse-logit link.

    Uses ``p = sigmoid(raw / scale)``. When ``scale`` is ``None``, set from the
    ``percentile`` of ``|raw|`` on finite values (robust across datasets).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if scale is None:
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            scale = 1.0
        else:
            scale = float(np.percentile(np.abs(finite), percentile))
        scale = max(scale, eps)
    prob = 1.0 / (1.0 + np.exp(-raw / scale))
    return prob, float(scale)


def compute_weighted_spatial_tls_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    tls_weights: TLSModelWeights,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    radius_px: float | None = None,
    logistic_scale: float | None = None,
    logistic_percentile: float = 75.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Model-weighted spatial TLS (cross-dataset biomarker version).

    Per cell ``i`` within radius ``R``:

      B_w_local = mean_j Σ_{c∈B} w_c · p_j(c)   (same for T, DC; ``w_c`` fixed)

      raw_i = signed_geomean(B_w_local, T_w_local, DC_w_local)

      score_i = sigmoid(raw_i / scale)  ∈ (0, 1)

    Returns
    -------
    score_prob, raw, b_local_w, t_local_w, dc_local_w, scale_used
    """
    if radius_px is None:
        radius_px = um_to_he_pixel(radius_um, um_per_pixel)
    cw = tls_weights.class_weights
    b_prob = weighted_compartment_prob_per_cell(probs, class_names, TLS_B_CELLS, cw)
    t_prob = weighted_compartment_prob_per_cell(probs, class_names, TLS_T_CELLS, cw)
    dc_prob = weighted_compartment_prob_per_cell(probs, class_names, TLS_DC_CELLS, cw)

    b_local = local_neighborhood_mean(b_prob, coords, radius_px)
    t_local = local_neighborhood_mean(t_prob, coords, radius_px)
    dc_local = local_neighborhood_mean(dc_prob, coords, radius_px)

    raw = signed_geometric_mean3(b_local, t_local, dc_local)
    score_prob, scale_used = logistic01_from_raw(
        raw, scale=logistic_scale, percentile=logistic_percentile
    )
    return score_prob, raw, b_local, t_local, dc_local, scale_used


def add_weighted_spatial_tls_to_auroc_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    tls_weights: TLSModelWeights | None = None,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    logistic_scale: float | None = None,
    logistic_percentile: float = 75.0,
    prefix: str = "idx_",
    weights_csv: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add model-weighted spatial TLS columns (biomarker scale in ``(0, 1)``).

    Original ``idx_TLS_spatial`` is unchanged; new columns:

    - ``idx_TLS_spatial_weighted_raw`` — signed geometric mean
    - ``idx_TLS_spatial_weighted`` — logistic score in ``(0, 1)``
    - ``idx_TLS_B_local_weighted``, ``idx_TLS_T_local_weighted``, ``idx_TLS_DC_local_weighted``

    Hotspot flag ``tls_candidate_weighted`` uses the weighted logistic score.
    """
    if tls_weights is None:
        tls_weights = load_or_build_tls_model_weights(
            weights_csv=weights_csv,
            checkpoint_path=checkpoint_path,
        )
    radius_um, radius_px, um_per_pixel = resolve_spatial_radius(
        radius_um, um_per_pixel=um_per_pixel
    )
    out = df.copy()
    score, raw, b_loc, t_loc, dc_loc, scale_used = compute_weighted_spatial_tls_per_cell(
        probs,
        class_names,
        coords,
        tls_weights,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
        logistic_scale=logistic_scale,
        logistic_percentile=logistic_percentile,
    )
    out[f"{prefix}TLS_spatial_weighted"] = score
    out[f"{prefix}TLS_spatial_weighted_raw"] = raw
    out[f"{prefix}TLS_B_local_weighted"] = b_loc
    out[f"{prefix}TLS_T_local_weighted"] = t_loc
    out[f"{prefix}TLS_DC_local_weighted"] = dc_loc
    out["tls_weighted_logistic_scale"] = scale_used
    out["tls_weighted_method"] = tls_weights.method
    out["tls_weighted_model_checkpoint"] = str(tls_weights.model_checkpoint)
    out["tls_spatial_radius_um"] = radius_um
    out["tls_spatial_radius_px"] = radius_px
    out["tls_um_per_he_pixel"] = um_per_pixel

    is_cand, thresh = detect_tls_hotspot_cells(
        score, coords, percentile=hotspot_percentile
    )
    out["tls_candidate_weighted"] = is_cand
    out["tls_spatial_weighted_threshold"] = thresh
    return out


##################################################################
# 2026.06.27try the TLS hotspot detection
#            该细胞所在位置的 Spatial TLS 分数（idx_TLS_spatial）位于全片 top 5%
##################################################################
def detect_tls_hotspot_cells(
    spatial_tls: np.ndarray,
    coords: np.ndarray,
    *,
    percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    min_score: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Flag TLS candidate cells with high ``TLS_spatial``.

    Returns ``(is_candidate bool array, threshold used)``.
    """
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
    """Basic summary of spatial TLS hotspot cells."""
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


def add_spatial_tls_to_auroc_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    *,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float = DEFAULT_UM_PER_HE_PIXEL,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """
    Add spatial TLS columns and hotspot flag to an AUROC dataframe.

    When ``use_entropy=True``, also stores compartment Shannon entropy
    (``idx_TLS_compartment_entropy``) and the combined score
    ``idx_TLS_spatial_combined = idx_TLS_spatial + entropy_weight * H / log(3)``.
    Hotspot detection always uses ``idx_TLS_spatial`` (95th percentile).
    """
    radius_um, radius_px, um_per_pixel = resolve_spatial_radius(
        radius_um, um_per_pixel=um_per_pixel
    )
    out = df.copy()
    tls_s, b_loc, t_loc, dc_loc = compute_spatial_tls_per_cell(
        probs,
        class_names,
        coords,
        radius_um=radius_um,
        um_per_pixel=um_per_pixel,
    )
    out[f"{prefix}TLS_spatial"] = tls_s
    out[f"{prefix}TLS_B_local"] = b_loc
    out[f"{prefix}TLS_T_local"] = t_loc
    out[f"{prefix}TLS_DC_local"] = dc_loc
    out["tls_spatial_radius_um"] = radius_um
    out["tls_spatial_radius_px"] = radius_px
    out["tls_um_per_he_pixel"] = um_per_pixel

    if use_entropy:
        entropy = compartment_entropy(b_loc, t_loc, dc_loc)
        tls_combined = combine_spatial_tls_with_entropy(
            tls_s, entropy, entropy_weight=entropy_weight
        )
        out[f"{prefix}TLS_compartment_entropy"] = entropy
        out[f"{prefix}TLS_spatial_combined"] = tls_combined
        out["tls_entropy_weight"] = float(entropy_weight)

    is_cand, thresh = detect_tls_hotspot_cells(
        tls_s, coords, percentile=hotspot_percentile
    )
    out["tls_candidate"] = is_cand
    out["tls_spatial_threshold"] = thresh
    return out


def _dedupe_cells_for_spatial_plot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``cell_id`` for spatial scatter / candidate overlays."""
    if "cell_id" not in df.columns:
        return df
    return df.drop_duplicates(subset=["cell_id"], keep="first").copy()


def _select_tls_candidate_cells(
    df: pd.DataFrame,
    candidate_col: str = "tls_candidate",
) -> pd.DataFrame:
    if candidate_col not in df.columns:
        return df.iloc[0:0]
    mask = df[candidate_col].fillna(False).astype(bool)
    return df.loc[mask]


def _plot_tls_candidate_markers(
    ax,
    df: pd.DataFrame,
    *,
    candidate_col: str = "tls_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    candidate_size: float = 2.0,
    candidate_color: str = "cyan",
    show_legend: bool = False,
    legend_markerscale: float = 2.0,
    zorder: int = 3,
) -> None:
    """
    Overlay TLS candidate nuclei.

    Call **after** ``ax.invert_yaxis()`` so markers align with the spatial
    heatmap in ``plot_tls_abundance_vs_spatial`` and pathologist overlays.
    """
    cand = _select_tls_candidate_cells(df, candidate_col=candidate_col)
    if len(cand) == 0:
        return
    ax.scatter(
        cand[coord_x],
        cand[coord_y],
        s=candidate_size,
        c=candidate_color,
        alpha=0.85,
        label="TLS candidate",
        zorder=zorder,
    )
    if show_legend:
        ax.legend(loc="upper right", markerscale=legend_markerscale)


def plot_tls_abundance_vs_spatial(
    df: pd.DataFrame,
    sample: str,
    *,
    radius_um: float | None = None,
    abundance_col: str = "idx_TLS",
    spatial_col: str = "idx_TLS_spatial",
    combined_col: str = "idx_TLS_spatial_combined",
    use_entropy: bool = False,
    candidate_col: str = "tls_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    vmin_percentile: float = 1.0,
    point_size: float = 0.3,
    candidate_size: float = 2.0,
    candidate_color: str = "cyan",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    save_path: str | Path | None = None,
    axes: Sequence | None = None,
):
    """
    Spatial maps: abundance TLS vs spatial TLS (and optional entropy-combined TLS).

    With ``use_entropy=True``, layout is 1×3 — abundance | spatial TLS |
    ``idx_TLS_spatial + w·H/log(3)``. Color limits use per-map percentiles
    (default vmax=p99) on ``spatial_HE`` coordinates.

    If ``save_path`` is set, saves the figure after ``tight_layout`` (only when
    this function creates the figure, i.e. ``axes is None``).
    """
    import matplotlib.pyplot as plt

    if abundance_col not in df.columns:
        raise KeyError(f"Missing column {abundance_col!r}")
    if spatial_col not in df.columns:
        raise KeyError(f"Missing column {spatial_col!r}")
    if use_entropy and combined_col not in df.columns:
        raise KeyError(
            f"Missing column {combined_col!r}; run add_spatial_tls_to_auroc_df(use_entropy=True)."
        )

    if radius_um is None and "tls_spatial_radius_um" in df.columns:
        radius_um = float(df["tls_spatial_radius_um"].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_TLS_SPATIAL_RADIUS_UM

    ncols = 3 if use_entropy else 2
    if figsize is None:
        figsize = (18.0, 5.0) if use_entropy else (12.0, 5.0)

    created_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    else:
        if len(axes) != ncols:
            raise ValueError(f"Expected {ncols} axes, got {len(axes)}.")
        fig = axes[0].figure

    abundance = df[abundance_col].to_numpy(dtype=np.float64)
    spatial = df[spatial_col].to_numpy(dtype=np.float64)
    x = df[coord_x].to_numpy(dtype=np.float64)
    y = df[coord_y].to_numpy(dtype=np.float64)

    panels: list[tuple[np.ndarray, str, str]] = [
        (abundance, f"{sample}: Abundance TLS ({abundance_col})", "viridis"),
        (spatial, f"{sample}: Spatial TLS (R={radius_um:g} μm)", "magma"),
    ]
    if use_entropy:
        combined = df[combined_col].to_numpy(dtype=np.float64)
        entropy_weight = float(df["tls_entropy_weight"].iloc[0]) if "tls_entropy_weight" in df.columns else 1.0
        panels.append(
            (
                combined,
                f"{sample}: Spatial TLS + Entropy (w={entropy_weight:g})",
                "plasma",
            )
        )

    for ax, (values, title, cmap) in zip(axes, panels):
        vmin, vmax = percentile_vlim(
            values, vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile
        )
        sc = ax.scatter(x, y, c=values, s=point_size, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    # ``sharey=True``: invert once; a second call on later axes cancels the flip.
    axes[0].invert_yaxis()

    candidate_axes = [axes[1]] if len(axes) > 1 else []
    if use_entropy and len(axes) > 2:
        candidate_axes.append(axes[2])
    for i, candidate_ax in enumerate(candidate_axes):
        _plot_tls_candidate_markers(
            candidate_ax,
            df,
            candidate_col=candidate_col,
            coord_x=coord_x,
            coord_y=coord_y,
            candidate_size=candidate_size,
            candidate_color=candidate_color,
            show_legend=(i == 0),
            legend_markerscale=3,
        )

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes


def plot_spatial_tls_vs_weighted(
    df: pd.DataFrame,
    sample: str,
    *,
    radius_um: float | None = None,
    spatial_col: str = "idx_TLS_spatial",
    weighted_col: str = "idx_TLS_spatial_weighted",
    candidate_col: str = "tls_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    vmin_percentile: float = 1.0,
    cmap: str = "magma",
    point_size: float = 0.3,
    candidate_size: float = 2.0,
    candidate_color: str = "cyan",
    figsize: tuple[float, float] = (12.0, 5.0),
    axes: Sequence | None = None,
):
    """
    Side-by-side classic vs weighted spatial TLS (same layout as ``plot_tls_abundance_vs_spatial``).

    Both panels use the same ``magma`` colormap; each panel gets its own percentile
    color limits and colorbar. Cyan TLS candidates overlay both maps.
    """
    import matplotlib.pyplot as plt

    for col in (spatial_col, weighted_col):
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}")

    if radius_um is None and "tls_spatial_radius_um" in df.columns:
        radius_um = float(df["tls_spatial_radius_um"].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_TLS_SPATIAL_RADIUS_UM

    created_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    else:
        if len(axes) != 2:
            raise ValueError(f"Expected 2 axes, got {len(axes)}.")
        fig = axes[0].figure

    x = df[coord_x].to_numpy(dtype=np.float64)
    y = df[coord_y].to_numpy(dtype=np.float64)

    panels: list[tuple[np.ndarray, str]] = [
        (
            df[spatial_col].to_numpy(dtype=np.float64),
            f"{sample}: Spatial TLS (classic) (R={radius_um:g} μm)",
        ),
        (
            df[weighted_col].to_numpy(dtype=np.float64),
            f"{sample}: Spatial TLS weighted (R={radius_um:g} μm)",
        ),
    ]

    for ax, (values, title) in zip(axes, panels):
        vmin, vmax = percentile_vlim(
            values, vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile
        )
        sc = ax.scatter(
            x, y, c=values, s=point_size, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)

    axes[0].invert_yaxis()

    for i, ax in enumerate(axes):
        _plot_tls_candidate_markers(
            ax,
            df,
            candidate_col=candidate_col,
            coord_x=coord_x,
            coord_y=coord_y,
            candidate_size=candidate_size,
            candidate_color=candidate_color,
            show_legend=(i == 0),
            legend_markerscale=3,
        )

    if created_fig:
        fig.tight_layout()
    return fig, axes


PATHOLOGIST_TLS_LABEL = "Mixed Inflammation"
PATHOLOGIST_FRI_LABELS: tuple[str, ...] = (
    "Fibrosis",
    "Severe Fibrosis",
    "Remodeled Epithelium",
    "Fibroblastic Focus",
)
PATHOLOGIST_ARI_LABELS: tuple[str, ...] = (
    "Remodeled Epithelium",
    "Fibrosis",
    "Severe Fibrosis",
    "Fibroblastic Focus",
)


def pathologist_annotation_csv(
    sample: str,
    cases_root: str | Path | None = None,
) -> Path:
    """Return ``{sample}_cells_matched_by_stardist.csv`` under Complete_Cases."""
    root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    return root / sample / f"{sample}_cells_matched_by_stardist.csv"


def load_pathologist_annotations(
    sample: str,
    cases_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load per-cell pathologist ROI labels matched to StarDist nuclei."""
    path = pathologist_annotation_csv(sample, cases_root)
    if not path.is_file():
        raise FileNotFoundError(f"Pathologist annotation CSV not found: {path}")
    usecols = [
        "cell_id",
        "Annotation_Type",
        "annotation_type",
        "annotation_type_instance",
        "X_pix_HE",
        "Y_pix_HE",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df["cell_id"] = df["cell_id"].astype(str)
    # A few nuclei appear twice when assigned to overlapping ROI labels.
    return df.drop_duplicates(subset=["cell_id"], keep="first")


def merge_pathologist_annotations(
    df: pd.DataFrame,
    sample: str,
    cases_root: str | Path | None = None,
) -> pd.DataFrame:
    """Join pathologist labels onto an AUROC / spatial-TLS dataframe by ``cell_id``."""
    anno = load_pathologist_annotations(sample, cases_root)
    out = df.copy()
    out["cell_id"] = out["cell_id"].astype(str)
    merged = out.merge(anno, on="cell_id", how="left", suffixes=("", "_path"))
    n_missing = int(merged["Annotation_Type"].isna().sum())
    if n_missing:
        # A few StarDist nuclei may lack pathologist ROI assignment.
        merged["Annotation_Type"] = merged["Annotation_Type"].fillna("Unassigned")
    merged["is_mixed_inflammation"] = (
        merged["Annotation_Type"].astype(str) == PATHOLOGIST_TLS_LABEL
    )
    merged["is_pathologist_fri"] = merged["Annotation_Type"].astype(str).isin(
        PATHOLOGIST_FRI_LABELS
    )
    merged["is_pathologist_ari"] = merged["Annotation_Type"].astype(str).isin(
        PATHOLOGIST_ARI_LABELS
    )
    return merged


def pathologist_region_hulls(
    df: pd.DataFrame,
    *,
    pathologist_label: str = PATHOLOGIST_TLS_LABEL,
    instance_col: str = "annotation_type_instance",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
) -> dict[str, np.ndarray]:
    """
    Convex-hull polygon boundaries for each pathologist ROI instance.

    Returns ``{instance_id: (M, 2) closed polygon in spatial_HE pixels}``.
    """
    from scipy.spatial import ConvexHull

    if instance_col not in df.columns:
        raise KeyError(f"Missing column {instance_col!r}")
    sub = df.loc[df["Annotation_Type"].astype(str) == pathologist_label]
    hulls: dict[str, np.ndarray] = {}
    for inst, grp in sub.groupby(instance_col, sort=True):
        pts = grp[[coord_x, coord_y]].to_numpy(dtype=np.float64)
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        verts = pts[hull.vertices]
        closed = np.vstack([verts, verts[:1]])
        hulls[str(inst)] = closed
    return hulls


def validate_tls_against_pathologist(
    df: pd.DataFrame,
    *,
    pathologist_label: str = PATHOLOGIST_TLS_LABEL,
    candidate_col: str = "tls_candidate",
    spatial_col: str = "idx_TLS_spatial",
    abundance_col: str = "idx_TLS",
    instance_col: str = "annotation_type_instance",
) -> dict:
    """
    Compare TLS candidates with pathologist ``Mixed Inflammation`` labels.

    Returns a dict with confusion counts, rates, score summaries, and
    per-instance tables suitable for plotting.
    """
    if "is_mixed_inflammation" not in df.columns:
        raise KeyError("Run merge_pathologist_annotations() first.")
    if candidate_col not in df.columns:
        raise KeyError(f"Missing column {candidate_col!r}")

    mi = df["is_mixed_inflammation"].to_numpy(dtype=bool)
    cand = df[candidate_col].fillna(False).to_numpy(dtype=bool)
    tp = int((cand & mi).sum())
    fp = int((cand & ~mi).sum())
    fn = int((~cand & mi).sum())
    tn = int((~cand & ~mi).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    mi_rate = float(mi.mean())
    enrichment = (tp / max(cand.sum(), 1)) / max(mi_rate, 1e-9)

    score_rows = []
    for col, label in ((spatial_col, "Spatial TLS"), (abundance_col, "Abundance TLS")):
        if col not in df.columns:
            continue
        mi_mean = float(df.loc[mi, col].mean())
        non_mean = float(df.loc[~mi, col].mean())
        score_rows.append(
            {
                "score": label,
                "column": col,
                "mixed_inflammation_mean": mi_mean,
                "non_mixed_inflammation_mean": non_mean,
                "fold_change": mi_mean / max(non_mean, 1e-9),
            }
        )

    inst = (
        df.loc[mi]
        .groupby(instance_col, sort=True)
        .agg(
            n_cells=("cell_id", "count"),
            n_candidate=(candidate_col, "sum"),
            mean_spatial=(spatial_col, "mean"),
            max_spatial=(spatial_col, "max"),
            mean_abundance=(abundance_col, "mean"),
            centroid_x=("coord_x", "mean"),
            centroid_y=("coord_y", "mean"),
        )
        .reset_index()
    )
    inst["fraction_candidate"] = inst["n_candidate"] / inst["n_cells"].clip(lower=1)

    return {
        "n_cells": int(len(df)),
        "n_mixed_inflammation": int(mi.sum()),
        "n_candidate": int(cand.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "enrichment": enrichment,
        "mixed_inflammation_rate": mi_rate,
        "score_summary": pd.DataFrame(score_rows),
        "instance_summary": inst,
    }


def validate_spatial_index_against_pathologist(
    df: pd.DataFrame,
    *,
    index_name: str,
    pathologist_labels: Sequence[str],
    pathologist_flag_col: str,
    candidate_col: str,
    spatial_col: str,
    abundance_col: str,
    instance_col: str = "annotation_type_instance",
) -> dict:
    """
    Compare spatial index candidates with pathologist ROI labels.

    Generic counterpart to ``validate_tls_against_pathologist`` for FRI / ARI.
    """
    if pathologist_flag_col not in df.columns:
        raise KeyError("Run merge_pathologist_annotations() first.")
    if candidate_col not in df.columns:
        raise KeyError(f"Missing column {candidate_col!r}")

    pos = df[pathologist_flag_col].to_numpy(dtype=bool)
    cand = df[candidate_col].fillna(False).to_numpy(dtype=bool)
    tp = int((cand & pos).sum())
    fp = int((cand & ~pos).sum())
    fn = int((~cand & pos).sum())
    tn = int((~cand & ~pos).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pos_rate = float(pos.mean())
    enrichment = (tp / max(cand.sum(), 1)) / max(pos_rate, 1e-9)

    score_rows = []
    for col, label in (
        (spatial_col, f"Spatial {index_name}"),
        (abundance_col, f"Abundance {index_name}"),
    ):
        if col not in df.columns:
            continue
        pos_mean = float(df.loc[pos, col].mean())
        non_mean = float(df.loc[~pos, col].mean())
        score_rows.append(
            {
                "score": label,
                "column": col,
                "pathologist_positive_mean": pos_mean,
                "pathologist_negative_mean": non_mean,
                "fold_change": pos_mean / max(non_mean, 1e-9),
            }
        )

    label_set = {str(l) for l in pathologist_labels}
    inst = (
        df.loc[df["Annotation_Type"].astype(str).isin(label_set)]
        .groupby(instance_col, sort=True)
        .agg(
            n_cells=("cell_id", "count"),
            n_candidate=(candidate_col, "sum"),
            mean_spatial=(spatial_col, "mean"),
            max_spatial=(spatial_col, "max"),
            mean_abundance=(abundance_col, "mean"),
            centroid_x=("coord_x", "mean"),
            centroid_y=("coord_y", "mean"),
            annotation_type=("Annotation_Type", "first"),
        )
        .reset_index()
    )
    inst["fraction_candidate"] = inst["n_candidate"] / inst["n_cells"].clip(lower=1)

    return {
        "index_name": index_name,
        "pathologist_labels": tuple(pathologist_labels),
        "n_cells": int(len(df)),
        "n_pathologist_positive": int(pos.sum()),
        "n_candidate": int(cand.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "enrichment": enrichment,
        "pathologist_positive_rate": pos_rate,
        "score_summary": pd.DataFrame(score_rows),
        "instance_summary": inst,
    }


def validate_fri_against_pathologist(
    df: pd.DataFrame,
    *,
    pathologist_labels: Sequence[str] = PATHOLOGIST_FRI_LABELS,
    candidate_col: str = "fri_candidate",
    spatial_col: str = "idx_FRI_spatial",
    abundance_col: str = "idx_FRI_ratio",
) -> dict:
    """Compare FRI candidates with pathologist fibrosis-associated ROIs."""
    return validate_spatial_index_against_pathologist(
        df,
        index_name="FRI",
        pathologist_labels=pathologist_labels,
        pathologist_flag_col="is_pathologist_fri",
        candidate_col=candidate_col,
        spatial_col=spatial_col,
        abundance_col=abundance_col,
    )


def validate_ari_against_pathologist(
    df: pd.DataFrame,
    *,
    pathologist_labels: Sequence[str] = PATHOLOGIST_ARI_LABELS,
    candidate_col: str = "ari_candidate",
    spatial_col: str = "idx_ARI_spatial",
    abundance_col: str = "idx_ARI_ratio",
) -> dict:
    """Compare ARI candidates with pathologist remodeling-associated ROIs."""
    return validate_spatial_index_against_pathologist(
        df,
        index_name="ARI",
        pathologist_labels=pathologist_labels,
        pathologist_flag_col="is_pathologist_ari",
        candidate_col=candidate_col,
        spatial_col=spatial_col,
        abundance_col=abundance_col,
    )


def plot_index_pathologist_overlay(
    df: pd.DataFrame,
    sample: str,
    *,
    index_name: str,
    pathologist_labels: Sequence[str],
    spatial_df: pd.DataFrame | None = None,
    spatial_col: str,
    candidate_col: str,
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    radius_um: float | None = None,
    radius_col: str = "fri_ari_spatial_radius_um",
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    point_size: float = 0.25,
    candidate_size: float = 2.5,
    candidate_color: str = "cyan",
    boundary_color: str = "magenta",
    boundary_linewidth: float = 1.5,
    figsize: tuple[float, float] = (8.0, 8.0),
):
    """Overlay pathologist ROI hull boundaries and spatial index candidates."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    plot_df = _dedupe_cells_for_spatial_plot(spatial_df if spatial_df is not None else df)
    if spatial_col not in plot_df.columns:
        raise KeyError(f"Missing column {spatial_col!r}")
    if radius_um is None and radius_col in plot_df.columns:
        radius_um = float(plot_df[radius_col].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_FRI_SPATIAL_RADIUS_UM

    label_str = " / ".join(pathologist_labels)
    fig, ax = plt.subplots(figsize=figsize)
    x = plot_df[coord_x].to_numpy(dtype=np.float64)
    y = plot_df[coord_y].to_numpy(dtype=np.float64)
    values = plot_df[spatial_col].to_numpy(dtype=np.float64)
    vmin, vmax = percentile_vlim(values, vmax_percentile=vmax_percentile)
    sc = ax.scatter(x, y, c=values, s=point_size, cmap="magma", alpha=0.55, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.046, label=spatial_col)

    label_set = {str(l) for l in pathologist_labels}
    patches = []
    if "Annotation_Type" in df.columns and "annotation_type_instance" in df.columns:
        for inst, grp in df.loc[df["Annotation_Type"].astype(str).isin(label_set)].groupby(
            "annotation_type_instance", sort=True
        ):
            pts = grp[[coord_x, coord_y]].to_numpy(dtype=np.float64)
            if len(pts) < 3:
                continue
            from scipy.spatial import ConvexHull

            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            verts = pts[hull.vertices]
            closed = np.vstack([verts, verts[:1]])
            patches.append(
                Polygon(
                    closed,
                    closed=True,
                    fill=False,
                    edgecolor=boundary_color,
                    linewidth=boundary_linewidth,
                )
            )
    if patches:
        ax.add_collection(PatchCollection(patches, match_original=True))
    ax.plot([], [], color=boundary_color, linewidth=boundary_linewidth, label=label_str)

    ax.set_title(f"{sample}: Spatial {index_name} + pathologist ROIs")
    ax.set_xlabel("spatial_HE x (px)")
    ax.set_ylabel("spatial_HE y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    _plot_index_candidate_markers(
        ax,
        plot_df,
        candidate_col=candidate_col,
        coord_x=coord_x,
        coord_y=coord_y,
        candidate_size=candidate_size,
        candidate_color=candidate_color,
        label=f"{index_name} candidate",
    )
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    return fig, ax


def plot_fri_pathologist_overlay(
    df: pd.DataFrame,
    sample: str,
    *,
    spatial_df: pd.DataFrame | None = None,
    pathologist_labels: Sequence[str] = PATHOLOGIST_FRI_LABELS,
    **kwargs,
):
    return plot_index_pathologist_overlay(
        df,
        sample,
        index_name="FRI",
        pathologist_labels=pathologist_labels,
        spatial_df=spatial_df,
        spatial_col=kwargs.pop("spatial_col", "idx_FRI_spatial"),
        candidate_col=kwargs.pop("candidate_col", "fri_candidate"),
        **kwargs,
    )


def plot_ari_pathologist_overlay(
    df: pd.DataFrame,
    sample: str,
    *,
    spatial_df: pd.DataFrame | None = None,
    pathologist_labels: Sequence[str] = PATHOLOGIST_ARI_LABELS,
    **kwargs,
):
    return plot_index_pathologist_overlay(
        df,
        sample,
        index_name="ARI",
        pathologist_labels=pathologist_labels,
        spatial_df=spatial_df,
        spatial_col=kwargs.pop("spatial_col", "idx_ARI_spatial"),
        candidate_col=kwargs.pop("candidate_col", "ari_candidate"),
        **kwargs,
    )


def plot_index_pathologist_validation(
    df: pd.DataFrame,
    sample: str,
    validation: dict,
    *,
    index_name: str,
    pathologist_labels: Sequence[str],
    spatial_df: pd.DataFrame | None = None,
    spatial_col: str,
    abundance_col: str,
    candidate_col: str,
    radius_um: float | None = None,
    radius_col: str = "fri_ari_spatial_radius_um",
    figsize: tuple[float, float] = (14.0, 10.0),
):
    """
    Four-panel validation figure for a spatial index vs pathologist ROIs.

    Row 0: A) spatial overlay | B) confusion metrics
    Row 1: C) abundance vs spatial score violins | D) per-instance candidate fraction
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plot_df = _dedupe_cells_for_spatial_plot(spatial_df if spatial_df is not None else df)
    if radius_um is None and radius_col in plot_df.columns:
        radius_um = float(plot_df[radius_col].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_FRI_SPATIAL_RADIUS_UM

    flag_col = (
        "is_pathologist_fri" if index_name.upper() == "FRI" else "is_pathologist_ari"
    )
    pos_mask = df[flag_col].to_numpy(dtype=bool)
    label_str = " / ".join(pathologist_labels)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.0, 0.9])

    ax_overlay = fig.add_subplot(gs[0, 0])
    x = plot_df["coord_x"].to_numpy(dtype=np.float64)
    y = plot_df["coord_y"].to_numpy(dtype=np.float64)
    values = plot_df[spatial_col].to_numpy(dtype=np.float64)
    vmin, vmax = percentile_vlim(values)
    ax_overlay.scatter(
        x, y, c=values, s=0.2, cmap="magma", alpha=0.55, vmin=vmin, vmax=vmax
    )
    label_set = {str(l) for l in pathologist_labels}
    if "Annotation_Type" in df.columns and "annotation_type_instance" in df.columns:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        from scipy.spatial import ConvexHull

        patches = []
        for _, grp in df.loc[df["Annotation_Type"].astype(str).isin(label_set)].groupby(
            "annotation_type_instance", sort=True
        ):
            pts = grp[["coord_x", "coord_y"]].to_numpy(dtype=np.float64)
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            verts = pts[hull.vertices]
            closed = np.vstack([verts, verts[:1]])
            patches.append(
                Polygon(
                    closed,
                    closed=True,
                    fill=False,
                    edgecolor="magenta",
                    linewidth=1.5,
                )
            )
        if patches:
            ax_overlay.add_collection(PatchCollection(patches, match_original=True))
    ax_overlay.set_title(f"A  Spatial {index_name} + {label_str}")
    ax_overlay.set_aspect("equal")
    ax_overlay.invert_yaxis()
    _plot_index_candidate_markers(
        ax_overlay, plot_df, candidate_col=candidate_col, candidate_size=2.0
    )

    ax_metrics = fig.add_subplot(gs[0, 1])
    metrics = [
        ("Precision", validation["precision"]),
        ("Recall", validation["recall"]),
        ("Enrichment", validation["enrichment"]),
    ]
    ax_metrics.barh(
        [m[0] for m in metrics],
        [m[1] for m in metrics],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.85,
    )
    ax_metrics.set_xlim(0, max(m[1] for m in metrics) * 1.2 + 0.05)
    ax_metrics.set_title(f"B  {index_name} candidate vs pathologist ROIs")
    for i, (_, val) in enumerate(metrics):
        ax_metrics.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=9)

    def _plot_pos_violin(ax, col: str, title: str) -> None:
        if col not in df.columns:
            ax.set_axis_off()
            return
        pos_vals = df.loc[pos_mask, col].dropna().to_numpy(dtype=np.float64)
        neg_vals = df.loc[~pos_mask, col].dropna().to_numpy(dtype=np.float64)
        parts = ax.violinplot(
            [neg_vals, pos_vals],
            positions=[0, 1],
            showmeans=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-ROI", "Pathologist ROI"])
        ax.set_title(title)

    ax_scores_ab = fig.add_subplot(gs[1, 0])
    _plot_pos_violin(ax_scores_ab, abundance_col, f"C  Abundance {index_name}")
    ax_scores_sp = fig.add_subplot(gs[1, 1])
    _plot_pos_violin(ax_scores_sp, spatial_col, f"C  Spatial {index_name}")

    ax_inst = fig.add_subplot(gs[2, :])
    inst = validation["instance_summary"].sort_values("fraction_candidate", ascending=True)
    if len(inst):
        y_pos = np.arange(len(inst))
        ax_inst.barh(y_pos, inst["fraction_candidate"], color="#9467bd", alpha=0.85)
        ax_inst.set_yticks(y_pos)
        ax_inst.set_yticklabels(inst["annotation_type_instance"].astype(str), fontsize=8)
        ax_inst.set_xlabel(f"Fraction {index_name} candidate")
        ax_inst.set_title("D  Per pathologist ROI instance")
        ax_inst.set_xlim(0, min(1.0, inst["fraction_candidate"].max() * 1.15 + 0.05))
    else:
        ax_inst.text(0.5, 0.5, "No ROI instances", ha="center", va="center", transform=ax_inst.transAxes)
        ax_inst.set_axis_off()

    fig.suptitle(
        f"{sample} — H&E {index_name} vs pathologist ROIs (R={radius_um:g} μm)",
        y=0.995,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.94, hspace=0.38, wspace=0.28)
    return fig, validation


def plot_fri_pathologist_validation(
    df: pd.DataFrame,
    sample: str,
    validation: dict | None = None,
    *,
    spatial_df: pd.DataFrame | None = None,
    **kwargs,
):
    if validation is None:
        validation = validate_fri_against_pathologist(df)
    return plot_index_pathologist_validation(
        df,
        sample,
        validation,
        index_name="FRI",
        pathologist_labels=PATHOLOGIST_FRI_LABELS,
        spatial_df=spatial_df,
        spatial_col=kwargs.pop("spatial_col", "idx_FRI_spatial"),
        abundance_col=kwargs.pop("abundance_col", "idx_FRI_ratio"),
        candidate_col=kwargs.pop("candidate_col", "fri_candidate"),
        **kwargs,
    )


def plot_ari_pathologist_validation(
    df: pd.DataFrame,
    sample: str,
    validation: dict | None = None,
    *,
    spatial_df: pd.DataFrame | None = None,
    **kwargs,
):
    if validation is None:
        validation = validate_ari_against_pathologist(df)
    return plot_index_pathologist_validation(
        df,
        sample,
        validation,
        index_name="ARI",
        pathologist_labels=PATHOLOGIST_ARI_LABELS,
        spatial_df=spatial_df,
        spatial_col=kwargs.pop("spatial_col", "idx_ARI_spatial"),
        abundance_col=kwargs.pop("abundance_col", "idx_ARI_ratio"),
        candidate_col=kwargs.pop("candidate_col", "ari_candidate"),
        **kwargs,
    )


def plot_tls_pathologist_overlay(
    df: pd.DataFrame,
    sample: str,
    *,
    spatial_df: pd.DataFrame | None = None,
    pathologist_label: str = PATHOLOGIST_TLS_LABEL,
    spatial_col: str = "idx_TLS_spatial",
    candidate_col: str = "tls_candidate",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    radius_um: float | None = None,
    vmax_percentile: float = DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    point_size: float = 0.25,
    candidate_size: float = 2.5,
    candidate_color: str = "cyan",
    boundary_color: str = "magenta",
    boundary_linewidth: float = 1.5,
    figsize: tuple[float, float] = (8.0, 8.0),
):
    """
    Overlay pathologist Mixed Inflammation hull boundaries and TLS candidates.

    Background: spatial TLS heatmap; magenta outlines: pathologist ROIs;
    cyan dots: TLS candidate cells (``idx_TLS_spatial`` top 5% percentiles).

    Pass ``spatial_df=df_spatial`` when ``df`` is a pathologist-merged table so
    heatmap / candidate markers use the same one-row-per-cell table as
    ``plot_tls_abundance_vs_spatial``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    plot_df = _dedupe_cells_for_spatial_plot(spatial_df if spatial_df is not None else df)
    if spatial_col not in plot_df.columns:
        raise KeyError(f"Missing column {spatial_col!r}")
    if radius_um is None and "tls_spatial_radius_um" in plot_df.columns:
        radius_um = float(plot_df["tls_spatial_radius_um"].iloc[0])
    if radius_um is None:
        radius_um = DEFAULT_TLS_SPATIAL_RADIUS_UM

    fig, ax = plt.subplots(figsize=figsize)
    x = plot_df[coord_x].to_numpy(dtype=np.float64)
    y = plot_df[coord_y].to_numpy(dtype=np.float64)
    values = plot_df[spatial_col].to_numpy(dtype=np.float64)
    vmin, vmax = percentile_vlim(values, vmax_percentile=vmax_percentile)
    sc = ax.scatter(x, y, c=values, s=point_size, cmap="magma", alpha=0.55, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.046, label=spatial_col)

    hulls = pathologist_region_hulls(df, pathologist_label=pathologist_label)
    patches = [
        Polygon(hull, closed=True, fill=False, edgecolor=boundary_color, linewidth=boundary_linewidth)
        for hull in hulls.values()
    ]
    if patches:
        ax.add_collection(PatchCollection(patches, match_original=True))
    ax.plot([], [], color=boundary_color, linewidth=boundary_linewidth, label=pathologist_label)

    ax.set_title(f"{sample}: Spatial TLS + pathologist {pathologist_label}")
    ax.set_xlabel("spatial_HE x (px)")
    ax.set_ylabel("spatial_HE y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    _plot_tls_candidate_markers(
        ax,
        plot_df,
        candidate_col=candidate_col,
        coord_x=coord_x,
        coord_y=coord_y,
        candidate_size=candidate_size,
        candidate_color=candidate_color,
        show_legend=False,
    )
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout()
    return fig, ax


def plot_tls_pathologist_validation(
    df: pd.DataFrame,
    sample: str,
    validation: dict | None = None,
    *,
    spatial_df: pd.DataFrame | None = None,
    pathologist_label: str = PATHOLOGIST_TLS_LABEL,
    spatial_col: str = "idx_TLS_spatial",
    abundance_col: str = "idx_TLS",
    candidate_col: str = "tls_candidate",
    instance_col: str = "annotation_type_instance",
    radius_um: float | None = None,
    figsize: tuple[float, float] = (12.0, 14.0),
):
    """
    Validation figure (3 rows × 2 columns):

    Row 0: A) spatial overlay | B) precision / recall / enrichment
    Row 1: C) Abundance TLS violin | C) Spatial TLS violin (each with Mann–Whitney p)
    Row 2: D) per pathologist MI instance (full width)
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    if validation is None:
        validation = validate_tls_against_pathologist(
            df,
            pathologist_label=pathologist_label,
            candidate_col=candidate_col,
            spatial_col=spatial_col,
            abundance_col=abundance_col,
            instance_col=instance_col,
        )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9], wspace=0.28, hspace=0.38)

    # A) spatial overlay — row 0, col 0
    ax_overlay = fig.add_subplot(gs[0, 0])
    plot_df = _dedupe_cells_for_spatial_plot(spatial_df if spatial_df is not None else df)
    x = plot_df["coord_x"].to_numpy(dtype=np.float64)
    y = plot_df["coord_y"].to_numpy(dtype=np.float64)
    values = plot_df[spatial_col].to_numpy(dtype=np.float64)
    vmin, vmax = percentile_vlim(values)
    sc = ax_overlay.scatter(x, y, c=values, s=0.2, cmap="magma", alpha=0.55, vmin=vmin, vmax=vmax)
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    hulls = pathologist_region_hulls(df, pathologist_label=pathologist_label)
    patches = [
        Polygon(hull, closed=True, fill=False, edgecolor="magenta", linewidth=1.5)
        for hull in hulls.values()
    ]
    if patches:
        ax_overlay.add_collection(PatchCollection(patches, match_original=True))
    ax_overlay.plot([], [], color="magenta", linewidth=1.5, label=pathologist_label)
    ax_overlay.set_title(f"A  {sample}: overlay")
    ax_overlay.set_aspect("equal")
    ax_overlay.invert_yaxis()
    _plot_tls_candidate_markers(
        ax_overlay,
        plot_df,
        candidate_col=candidate_col,
        candidate_size=2.0,
        show_legend=False,
    )
    ax_overlay.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.colorbar(sc, ax=ax_overlay, fraction=0.046, pad=0.02)

    # B) metrics bars — row 0, col 1
    ax_metrics = fig.add_subplot(gs[0, 1])
    metrics = {
        "Precision": validation["precision"],
        "Recall": validation["recall"],
        "Enrichment": validation["enrichment"],
    }
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    bars = ax_metrics.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.85)
    ax_metrics.set_ylim(0, max(max(metrics.values()) * 1.15, 1.05))
    ax_metrics.set_ylabel("Rate / fold")
    ax_metrics.set_title("B  Candidate vs pathologist MI")
    for bar, val in zip(bars, metrics.values()):
        ax_metrics.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
    conf_text = (
        f"TP={validation['tp']}  FP={validation['fp']}\n"
        f"FN={validation['fn']}  TN={validation['tn']}\n"
        f"MI cells={validation['n_mixed_inflammation']:,}  "
        f"candidates={validation['n_candidate']:,}"
    )
    ax_metrics.text(0.02, 0.98, conf_text, transform=ax_metrics.transAxes, va="top", fontsize=9)

    def _plot_mi_nonmi_violin(ax, score_col: str, panel_title: str) -> None:
        mi_mask = df["is_mixed_inflammation"].to_numpy(dtype=bool)
        non_vals = df.loc[~mi_mask, score_col].to_numpy(dtype=np.float64)
        mi_vals = df.loc[mi_mask, score_col].to_numpy(dtype=np.float64)
        data = [non_vals, mi_vals]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=False, widths=0.75)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor("#ff7f0e" if i else "#aec7e8")
            body.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["non-MI", "MI"], fontsize=9)
        ax.set_ylabel("Score")
        if len(non_vals) and len(mi_vals):
            _, p = stats.mannwhitneyu(mi_vals, non_vals, alternative="two-sided")
            stars = pvalue_to_stars(float(p))
            ax.set_title(
                f"{panel_title}\nMann-Whitney p={p:.3g} ({stars})",
                fontsize=10,
            )
        else:
            ax.set_title(panel_title, fontsize=10)

    # C) score distributions — row 1
    ax_scores_tls = fig.add_subplot(gs[1, 0])
    if abundance_col in df.columns:
        _plot_mi_nonmi_violin(ax_scores_tls, abundance_col, "C  Abundance TLS")
    else:
        ax_scores_tls.set_axis_off()

    ax_scores_spatial = fig.add_subplot(gs[1, 1])
    if spatial_col in df.columns:
        _plot_mi_nonmi_violin(ax_scores_spatial, spatial_col, "C  Spatial TLS")
    else:
        ax_scores_spatial.set_axis_off()

    # D) per-instance candidate fraction — row 2, full width
    ax_inst = fig.add_subplot(gs[2, :])
    inst = validation["instance_summary"].sort_values("fraction_candidate", ascending=True)
    if len(inst):
        y_pos = np.arange(len(inst))
        ax_inst.barh(y_pos, inst["fraction_candidate"], color="#9467bd", alpha=0.85)
        ax_inst.set_yticks(y_pos)
        ax_inst.set_yticklabels(inst[instance_col].astype(str), fontsize=8)
        ax_inst.set_xlabel("Fraction TLS candidate")
        ax_inst.set_title("D  Per pathologist MI instance")
        ax_inst.set_xlim(0, min(1.0, inst["fraction_candidate"].max() * 1.15 + 0.05))
    else:
        ax_inst.text(0.5, 0.5, "No MI instances", ha="center", va="center", transform=ax_inst.transAxes)
        ax_inst.set_axis_off()

    if radius_um is None and "tls_spatial_radius_um" in df.columns:
        radius_um = float(df["tls_spatial_radius_um"].iloc[0])
    fig.suptitle(
        f"{sample} — H&E TLS vs pathologist {pathologist_label} (R={radius_um:g} μm)",
        y=0.995,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.96, hspace=0.38, wspace=0.28)
    return fig, validation
##################################################################


def compute_fri_score_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    annotation: pd.DataFrame | None = None,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """
    Per-cell FRI score = sum_c w_c * p_i(c) over FRI niche member cell types.

    Default ``w_c = 1`` for each class in ``FRI_ALL_NUMERATOR_CELLS``.
    """
    del annotation  # explicit member sets; annotation kept for API compatibility
    idx_map = class_name_to_prob_index(class_names)
    out = np.zeros(probs.shape[0], dtype=np.float64)
    for ct in FRI_ALL_NUMERATOR_CELLS:
        if ct not in idx_map:
            continue
        j = idx_map[ct]
        w = 1.0 if weights is None else float(weights.get(ct, 1.0))
        out += w * probs[:, j]
    return out


def compute_fri_niche_abundance_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    niche_key: str,
) -> np.ndarray:
    """Per-cell abundance for one FRI niche compartment."""
    if niche_key not in FRI_NICHE_CELL_TYPES:
        raise KeyError(f"Unknown FRI niche {niche_key!r}; expected one of {FRI_NICHE_ORDER}.")
    return compute_cell_type_abundance_per_cell(
        probs, class_names, FRI_NICHE_CELL_TYPES[niche_key]
    )


def compute_niche_abundance_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    annotation: pd.DataFrame,
    niche: str,
) -> np.ndarray:
    """Legacy annotation-sheet niche lookup (prefer explicit FRI niche keys)."""
    niche_map = {
        "Fibrotic stromal": "fibrotic_stromal",
        "Injury epithelium": "injury_epithelium",
        "Profibrotic macrophages": "profibrotic_macrophages",
    }
    if niche in niche_map:
        return compute_fri_niche_abundance_per_cell(probs, class_names, niche_map[niche])
    members = cell_types_for_niches(annotation, [niche], class_names)
    cols = prob_columns_for_cell_types(class_names, members)
    return sum_prob_columns(probs, cols)


def compute_ratio_index_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    numerator_cell_types: Sequence[str],
    denominator_cell_types: Sequence[str],
) -> np.ndarray:
    num_cols = prob_columns_for_cell_types(class_names, numerator_cell_types)
    den_cols = prob_columns_for_cell_types(class_names, denominator_cell_types)
    return safe_ratio(sum_prob_columns(probs, num_cols), sum_prob_columns(probs, den_cols))


def compute_fri_ratio_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
    annotation: pd.DataFrame | None = None,
) -> np.ndarray:
    del annotation
    niche_sums = [
        compute_fri_niche_abundance_per_cell(probs, class_names, key)
        for key in FRI_NICHE_ORDER
    ]
    numer = np.sum(np.stack(niche_sums, axis=0), axis=0)
    den = compute_cell_type_abundance_per_cell(probs, class_names, ALVEOLAR_EPITHELIUM)
    return safe_ratio(numer, den)


def compute_ari_ratio_per_cell(
    probs: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    spec = INDEX_SPECS["ARI_ratio"]
    return compute_ratio_index_per_cell(
        probs,
        class_names,
        spec.numerator_cell_types,
        spec.denominator_cell_types,
    )


def add_histology_indices_to_auroc_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    class_names: Sequence[str],
    annotation: pd.DataFrame,
    *,
    prefix: str = "idx_",
) -> pd.DataFrame:
    """Attach per-cell index columns to an AUROC dataframe."""
    out = df.copy()
    out[f"{prefix}TLS"] = compute_tls_per_cell(probs, class_names)
    out[f"{prefix}FRI_ratio"] = compute_fri_ratio_per_cell(probs, class_names, annotation)
    out[f"{prefix}FRI_score"] = compute_fri_score_per_cell(probs, class_names, annotation)
    out[f"{prefix}ARI_ratio"] = compute_ari_ratio_per_cell(probs, class_names)
    for niche_key in FRI_NICHE_ORDER:
        out[f"{prefix}niche_{niche_key}"] = compute_fri_niche_abundance_per_cell(
            probs, class_names, niche_key
        )
    out[f"{prefix}alveolar_AT1_AT2"] = compute_cell_type_abundance_per_cell(
        probs, class_names, ALVEOLAR_EPITHELIUM
    )
    fri_num_cols = [f"{prefix}niche_{k}" for k in FRI_NICHE_ORDER]
    out[f"{prefix}FRI_numerator"] = out[fri_num_cols].sum(axis=1)
    out[f"{prefix}ARI_numerator"] = compute_cell_type_abundance_per_cell(
        probs, class_names, ARI_FB_CELLS
    )
    return out


def summarize_indices_by_sample(
    df: pd.DataFrame,
    sample_col: str = "sample",
    index_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Sample-level summary.

    Means for abundance-like indices; **FRI_ratio** and **ARI_ratio** recomputed as
    mean(numerator) / mean(alveolar) for interpretability.
    """
    if index_cols is None:
        index_cols = [c for c in df.columns if c.startswith("idx_")]

    def _aggregate(sub: pd.DataFrame) -> dict:
        row: dict = {"n_cells": len(sub)}
        for c in index_cols:
            if c in ("idx_FRI_ratio", "idx_ARI_ratio", "idx_FRI_spatial", "idx_ARI_spatial"):
                continue
            if c in sub.columns:
                row[c] = float(sub[c].mean())
        alv = float(sub["idx_alveolar_AT1_AT2"].mean()) if "idx_alveolar_AT1_AT2" in sub.columns else 1e-8
        if "idx_FRI_numerator" in sub.columns:
            row["idx_FRI_ratio"] = float(sub["idx_FRI_numerator"].mean()) / max(alv, 1e-8)
        if "idx_ARI_numerator" in sub.columns:
            row["idx_ARI_ratio"] = float(sub["idx_ARI_numerator"].mean()) / max(alv, 1e-8)
        if "idx_FRI_spatial" in sub.columns:
            row["idx_FRI_spatial"] = float(sub["idx_FRI_spatial"].mean())
        if "idx_ARI_spatial" in sub.columns:
            row["idx_ARI_spatial"] = float(sub["idx_ARI_spatial"].mean())
        return row

    if sample_col not in df.columns:
        row = _aggregate(df)
        row["sample"] = "all"
        return pd.DataFrame([row])

    rows = []
    for sample, sub in df.groupby(sample_col, sort=True):
        row = _aggregate(sub)
        row["sample"] = sample
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CompleteCasePaths:
    """Per-sample StarDist AUROC + matched h5ad under ``Complete_Cases``."""

    sample: str
    auroc_csv: Path
    class_names_csv: Path
    stardist_h5ad: Path

    def assert_ready(self, *, require_h5ad: bool = True) -> None:
        missing = []
        if not self.auroc_csv.is_file():
            missing.append(str(self.auroc_csv))
        if not self.class_names_csv.is_file():
            missing.append(str(self.class_names_csv))
        if require_h5ad and not self.stardist_h5ad.is_file():
            missing.append(str(self.stardist_h5ad))
        if missing:
            raise FileNotFoundError(f"{self.sample}: missing file(s):\n  " + "\n  ".join(missing))


def complete_case_paths(
    sample: str,
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
) -> CompleteCasePaths:
    """
    Resolve per-sample paths under ``Complete_Cases/{sample}/``.

    AUROC probabilities:
      ``{sample}{project_suffix}/result/validation_external_stardist_matched_AUROC.csv``
    StarDist h5ad (``spatial_HE``):
      ``{sample}/{sample}_matched_features_stardist.h5ad``
    """
    root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    sample_dir = root / sample
    result_dir = sample_dir / f"{sample}{project_suffix}" / "result"
    return CompleteCasePaths(
        sample=sample,
        auroc_csv=result_dir / "validation_external_stardist_matched_AUROC.csv",
        class_names_csv=result_dir / "validation_external_stardist_matched_AUROC_class_names.csv",
        stardist_h5ad=sample_dir / f"{sample}_matched_features_stardist.h5ad",
    )


def discover_complete_cases(
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    require_h5ad: bool = True,
    require_auroc: bool = True,
) -> list[str]:
    """
    List sample IDs under ``Complete_Cases`` with AUROC (+ optional h5ad) present.
    """
    root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Complete_Cases root not found: {root}")
    samples = []
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        paths = complete_case_paths(sample, root, project_suffix=project_suffix)
        if require_auroc and not paths.auroc_csv.is_file():
            continue
        if require_h5ad and not paths.stardist_h5ad.is_file():
            continue
        samples.append(sample)
    return samples


def load_complete_case_with_spatial(
    sample: str,
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    obsm_key: str = "spatial_HE",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, CompleteCasePaths]:
    """
    Load one Complete_Cases sample: AUROC probs + ``spatial_HE`` coordinates.

    Returns ``(df, class_names, probs, coords, paths)``.
    """
    paths = complete_case_paths(sample, cases_root, project_suffix=project_suffix)
    paths.assert_ready(require_h5ad=True)
    df, class_names, probs, coords = load_auroc_with_spatial_coords(
        paths.auroc_csv,
        paths.stardist_h5ad,
        paths.class_names_csv,
        obsm_key=obsm_key,
    )
    return df, class_names, probs, coords, paths


##################################################################
# 2026.06.27LLY: add the per-sample and cross-dataset API for loading the AUROC and spatial coordinates
##################################################################
@dataclass(frozen=True)
class CrossDatasetSpatialPaths:
    """
    Cross-dataset StarDist AUROC under ``result_all_spatial/stardist/{sample}/``,
    with ``spatial_HE`` still loaded from ``Complete_Cases/{sample}/``.
    """

    sample: str
    auroc_csv: Path
    class_names_csv: Path
    stardist_h5ad: Path
    cases_root: Path
    model_checkpoint: Path = DEFAULT_CROSS_DATASET_MODEL_CHECKPOINT

    def assert_ready(self, *, require_h5ad: bool = True) -> None:
        missing = []
        if not self.auroc_csv.is_file():
            missing.append(str(self.auroc_csv))
        if not self.class_names_csv.is_file():
            missing.append(str(self.class_names_csv))
        if require_h5ad and not self.stardist_h5ad.is_file():
            missing.append(str(self.stardist_h5ad))
        if missing:
            raise FileNotFoundError(f"{self.sample}: missing file(s):\n  " + "\n  ".join(missing))


def cross_dataset_spatial_paths(
    sample: str,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
) -> CrossDatasetSpatialPaths:
    """
    Resolve cross-dataset prediction paths for one sample.

    AUROC probabilities:
      ``result_all_spatial/stardist/{sample}/validation_external_stardist_matched_AUROC.csv``
    StarDist h5ad (``spatial_HE``; unchanged):
      ``Complete_Cases/{sample}/{sample}_matched_features_stardist.h5ad``
    """
    st_root = Path(stardist_root or DEFAULT_CROSS_DATASET_STARDIST_ROOT).expanduser().resolve()
    cc_root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    ckpt = Path(model_checkpoint or DEFAULT_CROSS_DATASET_MODEL_CHECKPOINT).expanduser().resolve()
    sample_dir = st_root / sample
    cc_sample_dir = cc_root / sample
    return CrossDatasetSpatialPaths(
        sample=sample,
        auroc_csv=sample_dir / "validation_external_stardist_matched_AUROC.csv",
        class_names_csv=sample_dir / "validation_external_stardist_matched_AUROC_class_names.csv",
        stardist_h5ad=cc_sample_dir / f"{sample}_matched_features_stardist.h5ad",
        cases_root=cc_root,
        model_checkpoint=ckpt,
    )


def discover_cross_dataset_spatial_samples(
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    *,
    require_h5ad: bool = True,
    require_auroc: bool = True,
) -> list[str]:
    """List sample IDs with cross-dataset AUROC (+ optional Complete_Cases h5ad)."""
    st_root = Path(stardist_root or DEFAULT_CROSS_DATASET_STARDIST_ROOT).expanduser().resolve()
    if not st_root.is_dir():
        raise FileNotFoundError(f"Cross-dataset stardist root not found: {st_root}")
    samples = []
    for sample_dir in sorted(st_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        paths = cross_dataset_spatial_paths(sample, stardist_root=st_root, cases_root=cases_root)
        if require_auroc and not paths.auroc_csv.is_file():
            continue
        if require_h5ad and not paths.stardist_h5ad.is_file():
            continue
        samples.append(sample)
    return samples


def load_cross_dataset_sample_with_spatial(
    sample: str,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
    obsm_key: str = "spatial_HE",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, CrossDatasetSpatialPaths]:
    """
    Load one sample using cross-dataset AUROC + Complete_Cases ``spatial_HE``.

    Returns ``(df, class_names, probs, coords, paths)``.
    """
    paths = cross_dataset_spatial_paths(
        sample,
        stardist_root=stardist_root,
        cases_root=cases_root,
        model_checkpoint=model_checkpoint,
    )
    paths.assert_ready(require_h5ad=True)
    df, class_names, probs, coords = load_auroc_with_spatial_coords(
        paths.auroc_csv,
        paths.stardist_h5ad,
        paths.class_names_csv,
        obsm_key=obsm_key,
    )
    return df, class_names, probs, coords, paths


def load_sample_with_spatial(
    sample: str,
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    model_checkpoint: str | Path | None = None,
    obsm_key: str = "spatial_HE",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, CompleteCasePaths | CrossDatasetSpatialPaths]:
    """Dispatch loader for cross-dataset (default) or per-sample prediction sources."""
    if prediction_source == PREDICTION_SOURCE_PER_SAMPLE:
        return load_complete_case_with_spatial(
            sample,
            cases_root,
            project_suffix=project_suffix,
            obsm_key=obsm_key,
        )
    if prediction_source == PREDICTION_SOURCE_CROSS_DATASET:
        return load_cross_dataset_sample_with_spatial(
            sample,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
            obsm_key=obsm_key,
        )
    raise ValueError(
        f"Unknown prediction_source={prediction_source!r}; "
        f"expected {PREDICTION_SOURCE_PER_SAMPLE!r} or {PREDICTION_SOURCE_CROSS_DATASET!r}."
    )


def load_sample_with_spatial_indices(
    sample: str,
    annotation: pd.DataFrame,
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    model_checkpoint: str | Path | None = None,
    obsm_key: str = "spatial_HE",
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    prefix: str = "idx_",
) -> tuple[
    pd.DataFrame,
    list[str],
    np.ndarray,
    np.ndarray,
    CompleteCasePaths | CrossDatasetSpatialPaths,
    float,
]:
    """
    Load one sample and attach abundance + spatial TLS / FRI / ARI indices.

    Returns ``(df, class_names, probs, coords, paths, um_per_he_pixel)``.
    """
    df, class_names, probs, coords, paths = load_sample_with_spatial(
        sample,
        prediction_source=prediction_source,
        cases_root=cases_root,
        stardist_root=stardist_root,
        project_suffix=project_suffix,
        model_checkpoint=model_checkpoint,
        obsm_key=obsm_key,
    )
    sample_um = um_per_pixel
    if sample_um is None:
        sample_um = infer_um_per_he_pixel(
            paths.stardist_h5ad.parent, h5ad_path=paths.stardist_h5ad
        )
    df = add_all_spatial_niche_indices_to_auroc_df(
        df,
        probs,
        class_names,
        coords,
        annotation,
        radius_um=radius_um,
        um_per_pixel=sample_um,
        hotspot_percentile=hotspot_percentile,
        use_entropy=use_entropy,
        entropy_weight=entropy_weight,
        prefix=prefix,
    )
    return df, class_names, probs, coords, paths, sample_um


def discover_spatial_samples(
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    require_h5ad: bool = True,
    require_auroc: bool = True,
) -> list[str]:
    """List samples available for the chosen prediction source."""
    if prediction_source == PREDICTION_SOURCE_PER_SAMPLE:
        return discover_complete_cases(
            cases_root,
            project_suffix=project_suffix,
            require_h5ad=require_h5ad,
            require_auroc=require_auroc,
        )
    if prediction_source == PREDICTION_SOURCE_CROSS_DATASET:
        return discover_cross_dataset_spatial_samples(
            stardist_root,
            cases_root,
            require_h5ad=require_h5ad,
            require_auroc=require_auroc,
        )
    if prediction_source == PREDICTION_SOURCE_STARDIST_ALL:
        return discover_stardist_all_label_samples(
            stardist_root=stardist_root,
        )
    raise ValueError(
        f"Unknown prediction_source={prediction_source!r}; "
        f"expected {PREDICTION_SOURCE_PER_SAMPLE!r}, "
        f"{PREDICTION_SOURCE_CROSS_DATASET!r}, or {PREDICTION_SOURCE_STARDIST_ALL!r}."
    )


##################################################################
# All StarDist nuclei — {sample}_all_features_stardist_label.h5ad
##################################################################
def is_valid_h5ad(path: str | Path) -> bool:
    """Return True when *path* exists and starts with the HDF5 magic bytes."""
    p = Path(path).expanduser()
    if not p.is_file() or p.stat().st_size < len(_HDF5_SIGNATURE):
        return False
    with open(p, "rb") as fh:
        return fh.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE


def validate_h5ad_or_raise(path: str | Path, *, sample: str | None = None) -> Path:
    """
    Raise an actionable error when an h5ad path is missing or not valid HDF5.

    Truncated files usually come from an interrupted ``--steps stardist_all`` write
    (file lock, killed job, or notebook keeping the file open).
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing h5ad: {p}")
    if is_valid_h5ad(p):
        return p
    sample_id = sample or p.name.split("_all_features_stardist_label.h5ad")[0]
    size = p.stat().st_size
    raise OSError(
        f"{p.name} is not a valid HDF5/h5ad file (size={size:,} bytes).\n"
        "Likely truncated from an interrupted write. Close any program using this "
        "file (Jupyter, h5ad viewer), delete the corrupt file, then regenerate:\n"
        "  python -u code/Xenium_lung/Lung_train_validate_cv_UNIlabel.py \\\n"
        "    --mode cross-dataset --sample "
        f"{sample_id} \\\n"
        "    --pooled-save-result result_all_spatial \\\n"
        "    --use-spatial-context --spatial-k 8 --spatial-mode mean \\\n"
        "    --ablation-tag D_emph_L2_spatial_bs4096 \\\n"
        "    --steps stardist_all\n"
        f"Path: {p}"
    )


def audit_stardist_all_label_h5ads(
    stardist_root: str | Path | None = None,
    *,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """List every sample folder and whether its label h5ad exists / is valid HDF5."""
    if stardist_root is None:
        data_root = Path(data_root or DEFAULT_COMPLETE_CASES_ROOT.parent).expanduser().resolve()
        st_root = data_root / save_result / "stardist"
    else:
        st_root = Path(stardist_root).expanduser().resolve()
    rows = []
    if not st_root.is_dir():
        return pd.DataFrame(columns=["sample", "exists", "size_bytes", "valid_h5ad", "path"])
    for sample_dir in sorted(st_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        label_h5ad = sample_dir / f"{sample}{STARDIST_ALL_LABEL_H5AD_SUFFIX}"
        exists = label_h5ad.is_file()
        size = int(label_h5ad.stat().st_size) if exists else 0
        rows.append(
            {
                "sample": sample,
                "exists": exists,
                "size_bytes": size,
                "valid_h5ad": is_valid_h5ad(label_h5ad) if exists else False,
                "path": str(label_h5ad),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class StardistAllLabelPaths:
    """Per-sample label h5ad with five-head probs for every StarDist nucleus."""

    sample: str
    label_h5ad: Path
    cases_root: Path
    stardist_root: Path
    save_result: str = DEFAULT_POOLED_SAVE_RESULT

    def assert_ready(self) -> None:
        validate_h5ad_or_raise(self.label_h5ad, sample=self.sample)


def stardist_all_label_paths(
    sample: str,
    *,
    data_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
) -> StardistAllLabelPaths:
    """
    Resolve ``Data/{save_result}/stardist/{sample}/{sample}_all_features_stardist_label.h5ad``.
    """
    data_root = Path(data_root or DEFAULT_COMPLETE_CASES_ROOT.parent).expanduser().resolve()
    cc_root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    st_root = Path(stardist_root or (data_root / save_result / "stardist")).expanduser().resolve()
    return StardistAllLabelPaths(
        sample=sample,
        label_h5ad=st_root / sample / f"{sample}{STARDIST_ALL_LABEL_H5AD_SUFFIX}",
        cases_root=cc_root,
        stardist_root=st_root,
        save_result=save_result,
    )


def discover_stardist_all_label_samples(
    stardist_root: str | Path | None = None,
    *,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    data_root: str | Path | None = None,
    require_valid_h5ad: bool = True,
) -> list[str]:
    """List sample IDs with ``{sample}_all_features_stardist_label.h5ad`` present."""
    if stardist_root is None:
        data_root = Path(data_root or DEFAULT_COMPLETE_CASES_ROOT.parent).expanduser().resolve()
        st_root = data_root / save_result / "stardist"
    else:
        st_root = Path(stardist_root).expanduser().resolve()
    if not st_root.is_dir():
        return []
    out = []
    for sample_dir in sorted(st_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        label_h5ad = sample_dir / f"{sample_dir.name}{STARDIST_ALL_LABEL_H5AD_SUFFIX}"
        if not label_h5ad.is_file():
            continue
        if require_valid_h5ad and not is_valid_h5ad(label_h5ad):
            continue
        out.append(sample_dir.name)
    return out


def load_probs_from_stardist_all_label_h5ad(
    h5ad_path: str | Path,
    *,
    head: str = "l2",
    obsm_key: str = "spatial_HE",
    sample: str | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """
    Load Level-2 (or other head) softmax probs + coordinates from label h5ad.

    Returns ``(df, class_names, probs, coords)`` where ``df`` has ``cell_id``,
    ``coord_x``, ``coord_y``, and ``predict_l2_final_CT`` (argmax label when head='l2').
    """
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    validate_h5ad_or_raise(path, sample=sample)
    adata = ad.read_h5ad(str(path))
    if "pred_prob_class_names" not in adata.uns:
        raise KeyError(f"{path.name}: missing uns['pred_prob_class_names']")
    names_map = adata.uns["pred_prob_class_names"]
    if head not in names_map:
        raise KeyError(
            f"head {head!r} not in pred_prob_class_names; available: {list(names_map.keys())}"
        )
    class_names = [str(x) for x in names_map[head]]
    prob_cols = [f"{head}_prob_{j}" for j in range(len(class_names))]
    missing = [c for c in prob_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"{path.name}: missing prob columns {missing[:5]} ...")
    probs = adata.obs[prob_cols].to_numpy(dtype=np.float64)

    coords = None
    for key in (obsm_key, "spatial_HE", "spatial"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=np.float64)[:, :2]
            break
    if coords is None and {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        coords = adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    if coords is None:
        raise KeyError(
            f"{path.name}: no obsm['spatial_HE'/'spatial'] and no centroid_x/y in obs."
        )

    if "cell_id" in adata.obs.columns:
        cell_ids = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        cell_ids = adata.obs_names.astype(str).to_numpy()

    pred_idx = probs.argmax(axis=1)
    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "coord_x": coords[:, 0],
            "coord_y": coords[:, 1],
            "predict_l2_idx": pred_idx.astype(int),
            "predict_l2_final_CT": [class_names[i] for i in pred_idx],
        }
    )
    if sample is not None:
        df["sample"] = sample
    if "probability" in adata.obs.columns:
        df["stardist_probability"] = pd.to_numeric(adata.obs["probability"], errors="coerce")
    return df, class_names, probs, coords


########################################################
## 2026.07.05 LLY: shown predicted cell type final_CT and final_lineage from the label h5ad [Incomplete_Cases]
########################################################
def load_stardist_all_label_final_ct_lineage(
    h5ad_path: str | Path,
    *,
    sample: str | None = None,
) -> pd.DataFrame:
    """
    Argmax L2 → ``final_CT`` and argmax L1 → ``final_lineage`` from label h5ad.

    Intended for ``plot_final_ct_by_lineage`` (2×2 bar panels by lineage).
    """
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    validate_h5ad_or_raise(path, sample=sample)
    adata = ad.read_h5ad(str(path))
    names_map = adata.uns.get("pred_prob_class_names")
    if not names_map:
        raise KeyError(f"{path.name}: missing uns['pred_prob_class_names']")
    for head in ("l2", "l1"):
        if head not in names_map:
            raise KeyError(
                f"{path.name}: head {head!r} not in pred_prob_class_names; "
                f"available: {list(names_map.keys())}"
            )

    l2_names = [str(x) for x in names_map["l2"]]
    l1_names = [str(x) for x in names_map["l1"]]
    l2_cols = [f"l2_prob_{j}" for j in range(len(l2_names))]
    l1_cols = [f"l1_prob_{j}" for j in range(len(l1_names))]
    missing = [c for c in l2_cols + l1_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"{path.name}: missing prob columns {missing[:5]} ...")

    l2_probs = adata.obs[l2_cols].to_numpy(dtype=np.float64)
    l1_probs = adata.obs[l1_cols].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "final_CT": [l2_names[i] for i in l2_probs.argmax(axis=1)],
            "final_lineage": [l1_names[i] for i in l1_probs.argmax(axis=1)],
        }
    )
    if "cell_id" in adata.obs.columns:
        out["cell_id"] = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        out["cell_id"] = adata.obs_names.astype(str).to_numpy()
    if sample is not None:
        out["sample"] = sample
    return out

########################################################
## 2026.07.05 LLY: shown predicted cell type final_CT and final_lineage from the label h5ad [Incomplete_Cases]
########################################################
STARDIST_ALL_LABEL_TIER_ORDER: tuple[str, ...] = ("l2", "l1", "l12", "l3", "l4")

_STARDIST_TIER_SPATIAL_SCHEMES: dict[str, str] = {
    "l2": "xenium_lung_fine",
    "l1": "xenium_lung_coarse",
    "l12": "xenium_lung_intermediate",
    "l3": "xenium_lung_CNiche",
    "l4": "xenium_lung_TNiche",
}

_STARDIST_TIER_DISPLAY: dict[str, str] = {
    "l2": "Level2 cell type",
    "l1": "Level1 lineage",
    "l12": "Level12 sublineage",
    "l3": "Level3 CNiche",
    "l4": "Level4 TNiche",
}


def load_stardist_all_label_tier_predictions(
    h5ad_path: str | Path,
    *,
    sample: str | None = None,
    heads: Sequence[str] = STARDIST_ALL_LABEL_TIER_ORDER,
    obsm_key: str = "spatial_HE",
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """
    Load argmax predictions for five label heads from a StarDist label h5ad.

    Returns ``(coords, tier_data)`` where each ``tier_data[head]`` has
    ``class_names`` (list[str]) and ``pred_encoded`` (int64 array).
    """
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    validate_h5ad_or_raise(path, sample=sample)
    adata = ad.read_h5ad(str(path))
    names_map = adata.uns.get("pred_prob_class_names")
    if not names_map:
        raise KeyError(f"{path.name}: missing uns['pred_prob_class_names']")

    coords = None
    for key in (obsm_key, "spatial_HE", "spatial"):
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=np.float64)[:, :2]
            break
    if coords is None and {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        coords = adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    if coords is None:
        raise KeyError(
            f"{path.name}: no obsm['spatial_HE'/'spatial'] and no centroid_x/y in obs."
        )

    tier_data: dict[str, dict[str, object]] = {}
    for head in heads:
        if head not in names_map:
            raise KeyError(
                f"{path.name}: head {head!r} not in pred_prob_class_names; "
                f"available: {list(names_map.keys())}"
            )
        class_names = [str(x) for x in names_map[head]]
        prob_cols = [f"{head}_prob_{j}" for j in range(len(class_names))]
        missing = [c for c in prob_cols if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"{path.name}: missing prob columns {missing[:5]} ...")
        probs = adata.obs[prob_cols].to_numpy(dtype=np.float64)
        tier_data[head] = {
            "class_names": class_names,
            "pred_encoded": probs.argmax(axis=1).astype(np.int64),
        }
    return coords, tier_data


def _resolve_stardist_tier_save_path(
    save_path: str | Path,
    tier: str,
    *,
    sample_label: str,
    roi_label: str | None = None,
) -> Path:
    """
    Resolve one output file per tier under a single subdirectory.

    - ``.../spatial_tiers`` (no suffix) → ``.../spatial_tiers/{sample}_stardist_pred_{tier}.jpg``
    - ``.../spatial_tiers.jpg`` → same folder layout (stem becomes subdir name)
    - With *roi_label*: ``.../spatial_tiers/{roi}/{sample}_stardist_pred_{tier}.jpg``
    """
    path = Path(save_path).expanduser()
    suffix = path.suffix or ".jpg"
    base_dir = path.parent / (path.stem if path.suffix else path.name)
    out_dir = base_dir / roi_label if roi_label else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return (out_dir / f"{sample_label}_stardist_pred_{tier}{suffix}").resolve()


def _load_global_l2_class_names() -> list[str]:
    """Global training ``final_CT`` list (47 classes) for stable L2 spatial colors."""
    path = (
        Path(__file__).resolve().parents[2]
        / "data/Xemium/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/result_all_spatial"
        / "validation_external_stardist_matched_AUROC_all_samples_class_names.csv"
    )
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    name_col = "final_CT" if "final_CT" in df.columns else df.columns[-1]
    if "class_index" in df.columns:
        df = df.sort_values("class_index")
    return [str(x) for x in df[name_col].tolist()]


def _heanno_stardist_tier_color_overrides(
    class_names: Sequence[str],
    tier: str,
) -> dict[str, tuple[float, float, float, float]] | None:
    """Resolve the canonical package palette for one StarDist tier."""
    import sys
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parents[1] / "Hist2Pheno_pkg"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    from plotting_palettes import stardist_tier_rgba_overrides  # noqa: WPS433

    return stardist_tier_rgba_overrides(class_names, tier)

########################################################
## 2026.07.05 LLY: plot the predicted cell type spatial distribution from the label h5ad [Incomplete_Cases]
########################################################
def _plot_stardist_tier_spatial_panel(
    plot_mod,
    *,
    pred_encoded: np.ndarray,
    class_names: Sequence[str],
    coords: np.ndarray,
    color_overrides: dict[str, tuple[float, float, float, float]] | None,
    spatial_color_scheme: str,
    spatial_point_size: float,
    fig_size: tuple[float, float],
    save_path_pred: str | Path | None,
    title_pred: str,
    show: bool,
    show_grid: bool = False,
    axes_size: tuple[float, float] | None = None,
) -> None:
    """Pred-only tier spatial panel via ``plot_celltype_spatial_distribution``."""
    names = list(class_names)
    n_cls = len(names)
    pred_enc = np.asarray(pred_encoded, dtype=np.int64)
    X_coords_all = np.asarray(coords, dtype=np.float64)
    n = min(len(X_coords_all), len(pred_enc))
    X_coords_all = X_coords_all[:n]
    pred_enc = pred_enc[:n]

    valid = (pred_enc >= 0) & (pred_enc < n_cls)
    if not np.all(valid):
        n_bad = int((~valid).sum())
        print(f"  ⚠ Dropping {n_bad} out-of-range tier predictions for spatial plot.")
    X_coords_all = X_coords_all[valid]
    pred_enc = pred_enc[valid]

    pred_names = np.array(
        [names[int(i)] if 0 <= int(i) < n_cls else f"Unknown_{int(i)}" for i in pred_enc]
    )
    pred_df = pd.DataFrame(
        {
            "X_pix_HE": X_coords_all[:, 0],
            "Y_pix_HE": X_coords_all[:, 1],
            "celltype": pred_names,
        }
    )
    panel_size = axes_size if axes_size is not None else None
    plot_kwargs = dict(
        x_col="X_pix_HE",
        y_col="Y_pix_HE",
        celltype_col="celltype",
        alpha=0.6,
        s=spatial_point_size,
        format="jpg",
        save_path=str(save_path_pred) if save_path_pred is not None else None,
        spatial_color_scheme=spatial_color_scheme,
        color_overrides=color_overrides,
        title=title_pred,
        show=show,
    )
    if panel_size is not None:
        plot_kwargs["axes_size"] = panel_size
    else:
        plot_kwargs["figsize"] = fig_size
    import inspect

    plot_fn = plot_mod.plot_celltype_spatial_distribution
    sig = inspect.signature(plot_fn)
    if "show_grid" in sig.parameters:
        plot_kwargs["show_grid"] = show_grid
    if panel_size is not None and "axes_size" not in sig.parameters:
        plot_kwargs.pop("axes_size", None)
        plot_kwargs["figsize"] = fig_size
    result = plot_fn(pred_df, **plot_kwargs)
    if "show_grid" not in sig.parameters and not show_grid and result is not None:
        _, ax = result
        ax.grid(False)
    if save_path_pred:
        print(f"  Saved predicted spatial plot: {save_path_pred}")
########################################################

def resolve_he_manual_annotations_csv(
    sample: str,
    *,
    cases_root: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> Path:
    """
    Resolve ``{sample}/{sample}-HE_annotations.csv`` under *cases_root*.

    Manual ROI CSV columns: ``annotation_name``, ``wkt`` (HE pixel polygon), …
    """
    if csv_path is not None:
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"HE manual annotations CSV not found: {path}")
        return path
    root = Path(cases_root or DEFAULT_COMPLETE_CASES_ROOT).expanduser().resolve()
    path = root / sample / f"{sample}-HE_annotations.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"HE manual annotations CSV not found: {path}\n"
            f"Expected {{sample}}/{{sample}}-HE_annotations.csv with WKT polygons."
        )
    return path


def load_he_manual_annotations(
    csv_path: str | Path,
    *,
    annotation_name_col: str = "annotation_name",
    wkt_col: str = "wkt",
) -> pd.DataFrame:
    """Load manual HE ROI table (``annotation_name`` + ``wkt`` in HE pixel coords)."""
    path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(path)
    for col in (annotation_name_col, wkt_col):
        if col not in df.columns:
            raise ValueError(f"{path.name}: missing required column {col!r}")
    out = df.copy()
    out[annotation_name_col] = out[annotation_name_col].astype(str)
    out = out[out[annotation_name_col].str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path.name}: no rows with {annotation_name_col!r}")
    return out


def _extract_coords_from_adata(
    adata,
    *,
    obsm_key: str = "spatial_HE",
) -> np.ndarray:
    """Return (n, 2) HE pixel coordinates from AnnData."""
    for key in (obsm_key, "spatial_HE", "spatial"):
        if key in adata.obsm:
            return np.asarray(adata.obsm[key], dtype=np.float64)[:, :2]
    if {"centroid_x", "centroid_y"}.issubset(adata.obs.columns):
        return adata.obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    raise KeyError(
        "AnnData has no obsm['spatial_HE'/'spatial'] and no centroid_x/y in obs."
    )


def extract_stardist_label_tier_predictions_from_adata(
    adata,
    *,
    heads: Sequence[str] = STARDIST_ALL_LABEL_TIER_ORDER,
    obsm_key: str = "spatial_HE",
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """
    Argmax tier predictions from an in-memory label h5ad (or ROI subset).

    Same return shape as ``load_stardist_all_label_tier_predictions``.
    """
    names_map = adata.uns.get("pred_prob_class_names")
    if not names_map:
        raise KeyError("AnnData missing uns['pred_prob_class_names']")
    coords = _extract_coords_from_adata(adata, obsm_key=obsm_key)
    tier_data: dict[str, dict[str, object]] = {}
    for head in heads:
        if head not in names_map:
            raise KeyError(
                f"head {head!r} not in pred_prob_class_names; "
                f"available: {list(names_map.keys())}"
            )
        class_names = [str(x) for x in names_map[head]]
        prob_cols = [f"{head}_prob_{j}" for j in range(len(class_names))]
        missing = [c for c in prob_cols if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"missing prob columns {missing[:5]} ...")
        probs = adata.obs[prob_cols].to_numpy(dtype=np.float64)
        tier_data[head] = {
            "class_names": class_names,
            "pred_encoded": probs.argmax(axis=1).astype(np.int64),
        }
    return coords, tier_data


def subset_adata_by_he_manual_rois(
    adata,
    annotations_df: pd.DataFrame,
    *,
    annotation_name_col: str = "annotation_name",
    wkt_col: str = "wkt",
    obsm_key: str = "spatial_HE",
    annotation_names: Sequence[str] | None = None,
    min_cells: int = 1,
) -> dict[str, object]:
    """
    Split label h5ad by manual HE WKT polygons (point-in-polygon on ``spatial_HE``).

    Returns ``{annotation_name: AnnData subset}``; empty ROIs are skipped unless
    *min_cells* is 0 (then an empty AnnData is returned).
    """
    from shapely import wkt as shapely_wkt
    from shapely.vectorized import contains

    import anndata as ad

    coords = _extract_coords_from_adata(adata, obsm_key=obsm_key)
    xs = coords[:, 0]
    ys = coords[:, 1]

    df = annotations_df.copy()
    if annotation_names is not None:
        wanted = {str(x) for x in annotation_names}
        df = df[df[annotation_name_col].astype(str).isin(wanted)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No annotation rows left after filtering annotation_names.")

    out: dict[str, object] = {}
    for _, row in df.iterrows():
        roi_name = str(row[annotation_name_col])
        wkt_str = row[wkt_col]
        if pd.isna(wkt_str) or not str(wkt_str).strip():
            print(f"  ⚠ Skip ROI {roi_name!r}: empty WKT")
            continue
        geom = shapely_wkt.loads(str(wkt_str))
        mask = contains(geom, xs, ys)
        n_hit = int(mask.sum())
        print(f"  ROI {roi_name!r}: {n_hit:,} / {len(mask):,} nuclei")
        if n_hit < min_cells:
            print(f"    ⚠ Skip (< {min_cells} cells)")
            continue
        sub = adata[mask].copy()
        sub.obs["he_manual_roi"] = roi_name
        out[roi_name] = sub
    if not out:
        raise ValueError("No nuclei fell inside any manual HE ROI polygon.")
    return out


def summarize_he_manual_roi_subsets(
    roi_adatas: Mapping[str, object],
) -> pd.DataFrame:
    """Tabulate nucleus counts per manual ROI subset."""
    rows = [{"annotation_name": str(name), "n_cells": int(ad.shape[0])} for name, ad in roi_adatas.items()]
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def compute_roi_fig_size_from_coords(
    coords: np.ndarray,
    *,
    base: float = 8.0,
    min_side: float = 5.0,
    max_side: float = 14.0,
) -> tuple[float, float]:
    """
    Pick scatter **panel** size (inches) from ROI point-cloud aspect ratio.

    Legend width is excluded; use with ``plot_celltype_spatial_distribution(axes_size=...)``.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        return (base, base)
    x_range = float(arr[:, 0].max() - arr[:, 0].min())
    y_range = float(arr[:, 1].max() - arr[:, 1].min())
    if x_range <= 0 or y_range <= 0:
        return (base, base)
    aspect = x_range / y_range
    if aspect >= 1.0:
        w = float(np.clip(base * aspect, min_side, max_side))
        h = float(np.clip(base, min_side, max_side))
    else:
        w = float(np.clip(base, min_side, max_side))
        h = float(np.clip(base / aspect, min_side, max_side))
    return (w, h)


def _resolve_roi_fig_size(
    roi_name: str,
    coords: np.ndarray,
    *,
    fig_size: tuple[float, float],
    fig_size_by_roi: Mapping[str, Sequence[float]] | None,
    auto_fig_size: bool,
) -> tuple[float, float]:
    """Per-ROI scatter panel size (inches, excl. legend): dict → auto → default."""
    if fig_size_by_roi and roi_name in fig_size_by_roi:
        fs = tuple(float(x) for x in fig_size_by_roi[roi_name][:2])
        return (fs[0], fs[1])
    if auto_fig_size:
        return compute_roi_fig_size_from_coords(coords)
    return fig_size


def _plot_stardist_tier_data_spatial_maps(
    coords: np.ndarray,
    tier_data: Mapping[str, Mapping[str, object]],
    *,
    sample_label: str,
    tiers: Sequence[str],
    save_path: str | Path | None = None,
    roi_label: str | None = None,
    spatial_point_size: float = 0.08,
    fig_size: tuple[float, float] = (10.0, 8.0),
    show: bool = True,
    show_grid: bool = False,
) -> dict[str, Path | None]:
    """Shared five-tier spatial plotting loop."""
    plot_mod = _hist2pheno_plot_module()
    saved: dict[str, Path | None] = {}
    title_prefix = sample_label if roi_label is None else f"{sample_label} — {roi_label}"
    for tier in tiers:
        if tier not in tier_data:
            raise KeyError(f"Tier {tier!r} not loaded; available: {list(tier_data)}")
        info = tier_data[tier]
        class_names = info["class_names"]
        pred_encoded = info["pred_encoded"]
        display = _STARDIST_TIER_DISPLAY.get(tier, tier.upper())
        tier_save_path = None
        if save_path is not None:
            tier_save_path = _resolve_stardist_tier_save_path(
                save_path,
                tier,
                sample_label=sample_label,
                roi_label=roi_label,
            )
        color_overrides = _heanno_stardist_tier_color_overrides(class_names, tier)
        _plot_stardist_tier_spatial_panel(
            plot_mod,
            pred_encoded=pred_encoded,
            class_names=class_names,
            coords=coords,
            color_overrides=color_overrides,
            spatial_color_scheme=_STARDIST_TIER_SPATIAL_SCHEMES.get(
                tier, "xenium_lung_fine"
            ),
            spatial_point_size=spatial_point_size,
            fig_size=fig_size,
            save_path_pred=tier_save_path,
            title_pred=f"{title_prefix} StarDist pred {display}",
            show=show,
            show_grid=show_grid,
            axes_size=fig_size if roi_label is not None else None,
        )
        saved[tier] = tier_save_path
    return saved


def plot_stardist_all_label_spatial_tiers(
    h5ad_path: str | Path,
    *,
    sample: str | None = None,
    tiers: Sequence[str] = STARDIST_ALL_LABEL_TIER_ORDER,
    save_path: str | Path | None = None,
    spatial_point_size: float = 0.08,
    fig_size: tuple[float, float] = (10.0, 8.0),
    show: bool = True,
    show_grid: bool = False,
    obsm_key: str = "spatial_HE",
) -> dict[str, Path | None]:
    """
    Pred-only five-tier spatial maps from label h5ad.

    Uses ``Hist2Pheno_pkg.plot.plot_tier_spatial_distribution`` — the same helper
    as ``plot_stardist_spatial_extra`` in ``Lung_train_validate_cv_UNIlabel.py``.

    Colors match HE-annotation GT spatial maps through the canonical
    ``plotting_palettes`` registry (lineage, sublineage, CNiche, TNiche, and
    stable ``final_CT`` colors).

    When ``save_path`` is set, writes one file per tier under a subdirectory:

    - ``.../spatial_tiers`` → ``.../spatial_tiers/{sample}_stardist_pred_l2.jpg``, etc.
    - ``.../spatial_tiers.jpg`` → same layout (filename stem is the folder name).
    """
    coords, tier_data = load_stardist_all_label_tier_predictions(
        h5ad_path,
        sample=sample,
        heads=tiers,
        obsm_key=obsm_key,
    )
    sample_label = sample or Path(h5ad_path).stem.split("_all_features")[0]
    return _plot_stardist_tier_data_spatial_maps(
        coords,
        tier_data,
        sample_label=sample_label,
        tiers=tiers,
        save_path=save_path,
        spatial_point_size=spatial_point_size,
        fig_size=fig_size,
        show=show,
        show_grid=show_grid,
    )


def plot_stardist_all_label_spatial_tiers_by_he_rois(
    h5ad_path: str | Path,
    *,
    annotations_csv: str | Path | None = None,
    sample: str | None = None,
    cases_root: str | Path | None = None,
    annotation_names: Sequence[str] | None = None,
    tiers: Sequence[str] = STARDIST_ALL_LABEL_TIER_ORDER,
    save_path: str | Path | None = None,
    spatial_point_size: float = 0.08,
    fig_size: tuple[float, float] = (10.0, 8.0),
    fig_size_by_roi: Mapping[str, Sequence[float]] | None = None,
    auto_fig_size: bool = True,
    show: bool = True,
    show_grid: bool = False,
    obsm_key: str = "spatial_HE",
    min_cells: int = 1,
) -> dict[str, dict[str, Path | None]]:
    """
    Five-tier spatial maps per manual HE ROI from ``{sample}-HE_annotations.csv``.

    1. Load label h5ad (all StarDist nuclei).
    2. Point-in-polygon filter using WKT ROIs (``Large_Airway``, …).
    3. Plot L2/L1/L12/L3/L4 spatial maps for each ROI subset.

    *fig_size_by_roi* supplies explicit ``{roi_name: (panel_w, panel_h)}`` in inches for
    the **scatter panel only** (legend sits in extra margin; not counted in panel size).
    When *auto_fig_size* is True (default), each ROI uses an aspect-matched panel size from
    its nucleus coordinates unless overridden in *fig_size_by_roi*.

    When ``save_path`` is set:

    - ``.../spatial_tiers`` → ``.../spatial_tiers/{roi}/{sample}_stardist_pred_l2.jpg``
    """
    import anndata as ad

    path = Path(h5ad_path).expanduser().resolve()
    sample_label = sample or path.stem.split("_all_features")[0]
    if sample is None:
        sample = sample_label

    anno_path = resolve_he_manual_annotations_csv(
        sample,
        cases_root=cases_root,
        csv_path=annotations_csv,
    )
    annotations_df = load_he_manual_annotations(anno_path)
    print(f"Manual HE ROIs: {anno_path}")
    validate_h5ad_or_raise(path, sample=sample)
    adata = ad.read_h5ad(str(path))

    roi_adatas = subset_adata_by_he_manual_rois(
        adata,
        annotations_df,
        obsm_key=obsm_key,
        annotation_names=annotation_names,
        min_cells=min_cells,
    )
    summary = summarize_he_manual_roi_subsets(roi_adatas)
    print(summary.to_string(index=False))

    saved_all: dict[str, dict[str, Path | None]] = {}
    for roi_name, roi_adata in roi_adatas.items():
        print(f"\n=== Spatial tiers: {sample_label} / {roi_name} ===")
        coords, tier_data = extract_stardist_label_tier_predictions_from_adata(
            roi_adata,
            heads=tiers,
            obsm_key=obsm_key,
        )
        roi_fig_size = _resolve_roi_fig_size(
            roi_name,
            coords,
            fig_size=fig_size,
            fig_size_by_roi=fig_size_by_roi,
            auto_fig_size=auto_fig_size,
        )
        print(f"  panel_size={roi_fig_size[0]:.1f} × {roi_fig_size[1]:.1f} in (excl. legend)")
        saved_all[roi_name] = _plot_stardist_tier_data_spatial_maps(
            coords,
            tier_data,
            sample_label=sample_label,
            tiers=tiers,
            save_path=save_path,
            roi_label=roi_name,
            spatial_point_size=spatial_point_size,
            fig_size=roi_fig_size,
            show=show,
            show_grid=show_grid,
        )
    return saved_all
########################################################

########################################################

def load_stardist_all_label_with_spatial(
    sample: str,
    *,
    data_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    head: str = "l2",
    obsm_key: str = "spatial_HE",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, StardistAllLabelPaths]:
    """Load all StarDist nuclei from label h5ad. Returns ``(df, class_names, probs, coords, paths)``."""
    paths = stardist_all_label_paths(
        sample,
        data_root=data_root,
        cases_root=cases_root,
        stardist_root=stardist_root,
        save_result=save_result,
    )
    paths.assert_ready()
    df, class_names, probs, coords = load_probs_from_stardist_all_label_h5ad(
        paths.label_h5ad,
        head=head,
        obsm_key=obsm_key,
        sample=sample,
    )
    return df, class_names, probs, coords, paths


def load_stardist_all_label_with_spatial_indices(
    sample: str,
    annotation: pd.DataFrame,
    *,
    data_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    save_result: str = DEFAULT_POOLED_SAVE_RESULT,
    head: str = "l2",
    obsm_key: str = "spatial_HE",
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    prefix: str = "idx_",
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, StardistAllLabelPaths, float]:
    """
    Load all StarDist nuclei label h5ad and attach abundance + spatial niche indices.

    Same index columns as matched AUROC workflow (TLS / FRI / ARI).
    """
    df, class_names, probs, coords, paths = load_stardist_all_label_with_spatial(
        sample,
        data_root=data_root,
        cases_root=cases_root,
        stardist_root=stardist_root,
        save_result=save_result,
        head=head,
        obsm_key=obsm_key,
    )
    sample_um = um_per_pixel
    if sample_um is None:
        sample_um = infer_um_per_he_pixel(paths.cases_root / sample)
    df = add_all_spatial_niche_indices_to_auroc_df(
        df,
        probs,
        class_names,
        coords,
        annotation,
        radius_um=radius_um,
        um_per_pixel=sample_um,
        hotspot_percentile=hotspot_percentile,
        use_entropy=use_entropy,
        entropy_weight=entropy_weight,
        prefix=prefix,
    )
    return df, class_names, probs, coords, paths, sample_um


def summarize_matched_vs_all_nuclei(
    df_matched: pd.DataFrame,
    df_all: pd.DataFrame,
    sample: str,
    *,
    index_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare cell counts and mean indices: matched AUROC subset vs all StarDist nuclei."""
    if index_cols is None:
        index_cols = [
            c
            for c in df_all.columns
            if c.startswith("idx_")
            and c not in ("idx_FRI_ratio", "idx_ARI_ratio", "idx_FRI_spatial", "idx_ARI_spatial")
        ]
    rows = []
    for label, sub in (("matched_auroc", df_matched), ("all_stardist", df_all)):
        row = {"sample": sample, "subset": label, "n_cells": len(sub)}
        for c in index_cols:
            if c in sub.columns:
                row[c] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _hist2pheno_plot_module():
    """Import and reload ``Hist2Pheno_pkg.plot`` (picks up ``show_grid`` etc. after edits)."""
    import importlib
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "Hist2Pheno_pkg"
    pkg_str = str(pkg)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)
    import plot as h2p_plot  # noqa: WPS433

    return importlib.reload(h2p_plot)


def plot_predicted_l2_spatial(
    df: pd.DataFrame,
    sample: str,
    *,
    label_col: str = "predict_l2_final_CT",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    class_names: Sequence[str] | None = None,
    spatial_color_scheme: str = "xenium_lung_fine",
    highlight_index_members: bool = True,
    highlight_cell_types: Sequence[str] | None = None,
    other_label: str = INDEX_HIGHLIGHT_OTHER_LABEL,
    other_color: tuple[float, float, float, float] = DEFAULT_INDEX_OTHER_RGBA,
    point_size: float = 0.15,
    other_point_size: float | None = None,
    alpha: float = 0.55,
    other_alpha: float = 0.35,
    figsize: tuple[float, float] = (12.0, 10.0),
    show: bool = True,
    ax=None,
):
    """
    Spatial scatter colored by argmax Level-2 predicted cell type (all nuclei).

    When ``highlight_index_members=True`` (default), only TLS / FRI / ARI member
    cell types keep their training colors; all other predicted types are grouped
    as ``other`` and drawn in light gray underneath.

    Colors for highlighted types use the canonical Xenium lung fine palette.
    """
    import matplotlib.pyplot as plt

    if label_col not in df.columns:
        raise KeyError(f"Missing column {label_col!r}")

    plot_mod = _hist2pheno_plot_module()
    plot_df = df[[coord_x, coord_y, label_col]].dropna(subset=[coord_x, coord_y]).copy()
    plot_df[label_col] = plot_df[label_col].astype(str)

    highlight_set = (
        set(highlight_cell_types)
        if highlight_cell_types is not None
        else set(index_highlight_cell_types())
    )

    if highlight_index_members:
        plot_col = f"{label_col}_index_highlight"
        plot_df[plot_col] = plot_df[label_col].where(
            plot_df[label_col].isin(highlight_set),
            other_label,
        )
        display_col = plot_col
    else:
        display_col = label_col

    canonical = [str(x) for x in class_names] if class_names is not None else None
    scheme = (spatial_color_scheme or "xenium_lung_fine").lower()
    color_map: dict[str, tuple[float, float, float, float]] = {}
    if highlight_index_members:
        color_map[other_label] = other_color
    xenium_schemes = {
        "xenium_lung_fine", "xenium_lung_intermediate", "xenium_lung_coarse",
        "xenium_lung_cniche", "xenium_lung_tniche",
        "xenium", "xenium_auto", "xenium_ct", "xenium_lineage",
    }
    if canonical is not None and scheme in xenium_schemes:
        overrides = plot_mod.build_xenium_spatial_color_overrides(
            canonical,
            spatial_color_scheme=scheme,
        )
        if overrides:
            target_types = highlight_set if highlight_index_members else plot_df[label_col].unique()
            for ct in target_types:
                ct = str(ct)
                if ct in overrides:
                    t = tuple(overrides[ct])
                    color_map[ct] = (
                        (float(t[0]), float(t[1]), float(t[2]), 1.0)
                        if len(t) == 3
                        else tuple(float(x) for x in t[:4])
                    )

    unique_celltypes = plot_df[display_col].astype(str).value_counts().index.tolist()
    missing = [ct for ct in unique_celltypes if ct not in color_map]
    if missing:
        fixed_color_map = plot_mod.build_ncrt_spatial_color_map("celltype")
        scheme_overrides = plot_mod._resolve_spatial_color_overrides(
            missing,
            None,
            spatial_color_scheme,
            canonical_labels=canonical,
        )
        for ct in missing:
            if scheme_overrides and ct in scheme_overrides:
                t = tuple(scheme_overrides[ct])
                color_map[ct] = (
                    (float(t[0]), float(t[1]), float(t[2]), 1.0)
                    if len(t) == 3
                    else tuple(float(x) for x in t[:4])
                )
            elif ct != other_label:
                color_map[ct] = fixed_color_map.get(ct, (0.5, 0.5, 0.5, 1.0))

    counts = plot_df[display_col].value_counts()
    plot_order = [other_label] if highlight_index_members and other_label in counts.index else []
    plot_order.extend(
        ct for ct in counts.index.astype(str).tolist()
        if ct not in plot_order and ct != other_label
    )

    n_highlight = sum(1 for ct in plot_order if ct != other_label)
    title = (
        f"{sample}: L2 prediction — index members (n={len(plot_df):,}; "
        f"{n_highlight} types highlighted, rest={other_label})"
        if highlight_index_members
        else f"{sample}: predicted Level-2 cell type (all StarDist nuclei, n={len(df):,})"
    )

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    o_size = other_point_size if other_point_size is not None else point_size
    for celltype in plot_order:
        sub = plot_df.loc[plot_df[display_col].astype(str) == str(celltype)]
        is_other = highlight_index_members and str(celltype) == other_label
        ax.scatter(
            sub[coord_x],
            sub[coord_y],
            c=[color_map.get(str(celltype), other_color if is_other else (0.5, 0.5, 0.5, 1.0))],
            label=f"{celltype} (n={len(sub):,})",
            alpha=other_alpha if is_other else alpha,
            s=o_size if is_other else point_size,
            zorder=0 if is_other else 1,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=7,
        scatterpoints=1,
        markerscale=4,
        frameon=False,
        ncol=1,
        title="Cell type count",
    )
    if created:
        fig.tight_layout()
    if show and created:
        plt.show()
    elif show and not created:
        plt.show()
    return fig, ax


def plot_matched_vs_all_spatial_overlay(
    df_matched: pd.DataFrame,
    df_all: pd.DataFrame,
    sample: str,
    *,
    value_col: str = "idx_TLS_spatial",
    coord_x: str = "coord_x",
    coord_y: str = "coord_y",
    point_size_all: float = 0.08,
    point_size_matched: float = 0.35,
    figsize: tuple[float, float] = (14.0, 6.0),
    dpi: int = 150,
    save_path: str | Path | None = None,
):
    """Side-by-side spatial maps: all nuclei vs matched AUROC subset for one index."""
    import matplotlib.pyplot as plt

    if value_col not in df_all.columns:
        raise KeyError(f"Missing {value_col!r} in all-nuclei dataframe.")
    vmin, vmax = percentile_vlim(
        df_all[value_col].to_numpy(dtype=np.float64),
        vmin_percentile=1.0,
        vmax_percentile=DEFAULT_TLS_PLOT_VMAX_PERCENTILE,
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, sub, title, psize in (
        (axes[0], df_all, f"{sample}: all nuclei (n={len(df_all):,})", point_size_all),
        (axes[1], df_matched, f"{sample}: matched AUROC (n={len(df_matched):,})", point_size_matched),
    ):
        vals = sub[value_col].to_numpy(dtype=np.float64)
        sc = ax.scatter(
            sub[coord_x], sub[coord_y], c=vals, s=psize, cmap="magma",
            alpha=0.65, vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046)
    axes[0].invert_yaxis()
    fig.suptitle(f"{sample}: {value_col} — full tissue vs matched subset", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes
##################################################################

def summarize_complete_cases_indices(
    annotation: pd.DataFrame,
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    samples: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Abundance-based index summary for all (or selected) Complete_Cases samples."""
    if samples is None:
        samples = discover_complete_cases(
            cases_root, project_suffix=project_suffix, require_h5ad=False
        )
    rows = []
    for sample in samples:
        paths = complete_case_paths(sample, cases_root, project_suffix=project_suffix)
        df, class_names, probs = load_stardist_auroc_csv(
            paths.auroc_csv, paths.class_names_csv
        )
        df_idx = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        row = summarize_indices_by_sample(df_idx.assign(sample=sample)).iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)

##################################################################
# 2026.06.27LLY: add the cross-dataset index summary
##################################################################
def summarize_cross_dataset_indices(
    annotation: pd.DataFrame,
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Abundance-based index summary for cross-dataset AUROC predictions."""
    if samples is None:
        samples = discover_cross_dataset_spatial_samples(
            stardist_root, cases_root, require_h5ad=False
        )
    rows = []
    for sample in samples:
        paths = cross_dataset_spatial_paths(
            sample,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
        )
        df, class_names, probs = load_stardist_auroc_csv(
            paths.auroc_csv, paths.class_names_csv
        )
        df_idx = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        row = summarize_indices_by_sample(df_idx.assign(sample=sample)).iloc[0].to_dict()
        row["prediction_source"] = PREDICTION_SOURCE_CROSS_DATASET
        row["auroc_csv"] = str(paths.auroc_csv)
        row["model_checkpoint"] = str(paths.model_checkpoint)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cross_dataset_spatial_tls(
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spatial TLS summary using cross-dataset AUROC + Complete_Cases coordinates."""
    if samples is None:
        samples = discover_cross_dataset_spatial_samples(
            stardist_root, cases_root, require_h5ad=True
        )
    rows = []
    for sample in samples:
        df, class_names, probs, coords, paths = load_cross_dataset_sample_with_spatial(
            sample,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
        )
        sample_um = um_per_pixel
        if sample_um is None:
            sample_um = infer_um_per_he_pixel(
                paths.stardist_h5ad.parent, h5ad_path=paths.stardist_h5ad
            )
        if annotation is not None:
            df = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        df = add_spatial_tls_to_auroc_df(
            df,
            probs,
            class_names,
            coords,
            radius_um=radius_um,
            um_per_pixel=sample_um,
            hotspot_percentile=hotspot_percentile,
            use_entropy=use_entropy,
            entropy_weight=entropy_weight,
        )
        hs = summarize_tls_hotspots(df)
        row = {
            "sample": sample,
            "prediction_source": PREDICTION_SOURCE_CROSS_DATASET,
            "n_cells": hs["n_cells_total"],
            "n_tls_candidates": hs["n_candidate_cells"],
            "fraction_tls_candidates": hs["fraction_candidate"],
            "max_idx_TLS_spatial": hs["max_tls_spatial"],
            "mean_idx_TLS_spatial_candidate": hs["mean_tls_spatial_candidate"],
            "tls_spatial_threshold": float(df["tls_spatial_threshold"].iloc[0]),
            "tls_spatial_radius_um": float(df["tls_spatial_radius_um"].iloc[0]),
            "tls_spatial_radius_px": float(df["tls_spatial_radius_px"].iloc[0]),
            "tls_um_per_he_pixel": float(df["tls_um_per_he_pixel"].iloc[0]),
            "tls_hotspot_centroid_x": hs["centroid_x"],
            "tls_hotspot_centroid_y": hs["centroid_y"],
            "auroc_csv": str(paths.auroc_csv),
            "stardist_h5ad": str(paths.stardist_h5ad),
            "model_checkpoint": str(paths.model_checkpoint),
        }
        if "idx_TLS" in df.columns:
            row["mean_idx_TLS"] = float(df["idx_TLS"].mean())
            row["mean_idx_TLS_spatial"] = float(df["idx_TLS_spatial"].mean())
        if use_entropy and "idx_TLS_spatial_combined" in df.columns:
            row["mean_idx_TLS_compartment_entropy"] = float(
                df["idx_TLS_compartment_entropy"].mean()
            )
            row["mean_idx_TLS_spatial_combined"] = float(
                df["idx_TLS_spatial_combined"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sample_indices(
    annotation: pd.DataFrame,
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Abundance index cohort summary for cross-dataset (default) or per-sample predictions."""
    if prediction_source == PREDICTION_SOURCE_PER_SAMPLE:
        out = summarize_complete_cases_indices(
            annotation,
            cases_root,
            project_suffix=project_suffix,
            samples=samples,
        )
        if len(out):
            out = out.copy()
            out["prediction_source"] = PREDICTION_SOURCE_PER_SAMPLE
        return out
    if prediction_source == PREDICTION_SOURCE_CROSS_DATASET:
        return summarize_cross_dataset_indices(
            annotation,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
            samples=samples,
        )
    raise ValueError(
        f"Unknown prediction_source={prediction_source!r}; "
        f"expected {PREDICTION_SOURCE_PER_SAMPLE!r} or {PREDICTION_SOURCE_CROSS_DATASET!r}."
    )


def summarize_sample_spatial_tls(
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spatial TLS cohort summary for cross-dataset (default) or per-sample predictions."""
    if prediction_source == PREDICTION_SOURCE_PER_SAMPLE:
        out = summarize_complete_cases_spatial_tls(
            cases_root,
            project_suffix=project_suffix,
            samples=samples,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
            hotspot_percentile=hotspot_percentile,
            use_entropy=use_entropy,
            entropy_weight=entropy_weight,
            annotation=annotation,
        )
        if len(out):
            out = out.copy()
            out["prediction_source"] = PREDICTION_SOURCE_PER_SAMPLE
        return out
    if prediction_source == PREDICTION_SOURCE_CROSS_DATASET:
        return summarize_cross_dataset_spatial_tls(
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
            samples=samples,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
            hotspot_percentile=hotspot_percentile,
            use_entropy=use_entropy,
            entropy_weight=entropy_weight,
            annotation=annotation,
        )
    raise ValueError(
        f"Unknown prediction_source={prediction_source!r}; "
        f"expected {PREDICTION_SOURCE_PER_SAMPLE!r} or {PREDICTION_SOURCE_CROSS_DATASET!r}."
    )


def summarize_sample_spatial_fri_ari(
    *,
    prediction_source: PredictionSource = PREDICTION_SOURCE_DEFAULT,
    cases_root: str | Path | None = None,
    stardist_root: str | Path | None = None,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spatial FRI / ARI cohort summary for cross-dataset (default) or per-sample predictions."""
    if prediction_source == PREDICTION_SOURCE_PER_SAMPLE:
        out = summarize_complete_cases_spatial_fri_ari(
            cases_root,
            project_suffix=project_suffix,
            samples=samples,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
            hotspot_percentile=hotspot_percentile,
            annotation=annotation,
        )
        if len(out):
            out = out.copy()
            out["prediction_source"] = PREDICTION_SOURCE_PER_SAMPLE
        return out
    if prediction_source == PREDICTION_SOURCE_CROSS_DATASET:
        return summarize_cross_dataset_spatial_fri_ari(
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
            samples=samples,
            radius_um=radius_um,
            um_per_pixel=um_per_pixel,
            hotspot_percentile=hotspot_percentile,
            annotation=annotation,
        )
    raise ValueError(
        f"Unknown prediction_source={prediction_source!r}; "
        f"expected {PREDICTION_SOURCE_PER_SAMPLE!r} or {PREDICTION_SOURCE_CROSS_DATASET!r}."
    )
##################################################################

def summarize_complete_cases_spatial_tls(
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_TLS_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    use_entropy: bool = False,
    entropy_weight: float = 1.0,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Spatial TLS summary for all (or selected) Complete_Cases samples.

    Also includes abundance ``idx_TLS`` when ``annotation`` is provided.
    """
    if samples is None:
        samples = discover_complete_cases(
            cases_root, project_suffix=project_suffix, require_h5ad=True
        )
    rows = []
    for sample in samples:
        df, class_names, probs, coords, paths = load_complete_case_with_spatial(
            sample, cases_root, project_suffix=project_suffix
        )
        sample_um = um_per_pixel
        if sample_um is None:
            sample_um = infer_um_per_he_pixel(
                paths.stardist_h5ad.parent, h5ad_path=paths.stardist_h5ad
            )
        if annotation is not None:
            df = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        df = add_spatial_tls_to_auroc_df(
            df,
            probs,
            class_names,
            coords,
            radius_um=radius_um,
            um_per_pixel=sample_um,
            hotspot_percentile=hotspot_percentile,
            use_entropy=use_entropy,
            entropy_weight=entropy_weight,
        )
        hs = summarize_tls_hotspots(df)
        row = {
            "sample": sample,
            "n_cells": hs["n_cells_total"],
            "n_tls_candidates": hs["n_candidate_cells"],
            "fraction_tls_candidates": hs["fraction_candidate"],
            "max_idx_TLS_spatial": hs["max_tls_spatial"],
            "mean_idx_TLS_spatial_candidate": hs["mean_tls_spatial_candidate"],
            "tls_spatial_threshold": float(df["tls_spatial_threshold"].iloc[0]),
            "tls_spatial_radius_um": float(df["tls_spatial_radius_um"].iloc[0]),
            "tls_spatial_radius_px": float(df["tls_spatial_radius_px"].iloc[0]),
            "tls_um_per_he_pixel": float(df["tls_um_per_he_pixel"].iloc[0]),
            "tls_hotspot_centroid_x": hs["centroid_x"],
            "tls_hotspot_centroid_y": hs["centroid_y"],
            "auroc_csv": str(paths.auroc_csv),
            "stardist_h5ad": str(paths.stardist_h5ad),
        }
        if "idx_TLS" in df.columns:
            row["mean_idx_TLS"] = float(df["idx_TLS"].mean())
            row["mean_idx_TLS_spatial"] = float(df["idx_TLS_spatial"].mean())
        if use_entropy and "idx_TLS_spatial_combined" in df.columns:
            row["mean_idx_TLS_compartment_entropy"] = float(
                df["idx_TLS_compartment_entropy"].mean()
            )
            row["mean_idx_TLS_spatial_combined"] = float(
                df["idx_TLS_spatial_combined"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_complete_cases_spatial_fri_ari(
    cases_root: str | Path | None = None,
    *,
    project_suffix: str = DEFAULT_PROJECT_SUFFIX,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spatial FRI / ARI summary for Complete_Cases samples."""
    if samples is None:
        samples = discover_complete_cases(
            cases_root, project_suffix=project_suffix, require_h5ad=True
        )
    rows = []
    for sample in samples:
        df, class_names, probs, coords, paths = load_complete_case_with_spatial(
            sample, cases_root, project_suffix=project_suffix
        )
        sample_um = um_per_pixel
        if sample_um is None:
            sample_um = infer_um_per_he_pixel(
                paths.stardist_h5ad.parent, h5ad_path=paths.stardist_h5ad
            )
        if annotation is not None:
            df = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        df = add_spatial_fri_ari_to_auroc_df(
            df,
            probs,
            class_names,
            coords,
            radius_um=radius_um,
            um_per_pixel=sample_um,
            hotspot_percentile=hotspot_percentile,
        )
        hs = summarize_fri_ari_hotspots(df)
        row = {
            "sample": sample,
            "n_cells": hs["FRI"]["n_cells_total"],
            "n_fri_candidates": hs["FRI"]["n_candidate_cells"],
            "fraction_fri_candidates": hs["FRI"]["fraction_candidate"],
            "max_idx_FRI_spatial": hs["FRI"]["max_idx_FRI_spatial"],
            "mean_idx_FRI_spatial_candidate": hs["FRI"]["mean_idx_FRI_spatial_candidate"],
            "fri_spatial_threshold": float(df["fri_spatial_threshold"].iloc[0]),
            "n_ari_candidates": hs["ARI"]["n_candidate_cells"],
            "fraction_ari_candidates": hs["ARI"]["fraction_candidate"],
            "max_idx_ARI_spatial": hs["ARI"]["max_idx_ARI_spatial"],
            "mean_idx_ARI_spatial_candidate": hs["ARI"]["mean_idx_ARI_spatial_candidate"],
            "ari_spatial_threshold": float(df["ari_spatial_threshold"].iloc[0]),
            "fri_ari_spatial_radius_um": float(df["fri_ari_spatial_radius_um"].iloc[0]),
            "fri_ari_spatial_radius_px": float(df["fri_ari_spatial_radius_px"].iloc[0]),
            "fri_ari_um_per_he_pixel": float(df["fri_ari_um_per_he_pixel"].iloc[0]),
            "auroc_csv": str(paths.auroc_csv),
            "stardist_h5ad": str(paths.stardist_h5ad),
        }
        if "idx_FRI_ratio" in df.columns:
            row["mean_idx_FRI_ratio"] = float(df["idx_FRI_ratio"].mean())
            row["mean_idx_FRI_spatial"] = float(df["idx_FRI_spatial"].mean())
        if "idx_ARI_ratio" in df.columns:
            row["mean_idx_ARI_ratio"] = float(df["idx_ARI_ratio"].mean())
            row["mean_idx_ARI_spatial"] = float(df["idx_ARI_spatial"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cross_dataset_spatial_fri_ari(
    *,
    stardist_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    model_checkpoint: str | Path | None = None,
    samples: Sequence[str] | None = None,
    radius_um: float = DEFAULT_FRI_SPATIAL_RADIUS_UM,
    um_per_pixel: float | None = None,
    hotspot_percentile: float = DEFAULT_TLS_HOTSPOT_PERCENTILE,
    annotation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spatial FRI / ARI summary using cross-dataset AUROC + Complete_Cases coordinates."""
    if samples is None:
        samples = discover_cross_dataset_spatial_samples(
            stardist_root, cases_root, require_h5ad=True
        )
    rows = []
    for sample in samples:
        df, class_names, probs, coords, paths = load_cross_dataset_sample_with_spatial(
            sample,
            stardist_root=stardist_root,
            cases_root=cases_root,
            model_checkpoint=model_checkpoint,
        )
        sample_um = um_per_pixel
        if sample_um is None:
            sample_um = infer_um_per_he_pixel(
                paths.stardist_h5ad.parent, h5ad_path=paths.stardist_h5ad
            )
        if annotation is not None:
            df = add_histology_indices_to_auroc_df(df, probs, class_names, annotation)
        df = add_spatial_fri_ari_to_auroc_df(
            df,
            probs,
            class_names,
            coords,
            radius_um=radius_um,
            um_per_pixel=sample_um,
            hotspot_percentile=hotspot_percentile,
        )
        hs = summarize_fri_ari_hotspots(df)
        row = {
            "sample": sample,
            "prediction_source": PREDICTION_SOURCE_CROSS_DATASET,
            "n_cells": hs["FRI"]["n_cells_total"],
            "n_fri_candidates": hs["FRI"]["n_candidate_cells"],
            "fraction_fri_candidates": hs["FRI"]["fraction_candidate"],
            "max_idx_FRI_spatial": hs["FRI"]["max_idx_FRI_spatial"],
            "mean_idx_FRI_spatial_candidate": hs["FRI"]["mean_idx_FRI_spatial_candidate"],
            "fri_spatial_threshold": float(df["fri_spatial_threshold"].iloc[0]),
            "n_ari_candidates": hs["ARI"]["n_candidate_cells"],
            "fraction_ari_candidates": hs["ARI"]["fraction_candidate"],
            "max_idx_ARI_spatial": hs["ARI"]["max_idx_ARI_spatial"],
            "mean_idx_ARI_spatial_candidate": hs["ARI"]["mean_idx_ARI_spatial_candidate"],
            "ari_spatial_threshold": float(df["ari_spatial_threshold"].iloc[0]),
            "fri_ari_spatial_radius_um": float(df["fri_ari_spatial_radius_um"].iloc[0]),
            "fri_ari_spatial_radius_px": float(df["fri_ari_spatial_radius_px"].iloc[0]),
            "fri_ari_um_per_he_pixel": float(df["fri_ari_um_per_he_pixel"].iloc[0]),
            "auroc_csv": str(paths.auroc_csv),
            "stardist_h5ad": str(paths.stardist_h5ad),
            "model_checkpoint": str(paths.model_checkpoint),
        }
        if "idx_FRI_ratio" in df.columns:
            row["mean_idx_FRI_ratio"] = float(df["idx_FRI_ratio"].mean())
            row["mean_idx_FRI_spatial"] = float(df["idx_FRI_spatial"].mean())
        if "idx_ARI_ratio" in df.columns:
            row["mean_idx_ARI_ratio"] = float(df["idx_ARI_ratio"].mean())
            row["mean_idx_ARI_spatial"] = float(df["idx_ARI_spatial"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def discover_pooled_auroc_concat(result_root: str | Path) -> Path | None:
    """Return pooled concat CSV if present under ``Data/{result_root}/``."""
    root = Path(result_root).expanduser().resolve()
    p = root / "validation_external_stardist_matched_AUROC_all_samples_concat.csv"
    return p if p.is_file() else None


def load_pooled_or_single_auroc(
    auroc_csv: str | Path,
    class_names_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Convenience loader (single sample or pre-concatenated pooled CSV)."""
    return load_stardist_auroc_csv(auroc_csv, class_names_csv)


def annotation_index_notes(annotation: pd.DataFrame) -> pd.DataFrame:
    """Extract non-empty ``Index`` formula strings from the annotation sheet."""
    if "Index" not in annotation.columns:
        return pd.DataFrame(columns=["final_CT", "Index"])
    mask = annotation["Index"].notna() & (annotation["Index"].astype(str).str.strip() != "")
    return annotation.loc[mask, ["final_CT", "Niches", "Functions", "Index"]].reset_index(drop=True)
