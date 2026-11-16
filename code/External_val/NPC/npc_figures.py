"""NPC figure helpers for the Pred_visual notebooks.

Sample ids, paths, and display settings stay in the notebook. This module holds
the functions those cells used to define inline: PDF/PNG export, the
cell2location grouping, and the expert-TLS outline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.path import Path as MplPath
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance_matrix
from skimage.draw import disk
from skimage.measure import find_contours

__all__ = [
    "C2L_TYPES",
    "L2_TO_C2L",
    "STACK_ORDER",
    "TLS_EDGE",
    "TLS_FILL",
    "c2l_palette",
    "configure_figure_dpi",
    "draw_spot_circles",
    "draw_tls",
    "group_fraction",
    "keep_leftmost_foci",
    "legend_handles",
    "load_tls_xy",
    "map_l2",
    "nuclei_in_contours",
    "save_figure",
    "l2_group_means",
    "load_aligned_l2_probs",
    "plot_abundance_maps",
    "plot_nucleus_abundance_confidence",
    "spline_closed",
    "tls_score_row",
    "tls_smooth_contours",
]

# cell2location types. Hist2Pheno L2 names that are not in this table become "other".
C2L_TYPES = ("tumor", "normal", "fibroblast", "T", "Treg", "B", "Myeloid")
STACK_ORDER = C2L_TYPES + ("other",)
L2_TO_C2L = {
    "Tumor": "tumor",
    "Epithelial": "normal",
    "Alveolar": "normal",
    "Injury_epithelial": "normal",
    "DCIS": "tumor",  # 2026.11.16, in-situ carcinoma is neoplastic, not normal epithelium
    "Myoepithelial": "normal",
    "Fibroblast": "fibroblast",
    "Myofibroblast": "fibroblast",
    "Stromal": "fibroblast",
    "CD4_T": "T",
    "CD8_T": "T",
    "NK": "T",
    "Tregs": "Treg",
    "B_cell": "B",
    "Plasma": "B",
    "Macrophage": "Myeloid",
    "DC": "Myeloid",
    "Neutrophil": "Myeloid",
}

TLS_EDGE = "black"
TLS_FILL = (0.0, 0.0, 0.0, 0.16)

_SAVE_DPI = 300
_PLOT_DPI = 200


def configure_figure_dpi(*, save_dpi: int | None = None, plot_dpi: int | None = None) -> None:
    """Point export and TLS spot markers at the notebook ``SAVE_DPI`` / ``DPI``."""
    global _SAVE_DPI, _PLOT_DPI
    if save_dpi is not None:
        _SAVE_DPI = int(save_dpi)
    if plot_dpi is not None:
        _PLOT_DPI = int(plot_dpi)


def save_figure(fig, path, *, dpi=None):
    """Write PDF (editable text) and a PNG preview."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(_SAVE_DPI if dpi is None else dpi)
    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(png, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(pdf)
    return pdf


def c2l_palette(l2_palette: Mapping) -> dict:
    """Colour cell2location groups with the matching pan-cancer L2 colours."""
    return {
        "tumor": l2_palette["Tumor"],
        "normal": l2_palette["Epithelial"],
        "fibroblast": l2_palette["Fibroblast"],
        "T": l2_palette["CD4_T"],
        "Treg": l2_palette["Tregs"],
        "B": l2_palette["B_cell"],
        "Myeloid": l2_palette["Macrophage"],
        "other": (0.62, 0.62, 0.62, 1.0),
    }


def map_l2(names, mapping: Mapping[str, str] | None = None) -> np.ndarray:
    table = L2_TO_C2L if mapping is None else mapping
    return np.array([table.get(str(n), "other") for n in names], dtype=object)


def group_fraction(labels: np.ndarray, order: Sequence[str] = STACK_ORDER) -> pd.Series:
    vc = pd.Series(labels).value_counts(normalize=True)
    return pd.Series({k: float(vc.get(k, 0.0)) for k in order})


def legend_handles(names, palette, markersize=8):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=(
                palette[name][:3] if not isinstance(palette[name], str) else palette[name]
            ),
            markeredgecolor="none",
            markersize=markersize,
            label=name,
        )
        for name in names
        if name in palette
    ]


def draw_spot_circles(ax, xy, labels, radius, palette, order):
    patches, colors = [], []
    labels = np.asarray(labels)
    for name in order:
        hit = labels == name
        if not np.any(hit):
            continue
        col = palette[name]
        rgb = col[:3] if not isinstance(col, str) else col
        for x, y in np.asarray(xy)[hit]:
            patches.append(Circle((x, y), radius))
            colors.append(rgb)
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none"))


def load_tls_xy(tls_path, visium_dir):
    raw = pd.read_csv(tls_path)
    barcodes = raw.iloc[:, -1].astype(str).tolist()
    pos = pd.read_csv(
        visium_dir / "tissue_positions_list.csv",
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    )
    pos["barcode"] = pos["barcode"].astype(str)
    hit = pos.set_index("barcode").reindex(barcodes)
    if hit["pxl_row"].isna().any():
        missing = hit.index[hit["pxl_row"].isna()].tolist()
        raise KeyError(f"TLS barcodes not in tissue_positions: {missing[:8]}")
    xy = hit[["pxl_col", "pxl_row"]].to_numpy(dtype=np.float64)
    in_t = pos.loc[pos["in_tissue"] == 1, ["pxl_col", "pxl_row"]].to_numpy(dtype=np.float64)
    pitch = float(np.median(cKDTree(in_t).query(in_t, k=2)[0][:, 1]))
    scale = json.loads((visium_dir / "scalefactors_json.json").read_text())
    spot_r = float(scale["spot_diameter_fullres"]) / 2.0
    return xy, np.asarray(barcodes), pitch, spot_r


def keep_leftmost_foci(xy, pitch, n_keep=3, link=2.2):
    """Drop the lower-right TLS clusters; keep the n leftmost connected foci."""
    labels = connected_components(distance_matrix(xy, xy) < link * pitch)[1]
    means = np.array([xy[labels == i, 0].mean() for i in range(int(labels.max()) + 1)])
    keep_ids = set(np.argsort(means)[: int(n_keep)].tolist())
    mask = np.isin(labels, list(keep_ids))
    print(
        f"TLS foci: {int(labels.max()) + 1} total, keep {n_keep} leftmost "
        f"({int(mask.sum())} / {len(xy)} spots)"
    )
    return mask


def spline_closed(poly, n=360, s_scale=12.0):
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 4:
        return p
    if np.linalg.norm(p[0] - p[-1]) > 1e-6:
        p = np.vstack([p, p[0]])
    step = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
    p = p[np.concatenate([[True], step > 1e-3])]
    if len(p) < 4:
        return poly
    tck, _ = splprep([p[:, 0], p[:, 1]], s=s_scale * len(p), per=True, k=3)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    xs, ys = splev(u, tck)
    return np.column_stack([xs, ys])


def tls_smooth_contours(xy, disk_r, sigma, level=0.22, step=None):
    pad = float(disk_r + 4.0 * sigma)
    origin = xy.min(axis=0) - pad
    extent = xy.max(axis=0) + pad
    step = float(max(6.0, disk_r / 10.0) if step is None else step)
    w = int(np.ceil((extent[0] - origin[0]) / step))
    h = int(np.ceil((extent[1] - origin[1]) / step))
    mask = np.zeros((h, w), dtype=np.float32)
    rad = disk_r / step
    for x, y in xy:
        rr, cc = disk(((y - origin[1]) / step, (x - origin[0]) / step), rad, shape=mask.shape)
        mask[rr, cc] = 1.0
    mask = gaussian_filter(mask, sigma=sigma / step)
    out = []
    for contour in find_contours(mask, float(level)):
        xs = contour[:, 1] * step + origin[0]
        ys = contour[:, 0] * step + origin[1]
        out.append(spline_closed(np.column_stack([xs, ys])))
    return out


def draw_tls(
    ax,
    contours,
    spots=None,
    spot_r=None,
    local=(0.0, 0.0),
    lw=2.0,
    fill=False,
    *,
    edge=None,
    fill_rgba=None,
    dpi=None,
):
    edge = TLS_EDGE if edge is None else edge
    fill_rgba = TLS_FILL if fill_rgba is None else fill_rgba
    dpi = _PLOT_DPI if dpi is None else float(dpi)
    x0, y0 = local
    for poly in contours:
        if fill:
            ax.fill(
                poly[:, 0] - x0,
                poly[:, 1] - y0,
                facecolor=fill_rgba,
                edgecolor="none",
                zorder=4,
            )
        ax.plot(
            poly[:, 0] - x0,
            poly[:, 1] - y0,
            color=edge,
            lw=lw,
            solid_capstyle="round",
            zorder=5,
        )
    if spots is not None:
        ax.scatter(
            spots[:, 0] - x0,
            spots[:, 1] - y0,
            s=28 if spot_r is None else (spot_r * 72.0 / float(dpi)) ** 2 * 0.08,
            facecolors="none",
            edgecolors=edge,
            linewidths=0.9,
            zorder=6,
        )


def nuclei_in_contours(xy, contours):
    """Nuclei whose HE coordinates fall inside any black TLS outline."""
    xy = np.asarray(xy, dtype=np.float64)
    inside = np.zeros(len(xy), dtype=bool)
    for poly in contours:
        poly = np.asarray(poly, dtype=np.float64)
        if len(poly) < 3:
            continue
        inside |= MplPath(poly).contains_points(xy)
    return inside

## 2026-11-11 — Per-nucleus L2 confidence (do not only argmax)
def l2_group_means(mean_probs, class_names, groups=None, mapping=None) -> np.ndarray:
    """Sum per-spot mean L2 probabilities into cell2location groups.

    Classes that do not map onto ``groups`` (Endothelial, Neural) are left
    out. The columns are not renormalized, so a row need not sum to 1.
    """
    groups = list(C2L_TYPES if groups is None else groups)
    mapping = L2_TO_C2L if mapping is None else mapping
    mean_probs = np.asarray(mean_probs, dtype=np.float64)
    class_names = [str(name) for name in class_names]
    if mean_probs.ndim != 2 or mean_probs.shape[1] != len(class_names):
        raise ValueError(
            f"mean_probs {mean_probs.shape} does not match {len(class_names)} classes"
        )
    out = np.zeros((len(mean_probs), len(groups)), dtype=np.float64)
    for j, group in enumerate(groups):
        cols = [i for i, name in enumerate(class_names) if mapping.get(name) == group]
        if cols:
            out[:, j] = mean_probs[:, cols].sum(axis=1)
    return out


def load_aligned_l2_probs(pred_h5ad, probs_npz):
    """L2 softmax in the same row order as ``obsm['spatial_HE']``."""
    import anndata as ad

    adata = ad.read_h5ad(pred_h5ad, backed="r")
    try:
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64).copy()
        names = np.asarray(adata.obs_names.astype(str))
    finally:
        closer = getattr(adata, "file", None)
        if closer is not None:
            closer.close()
    with np.load(probs_npz, allow_pickle=True) as z:
        probs = np.asarray(z["probs"], dtype=np.float64)
        classes = [str(name) for name in z["class_names"]]
        obs = np.asarray(z["obs_names"]).astype(str) if "obs_names" in z.files else None
    if probs.ndim != 2 or probs.shape[1] != len(classes):
        raise ValueError(f"{probs_npz} probs {probs.shape} vs {len(classes)} classes")
    if obs is None:
        if len(probs) != len(names):
            raise ValueError(
                f"{probs_npz} has {len(probs)} rows and the prediction has {len(names)}"
            )
    elif not np.array_equal(obs, names):
        order = pd.Index(obs).get_indexer(names)
        if np.any(order < 0):
            raise KeyError("L2 probability rows do not cover the prediction nuclei")
        probs = probs[order]
    return xy, probs, classes


def plot_abundance_maps(spot_xy, h2p, c2l, groups, *, radius, sample, dpi=200):
    """Hist2Pheno mean softmax (top) and cell2location counts (bottom).

    Each panel is scaled to its own 99th percentile. Spots with no value are
    gray. Returns the figure and a per-type Spearman table. The two rows are
    different units, so the rank correlation is the comparison, not the color.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    spot_xy = np.asarray(spot_xy, dtype=np.float64)
    h2p = np.asarray(h2p, dtype=np.float64)
    c2l = np.asarray(c2l, dtype=np.float64)
    groups = list(groups)
    if h2p.shape != c2l.shape or h2p.shape[1] != len(groups):
        raise ValueError(f"h2p {h2p.shape}, c2l {c2l.shape}, groups {len(groups)}")
    if len(spot_xy) != len(h2p):
        raise ValueError(f"spots {len(spot_xy)} != rows {len(h2p)}")

    def _spearman(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        n = int(ok.sum())
        if n < 3:
            return float("nan"), n
        result = spearmanr(a[ok], b[ok])
        rho = result.statistic if hasattr(result, "statistic") else result[0]
        return float(rho), n

    def _fill(ax, values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        colors = np.full((len(values), 4), (0.93, 0.93, 0.93, 1.0))
        ok = np.isfinite(values)
        colors[ok] = mpl.colormaps["magma"](np.clip(values[ok] / vmax, 0.0, 1.0))
        patches = [Circle((float(x), float(y)), radius) for x, y in spot_xy]
        ax.add_collection(
            PatchCollection(patches, facecolors=colors, edgecolors="none", rasterized=True)
        )
        return vmax

    fig, axes = plt.subplots(
        2, len(groups), figsize=(2.15 * len(groups), 5.6), dpi=dpi, squeeze=False,
    )
    rows = []
    for j, group in enumerate(groups):
        rho, n = _spearman(h2p[:, j], c2l[:, j])
        both = np.isfinite(h2p[:, j]) & np.isfinite(c2l[:, j])
        h_vmax = _fill(axes[0, j], h2p[:, j])
        c_vmax = _fill(axes[1, j], c2l[:, j])
        axes[0, j].set_title(f"{group}\nρ = {rho:.2f}" if np.isfinite(rho) else group, fontsize=9)
        axes[0, j].text(
            0.02, 0.98, f"p99 {h_vmax:.2f}", transform=axes[0, j].transAxes,
            ha="left", va="top", fontsize=7, color="0.25",
        )
        axes[1, j].text(
            0.02, 0.98, f"p99 {c_vmax:.2f}", transform=axes[1, j].transAxes,
            ha="left", va="top", fontsize=7, color="0.25",
        )
        rows.append({
            "group": group,
            "spearman": rho,
            "n_spots": n,
            "h2p_p50": float(np.median(h2p[both, j])) if both.any() else float("nan"),
            "h2p_p99": h_vmax,
            "c2l_p50": float(np.median(c2l[both, j])) if both.any() else float("nan"),
            "c2l_p99": c_vmax,
        })
    pad = 2.0 * float(radius)
    for ax in axes.ravel():
        ax.set_xlim(spot_xy[:, 0].min() - pad, spot_xy[:, 0].max() + pad)
        ax.set_ylim(spot_xy[:, 1].max() + pad, spot_xy[:, 1].min() - pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Hist2Pheno\nmean softmax", fontsize=9)
    axes[1, 0].set_ylabel("cell2location\ncounts", fontsize=9)
    fig.suptitle(
        f"{sample}   per-type abundance\n"
        "each panel scaled to its own 99th percentile; gray = no value",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def plot_nucleus_abundance_confidence(
    xy,
    abundance,
    confidence,
    class_names,
    *,
    sample,
    dpi=200,
    marker_s=0.15,
    per_figure=5,
):
    """Patient-level single-nucleus maps, one row per L2 type.

    Left is that type's softmax on ``turbo``, 0–1, the scale UTOPIA uses for
    predicted magnitude. Right is conformal confidence on ``hot``, 0–1.
    Each panel has its own colorbar. Returns one figure per page.
    """
    import matplotlib.pyplot as plt

    xy = np.asarray(xy, dtype=np.float64)
    abundance = np.asarray(abundance, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    class_names = [str(name) for name in class_names]
    if abundance.shape != confidence.shape or abundance.shape[1] != len(class_names):
        raise ValueError(
            f"abundance {abundance.shape}, confidence {confidence.shape}, names {len(class_names)}"
        )
    if len(xy) != len(abundance):
        raise ValueError(f"nuclei {len(xy)} != rows {len(abundance)}")
    winner = abundance.argmax(axis=1)
    pad = 0.02 * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    xlim = (float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad)
    ylim = (float(xy[:, 1].max()) + pad, float(xy[:, 1].min()) - pad)

    def _panel(fig, ax, values, cmap, label):
        order = np.argsort(values, kind="stable")
        sc = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=values[order],
            s=marker_s,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=2)
        return sc

    figs = []
    n_pages = int(np.ceil(len(class_names) / per_figure))
    for page in range(n_pages):
        names = class_names[page * per_figure : (page + 1) * per_figure]
        fig, axes = plt.subplots(
            len(names), 2, figsize=(11.2, 2.85 * len(names)), dpi=dpi,
            squeeze=False, layout="constrained",
        )
        for row, name in enumerate(names):
            j = class_names.index(name)
            abu = abundance[:, j]
            conf = confidence[:, j]
            n_arg = int((winner == j).sum())
            n_hi = int((conf >= 0.5).sum())
            _panel(fig, axes[row, 0], abu, "turbo", "softmax")
            _panel(fig, axes[row, 1], conf, "hot", "confidence")
            axes[row, 0].set_title(f"{name}\nsoftmax   argmax {n_arg:,}", fontsize=9)
            axes[row, 1].set_title(f"{name}\nconfidence   ≥0.5 {n_hi:,}", fontsize=9)
            print(
                f"{name:20}  argmax {n_arg:7,d}  softmax p50 {float(np.median(abu)):.3f} "
                f"p99 {float(np.quantile(abu, 0.99)):.3f}   "
                f"confidence p50 {float(np.median(conf)):.3f}  ≥0.5 {n_hi:,}",
                flush=True,
            )
        fig.suptitle(
            f"{sample}   single nuclei    softmax (turbo, 0–1)  |  confidence (hot, 0–1)",
            fontsize=12,
        )
        figs.append(fig)
    return figs


def tls_score_row(df, sample: str) -> dict:
    """Quantify abundance TLS vs spatial TLS on the same nuclei."""
    a = df["idx_TLS"].to_numpy(dtype=float)
    s = df["idx_TLS_spatial"].to_numpy(dtype=float)
    a_hi = a >= np.percentile(a, 95)
    s_hi = df["tls_candidate"].fillna(False).astype(bool).to_numpy()
    inter = int(np.logical_and(a_hi, s_hi).sum())
    union = int(np.logical_or(a_hi, s_hi).sum())
    n_a = int(a_hi.sum())
    return {
        "sample": sample,
        "n": int(len(df)),
        "TLS_mean": float(np.mean(a)),
        "TLS_median": float(np.median(a)),
        "TLS_p95": float(np.percentile(a, 95)),
        "TLS_max": float(np.max(a)),
        "TLS_spatial_mean": float(np.mean(s)),
        "TLS_spatial_median": float(np.median(s)),
        "TLS_spatial_p95": float(np.percentile(s, 95)),
        "TLS_spatial_max": float(np.max(s)),
        "pearson": float(np.corrcoef(a, s)[0, 1]),
        "spearman": float(pd.Series(a).corr(pd.Series(s), method="spearman")),
        "top5_jaccard": (inter / union) if union else float("nan"),
        "abundance_top5_also_spatial_top5": (inter / n_a) if n_a else float("nan"),
        "TLS_spatial_mean_in_abundance_top5": float(s[a_hi].mean()) if n_a else float("nan"),
        "TLS_mean_in_spatial_top5": float(a[s_hi].mean()) if s_hi.any() else float("nan"),
    }
