"""Pan-cancer L1 / L12 / L2 colors and 4×3 GT / pred / ROC / StarDist-all panels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1] / "Hist2Pheno_pkg"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from plotting_palettes import get_palette  # noqa: E402


def palette_for(classes: dict[str, list[str]]) -> dict[str, dict[str, tuple]]:
    """Name→RGBA for each head; same maps as ``plotting_palettes`` pan-cancer schemes."""
    maps = {
        "L1": get_palette("pancancer", "coarse"),
        "L12": get_palette("pancancer", "intermediate"),
        "L2": get_palette("pancancer", "fine"),
    }
    out = {}
    for key, pal in maps.items():
        names = list(classes[key])
        out[key] = {n: pal.get(n, (0.55, 0.55, 0.55, 1.0)) for n in names}
    return out


def _scatter(ax, xy, labels, names, palette, s: float) -> None:
    colors = np.array(
        [
            palette.get(names[int(k)], (0.55, 0.55, 0.55, 1.0))
            if 0 <= int(k) < len(names)
            else (0.75, 0.75, 0.75, 1.0)
            for k in np.asarray(labels, dtype=np.int64)
        ]
    )
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, alpha=0.75, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def _roc(ax, y_true, probs, names, palette) -> float:
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    n_classes = len(names)
    valid = (y >= 0) & (y < n_classes)
    if not np.any(valid) or p.ndim != 2:
        ax.text(0.5, 0.5, "no GT", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return float("nan")
    y, p = y[valid], p[valid]
    Y = label_binarize(y, classes=np.arange(n_classes))
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    aucs = []
    for k, name in enumerate(names):
        if k >= Y.shape[1]:
            break
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == len(Y):
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], p[:, k], drop_intermediate=True)
        ak = float(auc(fpr, tpr))
        aucs.append(ak)
        col = palette.get(str(name), (0.3, 0.3, 0.3, 1.0))
        ax.plot(fpr, tpr, color=col[:3], lw=1.0, alpha=0.9)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("FPR", fontsize=7)
    ax.set_ylabel("TPR", fontsize=7)
    macro = float(np.mean(aucs)) if aucs else float("nan")
    ax.set_title(f"macro AUROC={macro:.3f}" if np.isfinite(macro) else "ROC", fontsize=8)
    return macro


def plot_section_panel(
    *,
    save_path: Path,
    xy_gt,
    xy_star,
    classes: dict[str, list[str]],
    palettes: dict[str, dict[str, tuple]],
    gt: dict[str, np.ndarray] | None,
    pred: dict[str, np.ndarray],
    probs: dict[str, np.ndarray] | None,
    star: dict[str, np.ndarray] | None,
    title: str,
    point_size: float = 0.6,
    dpi: int = 130,
) -> dict[str, float]:
    """4 rows × 3 cols: GT, matched pred, ROC, StarDist-all; cols L1 / L12 / L2."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    heads = ("L1", "L12", "L2")
    n_leg = max(len(classes[h]) for h in heads)
    extra_h = 0.22 * int(np.ceil(n_leg / 2)) + 0.4
    fig, axes = plt.subplots(
        5,
        3,
        figsize=(12.5, 14.5 + extra_h * 0.35),
        gridspec_kw={"height_ratios": [3.0, 3.0, 2.6, 3.0, extra_h]},
    )
    row_names = ["GT (matched UNI)", "Hist2Pheno pred", "ROC (matched)", "StarDist all nuclei"]
    macros: dict[str, float] = {}
    for j, head in enumerate(heads):
        names = list(classes[head])
        pal = palettes[head]
        key = head
        ax0 = axes[0, j]
        if gt is not None and xy_gt is not None and len(xy_gt):
            _scatter(ax0, xy_gt, gt[key], names, pal, point_size)
        else:
            ax0.axis("off")
            ax0.text(0.5, 0.5, "no GT", ha="center", va="center", transform=ax0.transAxes)
        if j == 0:
            ax0.set_ylabel(row_names[0], fontsize=8)
        ax0.set_title(head, fontsize=11)

        ax1 = axes[1, j]
        if xy_gt is not None and len(xy_gt):
            _scatter(ax1, xy_gt, pred[key], names, pal, point_size)
        else:
            ax1.axis("off")
        if j == 0:
            ax1.set_ylabel(row_names[1], fontsize=8)

        ax2 = axes[2, j]
        if gt is not None and probs is not None and key in probs:
            macros[head] = _roc(ax2, gt[key], probs[key], names, pal)
        else:
            ax2.axis("off")
            macros[head] = float("nan")
        if j == 0:
            ax2.set_ylabel(row_names[2], fontsize=8)

        ax3 = axes[3, j]
        if star is not None and xy_star is not None and len(xy_star):
            _scatter(ax3, xy_star, star[key], names, pal, max(point_size * 0.7, 0.15))
        else:
            ax3.axis("off")
            ax3.text(0.5, 0.5, "no StarDist-all", ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        if j == 0:
            ax3.set_ylabel(row_names[3], fontsize=8)

        axL = axes[4, j]
        axL.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=pal[n][:3],
                markeredgecolor="none",
                label=n,
            )
            for n in names
        ]
        axL.legend(
            handles=handles,
            loc="upper center",
            fontsize=6.5,
            frameon=False,
            ncol=2 if len(names) > 6 else 1,
            columnspacing=0.7,
            handletextpad=0.3,
            borderaxespad=0.0,
        )

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return macros
