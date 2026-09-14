"""Plotting helpers for Xenium CRC / Visium HD notebooks.

Twin of ``code/Xenium_brca/BRCA_plot.py`` (smaller, CRC-specific).
"""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.pyplot import rcParams

from crc_paths import PERIPH_COLORS, on_tissue


def configure_matplotlib() -> None:
    rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 10,
        }
    )


def subsample_df(df: pd.DataFrame, n: int = 80_000, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def legend_handles(palette: dict):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markersize=8,
            label=lab,
        )
        for lab, c in palette.items()
    ]


def scatter_cats(
    ax: Axes,
    df: pd.DataFrame,
    color_col: str,
    palette: dict,
    *,
    s: float = 0.12,
    highlight_s: float | None = None,
    max_bg: int = 180_000,
    bg_color: str = "#efefef",
) -> None:
    on = on_tissue(df)
    labeled = on[color_col].isin(list(palette))
    bg = on.loc[~labeled]
    if len(bg) > max_bg:
        bg = bg.sample(max_bg, random_state=0)
    if len(bg):
        ax.scatter(bg["X"], bg["Y"], s=s, c=bg_color, linewidths=0, rasterized=True)
    hs = s if highlight_s is None else highlight_s
    for lab, color in palette.items():
        sub = on.loc[on[color_col] == lab]
        if sub.empty:
            continue
        ax.scatter(sub["X"], sub["Y"], s=hs, c=color, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_axis_off()


def scatter_periphery(ax: Axes, df: pd.DataFrame, s: float = 0.12) -> None:
    """Draw Tissue, then 50 µm rim, then Tumor so the rim stays visible."""
    on = on_tissue(df)
    for lab in ("Tissue", "50 micron", "Tumor"):
        sub = on.loc[on["Periphery"] == lab]
        if lab == "Tissue" and len(sub) > 180_000:
            sub = sub.sample(180_000, random_state=0)
        if sub.empty:
            continue
        ax.scatter(
            sub["X"], sub["Y"], s=s, c=PERIPH_COLORS[lab], linewidths=0, rasterized=True
        )
    ax.set_aspect("equal")
    ax.set_axis_off()


def scatter_cats_xy(
    ax: Axes,
    df: pd.DataFrame,
    color_col: str,
    palette: dict,
    *,
    x: str = "X",
    y: str = "Y",
    s: float = 0.15,
    max_n: int | None = 180_000,
) -> None:
    """Scatter ``df`` in arbitrary x/y columns (e.g. Xenium visium-scale pixels)."""
    plot_df = df if max_n is None or len(df) <= max_n else df.sample(max_n, random_state=0)
    for lab, color in palette.items():
        sub = plot_df.loc[plot_df[color_col] == lab]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], s=s, c=color, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_axis_off()


def axis_legend(ax: Axes, palette: dict, ncol: int = 2, fontsize: int = 7) -> None:
    ax.legend(
        handles=legend_handles(palette),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
    )
