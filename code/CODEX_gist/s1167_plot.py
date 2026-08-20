## 2026.08.20 plotting wrappers for s1167 GIST TMA
## Analog of code/CODEX_pdac/s1167_plot.py. Group cores by coverslip (c009/c011/c013).

"""Plotting wrappers for the s1167 CODEX GIST TMA dataset (550 annotated cores)."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent.parent / "Hist2Pheno_pkg"
_PDAC_DIR = Path(__file__).resolve().parent.parent / "CODEX_pdac"
for _p in (_PKG_DIR, _PDAC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from s1167_img_cell_mapping import coverslip_from_acq  # noqa: E402

GIST_COVERSLIP_ORDER = ("c009", "c011", "c013")
GIST_STARDIST_MACRO_AUROC_TIERS: tuple[str, ...] = ("l2", "l12", "l1")


def short_gist_acq_label(sample: str) -> str:
    """Short y-axis / title label: ``c013 reg002``."""
    import re

    sample = str(sample)
    cs = coverslip_from_acq(sample)
    match = re.search(r"reg(\d+)", sample, flags=re.IGNORECASE)
    return f"{cs} reg{match.group(1)}" if match else sample


def _subset_loaded_by_coverslip(
    loaded: Mapping[str, Mapping],
    *,
    max_per_coverslip: int | None,
) -> dict[str, Mapping]:
    if max_per_coverslip is None:
        return dict(loaded)
    counts: dict[str, int] = {}
    subset: dict[str, Mapping] = {}
    for sample, rec in loaded.items():
        cs = coverslip_from_acq(sample)
        n = counts.get(cs, 0)
        if n >= max_per_coverslip:
            continue
        subset[sample] = rec
        counts[cs] = n + 1
    if len(subset) < len(loaded):
        print(
            f"Overview subset: {len(subset)}/{len(loaded)} cores "
            f"({max_per_coverslip} per coverslip)"
        )
    return subset


def plot_gist_stardist_spatial_maps(
    data_root,
    samples: Sequence[str],
    save_result: str = "result_all_spatial",
    *,
    heads: Sequence[str] = GIST_STARDIST_MACRO_AUROC_TIERS,
    pan_organ: str = "codex_gist",
    spatial_point_size: float = 0.25,
    fig_size: tuple[float, float] = (10, 8),
    show: bool = False,
) -> dict[str, dict]:
    """Full-size pred-only L2/L12/L1 maps from annotated StarDist-all label h5ads."""
    from uni_label_cv_helpers import (
        plot_stardist_label_spatial_maps,
        stardist_all_label_h5ad_path,
    )

    return plot_stardist_label_spatial_maps(
        data_root,
        samples,
        save_result,
        label_h5ad_path_fn=stardist_all_label_h5ad_path,
        heads=heads,
        pan_organ=pan_organ,
        spatial_point_size=spatial_point_size,
        fig_size=fig_size,
        show=show,
        title_prefix_fn=short_gist_acq_label,
        missing_error="No GIST StarDist-all label h5ads. Run the §5 inference cell first.",
    )


def plot_gist_stardist_spatial_overview(
    loaded: Mapping[str, Mapping],
    data_root,
    save_result: str = "result_all_spatial",
    *,
    heads: Sequence[str] = GIST_STARDIST_MACRO_AUROC_TIERS,
    pan_organ: str = "codex_gist",
    point_size: float = 0.5,
    show: bool = True,
    save_path=None,
    max_per_coverslip: int | None = 2,
):
    """n×3 overview of GIST StarDist-all predicted spatial maps.

    Defaults to ``max_per_coverslip=2`` so 550 cores do not make a huge grid.
    Pass ``max_per_coverslip=None`` to plot every loaded core.
    """
    from uni_label_cv_helpers import plot_stardist_label_spatial_overview

    plot_loaded = _subset_loaded_by_coverslip(
        loaded, max_per_coverslip=max_per_coverslip,
    )
    if save_path is None:
        save_path = (
            Path(data_root)
            / save_result
            / "stardist"
            / "gist_spatial_pred_overview_l2_l12_l1.jpg"
        )
    return plot_stardist_label_spatial_overview(
        plot_loaded,
        heads=heads,
        pan_organ=pan_organ,
        save_path=save_path,
        point_size=point_size,
        sample_labels={s: short_gist_acq_label(s) for s in plot_loaded},
        suptitle="GIST StarDist-all predicted spatial maps (pred only)",
        show=show,
    )
