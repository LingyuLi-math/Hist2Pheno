"""Assign single nuclei to Visium-style spots.

The capture rule follows ``CTA_align`` (Li et al., npj Precision Oncology 2025,
``data/Visium/BRCA/npj_data/BreastCancer_CTA/CTA_align.R``): each nucleus is
given to the nearest spot center, and is kept only when it falls inside that
spot's capture window. Nuclei outside every window are not captured.

Nuclei in this project already sit on the full-resolution H&E, so there is
no ``scaling_factor`` and no second y-axis flip. Visium and ST spots use a
disc of radius ``spot diameter / 2``. Bins that tile the tissue use the
axis-aligned square from ``CTA_align``, with half-side equal to half the bin.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def nuclei_to_spots(
    nuc_xy,
    spot_xy,
    radius_px,
    *,
    shape="circle",
    labels=None,
    class_names=None,
    values=None,
):
    """Map nuclei onto spots and summarise the nuclei inside each capture area.

    Parameters
    ----------
    nuc_xy, spot_xy :
        ``(n, 2)`` and ``(m, 2)`` coordinates in the same pixel frame.
        Column 0 is x, column 1 is y.
    radius_px :
        Capture radius in pixels. For ``shape="circle"`` this is the disc
        radius. For ``shape="square"`` it is the half-side of the
        axis-aligned window.
    shape :
        ``"circle"`` for Visium and ST spots. ``"square"`` for tiling bins,
        matching the square in ``CTA_align``.
    labels :
        Hard class name for each nucleus. The spot label is the class with
        the most nuclei inside the window.
    class_names :
        Class order used for ``class_counts``. Required when ``labels`` is set.
    values :
        ``(n, k)`` array, for example L2 probabilities. The spot profile is
        the mean of the rows that fall inside the window.

    Returns
    -------
    dict
        ``spot_index`` ``(n,)`` nearest spot for every nucleus.
        ``inside`` ``(n,)`` True when that nucleus is inside the window.
        ``n_nuclei`` ``(m,)`` nuclei kept in each spot.
        ``dominant`` ``(m,)`` majority label, or ``None`` if ``labels`` is omitted.
        Empty spots are ``""``.
        ``class_counts`` ``(m, n_classes)`` hard counts, or ``None``.
        ``mean`` ``(m, k)`` mean of ``values``, or ``None``. Spots with no
        nucleus are zeros.
    """
    nuc_xy = np.asarray(nuc_xy, dtype=np.float64)
    spot_xy = np.asarray(spot_xy, dtype=np.float64)
    if nuc_xy.ndim != 2 or nuc_xy.shape[1] != 2:
        raise ValueError("nuc_xy must have shape (n, 2)")
    if spot_xy.ndim != 2 or spot_xy.shape[1] != 2:
        raise ValueError("spot_xy must have shape (m, 2)")
    radius_px = float(radius_px)
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")

    dist, spot_index = cKDTree(spot_xy).query(nuc_xy, k=1)
    if shape == "circle":
        inside = dist <= radius_px
    elif shape == "square":
        delta = np.abs(nuc_xy - spot_xy[spot_index])
        inside = (delta.max(axis=1) <= radius_px) & np.isfinite(dist)
    else:
        raise ValueError("shape must be 'circle' or 'square'")

    n_spots = len(spot_xy)
    n_nuclei = np.zeros(n_spots, dtype=np.int32)
    if inside.any():
        np.add.at(n_nuclei, spot_index[inside], 1)

    dominant = None
    class_counts = None
    if labels is not None:
        if class_names is None:
            raise ValueError("class_names is required when labels is set")
        labels = np.asarray(labels)
        if len(labels) != len(nuc_xy):
            raise ValueError(f"labels {len(labels)} != nuclei {len(nuc_xy)}")
        names = [str(name) for name in class_names]
        lookup = {name: i for i, name in enumerate(names)}
        missing = sorted(set(map(str, labels)) - set(lookup))
        if missing:
            raise RuntimeError("labels outside class_names: " + ", ".join(missing))
        lab = np.fromiter((lookup[str(x)] for x in labels), dtype=np.int64, count=len(labels))
        class_counts = np.zeros((n_spots, len(names)), dtype=np.int32)
        if inside.any():
            np.add.at(class_counts, (spot_index[inside], lab[inside]), 1)
        dominant = np.array(names, dtype=object)[class_counts.argmax(axis=1)]
        dominant = np.where(n_nuclei > 0, dominant, "")

    mean = None
    if values is not None:
        values = np.asarray(values)
        if len(values) != len(nuc_xy):
            raise ValueError(f"values {len(values)} != nuclei {len(nuc_xy)}")
        if values.ndim == 1:
            values = values[:, None]
        sums = np.zeros((n_spots, values.shape[1]), dtype=np.float64)
        if inside.any():
            np.add.at(sums, spot_index[inside], values[inside])
        mean = np.divide(
            sums, n_nuclei[:, None], out=np.zeros_like(sums), where=n_nuclei[:, None] > 0,
        )

    return {
        "spot_index": spot_index,
        "inside": inside,
        "n_nuclei": n_nuclei,
        "dominant": dominant,
        "class_counts": class_counts,
        "mean": mean,
    }
