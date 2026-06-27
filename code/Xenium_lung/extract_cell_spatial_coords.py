## 20260518 LLY
##          conda run -n SeededNTM python code/Xenium_lung/extract_cell_spatial_coords.py
## 20260621 align with Data_process_visual_xenium_all.ipynb (Complete_Cases_Select4)
# cd /home/lingyu/ssd2/Python/Collaborate/esccAI
# conda run -n SeededNTM python code/Xenium_lung/extract_cell_spatial_coords.py    # default: Complete_Cases_Select4, for example

# conda run -n SeededNTM python code/Xenium_lung/extract_cell_spatial_coords.py \
#   --data-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases    # Complete_Cases: 25 samples, not necessary

## Incomplete_Cases 下没有找到 data.zarr，所以无法提取 spatial coords
# conda run -n SeededNTM python code/Xenium_lung/extract_cell_spatial_coords.py \
#   --data-dir data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Incomplete_Cases    # Incomplete_Cases: 20 samples


"""Extract per-cell spatial coords (um + HE pixel) from Xenium SpatialData zarr."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.transformations import get_transformation
from tqdm import tqdm

from xenium_coords import add_he_pixel_columns

COMBINED_LABELS_CSV_NAME = "Cell_spatial_coords_um_pix_from_zarr_all.csv"


def _scale_factor(sdata: sd.SpatialData) -> float:
    t = get_transformation(sdata.shapes["cell_boundaries"], to_coordinate_system="global")
    return float(np.asarray(t.scale).ravel()[0])


def _dominant_label_in_polygon(label_img: np.ndarray, geom, scale: float) -> int:
    from skimage.draw import polygon

    xs = np.asarray(geom.exterior.coords.xy[0], dtype=float) * scale
    ys = np.asarray(geom.exterior.coords.xy[1], dtype=float) * scale
    rr, cc = polygon(ys, xs, shape=label_img.shape)
    vals = label_img[rr, cc]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0
    return int(pd.Series(vals).mode().iloc[0])


def _label_at_point(label_img: np.ndarray, x: float, y: float, scale: float) -> int:
    xi = int(round(x * scale)) - 1
    yi = int(round(y * scale)) - 1
    yi = int(np.clip(yi, 0, label_img.shape[0] - 1))
    xi = int(np.clip(xi, 0, label_img.shape[1] - 1))
    return int(label_img[yi, xi])


def extract_labels(zarr_path: Path, method: str = "centroid") -> pd.DataFrame:
    """Return DataFrame with cell_id, labels, x/y centroids (um) and HE pixel columns."""
    sdata = sd.read_zarr(zarr_path)
    ad = sdata.tables["table"]
    scale = _scale_factor(sdata)

    cell_img = sdata.labels["cell_labels"]["scale0"]["image"].compute().values
    nuc_img = sdata.labels["nucleus_labels"]["scale0"]["image"].compute().values

    cell_gdf = sdata.shapes["cell_boundaries"]
    nuc_gdf = sdata.shapes["nucleus_boundaries"]
    coords = ad.obsm["spatial"]

    rows = []
    n = ad.n_obs
    iterator = range(n)
    if n > 500:
        iterator = tqdm(iterator, desc=zarr_path.parent.name, leave=False)

    for i in iterator:
        cell_id = ad.obs["cell_id"].iloc[i]
        x, y = float(coords[i, 0]), float(coords[i, 1])
        if method == "polygon":
            cell_label = _dominant_label_in_polygon(cell_img, cell_gdf.geometry.iloc[i], scale)
            nucleus_label = _dominant_label_in_polygon(nuc_img, nuc_gdf.geometry.iloc[i], scale)
        else:
            cell_label = _label_at_point(cell_img, x, y, scale)
            nucleus_label = _label_at_point(nuc_img, x, y, scale)
            if nucleus_label == 0:
                nucleus_label = _dominant_label_in_polygon(nuc_img, nuc_gdf.geometry.iloc[i], scale)

        rows.append(
            {
                "cell_id": cell_id,
                "cell_label": cell_label,
                "nucleus_label": nucleus_label,
                "x_centroid": x,
                "y_centroid": y,
            }
        )

    out = pd.DataFrame(rows)
    out.insert(0, "sample", zarr_path.parent.name)
    return add_he_pixel_columns(out, coord_units="um")


def labels_csv_path(sample_dir: Path) -> Path:
    """Per-sample output CSV, e.g. VUILD102LA/VUILD102LA_spatial_coords_um_pix_from_zarr.csv."""
    return sample_dir / f"{sample_dir.name}_spatial_coords_um_pix_from_zarr.csv"


def combined_labels_csv_path(data_dir: Path) -> Path:
    """Combined output CSV for all samples under a data root."""
    return data_dir / COMBINED_LABELS_CSV_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data/Complete_Cases_Select4"
        ),
    )
    parser.add_argument("--method", choices=["centroid", "polygon"], default="centroid")
    args = parser.parse_args()

    root = args.data_dir.resolve()
    zarr_paths = sorted(root.glob("*/data.zarr"))
    if not zarr_paths:
        raise SystemExit(
            f"No */data.zarr found under {root}. "
            "This script requires Xenium zarr (e.g. Complete_Cases_Select4). "
            "Incomplete_Cases has no zarr; use match_HEanno_with_sample_pix.py for cell positions."
        )

    frames: list[pd.DataFrame] = []
    for zarr_path in zarr_paths:
        sample = zarr_path.parent.name
        df_labels = extract_labels(zarr_path, method=args.method)
        out_csv = labels_csv_path(zarr_path.parent)
        df_labels.to_csv(out_csv, index=False)
        frames.append(df_labels)
        print(f"{sample}: {len(df_labels):,} cells -> {out_csv.name}")

    combined = pd.concat(frames, ignore_index=True)
    combined_path = combined_labels_csv_path(root)
    combined.to_csv(combined_path, index=False)
    print(f"Combined: {combined_path} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
