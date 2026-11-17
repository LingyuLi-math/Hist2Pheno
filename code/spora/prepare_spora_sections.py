#!/usr/bin/env python3
"""Export SPORA H&E, GT centroids, and matched h5ads for pan-cancer training.

Schurch and Hoefflin already have 1 µm instance masks whose labels match
``cell_id``. This step reads those masks. It does not run StarDist.
``run_spora_pancancer.sh segment`` is the 4-GPU StarDist job.

Outputs, per tissue, under ``data/CODEX/spora/prepared/<dataset>/<tissue>/``:

- ``<tissue>_he.tif`` RGB uint8 at 1 µm/pixel
- ``<tissue>_cells_with_pixel.csv`` kept cells, ``centroid_x`` = column, ``centroid_y`` = row
- ``<tissue>_matched_features.h5ad`` ``obs['final_CT']`` and ``obsm['spatial_HE']``

``Image_feature_extraction.py`` crops with the first coordinate as the horizontal
pixel (image width) and the second as the vertical pixel (image height). The
column is still named ``pxl_row_in_fullres`` after that file's historical swap.
``centroid_x`` is therefore the column, and ``centroid_y`` is the row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_DATASET_PY = _REPO / "code" / "Hist2Pheno_pkg" / "dataset"
if str(_DATASET_PY) not in sys.path:
    sys.path.insert(0, str(_DATASET_PY))

from build_pancancer_celltype_v1 import pancancer_keeps_native  # noqa: E402

DATA_ROOT = _REPO / "data" / "CODEX" / "spora"
PREPARED = DATA_ROOT / "prepared"
DATASETS = ("schurch2020coordinated", "hoefflin2026spatial")
LABEL_COL = {
    "schurch2020coordinated": "cell_type",
    "hoefflin2026spatial": "cell_type",
}


def _read_he(zarr_path: Path) -> np.ndarray:
    import zarr

    array = zarr.open_group(str(zarr_path), mode="r")["0"]
    volume = np.asarray(array[:])
    if volume.ndim != 3 or volume.shape[0] != 3:
        raise ValueError(f"{zarr_path} expected CYX with 3 channels, got {volume.shape}")
    return np.moveaxis(volume, 0, -1)


def _centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return label, row, col for every positive instance. One pass over foreground."""
    rows, cols = np.nonzero(mask)
    labels = mask[rows, cols].astype(np.int64, copy=False)
    keep = labels > 0
    rows, cols, labels = rows[keep], cols[keep], labels[keep]
    counts = np.bincount(labels)
    sum_row = np.bincount(labels, weights=rows.astype(np.float64))
    sum_col = np.bincount(labels, weights=cols.astype(np.float64))
    present = np.flatnonzero(counts > 0)
    present = present[present > 0]
    return present, sum_row[present] / counts[present], sum_col[present] / counts[present]


def _write_tif(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile

        tifffile.imwrite(path, image, photometric="rgb")
    except ImportError:
        from PIL import Image

        Image.fromarray(image).save(path)


def _write_h5ad(path: Path, frame: pd.DataFrame, uni2_dir: Path) -> None:
    import anndata as ad

    xy = frame[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    obs = frame.set_index(frame["cell_id"].astype(str), drop=False)
    obs.index.name = None
    adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial_HE"] = xy
    adata.uns["embedding_dir"] = str(uni2_dir)
    adata.uns["match_tolerance"] = 1.0
    adata.uns["mpp"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def prepare_tissue(dataset: str, tissue: str, force: bool) -> str:
    source = DATA_ROOT / dataset / dataset
    out = PREPARED / dataset / tissue
    he_tif = out / f"{tissue}_he.tif"
    csv_path = out / f"{tissue}_cells_with_pixel.csv"
    h5ad_path = out / f"{tissue}_matched_features.h5ad"
    uni2_dir = out / f"{tissue}_project_all_UNI2" / "ImgEmbeddings_all" / "sc_pth_14_14"
    if not force and he_tif.is_file() and csv_path.is_file() and h5ad_path.is_file():
        return f"skip {tissue}"
    out.mkdir(parents=True, exist_ok=True)

    cells = pd.read_parquet(
        source / "metadata" / "cells.parquet",
        columns=["tissue_id", "cell_id", LABEL_COL[dataset]],
    )
    cells = cells.loc[cells["tissue_id"] == tissue].copy()
    native = cells[LABEL_COL[dataset]].astype(str)
    cells = cells.loc[native.map(pancancer_keeps_native)].copy()
    cells["final_CT"] = native.loc[cells.index].to_numpy()
    if cells.empty:
        return f"skip {tissue}: no kept labels"

    tissues = pd.read_parquet(source / "metadata" / "tissues.parquet")
    he_meta = tissues.loc[
        (tissues["modality"] == "he") & (tissues["tissue_id"] == tissue),
        ["patient_id", "split"],
    ].drop_duplicates()
    if len(he_meta) == 1:
        cells["patient_id"] = he_meta["patient_id"].iloc[0]
        cells["split"] = he_meta["split"].iloc[0]
    else:
        cells["patient_id"] = ""
        cells["split"] = ""

    mask_path = source / "segmentations" / "1_0mpp" / "cell_masks" / "instances" / f"{tissue}.npz"
    with np.load(mask_path, allow_pickle=False) as handle:
        mask = handle["mask"]
    label, row, col = _centroids(mask)
    del mask
    # centroid_x is horizontal (column). The extractor compares it with image width.
    lookup = pd.DataFrame({"cell_id": label.astype(np.int64), "centroid_x": col, "centroid_y": row})
    merged = cells.merge(lookup, on="cell_id", how="inner")
    if merged.empty:
        return f"skip {tissue}: no mask hits"
    merged = merged.sort_values("cell_id")
    keep_cols = ["cell_id", "centroid_x", "centroid_y", "final_CT", "tissue_id", "patient_id", "split"]
    merged[keep_cols].to_csv(csv_path, index=False)

    zarr_path = source / "he" / "1_0mpp" / "images" / f"{tissue}.ome.zarr"
    _write_tif(he_tif, _read_he(zarr_path))
    _write_h5ad(h5ad_path, merged[keep_cols], uni2_dir)
    return f"wrote {tissue}: {len(merged)} cells"


def _job(payload: tuple[str, str, bool]) -> str:
    dataset, tissue, force = payload
    try:
        return prepare_tissue(dataset, tissue, force)
    except Exception as exc:
        return f"FAIL {tissue}: {exc}"


def _extractor_outside(horizontal: np.ndarray, vertical: np.ndarray, width: int, height: int, half: int = 7) -> np.ndarray:
    """Cells whose patch the extractor rejects. horizontal is compared with width."""
    return (horizontal < half) | (vertical < half) | (horizontal + half > width) | (vertical + half > height)


########################################################
# 2026.11.17, decide the axis from the instance mask, not from bounds
# 用 mask 质心判定方向：SPORA TMA core 接近正方形，越界计数两种方向都可能是 0，
# 不能区分。这里重新计算每个 cell_id 的 (row, col)，与表中坐标逐个比较。
# 重复运行安全（已正确的表不会被再次交换）。
########################################################
def axis_state(frame: pd.DataFrame, mask_centroids: pd.DataFrame, tol: float = 0.51) -> tuple[str, float, float]:
    """'ok' if centroid_x == column, 'swapped' if centroid_x == row, else 'unknown'.

    Returns (state, share matching column-first, share matching row-first).
    """
    m = frame[["cell_id", "centroid_x", "centroid_y"]].merge(mask_centroids, on="cell_id", how="inner")
    if m.empty:
        return "unknown", 0.0, 0.0
    hx = m["centroid_x"].to_numpy(np.float64)
    vy = m["centroid_y"].to_numpy(np.float64)
    col = m["col"].to_numpy(np.float64)
    row = m["row"].to_numpy(np.float64)
    ok = float(np.mean((np.abs(hx - col) <= tol) & (np.abs(vy - row) <= tol)))
    bad = float(np.mean((np.abs(hx - row) <= tol) & (np.abs(vy - col) <= tol)))
    if ok >= 0.99 and ok > bad:
        return "ok", ok, bad
    if bad >= 0.99 and bad > ok:
        return "swapped", ok, bad
    if ok >= 0.99 and bad >= 0.99:          # every centroid on the diagonal: harmless either way
        return "ok", ok, bad
    return "unknown", ok, bad


def _newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            pass
    return newest


def _stale_store(tissue_dir: Path, tissue: str, csv_path: Path) -> Path | None:
    """The UNI2 store if it was built before the current table, else None."""
    store = tissue_dir / f"{tissue}_project_all_UNI2"
    if not store.is_dir() or _newest_mtime(store) >= csv_path.stat().st_mtime:
        return None
    return store


def _retire_stale_store(tissue_dir: Path, tissue: str, csv_path: Path) -> str:
    """Rename a UNI2 store built before the current table so the worker re-extracts it."""
    store = _stale_store(tissue_dir, tissue, csv_path)
    if store is None:
        return ""
    import time

    target = store.with_name(f"{store.name}.stale_axis_{time.strftime('%Y%m%d%H%M%S')}")
    store.rename(target)
    return target.name


def fix_centroid_axis(dataset: str, dry_run: bool = False) -> str:
    """Make ``centroid_x`` the horizontal pixel on every prepared table, checked
    cell by cell against the instance mask. UNI2 stores older than their table
    are renamed ``*.stale_axis_<time>`` so ``uni2`` re-extracts them. The H&E
    TIFF is not rewritten. Returns a one-line status.
    """
    import anndata as ad

    root = PREPARED / dataset
    source = DATA_ROOT / dataset / dataset
    tissues = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    counts = {"ok": 0, "swapped": 0, "unknown": 0, "retired": 0, "missing": 0}
    for tissue_dir in tissues:
        tissue = tissue_dir.name
        csv_path = tissue_dir / f"{tissue}_cells_with_pixel.csv"
        h5ad_path = tissue_dir / f"{tissue}_matched_features.h5ad"
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path)
        mask_path = source / "segmentations" / "1_0mpp" / "cell_masks" / "instances" / f"{tissue}.npz"
        with np.load(mask_path, allow_pickle=False) as handle:
            label, row, col = _centroids(handle["mask"])
        mc = pd.DataFrame({"cell_id": label.astype(np.int64), "row": row, "col": col})
        state, ok, bad = axis_state(frame, mc)
        counts[state] += 1
        if state == "unknown":
            print(f"  ?? {tissue}: column-first {ok:.3f}, row-first {bad:.3f}; re-run prepare --force for this tissue",
                  flush=True)
            continue
        if state == "swapped" and not dry_run:
            frame["centroid_x"], frame["centroid_y"] = frame["centroid_y"].copy(), frame["centroid_x"].copy()
            frame.to_csv(csv_path, index=False)
            adata = ad.read_h5ad(h5ad_path)
            aligned = frame.set_index(frame["cell_id"].astype(str)).loc[adata.obs["cell_id"].astype(str).to_numpy()]
            adata.obs["centroid_x"] = aligned["centroid_x"].to_numpy()
            adata.obs["centroid_y"] = aligned["centroid_y"].to_numpy()
            adata.obsm["spatial_HE"] = aligned[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
            adata.write_h5ad(h5ad_path)
        # 2026.11.17, dry-run also reports stale UNI2 stores (and missing ones)
        # 一个已正确的表仍可能配着修正前提取的 UNI2；dry-run 也要报告
        if not (tissue_dir / f"{tissue}_project_all_UNI2").is_dir():
            counts["missing"] += 1
        elif dry_run:
            if state == "swapped" or _stale_store(tissue_dir, tissue, csv_path) is not None:
                counts["retired"] += 1
                print(f"  {tissue}: UNI2 store older than its table (would be re-extracted)", flush=True)
        else:
            retired = _retire_stale_store(tissue_dir, tissue, csv_path)
            if retired:
                counts["retired"] += 1
                print(f"  {tissue}: stale UNI2 store -> {retired}", flush=True)
    verb = "would swap" if dry_run else "swapped"
    verb2 = "would retire" if dry_run else "retired"
    return (f"{dataset}: {verb} {counts['swapped']}, already correct {counts['ok']}, "
            f"unresolved {counts['unknown']}, stale UNI2 stores {verb2} {counts['retired']}, "
            f"no UNI2 store yet {counts['missing']}")


def tissues_of(dataset: str) -> list[str]:
    cells = pd.read_parquet(
        DATA_ROOT / dataset / dataset / "metadata" / "cells.parquet",
        columns=["tissue_id"],
    )
    return sorted(cells["tissue_id"].unique())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--tissue", default="", help="One tissue_id. Empty means every annotated tissue.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many tissues per dataset. 0 means all.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fix-centroid-axis",
        action="store_true",
        help="Check every prepared table against its instance mask; swap tables whose centroid_x is the row, "
             "and rename UNI2 stores older than their table. Does not re-read the OME-Zarr.",
    )
    parser.add_argument("--dry-run", action="store_true", help="With --fix-centroid-axis: report only.")
    args = parser.parse_args(argv)
    if args.fix_centroid_axis:
        for dataset in args.datasets:
            if dataset not in DATASETS:
                raise SystemExit(f"unknown dataset {dataset}")
            print(fix_centroid_axis(dataset, dry_run=args.dry_run), flush=True)
        return 0

    jobs: list[tuple[str, str, bool]] = []
    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"unknown dataset {dataset}")
        names = [args.tissue] if args.tissue else tissues_of(dataset)
        if args.limit:
            names = names[: args.limit]
        jobs.extend((dataset, name, args.force) for name in names)
    print(f"prepare {len(jobs)} tissues -> {PREPARED}", flush=True)
    if args.workers <= 1 or len(jobs) <= 1:
        lines = [_job(job) for job in jobs]
    else:
        from multiprocessing import Pool

        with Pool(args.workers) as pool:
            lines = pool.map(_job, jobs)
    failed = 0
    for line in lines:
        print(line, flush=True)
        if line.startswith("FAIL"):
            failed += 1
    print(f"done failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
