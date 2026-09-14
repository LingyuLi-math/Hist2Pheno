## 2026.08.12 LLY, copy from Xenium_lung/transer_embedding_label_h5ad.py

## Run script
# conda run --no-capture-output -n SeededNTM python -u \
#   code/Xenium_crc/transer_embedding_label_h5ad.py \
#   --sample P2CRC \
#   --steps stardist_all_h5ad \
#   --match-tolerance 1.0


# conda run --no-capture-output -n SeededNTM python -u \
#   code/Xenium_crc/transer_embedding_label_h5ad.py \
#   --steps stardist_all_h5ad


#!/usr/bin/env python3
"""
Build matched AnnData files for Xenium CRC HE regions.

The four explicit pipeline steps are:

1. ``he_h5ad``: ``*_cells_with_pixel.csv`` + HE embeddings
   -> ``*_matched_features.h5ad``.
2. ``stardist_csv``: transfer GT labels to raw StarDist centroids
   -> ``*_cells_matched_by_stardist.csv``.
3. ``stardist_h5ad``: the matched StarDist table + StarDist embeddings
   -> ``*_matched_features_stardist.h5ad``.
4. ``stardist_all_h5ad``: the raw StarDist table + StarDist embeddings
   -> ``*_all_features_stardist.h5ad``.

``--steps all`` means steps 1-3, matching the Xenium CLI. Step 4 remains
explicit because it builds an unlabeled, all-nuclei data set.

When multiple samples are requested, failures are reported per sample and the
remaining samples continue. The process exits nonzero if any sample fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
_SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (_PKG_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from base import (  # noqa: E402
    HCC_H5AD_OBS_COLUMNS,
    load_hist_embeddings,
    match_embeddings,
    match_hist2cell_h5ad,
)
from match_xenium_cells_with_pixel import (  # noqa: E402
    HCC_COLUMN_RENAME,
    build_cells_matched_by_stardist,
    cells_matched_stardist_path,
    cells_with_pixel_path,
    stardist_csv_path,
)

DEFAULT_CASES_ROOT = _REPO_ROOT / "data/Xemium/CRC/Cases"
DEFAULT_STARDIST_ROOT = _REPO_ROOT / "data/Xemium/CRC/StarDist_Segment"
DEFAULT_THERAPY_MODEL = "project_all_UNI"
MAPPING_CSV_NAME = None  # CRC sample id is P2CRC

HE_H5AD_SUFFIX = "_matched_features.h5ad"
STARDIST_H5AD_SUFFIX = "_matched_features_stardist.h5ad"
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"
HE_EMBEDDING_SUBDIR = "ImgEmbeddings_all/sc_pth_16_16"
STARDIST_EMBEDDING_SUBDIR = "ImgEmbeddings_all_stardist/sc_pth_16_16"
STARDIST_REQUIRED_COLUMNS = ("centroid_x", "centroid_y", "probability")


@dataclass(frozen=True)
class SamplePaths:
    """Resolved inputs and outputs for one MATCHED_HE region."""

    he_key: str
    acq_id: str
    cases_root: Path
    stardist_root: Path
    therapy_model: str

    @property
    def sample_dir(self) -> Path:
        return self.cases_root / self.he_key

    @property
    def cells_csv(self) -> Path:
        return cells_with_pixel_path(self.sample_dir, self.he_key)

    @property
    def stardist_raw_csv(self) -> Path:
        return stardist_csv_path(self.stardist_root, self.he_key)

    @property
    def stardist_matched_csv(self) -> Path:
        return cells_matched_stardist_path(self.sample_dir, self.he_key)

    @property
    def he_embedding_dir(self) -> Path:
        return self.sample_dir / self.therapy_model / HE_EMBEDDING_SUBDIR

    @property
    def stardist_embedding_dir(self) -> Path:
        return self.sample_dir / self.therapy_model / STARDIST_EMBEDDING_SUBDIR

    @property
    def he_h5ad(self) -> Path:
        return self.sample_dir / f"{self.he_key}{HE_H5AD_SUFFIX}"

    @property
    def stardist_h5ad(self) -> Path:
        return self.sample_dir / f"{self.he_key}{STARDIST_H5AD_SUFFIX}"

    @property
    def stardist_all_h5ad(self) -> Path:
        return self.sample_dir / f"{self.he_key}{STARDIST_ALL_H5AD_SUFFIX}"

    @property
    def he_pth_prefix(self) -> str:
        return f"sc_{self.acq_id}"

    @property
    def stardist_pth_prefix(self) -> str:
        return f"sc_{self.he_key}"


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _require_embedding_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not any(path.glob("*.pth")):
        raise FileNotFoundError(f"No .pth embeddings in {label}: {path}")


def _source_row_count(csv_path: Path) -> int:
    return len(pd.read_csv(csv_path, usecols=[0]))


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _h5ad_cache_usable(
    h5ad_path: Path,
    *,
    source_csv: Path,
    source_rows: int,
    required_obs: tuple[str, ...],
) -> bool:
    """Validate schema and source identity before accepting a cached h5ad."""
    if not h5ad_path.is_file():
        return False
    # Label rematches rewrite the CSV in place (same path / row count). Treat a
    # newer source CSV as stale so DCIS / hierarchy fixes are not skipped.
    try:
        if source_csv.is_file() and source_csv.stat().st_mtime > h5ad_path.stat().st_mtime:
            return False
    except OSError:
        return False
    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
            ok = adata.n_obs > 0 and adata.n_vars > 0
            ok = ok and all(col in adata.obs.columns for col in required_obs)
            ok = ok and adata.obs_names.is_unique
            ok = ok and "spatial" in adata.obsm and "spatial_HE" in adata.obsm
            ok = ok and np.dtype(adata.X.dtype) == np.dtype(np.float32)
            ok = ok and _as_int(adata.uns.get("source_row_count")) == source_rows
            cached_source = str(adata.uns.get("source_csv", ""))
            ok = ok and cached_source == str(source_csv.resolve())
            return bool(ok)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
    except Exception:
        return False


def _write_build_metadata(
    adata: "AnnData",
    *,
    source_csv: Path,
    embedding_dir: Path,
    pth_prefix: str,
    source_rows: int,
    match_tolerance: float,
) -> None:
    """Attach reproducibility and matching metadata without changing row order."""
    adata.uns["source_csv"] = str(source_csv.resolve())
    adata.uns["embedding_dir"] = str(embedding_dir.resolve())
    adata.uns["pth_prefix"] = pth_prefix
    adata.uns["source_row_count"] = int(source_rows)
    adata.uns["raw_n_cells"] = int(source_rows)
    adata.uns["matched_n_cells"] = int(adata.n_obs)
    adata.uns["embedding_dim"] = int(adata.n_vars)
    adata.uns["match_tolerance"] = float(match_tolerance)
    adata.uns["match_rate"] = float(adata.n_obs / max(source_rows, 1))


def _write_h5ad(adata: "AnnData", output: Path) -> None:
    """Write through a temporary file so interrupted builds do not look valid."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        adata.write_h5ad(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

## 2026.09.03 CRC: sample id is the UNI embedding prefix (sc_P2CRC)
def load_acquisition_map(
    cases_root: Path, *, aligned_only: bool = True
) -> dict[str, str]:
    """P1CRC / P2CRC / P5CRC maps to itself (no CODEX acquisition table)."""
    from match_xenium_cells_with_pixel import list_aligned_annotated_regions

    return {
        str(k): str(k)
        for k in list_aligned_annotated_regions()["MATCHED_HE"].tolist()
    }


def discover_samples(
    cases_root: Path, requested_sample: str | None, acq_map: dict[str, str]
) -> list[str]:
    """Return mapped CRC regions with directories under cases_root."""
    if requested_sample is not None:
        if requested_sample not in acq_map:
            acq_map = load_acquisition_map(cases_root, aligned_only=False)
        if requested_sample not in acq_map:
            raise KeyError(
                f"Unknown CRC sample {requested_sample!r}; choose P1CRC, P2CRC, or P5CRC"
            )
        if not (cases_root / requested_sample).is_dir():
            raise FileNotFoundError(
                f"Sample directory not found: {cases_root / requested_sample}"
            )
        return [requested_sample]

    samples = sorted(key for key in acq_map if (cases_root / key).is_dir())
    if not samples:
        raise FileNotFoundError(
            f"No mapped CRC sample directories found under {cases_root}"
        )
    return samples


def _build_matched_h5ad(
    *,
    source_csv: Path,
    embedding_dir: Path,
    output: Path,
    coord_cols: tuple[str, str],
    spatial_cols: tuple[str, str],
    spatial_he_cols: tuple[str, str],
    pth_prefix: str,
    match_tolerance: float,
    force_rebuild: bool,
) -> None:
    """Build a CRC labeled h5ad with stricter cache metadata than base.py."""
    _require_file(source_csv, "cell coordinate CSV")
    _require_embedding_dir(embedding_dir, "embedding directory")
    source_rows = _source_row_count(source_csv)
    cache_ok = _h5ad_cache_usable(
        output,
        source_csv=source_csv,
        source_rows=source_rows,
        required_obs=tuple(HCC_H5AD_OBS_COLUMNS),
    )
    if cache_ok and not force_rebuild:
        print(f"  cache valid, skip: {output.name}")
        return
    if output.is_file() and not force_rebuild:
        print(f"  cache incomplete or stale, rebuilding: {output.name}")

    adata = match_hist2cell_h5ad(
        cell_coords_path=str(source_csv),
        hist_embedding_dir=embedding_dir,
        matched_h5ad_path=str(output),
        coord_cols=coord_cols,
        tolerance=match_tolerance,
        pth_prefix=pth_prefix,
        level1_name="celltype_level1",
        column_rename=dict(HCC_COLUMN_RENAME),
        auto_rename=False,
        force_rebuild=True,
        obs_columns=HCC_H5AD_OBS_COLUMNS,
        cell_id_col="cell_id",
        spatial_cols=spatial_cols,
        spatial_he_cols=spatial_he_cols,
    )
    _write_build_metadata(
        adata,
        source_csv=source_csv,
        embedding_dir=embedding_dir,
        pth_prefix=pth_prefix,
        source_rows=source_rows,
        match_tolerance=match_tolerance,
    )
    _write_h5ad(adata, output)
    print(f"  -> {adata.n_obs:,} cells x {adata.n_vars} features")


def build_he_h5ad(
    paths: SamplePaths, *, match_tolerance: float, force_rebuild: bool
) -> None:
    """Step 1: build the labeled HE embedding h5ad."""
    print(f"  [HE h5ad] {paths.he_h5ad.name}")
    _build_matched_h5ad(
        source_csv=paths.cells_csv,
        embedding_dir=paths.he_embedding_dir,
        output=paths.he_h5ad,
        coord_cols=("X_pix_HE", "Y_pix_HE"),
        spatial_cols=("x_centroid", "y_centroid"),
        spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
        pth_prefix=paths.he_pth_prefix,
        match_tolerance=match_tolerance,
        force_rebuild=force_rebuild,
    )


def _matched_csv_usable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
        required = {
            "cell_id",
            "centroid_x",
            "centroid_y",
            *HCC_H5AD_OBS_COLUMNS,
        }
        return required.issubset(columns) and _source_row_count(path) > 0
    except Exception:
        return False


def build_stardist_csv(paths: SamplePaths, *, overwrite: bool) -> Path:
    """Step 2: reuse the one-to-one GT-to-StarDist matching helper."""
    _require_file(paths.cells_csv, "GT cells-with-pixel CSV")
    _require_file(paths.stardist_raw_csv, "raw StarDist CSV")
    if not overwrite and _matched_csv_usable(paths.stardist_matched_csv):
        print(f"  [StarDist CSV] cache valid, skip: {paths.stardist_matched_csv.name}")
        return paths.stardist_matched_csv
    if paths.stardist_matched_csv.is_file() and not overwrite:
        print(
            f"  [StarDist CSV] cache incomplete, rebuilding: "
            f"{paths.stardist_matched_csv.name}"
        )
    else:
        print(f"  [StarDist CSV] {paths.stardist_matched_csv.name}")
    matched = build_cells_matched_by_stardist(
        paths.cells_csv,
        paths.he_key,
        stardist_root=paths.stardist_root,
    )
    if matched.empty:
        raise ValueError("GT-to-StarDist matching produced zero rows")
    paths.stardist_matched_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(paths.stardist_matched_csv, index=False)
    print(f"  -> {len(matched):,} rows")
    return paths.stardist_matched_csv


def build_stardist_h5ad(
    paths: SamplePaths,
    *,
    match_tolerance: float,
    force_rebuild: bool,
    overwrite_csv: bool,
) -> None:
    """Step 3: build the labeled StarDist embedding h5ad."""
    matched_csv = build_stardist_csv(paths, overwrite=overwrite_csv)
    print(f"  [StarDist h5ad] {paths.stardist_h5ad.name}")
    _build_matched_h5ad(
        source_csv=matched_csv,
        embedding_dir=paths.stardist_embedding_dir,
        output=paths.stardist_h5ad,
        coord_cols=("centroid_x", "centroid_y"),
        spatial_cols=("centroid_x", "centroid_y"),
        spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
        pth_prefix=paths.stardist_pth_prefix,
        match_tolerance=match_tolerance,
        force_rebuild=force_rebuild,
    )


def load_raw_stardist_table(csv_path: Path, he_key: str) -> pd.DataFrame:
    """Load raw StarDist rows and synthesize deterministic cell IDs if needed."""
    raw = pd.read_csv(csv_path)
    missing = [col for col in STARDIST_REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise KeyError(f"Raw StarDist CSV missing columns {missing}: {csv_path}")
    if "cell_id" not in raw.columns:
        width = max(8, len(str(max(len(raw) - 1, 0))))
        raw.insert(
            0,
            "cell_id",
            [f"{he_key}-stardist-{i:0{width}d}" for i in range(len(raw))],
        )
    return raw


def _make_unique_names(values: pd.Series) -> pd.Index:
    """Create stable unique obs names while retaining the original cell_id column."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for i, value in enumerate(values):
        base = str(value).strip() if pd.notna(value) else ""
        if not base:
            base = f"stardist-{i:08d}"
        occurrence = seen.get(base, 0)
        names.append(base if occurrence == 0 else f"{base}__dup{occurrence}")
        seen[base] = occurrence + 1
    return pd.Index(names, dtype=str)


def build_stardist_all_h5ad(
    paths: SamplePaths, *, match_tolerance: float, force_rebuild: bool
) -> None:
    """Step 4: match every raw StarDist row for which a UNI embedding exists."""
    _require_file(paths.stardist_raw_csv, "raw StarDist CSV")
    _require_embedding_dir(paths.stardist_embedding_dir, "StarDist embedding directory")
    raw = load_raw_stardist_table(paths.stardist_raw_csv, paths.he_key)
    source_rows = len(raw)
    required_obs = tuple(raw.columns)
    if not force_rebuild and _h5ad_cache_usable(
        paths.stardist_all_h5ad,
        source_csv=paths.stardist_raw_csv,
        source_rows=source_rows,
        required_obs=required_obs,
    ):
        print(
            f"  [StarDist all h5ad] cache valid, skip: "
            f"{paths.stardist_all_h5ad.name}"
        )
        return
    if paths.stardist_all_h5ad.is_file() and not force_rebuild:
        print(
            f"  [StarDist all h5ad] cache incomplete or stale, rebuilding: "
            f"{paths.stardist_all_h5ad.name}"
        )
    else:
        print(f"  [StarDist all h5ad] {paths.stardist_all_h5ad.name}")

    # match_embeddings expects a celltype column, but the synthetic helper column
    # is kept out of obs so all original StarDist columns remain unmodified.
    matching = raw.copy()
    if "celltype" not in matching.columns:
        matching["celltype"] = "stardist_unlabeled"
    embeddings = load_hist_embeddings(
        paths.stardist_embedding_dir,
        celltype_df=matching,
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=match_tolerance,
        pth_prefix=paths.stardist_pth_prefix,
    )
    matched = match_embeddings(
        matching,
        embeddings,
        tolerance=match_tolerance,
        coord_cols=("centroid_x", "centroid_y"),
    )
    if not matched:
        raise ValueError(
            "Matched zero StarDist embeddings; check the embedding directory, "
            f"prefix {paths.stardist_pth_prefix!r}, coordinates, and tolerance."
        )

    import anndata as ad

    indices = np.asarray([int(item["idx"]) for item in matched], dtype=np.int64)
    obs = raw.iloc[indices].copy().reset_index(drop=True)
    obs.index = _make_unique_names(obs["cell_id"])
    X = np.vstack([np.asarray(item["embedding"]).reshape(-1) for item in matched])
    X = X.astype(np.float32, copy=False)
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"feat_{i}" for i in range(X.shape[1])]),
    )
    spatial = obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    adata.obsm["spatial"] = spatial
    adata.obsm["spatial_HE"] = spatial.copy()
    _write_build_metadata(
        adata,
        source_csv=paths.stardist_raw_csv,
        embedding_dir=paths.stardist_embedding_dir,
        pth_prefix=paths.stardist_pth_prefix,
        source_rows=source_rows,
        match_tolerance=match_tolerance,
    )
    _write_h5ad(adata, paths.stardist_all_h5ad)
    print(
        f"  -> {adata.n_obs:,}/{source_rows:,} rows matched "
        f"({adata.uns['match_rate']:.2%}), {adata.n_vars} features"
    )


def _validate_dry_run_inputs(paths: SamplePaths, steps: set[str]) -> None:
    """Validate requested inputs during dry-run without loading embeddings."""
    if "he_h5ad" in steps:
        _require_file(paths.cells_csv, "GT cells-with-pixel CSV")
        _require_embedding_dir(paths.he_embedding_dir, "HE embedding directory")
    if {"stardist_csv", "stardist_h5ad"} & steps:
        _require_file(paths.cells_csv, "GT cells-with-pixel CSV")
        _require_file(paths.stardist_raw_csv, "raw StarDist CSV")
    if "stardist_h5ad" in steps:
        _require_embedding_dir(
            paths.stardist_embedding_dir, "StarDist embedding directory"
        )
    if "stardist_all_h5ad" in steps:
        _require_file(paths.stardist_raw_csv, "raw StarDist CSV")
        _require_embedding_dir(
            paths.stardist_embedding_dir, "StarDist embedding directory"
        )


def process_sample(
    paths: SamplePaths,
    steps: set[str],
    *,
    match_tolerance: float,
    force_rebuild: bool,
    overwrite_csv: bool,
    dry_run: bool,
) -> None:
    """Run requested steps for one sample; exceptions are handled by main."""
    print(f"\n{'=' * 72}\nSample: {paths.he_key} (acq={paths.acq_id})\n{'=' * 72}")
    if dry_run:
        _validate_dry_run_inputs(paths, steps)
        outputs = {
            "he_h5ad": paths.he_h5ad,
            "stardist_csv": paths.stardist_matched_csv,
            "stardist_h5ad": paths.stardist_h5ad,
            "stardist_all_h5ad": paths.stardist_all_h5ad,
        }
        for step in sorted(steps):
            print(f"  [dry-run] {step} -> {outputs[step]}")
        return

    if "he_h5ad" in steps:
        build_he_h5ad(
            paths,
            match_tolerance=match_tolerance,
            force_rebuild=force_rebuild,
        )
    if "stardist_csv" in steps and "stardist_h5ad" not in steps:
        build_stardist_csv(paths, overwrite=overwrite_csv)
    if "stardist_h5ad" in steps:
        build_stardist_h5ad(
            paths,
            match_tolerance=match_tolerance,
            force_rebuild=force_rebuild,
            overwrite_csv=overwrite_csv,
        )
    if "stardist_all_h5ad" in steps:
        build_stardist_all_h5ad(
            paths,
            match_tolerance=match_tolerance,
            force_rebuild=force_rebuild,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Xenium CRC matched and all-nuclei UNI h5ad files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=DEFAULT_CASES_ROOT,
        help="Root containing Cases/{sample} directories (repository default).",
    )
    parser.add_argument(
        "--stardist-root",
        type=Path,
        default=DEFAULT_STARDIST_ROOT,
        help="Root containing StarDist_Segment/{stardist_key} directories.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="One sample id (P1CRC / P2CRC / P5CRC); default processes on-disk CRC regions.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=(
            "he_h5ad",
            "stardist_csv",
            "stardist_h5ad",
            "stardist_all_h5ad",
            "all",
        ),
        default=["all"],
        help="'all' runs steps 1-3; stardist_all_h5ad is always explicit.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=1.0,
        help="Maximum coordinate distance for embedding matching (default: 1.0).",
    )
    parser.add_argument(
        "--therapy-model",
        default=DEFAULT_THERAPY_MODEL,
        help="Model directory below each HE region (default: project_all_UNI).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild h5ad files even when their validated cache is usable.",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Rebuild the GT-labeled StarDist CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print outputs without writing.",
    )
    return parser.parse_args(argv)


def resolve_steps(raw_steps: list[str]) -> set[str]:
    if "all" in raw_steps:
        return {"he_h5ad", "stardist_csv", "stardist_h5ad"}
    return set(raw_steps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.match_tolerance < 0:
        raise ValueError("--match-tolerance must be nonnegative")
    cases_root = args.cases_root.expanduser().resolve()
    stardist_root = args.stardist_root.expanduser().resolve()
    steps = resolve_steps(args.steps)
    acq_map = load_acquisition_map(
        cases_root, aligned_only=args.sample is None
    )
    samples = discover_samples(cases_root, args.sample, acq_map)

    print(f"Cases root:     {cases_root}")
    print(f"StarDist root:  {stardist_root}")
    print(f"Therapy model:  {args.therapy_model}")
    print(f"Samples:        {len(samples)}")
    print(f"Steps:          {', '.join(sorted(steps))}")
    print(f"Tolerance:      {args.match_tolerance}")

    failures: list[tuple[str, str]] = []
    for he_key in samples:
        paths = SamplePaths(
            he_key=he_key,
            acq_id=acq_map[he_key],
            cases_root=cases_root,
            stardist_root=stardist_root,
            therapy_model=args.therapy_model,
        )
        try:
            process_sample(
                paths,
                steps,
                match_tolerance=args.match_tolerance,
                force_rebuild=args.force_rebuild,
                overwrite_csv=args.overwrite_csv,
                dry_run=args.dry_run,
            )
            print(f"  OK: {he_key}")
        except Exception as exc:
            failures.append((he_key, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {he_key}: {type(exc).__name__}: {exc}", file=sys.stderr)

    succeeded = len(samples) - len(failures)
    print(f"\nDone: {succeeded} succeeded, {len(failures)} failed, {len(samples)} total.")
    if failures:
        print("Failures:", file=sys.stderr)
        for he_key, message in failures:
            print(f"  - {he_key}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
