## 2026.06.24 LLY Transer embedding+label to h5ad for Xenium lung
## cd /home/lingyu/ssd2/Python/Collaborate/esccAI

## using python -u
## conda activate SeededNTM
## python -u code/Xenium_lung/transer_embedding_label_h5ad.py
## python -u code/Xenium_lung/transer_embedding_label_h5ad.py --cases-set incomplete

## Or using conda run --no-capture-output
## conda run --no-capture-output -n SeededNTM python -u code/Xenium_lung/transer_embedding_label_h5ad.py

########################################################
## 2026.06.27 LLY: Add the function to load the raw StarDist CSV
########################################################
## cd /home/lingyu/ssd2/Python/Collaborate/esccAI
## conda activate SeededNTM
## python -u code/Xenium_lung/transer_embedding_label_h5ad.py --steps stardist_all_h5ad

# 单样本示例：
# conda run --no-capture-output -n SeededNTM python -u \
#     code/Xenium_lung/transer_embedding_label_h5ad.py \
#     --sample VUILD107MA \
#     --steps stardist_all_h5ad


#!/usr/bin/env python3
"""
Build matched AnnData (.h5ad) and StarDist matched CSV for Xenium lung Complete_Cases.

Pipeline (mirrors ``Lung_train_validate_cv_UNIlabel_all_clean.ipynb`` cells 8 & 24):

  Step 1 — HE/GT matched h5ad
      {sample}_cells_partitioned_by_annotation_sample_match_with_pixel.csv
      + ImgEmbeddings_all/sc_pth_16_16/*.pth
      → {sample}_matched_features.h5ad

  Step 2 — StarDist cell table (Xenium labels + StarDist centroids)
      GT CSV + {sample}_Float_prob0.01_nms_0.3.csv
      → {sample}_cells_matched_by_stardist.csv

  Step 3 — StarDist matched h5ad
      cells_matched_by_stardist.csv
      + ImgEmbeddings_all_stardist/sc_pth_16_16/*.pth
      → {sample}_matched_features_stardist.h5ad

  Step 4 — StarDist all h5ad
      {sample}_Float_prob0.01_nms_0.3.csv
      + ImgEmbeddings_all_stardist/sc_pth_16_16/*.pth
      → {sample}_all_features_stardist.h5ad

Row order inside each output file is the embedding-match order: row *i* of ``.X``,
``obs``, and ``obsm`` always refer to the same cell.  HE and StarDist outputs are
**not** row-aligned with each other; join on ``obs_names`` / ``cell_id`` if needed.

Terminal usage (from repo root ``esccAI``):

  Use ``conda run --no-capture-output`` (or ``conda activate SeededNTM`` first).
  Plain ``conda run`` buffers stdout until the process exits, so long batch runs
  look silent even though work is in progress.

  # All Complete_Cases samples (skip outputs that already look valid)
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py

  # One sample, all three steps
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py --sample VUILD107MA

  # Force rebuild h5ad even when cache exists
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py \\
      --sample VUILD107MA --force-rebuild

  # Only StarDist steps (CSV + h5ad)
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py \\
      --steps stardist_csv stardist_h5ad

  # Only Step 4 — all StarDist nuclei h5ad (does not touch steps 1–3)
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py \\
      --steps stardist_all_h5ad

  # Incomplete_Cases
  conda run --no-capture-output -n SeededNTM python -u \\
      code/Xenium_lung/transer_embedding_label_h5ad.py --cases-set incomplete
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Show progress immediately when run via ``conda run --no-capture-output``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
print("Loading Hist2Pheno dependencies...", flush=True)

import pandas as pd

import numpy as np

# ---------------------------------------------------------------------------
# Import Hist2Pheno helpers (same paths as the training notebook).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_DIR = _REPO_ROOT / "code" / "Hist2Pheno_pkg"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import base  # noqa: E402
from base import (  # noqa: E402
    XENIUM_CELL_COORD_COLUMN_RENAME_FULL,
    load_cell_pixcoords,
    load_hist_embeddings,
    match_celltype2stardist,
    match_embeddings,
    match_hist2cell_h5ad,
)

DEFAULT_DATA_ROOT = (
    _REPO_ROOT
    / "data/Xemiun/weiqin/SpatialPF-NGenetics/Spatial-PF-Processed/Data"
)
GT_CSV_SUFFIX = "_cells_partitioned_by_annotation_sample_match_with_pixel.csv"
STARDIST_RAW_SUFFIX = "_Float_prob0.01_nms_0.3.csv"
STARDIST_MATCHED_SUFFIX = "_cells_matched_by_stardist.csv"
HE_H5AD_SUFFIX = "_matched_features.h5ad"
STARDIST_H5AD_SUFFIX = "_matched_features_stardist.h5ad"
STARDIST_ALL_H5AD_SUFFIX = "_all_features_stardist.h5ad"
STARDIST_ALL_OBS_COLUMNS = ("centroid_x", "centroid_y", "probability")
EMB_SUBDIR_HE = "ImgEmbeddings_all/sc_pth_16_16"
EMB_SUBDIR_STARDIST = "ImgEmbeddings_all_stardist/sc_pth_16_16"
STARDIST_ALL_CELLTYPE_PLACEHOLDER = "stardist_unlabeled"


@dataclass(frozen=True)
class SamplePaths:
    """Standard Complete_Cases paths for one sample."""

    sample: str
    sample_dir: Path
    therapy_model: str

    @property
    def gt_csv(self) -> Path:
        return self.sample_dir / f"{self.sample}{GT_CSV_SUFFIX}"

    @property
    def stardist_raw_csv(self) -> Path:
        return self.sample_dir / f"{self.sample}{STARDIST_RAW_SUFFIX}"

    @property
    def stardist_matched_csv(self) -> Path:
        return self.sample_dir / f"{self.sample}{STARDIST_MATCHED_SUFFIX}"

    @property
    def he_h5ad(self) -> Path:
        return self.sample_dir / f"{self.sample}{HE_H5AD_SUFFIX}"

    @property
    def stardist_h5ad(self) -> Path:
        return self.sample_dir / f"{self.sample}{STARDIST_H5AD_SUFFIX}"

    @property
    def stardist_all_h5ad(self) -> Path:
        return self.sample_dir / f"{self.sample}{STARDIST_ALL_H5AD_SUFFIX}"

    @property
    def he_embedding_dir(self) -> Path:
        return self.sample_dir / self.therapy_model / EMB_SUBDIR_HE

    @property
    def stardist_embedding_dir(self) -> Path:
        return self.sample_dir / self.therapy_model / EMB_SUBDIR_STARDIST


def _therapy_model_name(sample: str) -> str:
    return f"{sample}_project_all_UNI"


def discover_samples(cases_root: Path, sample: str | None) -> list[str]:
    """List sample IDs under *cases_root* (sorted)."""
    if sample is not None:
        return [sample]
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")
    out = sorted(
        p.name
        for p in cases_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return out


def sample_paths(sample: str, cases_root: Path) -> SamplePaths:
    return SamplePaths(
        sample=sample,
        sample_dir=cases_root / sample,
        therapy_model=_therapy_model_name(sample),
    )


def _require(path: Path, label: str) -> None:
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _has_pth_files(emb_dir: Path) -> bool:
    return emb_dir.is_dir() and any(emb_dir.glob("*.pth"))


def build_he_matched_h5ad(
    paths: SamplePaths,
    *,
    match_tolerance: float,
    column_rename: dict,
    force_rebuild: bool,
) -> None:
    """Step 1: GT/HE embeddings → ``{sample}_matched_features.h5ad``."""
    _require(paths.gt_csv, "GT cell CSV")
    _require(paths.he_embedding_dir, "HE embedding directory")
    if not _has_pth_files(paths.he_embedding_dir):
        raise FileNotFoundError(f"No .pth files in {paths.he_embedding_dir}")

    print(f"  [HE h5ad] {paths.he_h5ad.name}")
    adata = match_hist2cell_h5ad(
        cell_coords_path=str(paths.gt_csv),
        hist_embedding_dir=paths.he_embedding_dir,
        matched_h5ad_path=str(paths.he_h5ad),
        coord_cols=("X_pix_HE", "Y_pix_HE"),
        tolerance=match_tolerance,
        pth_prefix=paths.sample,
        column_rename=column_rename,
        auto_rename=False,
        force_rebuild=force_rebuild,
    )
    print(f"    → {adata.n_obs:,} cells × {adata.n_vars} features")


def build_stardist_matched_csv(
    paths: SamplePaths,
    *,
    column_rename: dict,
    overwrite: bool,
) -> Path:
    """
    Step 2: Match Xenium GT labels to StarDist centroids.

    Saves ``{sample}_cells_matched_by_stardist.csv`` with **original** Xenium
    column names (``final_CT``, ``final_lineage``, …) for h5ad ``obs``.
    """
    if paths.stardist_matched_csv.is_file() and not overwrite:
        print(f"  [StarDist CSV] exists, skip: {paths.stardist_matched_csv.name}")
        return paths.stardist_matched_csv

    _require(paths.gt_csv, "GT cell CSV")
    _require(paths.stardist_raw_csv, "StarDist raw CSV")

    print(f"  [StarDist CSV] {paths.stardist_matched_csv.name}")
    celltype_xenium = load_cell_pixcoords(
        str(paths.gt_csv),
        column_rename=column_rename,
        auto_rename=False,
    )
    star_raw = pd.read_csv(paths.stardist_raw_csv)
    matched = match_celltype2stardist(
        celltype_xenium,
        star_raw,
        celltype_pixel_coords_cols=("X_pix_HE", "Y_pix_HE"),
        stardist_pixel_coords_cols=("centroid_x", "centroid_y"),
    )
    matched = matched.dropna(subset=["centroid_x", "centroid_y"])
    restore = {
        tgt: src
        for src, tgt in column_rename.items()
        if tgt in matched.columns
    }
    matched.rename(columns=restore).to_csv(paths.stardist_matched_csv, index=False)
    print(f"    → {len(matched):,} rows")
    return paths.stardist_matched_csv


def build_stardist_matched_h5ad(
    paths: SamplePaths,
    *,
    match_tolerance: float,
    column_rename: dict,
    level1_name: str,
    force_rebuild: bool,
    overwrite_csv: bool,
) -> None:
    """Step 3: StarDist embeddings → ``{sample}_matched_features_stardist.h5ad``."""
    csv_path = build_stardist_matched_csv(
        paths,
        column_rename=column_rename,
        overwrite=overwrite_csv,
    )
    _require(paths.stardist_embedding_dir, "StarDist embedding directory")
    if not _has_pth_files(paths.stardist_embedding_dir):
        raise FileNotFoundError(f"No .pth files in {paths.stardist_embedding_dir}")

    print(f"  [StarDist h5ad] {paths.stardist_h5ad.name}")
    adata = match_hist2cell_h5ad(
        cell_coords_path=str(csv_path),
        hist_embedding_dir=paths.stardist_embedding_dir,
        matched_h5ad_path=str(paths.stardist_h5ad),
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=match_tolerance,
        pth_prefix=paths.sample,
        level1_name=level1_name,
        column_rename=column_rename,
        auto_rename=False,
        spatial_cols=("centroid_x", "centroid_y"),
        spatial_he_cols=("X_pix_HE", "Y_pix_HE"),
        force_rebuild=force_rebuild,
    )
    print(f"    → {adata.n_obs:,} cells × {adata.n_vars} features")


########################################################
## 2026.06.27 LLY: Add the function to load the raw StarDist CSV
########################################################
def load_stardist_raw_table(csv_path: Path, sample: str) -> pd.DataFrame:
    """Load raw StarDist CSV and add fields required for embedding matching."""
    df = pd.read_csv(csv_path)
    _c0 = df.columns[0]
    if str(_c0).startswith("Unnamed"):
        df = df.drop(columns=[_c0])
    if "cell_id" not in df.columns:
        df.insert(0, "cell_id", [f"{sample}-{i}" for i in range(len(df))])
    if "celltype" not in df.columns:
        df["celltype"] = STARDIST_ALL_CELLTYPE_PLACEHOLDER
    return df


def _unique_obs_names(values, *, prefix: str = "cell") -> np.ndarray:
    seen: dict[str, int] = {}
    out: list[str] = []
    for v in values:
        base = str(v).strip() if pd.notna(v) and str(v).strip() else ""
        if not base:
            base = prefix
        if base not in seen:
            seen[base] = 0
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}__dup{seen[base]}")
    return np.asarray(out, dtype=object)


def _stardist_all_h5ad_usable(h5ad_path: Path) -> bool:
    if not h5ad_path.is_file():
        return False
    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        ok = all(c in adata.obs.columns for c in STARDIST_ALL_OBS_COLUMNS)
        ok = ok and adata.n_obs > 0 and adata.n_vars > 0
        ok = ok and "spatial" in adata.obsm and "spatial_HE" in adata.obsm
        return ok
    except Exception:
        return False


def _build_stardist_all_h5ad(
    stardist_df: pd.DataFrame,
    matched_data: list[dict],
    h5ad_path: Path,
    *,
    cell_id_col: str = "cell_id",
    spatial_cols: tuple[str, str] = ("centroid_x", "centroid_y"),
) -> "AnnData":
    """Build h5ad from raw StarDist CSV rows matched to embeddings."""
    import anndata as ad

    if not matched_data:
        raise ValueError("No matched StarDist rows — check embedding paths and tolerance.")

    indices = [int(row["idx"]) for row in matched_data]
    X = np.vstack([row["embedding"] for row in matched_data]).astype(np.float32)
    obs_df = stardist_df.iloc[indices].copy().reset_index(drop=True)
    if "celltype" in obs_df.columns and (
        obs_df["celltype"] == STARDIST_ALL_CELLTYPE_PLACEHOLDER
    ).all():
        obs_df = obs_df.drop(columns=["celltype"])

    if cell_id_col in obs_df.columns:
        obs_names = _unique_obs_names(obs_df[cell_id_col].tolist(), prefix="stardist")
    else:
        obs_names = _unique_obs_names(
            [f"stardist-{i}" for i in range(len(obs_df))],
            prefix="stardist",
        )

    adata = ad.AnnData(X=X, obs=obs_df)
    adata.obs_names = obs_names
    adata.var_names = [f"feat_{i}" for i in range(X.shape[1])]

    sx, sy = spatial_cols
    if sx in obs_df.columns and sy in obs_df.columns:
        spatial = obs_df[[sx, sy]].to_numpy(dtype=np.float64)
        adata.obsm["spatial"] = spatial
        adata.obsm["spatial_HE"] = spatial.copy()
    else:
        raise KeyError(
            f"spatial_cols {spatial_cols} not found in StarDist CSV columns: "
            f"{obs_df.columns.tolist()}"
        )

    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(h5ad_path)
    print(f"  ✓ Saved StarDist-all h5ad: {h5ad_path} ({adata.n_obs:,} cells)")
    return adata


def build_stardist_all_h5ad(
    paths: SamplePaths,
    *,
    match_tolerance: float,
    force_rebuild: bool,
) -> None:
    """
    Step 4: Raw StarDist CSV + all StarDist embeddings → ``{sample}_all_features_stardist.h5ad``.

    All columns from ``{sample}_Float_prob0.01_nms_0.3.csv`` are stored in ``adata.obs``.
    Does not read or overwrite outputs from steps 1–3.
    """
    if not force_rebuild and _stardist_all_h5ad_usable(paths.stardist_all_h5ad):
        print(f"  [StarDist all h5ad] cache found, skip: {paths.stardist_all_h5ad.name}")
        return

    _require(paths.stardist_raw_csv, "StarDist raw CSV")
    _require(paths.stardist_embedding_dir, "StarDist embedding directory")
    if not _has_pth_files(paths.stardist_embedding_dir):
        raise FileNotFoundError(f"No .pth files in {paths.stardist_embedding_dir}")

    print(f"  [StarDist all h5ad] {paths.stardist_all_h5ad.name}")
    stardist_df = load_stardist_raw_table(paths.stardist_raw_csv, paths.sample)
    embeddings_dict = load_hist_embeddings(
        paths.stardist_embedding_dir,
        celltype_df=stardist_df,
        coord_cols=("centroid_x", "centroid_y"),
        tolerance=match_tolerance,
        pth_prefix=paths.sample,
    )
    matched_data = match_embeddings(
        stardist_df,
        embeddings_dict,
        tolerance=match_tolerance,
        coord_cols=("centroid_x", "centroid_y"),
    )
    adata = _build_stardist_all_h5ad(
        stardist_df,
        matched_data,
        paths.stardist_all_h5ad,
        spatial_cols=("centroid_x", "centroid_y"),
    )
    print(f"    → {adata.n_obs:,} cells × {adata.n_vars} features")
########################################################

def process_sample(
    sample: str,
    cases_root: Path,
    steps: set[str],
    *,
    match_tolerance: float,
    column_rename: dict,
    level1_name: str,
    force_rebuild: bool,
    overwrite_csv: bool,
    dry_run: bool,
) -> bool:
    """
    Run selected pipeline steps for one sample.

    Returns True on success, False if skipped (missing inputs).
    """
    paths = sample_paths(sample, cases_root)
    print(f"\n{'=' * 60}\nSample: {sample}\n{'=' * 60}")

    if not paths.sample_dir.is_dir():
        print(f"  SKIP: sample directory not found: {paths.sample_dir}")
        return False

    try:
        if "he_h5ad" in steps:
            if dry_run:
                print(f"  [dry-run] would build {paths.he_h5ad}")
            else:
                build_he_matched_h5ad(
                    paths,
                    match_tolerance=match_tolerance,
                    column_rename=column_rename,
                    force_rebuild=force_rebuild,
                )

        if "stardist_csv" in steps and "stardist_h5ad" not in steps:
            if dry_run:
                print(f"  [dry-run] would build {paths.stardist_matched_csv}")
            else:
                build_stardist_matched_csv(
                    paths,
                    column_rename=column_rename,
                    overwrite=overwrite_csv,
                )

        if "stardist_h5ad" in steps:
            if dry_run:
                print(f"  [dry-run] would build {paths.stardist_matched_csv}")
                print(f"  [dry-run] would build {paths.stardist_h5ad}")
            else:
                build_stardist_matched_h5ad(
                    paths,
                    match_tolerance=match_tolerance,
                    column_rename=column_rename,
                    level1_name=level1_name,
                    force_rebuild=force_rebuild,
                    overwrite_csv=overwrite_csv,
                )

        if "stardist_all_h5ad" in steps:
            if dry_run:
                print(f"  [dry-run] would build {paths.stardist_all_h5ad}")
            else:
                build_stardist_all_h5ad(
                    paths,
                    match_tolerance=match_tolerance,
                    force_rebuild=force_rebuild,
                )
    except FileNotFoundError as exc:
        print(f"  SKIP: {exc}")
        return False

    print(f"  OK: {sample}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build {sample}_matched_features.h5ad, "
            "{sample}_cells_matched_by_stardist.csv, "
            "{sample}_matched_features_stardist.h5ad, and "
            "{sample}_all_features_stardist.h5ad for Complete_Cases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=None,
        help="Root directory containing per-sample folders (default: Complete_Cases).",
    )
    parser.add_argument(
        "--cases-set",
        choices=("complete", "incomplete"),
        default="complete",
        help="Which cases folder under the processed Data tree (default: complete).",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Process only this sample ID (default: all samples under cases-root).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=("he_h5ad", "stardist_csv", "stardist_h5ad", "stardist_all_h5ad", "all"),
        default=["all"],
        help=(
            "Pipeline steps to run (default: all = steps 1–3 only). "
            "Use stardist_all_h5ad to run Step 4 without touching steps 1–3."
        ),
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=1.0,
        help="Coordinate tolerance for embedding matching (default: 1.0 HE pixels).",
    )
    parser.add_argument(
        "--level1-name",
        type=str,
        default="celltype_level1",
        help="Level-1 column name after in-memory rename (default: celltype_level1).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild h5ad even when a usable cache file already exists.",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Overwrite existing {sample}_cells_matched_by_stardist.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    return parser.parse_args(argv)


def resolve_cases_root(args: argparse.Namespace) -> Path:
    if args.cases_root is not None:
        return args.cases_root.expanduser().resolve()
    sub = "Complete_Cases" if args.cases_set == "complete" else "Incomplete_Cases"
    return (DEFAULT_DATA_ROOT / sub).resolve()


def resolve_steps(raw: list[str]) -> set[str]:
    if "all" in raw:
        return {"he_h5ad", "stardist_csv", "stardist_h5ad"}
    return set(raw)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases_root = resolve_cases_root(args)
    steps = resolve_steps(args.steps)
    column_rename = dict(XENIUM_CELL_COORD_COLUMN_RENAME_FULL)

    samples = discover_samples(cases_root, args.sample)
    print(f"Cases root: {cases_root}")
    print(f"Samples:   {len(samples)}")
    print(f"Steps:     {', '.join(sorted(steps))}")
    print(f"Tolerance: {args.match_tolerance}")
    if args.force_rebuild:
        print("Force rebuild h5ad: yes")

    ok = 0
    skipped = 0
    for sample in samples:
        if process_sample(
            sample,
            cases_root,
            steps,
            match_tolerance=args.match_tolerance,
            column_rename=column_rename,
            level1_name=args.level1_name,
            force_rebuild=args.force_rebuild,
            overwrite_csv=args.overwrite_csv,
            dry_run=args.dry_run,
        ):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} succeeded, {skipped} skipped, {len(samples)} total.")
    return 0 if ok > 0 or len(samples) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
