#!/usr/bin/env python3
"""Scan seven-cohort HE h5ads and write a pan-cancer section index.

Does not copy UNI embeddings. Training reads the original matched h5ads and remaps
labels on the fly using ``celltype_pancancer``.

  conda activate SeededNTM
  python -u code/PanCancer/prepare_pancancer_index.py
  python -u code/PanCancer/prepare_pancancer_index.py --datasets BRCA GBM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pancancer_cv import cache_path_for, fold_plan_frame, ok_mask  # noqa: E402
from pancancer_labels import map_native_series, ontology_class_names  # noqa: E402
from pancancer_paths import (  # noqa: E402
    COHORT_ORDER,
    DEFAULT_DATA_ROOT,
    discover_h5ads,
)


def _obs_native(obs: pd.DataFrame, native_col: str) -> pd.Series:
    if native_col in obs.columns:
        return obs[native_col]
    if "celltype" in obs.columns:
        return obs["celltype"]
    raise KeyError(f"no native label column among {list(obs.columns)}")


def _close_backed(adata) -> None:
    fh = getattr(adata, "file", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def inspect_h5ad(
    path: Path,
    native_col: str,
    *,
    cache_path: Path | None = None,
) -> dict:
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(path, backed="r")
    try:
        native = _obs_native(adata.obs, native_col)
        n_raw = int(adata.n_obs)
        mapped = map_native_series(native)
        keep_mask = mapped.notna().to_numpy()
        n_keep = int(keep_mask.sum())
        has_xy = "spatial_HE" in adata.obsm or "spatial" in adata.obsm
        n_feat = int(adata.n_vars) if adata.n_vars else -1
        if cache_path is not None and n_keep > 0 and has_xy:
            idx = np.flatnonzero(keep_mask)
            triples = mapped.to_numpy()[keep_mask]
            key = "spatial_HE" if "spatial_HE" in adata.obsm else "spatial"
            xy = np.asarray(adata.obsm[key], dtype=np.float64)[idx]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                keep_idx=idx.astype(np.int64),
                y=np.asarray([t[2] for t in triples], dtype=object),
                y_l1=np.asarray([t[0] for t in triples], dtype=object),
                y_l12=np.asarray([t[1] for t in triples], dtype=object),
                xy=xy,
                n_raw=np.int64(n_raw),
            )
        return {
            "n_raw": n_raw,
            "n_keep": n_keep,
            "n_drop": n_raw - n_keep,
            "has_spatial": bool(has_xy),
            "n_features": n_feat,
            "ok": n_keep > 0 and has_xy,
            "label_cache": str(cache_path) if cache_path is not None else "",
        }
    finally:
        _close_backed(adata)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="*", default=None, help=f"Subset of {COHORT_ORDER}")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--xlsx", type=Path, default=None)
    p.add_argument("--cv-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-label-cache", action="store_true")
    args = p.parse_args(argv)

    classes = ontology_class_names(args.xlsx)
    rows_meta = discover_h5ads(args.datasets)
    print(f"Discovered {len(rows_meta)} h5ad files", flush=True)

    out_dir = args.out_dir.expanduser().resolve()
    cache_dir = out_dir / "label_cache"
    if not args.no_label_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in enumerate(rows_meta, 1):
        path = Path(row["h5ad"])
        cache_path = None if args.no_label_cache else cache_path_for(row["section_id"], cache_dir)
        try:
            stats = inspect_h5ad(path, row["native_col"], cache_path=cache_path)
        except Exception as exc:
            stats = {
                "n_raw": 0,
                "n_keep": 0,
                "n_drop": 0,
                "has_spatial": False,
                "n_features": -1,
                "ok": False,
                "error": str(exc),
            }
            print(f"  FAIL {row['section_id']}: {exc}", flush=True)
        rec = {**row, **stats}
        records.append(rec)
        if i % 40 == 0 or i == len(rows_meta):
            print(f"  inspected {i}/{len(rows_meta)}", flush=True)

    index = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "section_index.csv"
    index.to_csv(index_path, index=False)
    plan = fold_plan_frame(index, n_splits=args.cv_k, seed=args.seed)
    plan_path = out_dir / "fold_plan.csv"
    plan.to_csv(plan_path, index=False)

    ok = index.loc[ok_mask(index)]
    summary = (
        ok.groupby("dataset", dropna=False)
        .agg(n_sections=("section_id", "nunique"), n_keep=("n_keep", "sum"), n_raw=("n_raw", "sum"))
        .reindex(COHORT_ORDER)
        .dropna(how="all")
    )
    summary_path = out_dir / "section_index_summary.csv"
    summary.to_csv(summary_path)
    meta = {
        "n_h5ad": int(len(index)),
        "n_ok": int(ok.shape[0]),
        "n_keep_cells": int(ok["n_keep"].sum()) if len(ok) else 0,
        "L1": classes["L1"],
        "L12": classes["L12"],
        "L2": classes["L2"],
        "index_csv": str(index_path),
        "fold_plan_csv": str(plan_path),
        "cv_k": int(args.cv_k),
        "label_cache_dir": str(cache_dir) if not args.no_label_cache else "",
    }
    (out_dir / "ontology_heads.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {index_path}")
    print(f"Wrote {plan_path}  ({len(plan)} OK sections)")
    print(f"OK sections {meta['n_ok']}/{meta['n_h5ad']}, kept cells {meta['n_keep_cells']:,}")
    print(summary.fillna(0).astype({"n_sections": int}, errors="ignore").to_string())
    if not plan.empty:
        print("\nVal sections per fold × dataset:")
        print(pd.crosstab(plan["fold"], plan["dataset"]).reindex(columns=COHORT_ORDER).fillna(0).astype(int).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
