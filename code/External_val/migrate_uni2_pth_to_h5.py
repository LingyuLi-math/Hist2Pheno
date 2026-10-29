#!/usr/bin/env python3
"""Convert verified External_val legacy stores to standard embeddings.h5ad.

Does not run the image encoder. Each .pth file is moved into
previous_embeddings/ after embeddings.h5 and a completed manifest exist.
Lung_Incomplete and PDAC_unlabel are included once their extracts have finished.

  python -u code/External_val/migrate_uni2_pth_to_h5.py --only BRCA_ST/5714_HE_BT_C1
  python -u code/External_val/migrate_uni2_pth_to_h5.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "code" / "Hist2Pheno_pkg"
_PAN = _REPO / "code" / "PanCancer"
for _p in (
    _PKG,
    _PAN,
    _REPO / "code" / "External_val",
    _REPO / "code" / "External_val" / "BRCA",
    _REPO / "code" / "External_val" / "BRCA_ST",
    _REPO / "code" / "External_val" / "Prostate",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

import pipeline_provenance as pp  # noqa: E402
from embedding_store import (  # noqa: E402
    FILENAME,
    extraction_metadata,
    migrate_pth,
    uses_hdf5,
)
from feature_integrity import feature_manifest  # noqa: E402

WEIGHT = Path(
    "/nobackup2/users/lingyu/.cache/huggingface/hub/blobs/d2/"
    "d218ea3fde0f9dca913294fd93925444679c8a87e10750dd55ff3ed9f5c80cc4"
)
EXTRACTOR = _REPO / "code" / "Image_feature_extraction.py"
HCC_HE = _REPO / "data" / "CODEX" / "HCC" / "Michael_data_transfer" / "s4769" / "HE"
HCC_CSV = _REPO / "data" / "CODEX" / "HCC" / "StarDist_Segment"
LUNG_CASES = (
    _REPO
    / "data"
    / "Xemium"
    / "weiqin"
    / "SpatialPF-NGenetics"
    / "Spatial-PF-Processed"
    / "Data"
    / "Incomplete_Cases"
)
PDAC_ROOT = _REPO / "data" / "CODEX" / "HCC" / "Michael_data_transfer" / "s1167"
PDAC_CSV = _REPO / "data" / "CODEX" / "HCC" / "StarDist_Segment_pdac" / "pdac_result"
# A finished slide can omit a few edge crops. Below this, the extract did not finish.
_MIN_SAVED_RATIO = 0.95


def _has_pth(emb: Path) -> bool:
    return any(entry.name.endswith(".pth") for entry in os.scandir(emb))


def _pth_count(emb: Path) -> int:
    return sum(1 for entry in os.scandir(emb) if entry.name.endswith(".pth"))


def _csv_rows(path: Path) -> int:
    with path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _saved_enough(n_pth: int, n_csv: int) -> bool:
    if n_csv <= 0:
        return False
    return n_pth / n_csv >= _MIN_SAVED_RATIO


def _encoder(cache: Path) -> dict:
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    return pp.file_record(WEIGHT, cache)


def _extractor_sha() -> str:
    return pp.sha256(EXTRACTOR)


def _contract(he: Path, csv: Path, scale: float, scale_image: bool, encoder: dict, extractor_sha: str, cache: Path) -> dict:
    import torch

    return {
        "inputs": {"image": pp.file_record(he, cache), "positions": pp.file_record(csv, cache)},
        "method": "UNI2",
        "patch_size": 14,
        "scale_image": bool(scale_image),
        "scale": float(scale),
        "encoder_weights": encoder,
        "extractor_sha256": extractor_sha,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
    }


def _drop_stale_hdf5_marker(emb: Path) -> None:
    """A writing marker with storage_format=hdf5 and no file blocks .pth migration."""
    marker = emb / "embedding_manifest.json"
    if not marker.is_file() or (emb / FILENAME).is_file():
        return
    meta = extraction_metadata(emb)
    if meta.get("storage_format") == "hdf5" and meta.get("state") != "completed":
        marker.unlink()


def _seal(emb: Path, contract: dict, cache: Path) -> None:
    # Historical sealing entry retained, intentionally disabled / 禁止补造当前来源。
    raise RuntimeError("Cannot certify legacy tokens from a current contract; provide a verified original extraction manifest or re-extract")


def _stamp_extractor(emb: Path, extractor_sha: str) -> None:
    # No SHA rewriting, even for format-only conversions / 保留真实历史提取 SHA。
    if extraction_metadata(emb).get("extractor_sha256")!=extractor_sha:
        raise RuntimeError("Do not overwrite historical extractor SHA; use reviewed compatibility or re-extract")


def convert_one(emb: Path, contract: dict | None, cache: Path, extractor_sha: str) -> str:
    # 2026.10.29: current image/CSV/weight facts cannot certify historical vectors.
    from embedding_store import storage_format
    emb=emb.resolve()
    feature_manifest(emb,cache)  # Requires original completed, current-source-valid manifest.
    if storage_format(emb)=="h5ad":return "already"
    migrate_pth(emb,archive_pth=True)
    feature_manifest(emb,cache)
    return "converted"


def _cohorts():
    from brca_external import cohort_for as brca_for
    from brca_st_external import cohort_for as brca_st_for
    from external_validation import simple_cohort
    from prostate_external import cohort_for as prostate_for

    saved = os.environ.get("MPP_UM")
    os.environ["MPP_UM"] = "0.5"
    hcc = simple_cohort("HCC", _REPO, "uni2")
    if saved is None:
        os.environ.pop("MPP_UM", None)
    else:
        os.environ["MPP_UM"] = saved
    return [
        ("BRCA", brca_for("uni2")),
        ("BRCA_ST", brca_st_for("uni2")),
        ("Prostate", prostate_for("uni2")),
        ("NPC", simple_cohort("NPC", _REPO, "uni2")),
        ("CRC", simple_cohort("CRC", _REPO, "uni2")),
        ("HCC", hcc),
        ("HCC_nature", simple_cohort("HCC_NATURE", _REPO, "uni2")),
    ]


def _seal_existing_hdf5(emb: Path, contract: dict | None, cache: Path) -> None:
    raise RuntimeError("Incomplete stores cannot be resealed from current sources; restore a verified transaction or re-extract")


def _needs_work(emb: Path) -> bool:
    from embedding_store import storage_format
    if not emb.is_dir():return False
    if storage_format(emb)=="h5ad":return False
    return _has_pth(emb) or (emb/"embeddings.h5").is_file()


def _job_record(name, sample, emb, he, csv, scale, scale_image, cache, seal: bool):
    return {
        "name": name,
        "sample": sample,
        "emb": str(emb),
        "he": str(he) if he is not None else "",
        "csv": str(csv) if csv is not None else "",
        "scale": float(scale) if scale is not None else 0.0,
        "scale_image": bool(scale_image),
        "cache": str(cache),
        "seal": seal,
    }


def iter_jobs(only: str | None):
    wanted = None
    if only:
        cohort_name, _, sample = only.partition("/")
        wanted = (cohort_name, sample)
    for name, cohort in _cohorts():
        cache = cohort.uni_root / ".provenance"
        for sample in cohort.discover_samples():
            if wanted and (name, sample) != wanted:
                continue
            paths = cohort.sample_paths(sample)
            emb = paths["emb"]
            if not _needs_work(emb):
                continue
            meta = extraction_metadata(emb)
            seal = not (meta.get("state") == "completed" and meta.get("method") == "UNI2" and not uses_hdf5(emb))
            geo = cohort.geometry(sample) if seal else {"scale": 0.0, "scale_image": False}
            yield _job_record(
                name, sample, emb, paths["he"], paths["csv"], geo["scale"], geo["scale_image"], cache, seal
            )
    if wanted and wanted[0] not in {"HCC_unlabel", "Lung_Incomplete", "PDAC_unlabel"}:
        return
    if (wanted is None or wanted[0] == "HCC_unlabel") and HCC_HE.is_dir():
        yield from _hcc_unlabel_jobs(wanted)
    if wanted is None or wanted[0] == "Lung_Incomplete":
        yield from _lung_jobs(wanted)
    if wanted is None or wanted[0] == "PDAC_unlabel":
        yield from _pdac_jobs(wanted)


def _accept(name, sample, emb, he, csv, scale, scale_image, cache):
    if not emb.is_dir() or not _needs_work(emb):
        if emb.is_dir() and not _has_pth(emb) and not (emb / FILENAME).is_file():
            print(f"SKIP {name}/{sample}: no UNI2 embeddings", flush=True)
        return None
    meta = extraction_metadata(emb)
    seal = not (meta.get("state") == "completed" and meta.get("method") == "UNI2" and not uses_hdf5(emb))
    if seal:
        if not he.is_file() or not csv.is_file():
            print(f"SKIP {name}/{sample}: missing HE or CSV", flush=True)
            return None
        if _has_pth(emb):
            saved = _pth_count(emb)
        else:
            import h5py

            with h5py.File(emb / FILENAME, "r") as handle:
                saved = int(handle["embeddings"].shape[0]) if handle.attrs.get("complete") else 0
        n_csv = _csv_rows(csv)
        if not _saved_enough(saved, n_csv):
            print(f"SKIP {name}/{sample}: saved {saved} of {n_csv} CSV rows", flush=True)
            return None
    return _job_record(name, sample, emb, he, csv, scale, scale_image, cache, seal)


def _hcc_unlabel_jobs(wanted):
    cache = HCC_HE / ".provenance"
    for sample_dir in sorted(p for p in HCC_HE.iterdir() if p.is_dir()):
        sample = sample_dir.name
        if wanted and sample != wanted[1]:
            continue
        emb = sample_dir / "project_all_UNI2" / "ImgEmbeddings_all_stardist" / "sc_pth_14_14"
        if not _needs_work(emb):
            continue
        he = sample_dir / "figures" / f"{sample}.tif"
        csv = HCC_CSV / sample / f"{sample}_Float_prob0.01_nms_0.3.csv"
        if not he.is_file() or not csv.is_file():
            print(f"SKIP HCC_unlabel/{sample}: missing HE or CSV", flush=True)
            continue
        # These cores were extracted with scale_image False.
        yield _job_record("HCC_unlabel", sample, emb, he, csv, 1.0, False, cache, True)


def _lung_jobs(wanted):
    """Xenium Incomplete_Cases. Finished slides used scale 0.425, scale_image True."""
    cache = LUNG_CASES / ".provenance"
    if not LUNG_CASES.is_dir():
        return
    for sample_dir in sorted(p for p in LUNG_CASES.iterdir() if p.is_dir()):
        sample = sample_dir.name
        if wanted and sample != wanted[1]:
            continue
        emb = sample_dir / f"{sample}_project_all_UNI2" / "ImgEmbeddings_all_stardist" / "sc_pth_14_14"
        job = _accept(
            "Lung_Incomplete",
            sample,
            emb,
            sample_dir / f"{sample}-HE.tif",
            sample_dir / f"{sample}_Float_prob0.01_nms_0.3.csv",
            0.425,
            True,
            cache,
        )
        if job is not None:
            yield job


def _pdac_jobs(wanted):
    """195 unlabeled PDAC cores. The extractor was run with scale_image False, default scale 0.5."""
    import sys

    code = _REPO / "code" / "CODEX_pdac"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from match_codex_cells_with_pixel import list_incomplete_pdac_regions

    cache = PDAC_ROOT / ".provenance"
    names = list_incomplete_pdac_regions()["ACQUISITION_ID"].astype(str).str.strip().tolist()
    for sample in names:
        if wanted and sample != wanted[1]:
            continue
        emb = PDAC_ROOT / sample / "project_all_UNI2" / "ImgEmbeddings_all_stardist" / "sc_pth_14_14"
        job = _accept(
            "PDAC_unlabel",
            sample,
            emb,
            PDAC_ROOT / sample / "figures" / f"{sample}.tif",
            PDAC_CSV / sample / f"{sample}_Float_prob0.01_nms_0.3.csv",
            0.5,
            False,
            cache,
        )
        if job is not None:
            yield job


_COHORTS = None


def _cohort_map():
    global _COHORTS
    if _COHORTS is None:
        _COHORTS = {label: cohort for label, cohort in _cohorts()}
    return _COHORTS


def _verify_cohort(name: str, sample: str) -> None:
    if name in {"HCC_unlabel", "Lung_Incomplete", "PDAC_unlabel"}:
        return
    from external_consistency import extraction_complete

    if not extraction_complete(_cohort_map()[name], sample):
        raise RuntimeError(f"extraction_complete is false after conversion: {name}/{sample}")


def run_job(payload: dict) -> str:
    # Conversion consumes the original verified contract, never current facts.
    return convert_one(Path(payload["emb"]),None,Path(payload["cache"]),payload["extractor_sha"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Cohort/sample, for example BRCA_ST/5714_HE_BT_C1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    extractor_sha = _extractor_sha()
    jobs = list(iter_jobs(args.only))
    for job in jobs:
        job["extractor_sha"] = extractor_sha
    print(
        f"jobs {len(jobs)}  workers {args.workers}  extractor {extractor_sha[:12]}",
        flush=True,
    )
    if not jobs:
        print("nothing to convert", flush=True)
        return 0
    failures = 0
    if args.workers <= 1 or len(jobs) == 1:
        for job in jobs:
            label = f"{job['name']}/{job['sample']}"
            print(f"START {label}", flush=True)
            try:
                status = run_job(job)
                _verify_cohort(job["name"], job["sample"])
                print(f"DONE {label} {status}", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL {label}\n{traceback.format_exc()}", flush=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            pending = [
                (f"{job['name']}/{job['sample']}", job["name"], job["sample"], pool.apply_async(run_job, (job,)))
                for job in jobs
            ]
            for label, name, sample, result in pending:
                print(f"WAIT {label}", flush=True)
                try:
                    status = result.get()
                    _verify_cohort(name, sample)
                    print(f"DONE {label} {status}", flush=True)
                except Exception:
                    failures += 1
                    print(f"FAIL {label}\n{traceback.format_exc()}", flush=True)
    print(f"finished failures={failures} jobs={len(jobs)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
