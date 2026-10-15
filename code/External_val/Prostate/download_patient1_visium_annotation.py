#!/usr/bin/env python3
"""Download Patient_1/Visium_with_annotation from Mendeley Data.

Source: Erickson et al., Nature 2022, dataset svw96g68dv version 5
  https://data.mendeley.com/datasets/svw96g68dv/5
  Patient 1 > Visium_with_annotation > {H1_2, H1_4, H1_5, H2_1, H2_2, H2_5, V1_2}

Each section folder has:
  filtered_feature_bc_matrix.h5
  {section}_Final_Consensus_Annotations.csv
  scalefactors_json.json
  tissue_positions_list.csv

Usage (repo root):
  python -u code/External_val/Prostate/download_patient1_visium_annotation.py --list
  python -u code/External_val/Prostate/download_patient1_visium_annotation.py
  python -u code/External_val/Prostate/download_patient1_visium_annotation.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATASET_ID = "svw96g68dv"
DATASET_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
UA = "Hist2Pheno/prostate-visium-download (+https://data.mendeley.com/datasets/svw96g68dv/5)"
P1_SECTIONS = ("H1_2", "H1_4", "H1_5", "H2_1", "H2_2", "H2_5", "V1_2")
DEFAULT_DEST = REPO / "data" / "Visium" / "Prostate" / "Patient_1" / "Visium_with_annotation"
ANN_SUFFIX = "_Final_Consensus_Annotations.csv"
EXPECTED_FILES = 4


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_p1_visium(files: list[dict]) -> dict[str, list[dict]]:
    """Map section name -> files in that Mendeley folder."""
    by_folder: dict[str, list[dict]] = {}
    for rec in files:
        fid = rec.get("folder_id")
        if not fid:
            continue
        by_folder.setdefault(fid, []).append(rec)

    out: dict[str, list[dict]] = {}
    for recs in by_folder.values():
        anns = [
            r["filename"]
            for r in recs
            if str(r.get("filename", "")).endswith(ANN_SUFFIX)
        ]
        if len(anns) != 1:
            continue
        section = anns[0][: -len(ANN_SUFFIX)]
        if section in P1_SECTIONS:
            out[section] = recs
    missing = [s for s in P1_SECTIONS if s not in out]
    if missing:
        raise SystemExit(f"Mendeley listing is missing Patient_1 sections: {missing}")
    return {s: out[s] for s in P1_SECTIONS}


def download_one(url: str, dest: Path, expected_size: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        n = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            n += len(chunk)
    if expected_size and n != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: got {n} bytes, expected {expected_size}")
    tmp.replace(dest)


def already_ok(path: Path, size: int, sha: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    if sha and _sha256(path) != sha:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="local Visium_with_annotation directory",
    )
    ap.add_argument("--list", action="store_true", help="print remote files and exit")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download even if size and sha256 already match",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="do not keep dest.bak.YYYYMMDD_HHMMSS if dest already has files",
    )
    args = ap.parse_args()
    dest: Path = args.dest.resolve()

    print(f"dataset  {DATASET_URL}", flush=True)
    ds = _get_json(DATASET_URL)
    print(
        f"name     {ds.get('name')}  version={ds.get('version')}  n_files={len(ds.get('files') or [])}",
        flush=True,
    )
    groups = discover_p1_visium(ds["files"])
    total = sum(int(r.get("size") or 0) for recs in groups.values() for r in recs)
    print(
        f"select   Patient 1 / Visium_with_annotation  "
        f"{len(groups)} sections  {sum(len(v) for v in groups.values())} files  "
        f"{total / 1e6:.1f} MB",
        flush=True,
    )
    for section, recs in groups.items():
        names = ", ".join(sorted(r["filename"] for r in recs))
        print(f"  {section}/  ({len(recs)})  {names}", flush=True)
    if args.list:
        return 0

    staging = dest.parent / (dest.name + ".downloading")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_ok = n_get = 0
    t0 = time.time()
    try:
        for section, recs in groups.items():
            if len(recs) != EXPECTED_FILES:
                raise RuntimeError(f"{section}: expected {EXPECTED_FILES} files, got {len(recs)}")
            for rec in recs:
                name = rec["filename"]
                size = int(rec["size"])
                details = rec.get("content_details") or {}
                sha = details.get("sha256_hash")
                url = details.get("download_url") or (
                    f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}"
                    f"/files/{rec['id']}/file_downloaded"
                )
                out = staging / section / name
                final = dest / section / name
                if (
                    not args.force
                    and already_ok(final, size, sha)
                ):
                    shutil.copy2(final, out)
                    n_ok += 1
                    print(f"  skip  {section}/{name}", flush=True)
                    continue
                print(f"  get   {section}/{name}  {size / 1e6:.1f} MB", flush=True)
                download_one(url, out, size)
                if sha:
                    got = _sha256(out)
                    if got != sha:
                        raise RuntimeError(f"sha256 mismatch {section}/{name}\n  want {sha}\n  got  {got}")
                n_get += 1
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if dest.exists() and any(dest.iterdir()) and not args.no_backup:
        bak = dest.parent / f"{dest.name}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dest.rename(bak)
        print(f"backup  {bak}", flush=True)
    elif dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)

    elapsed = time.time() - t0
    print(
        f"done    {dest}\n"
        f"        downloaded={n_get}  reused={n_ok}  wall={elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
