#!/usr/bin/env python3
"""Run Internal_val Pred_visual_*.ipynb with IPython + Agg. Does not rewrite sources."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_ORDER = (
    "Pred_visual_fold0.ipynb",
    "Pred_visual_lung.ipynb",
    "Pred_visual_brca.ipynb",
    "Pred_visual_crc.ipynb",
    "Pred_visual_hcc.ipynb",
    "Pred_visual_pdac.ipynb",
    "Pred_visual_gist.ipynb",
    "Pred_visual_gbm.ipynb",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma substrings of the filename")
    args = p.parse_args()
    keys = [s.strip() for s in args.only.split(",") if s.strip()]
    nbs = [_HERE / name for name in _ORDER if (_HERE / name).is_file()]
    if keys:
        nbs = [p for p in nbs if any(k.lower() in p.name.lower() for k in keys)]
    if not nbs:
        print("no notebooks")
        return 1

    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    shell.run_cell("import matplotlib; matplotlib.use('Agg')")
    t0 = time.time()
    failed = []
    for i, nb_path in enumerate(nbs, 1):
        print(f"===== [{i}/{len(nbs)}] {nb_path.name} =====", flush=True)
        import json

        nb = json.loads(nb_path.read_text())
        try:
            for cell in nb["cells"]:
                if cell.get("cell_type") != "code":
                    continue
                src = "".join(cell.get("source") or [])
                if not src.strip():
                    continue
                result = shell.run_cell(src)
                if result.error_in_exec is not None:
                    raise RuntimeError(result.error_in_exec)
        except Exception:
            traceback.print_exc()
            failed.append(nb_path.name)
    print(f"done wall={time.time() - t0:.1f}s  failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
