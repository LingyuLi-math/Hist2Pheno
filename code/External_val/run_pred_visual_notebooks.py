#!/usr/bin/env python3
"""Run every Pred_visual_*.ipynb in External_val, one after another.

Does not write outputs back into the source notebooks (those files stay small).
Side effects go to each cohort's pred_pancancer_section/figures/ when
SAVE_FIGURES=1 (default here). Index-only Prostate notebooks are skipped.

From the repo root:

    conda activate SeededNTM
    bash code/External_val/demo.sh notebooks
    ONLY=BRCA,CRC bash code/External_val/demo.sh notebooks
    SAVE_FIGURES=0 bash code/External_val/demo.sh notebooks
"""

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

# Small cohorts first. Index-only Prostate files are skipped (no code cells).
_ORDER = (
    "BRCA/Pred_visual_brca.ipynb",
    "NPC/Pred_visual_npc.ipynb",
    "CRC/Pred_visual_crc.ipynb",
    "HCC/Pred_visual_hcc.ipynb",
    "HCC_nature/Pred_visual_hcc_nature_ready.ipynb",
    "HCC_nature/Pred_visual_hcc_nature.ipynb",
    "BRCA_ST/Pred_visual_brca_st_A.ipynb",
    "BRCA_ST/Pred_visual_brca_st_B.ipynb",
    "BRCA_ST/Pred_visual_brca_st_C.ipynb",
    "BRCA_ST/Pred_visual_brca_st_D.ipynb",
    "BRCA_ST/Pred_visual_brca_st_E.ipynb",
    "BRCA_ST/Pred_visual_brca_st_F.ipynb",
    "BRCA_ST/Pred_visual_brca_st_G.ipynb",
    "BRCA_ST/Pred_visual_brca_st_H.ipynb",
    "Prostate/Pred_visual_prostate_patient1_H1.ipynb",
    "Prostate/Pred_visual_prostate_patient1_H2.ipynb",
    "Prostate/Pred_visual_prostate_patient1_V1.ipynb",
    "Prostate/Pred_visual_prostate_patient2_H1.ipynb",
    "Prostate/Pred_visual_prostate_patient2_H2.ipynb",
    "Prostate/Pred_visual_prostate_patient2_H3.ipynb",
    "Prostate/Pred_visual_prostate_patient2_V1.ipynb",
    "Prostate/Pred_visual_prostate_patient2_V2.ipynb",
    "Lung_Incomplete/Pred_visual_lung_incomplete.ipynb",
    "HCC_unlabel/Pred_visual_hcc_unlabel.ipynb",
    "PDAC_unlabel/Pred_visual_pdac_unlabel.ipynb",
)


def _hms(seconds: float) -> str:
    t = int(round(seconds))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def _truthy(value: str | None, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _inject_save_figures(source: str, save_figures: bool) -> str:
    if not save_figures:
        return source
    return source.replace("SAVE_FIGURES = False", "SAVE_FIGURES = True")


def discover_notebooks(only: list[str]) -> list[Path]:
    encoder=os.environ.get("EMBEDDER","uni").lower()
    if encoder not in ("uni","uni2","virchow2"):raise ValueError("EMBEDDER must be uni, uni2 or virchow2")
    found: list[Path] = []
    for rel in _ORDER:
        base=_HERE/rel
        suffixes=["_uni2_A2", "_uni2"] if encoder=="uni2" else ["_"+encoder]
        for suffix in suffixes:
            path=base.with_name(base.stem+suffix+base.suffix)
            if path.is_file():found.append(path);break
    if not only:
        return found
    keys = [k.strip().lower() for k in only if k.strip()]
    return [p for p in found if any(k in str(p.relative_to(_HERE)).lower() for k in keys)]


def run_notebook(path: Path, *, save_figures: bool) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    if not code_cells:
        print(f"  skip (index / no code cells): {path.name}", flush=True)
        return

    os.chdir(path.parent)
    shell = InteractiveShell.instance()
    shell.reset(new_session=True)
    # Notebooks call display(); keep it a no-op so Agg does not keep rasters.
    try:
        from IPython.display import display as _display  # noqa: F401

        shell.user_ns["display"] = lambda *a, **k: None
    except Exception:
        pass

    for i, cell in enumerate(code_cells, start=1):
        src = "".join(cell.get("source") or [])
        src = _inject_save_figures(src, save_figures)
        if not src.strip():
            continue
        print(f"  cell {i}/{len(code_cells)}", flush=True)
        result = shell.run_cell(src, store_history=False, silent=True)
        err = result.error_before_exec or result.error_in_exec
        if err is not None:
            raise RuntimeError(f"{path.name} cell {i} failed: {err!r}") from err


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        default=os.environ.get("ONLY", ""),
        help="comma-separated path substrings, e.g. BRCA,Prostate_patient1_H1",
    )
    p.add_argument(
        "--save-figures",
        dest="save_figures",
        action="store_true",
        default=_truthy(os.environ.get("SAVE_FIGURES"), True),
    )
    p.add_argument(
        "--no-save-figures",
        dest="save_figures",
        action="store_false",
        help="keep SAVE_FIGURES=False (smoke run; no PNGs)",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        default=_truthy(os.environ.get("STOP_ON_ERROR"), False),
    )
    args = p.parse_args(argv)

    only = [x for x in str(args.only).split(",") if x.strip()]
    notebooks = discover_notebooks(only)
    if not notebooks:
        print("No Pred_visual notebooks matched.", file=sys.stderr)
        return 1

    log_path = _HERE / "Pred_visual_batch.log"
    print(f"n={len(notebooks)}  SAVE_FIGURES={int(args.save_figures)}  log={log_path}", flush=True)
    failed: list[str] = []
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n========== notebooks  {time.strftime('%F %T')}  "
            f"SAVE_FIGURES={int(args.save_figures)}  ONLY={args.only or 'all'} ==========\n"
        )
        for i, path in enumerate(notebooks, start=1):
            rel = path.relative_to(_HERE)
            print(f"\n[{i}/{len(notebooks)}] {rel}", flush=True)
            log.write(f"start {rel}  {time.strftime('%F %T')}\n")
            log.flush()
            start = time.time()
            try:
                run_notebook(path, save_figures=args.save_figures)
            except Exception as exc:
                elapsed = time.time() - start
                print(f"  FAIL  {_hms(elapsed)}  {exc}", flush=True)
                traceback.print_exc()
                log.write(f"FAIL  {rel}  {_hms(elapsed)}  {exc}\n")
                failed.append(str(rel))
                if args.stop_on_error:
                    break
            else:
                elapsed = time.time() - start
                print(f"  ok    {_hms(elapsed)}", flush=True)
                log.write(f"ok    {rel}  {_hms(elapsed)}\n")
            log.flush()
        total = time.time() - t0
        summary = (
            f"finished {time.strftime('%F %T')}  elapsed={_hms(total)}  "
            f"ok={len(notebooks) - len(failed)}/{len(notebooks)}"
        )
        if failed:
            summary += "  failed=" + ", ".join(failed)
        print("\n" + summary, flush=True)
        log.write(summary + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
