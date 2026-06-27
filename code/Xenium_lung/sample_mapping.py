"""Paper (Supplementary Table 6) sample IDs <-> Weiqin HE annotation sample IDs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAPER_TO_WEIQIN = {
    "THD0008": "THD0008",
    "THD0011": "THD0011",
    "VUHD038": "VUHD038",
    "VUHD049": "VUHD049",
    "VUHD069": "VUHD069",
    "VUHD090": "VUHD090",
    "VUHD095": "VUHD095",
    "VUHD113": "VUHD113",
    "VUHD116A": "VUHD116A",
    "VUHD116B": "VUHD116B",
    "TILD111LA": "TILD111LF",
    "TILD117LA": "TILD117LF",
    "TILD167LA": "TILD167LF",
    "TILD117MA1": "TILD117MF",
    "TILD117MA2": "TILD117MFB",
    "TILD028LA": "TILD028MF",
    "TILD049MA": "TILD049LF",
    "TILD080LA": "TILD080MF",
    "TILD113LA": "TILD113MF",
    "TILD130LA": "TILD130MF",
    "TILD299MA": "TILD299LF",
    "TILD315MA": "TILD315LF",
    "TILD175MA": "TILD175",
    "VUILD78LA": "VUILD78LF",
    "VUILD91LA": "VUILD91LF",
    "VUILD96LA": "VUILD96LF",
    "VUILD102LA": "VUILD102LF",
    "VUILD78MA": "VUILD78MF",
    "VUILD91MA": "VUILD91MF",
    "VUILD96MA": "VUILD96MF",
    "VUILD102MA": "VUILD102MF",
    "VUILD107MA": "VUILD107MF",
    "VUILD48LA1": "VUILD48LF",
    "VUILD48LA2": "VUILD48MF",
    # 2026.06.22 — LF/MF swap for MA1 vs MA2 (aligned with Data_process_visual_xenium_all.ipynb)
    "VUILD104MA1": "VUILD104LF",
    "VUILD104MA2": "VUILD104MF",
    "VUILD105MA1": "VUILD105LF",
    "VUILD105MA2": "VUILD105MF",
    "VUILD49LA": "VUILD49",
    "VUILD58MA": "VUILD58",
    "VUILD106MA": "VUILD106",
    "VUILD110LA": "VUILD110",
    "VUILD115MA": "VUILD115",
    "VUILD141MA": "VUILD141",
    "VUILD142MA": "VUILD142",
}
WEIQIN_TO_PAPER = {v: k for k, v in PAPER_TO_WEIQIN.items()}

DEFAULT_PAPER_SHEET = "Supplementary Table 6"


def load_select_samples_from_paper_xlsx(
    xlsx_path: str | Path,
    *,
    sheet_name: str = DEFAULT_PAPER_SHEET,
    header: int = 1,
) -> list[str]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header)
    return df["Sample"].dropna().unique().tolist()


def match_weiqin_cells_to_paper_samples(
    df: pd.DataFrame,
    select_samples: list[str],
    *,
    paper_to_weiqin: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Filter Weiqin cells and rename sample / full_cell_id to paper IDs."""
    paper_to_weiqin = paper_to_weiqin or PAPER_TO_WEIQIN
    weiqin_to_paper = {v: k for k, v in paper_to_weiqin.items()}

    weiqin_ids = sorted({paper_to_weiqin[s] for s in select_samples if s in paper_to_weiqin})
    unknown_paper = sorted(set(select_samples) - set(paper_to_weiqin))

    out = df[df["sample"].isin(weiqin_ids)].copy()
    out["sample"] = out["sample"].replace(weiqin_to_paper)

    full_cell_id = out["full_cell_id"]
    for weiqin_id, paper_id in sorted(weiqin_to_paper.items(), key=lambda kv: len(kv[0]), reverse=True):
        if weiqin_id != paper_id:
            full_cell_id = full_cell_id.str.replace(f"{weiqin_id}_", f"{paper_id}_", regex=False)
    out["full_cell_id"] = full_cell_id

    return out, weiqin_ids, unknown_paper
