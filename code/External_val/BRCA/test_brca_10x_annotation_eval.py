"""2026.11.17 synthetic checks of brca_10x_annotation_eval (no data needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brca_10x_annotation_eval as ev  # noqa: E402


def test_score_calls_matches_definitions():
    region = np.array(["IDC"] * 4 + ["DCIS/LCIS"] * 2 + ["Healthy"] * 2 + ["Tumor edge"])
    call = np.array(["Tumor", "DCIS", "DCIS", "Fibroblast", "DCIS", "Tumor", "Endothelial", "Tumor", "Endothelial"])
    s = ev.score_calls(region, call)
    assert s["IDC_called_Tumor"] == 0.25
    assert s["DCIS_called_DCIS"] == 0.5
    assert np.isclose(s["cancer_called_neoplastic"], 5 / 6)
    assert s["healthy_not_neoplastic"] == 0.5
    assert np.isclose(s["name_correct"], 2 / 5)          # IDC->Tumor once, DCIS->DCIS once, of 5 named
    assert np.isclose(s["score"], np.mean([5 / 6, 0.5, 0.4]))


def test_annotated_spots_radius_and_majority(tmp_path, monkeypatch):
    sp = tmp_path / "spatial"; sp.mkdir()
    (sp / "scalefactors_json.json").write_text(json.dumps({"spot_diameter_fullres": 130.0}))
    pd.DataFrame([["A", 1, 0, 0, 100, 100], ["B", 1, 0, 1, 100, 400], ["C", 0, 0, 2, 100, 700]]).to_csv(
        sp / "tissue_positions_list.csv", header=False, index=False)
    pd.DataFrame({"ID": ["A", "B", "C"], "fine_annot_type": ["IDC_1", "Healthy_2", "IDC_3"]}).to_csv(
        tmp_path / "metadata.tsv", sep="\t", index=False)
    monkeypatch.setattr(ev, "DATA_10X", tmp_path)
    spots, r = ev.annotated_spots(55.0)
    assert list(spots.barcode) == ["A", "B"] and np.isclose(r, 55.0)      # 130 * 55/65 / 2
    xy = np.array([[100, 100], [110, 100], [100, 150], [400, 100], [400, 300]], float)
    lab = np.array(["Tumor", "Tumor", "DCIS", "Endothelial", "Tumor"])
    call, n = ev.spot_majority(xy, lab, spots, r)
    assert list(n) == [3, 1] and list(call) == ["Tumor", "Endothelial"]
