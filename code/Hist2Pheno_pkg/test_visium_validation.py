# 2026.11.16, spot-level validation helpers
import json
import numpy as np
import pandas as pd
import pytest

import visium_validation as vv


def _h5(path, genes, bcs, M):
    import h5py
    from scipy.sparse import csc_matrix
    X = csc_matrix(M)  # genes × barcodes
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"], m["indices"], m["indptr"] = X.data, X.indices, X.indptr
        m["shape"] = np.array(X.shape)
        m["barcodes"] = np.array([b.encode() for b in bcs])
        m.create_group("features")["name"] = np.array([g.encode() for g in genes])


def test_capture_radius_and_positions(tmp_path):
    (tmp_path / "scalefactors_json.json").write_text(json.dumps({"spot_diameter_fullres": 130.0}))
    assert vv.capture_radius_px(tmp_path / "scalefactors_json.json") == pytest.approx(55.0)
    pd.DataFrame([["A", 1, 0, 0, 10.0, 20.0], ["B", 0, 0, 1, 1, 1]]).to_csv(
        tmp_path / "tissue_positions_list.csv", header=False, index=False)
    sp = vv.read_tissue_positions(tmp_path)
    assert sp.barcode.tolist() == ["A"] and sp.x[0] == 20.0 and sp.y[0] == 10.0


def test_spot_h2p_groups_and_counts():
    names = ["Tumor", "DCIS", "B_cell", "CD4_T", "Endothelial"]
    xy = np.array([[0, 0], [1, 0], [100, 100], [101, 100]], float)
    P = np.array([[.5, .5, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, .6, .4, 0], [0, 0, 1, 0, 0]])
    spots = pd.DataFrame({"barcode": ["s1", "s2", "s3"], "x": [0, 100, 500], "y": [0, 100, 500]})
    out = vv.spot_h2p(xy, P, names, spots, radius_px=5)
    assert out.n_nuclei.tolist() == [2, 2, 0]
    assert out.frac_Tumor[0] == pytest.approx(1.0)          # DCIS counts as tumour
    assert out.frac_B[1] == pytest.approx(0.8) and out.count_B[1] == pytest.approx(1.6)
    assert np.isnan(out.frac_T[2]) and out.count_T[2] == 0


def test_proportions_and_cell_fractions():
    c = pd.DataFrame({"a": [1.0, 0, 3], "b": [1.0, 0, 1]})
    p = vv.proportions(c)
    assert p.a[0] == .5 and np.isnan(p.a[1]) and p.a[2] == .75
    assert vv.cell_fractions(c).a == pytest.approx(4 / 6)


def test_marker_scores_and_agreement(tmp_path):
    genes = ["MS4A1", "CD79A", "CD3E", "EPCAM", "OTHER"]
    bcs = [f"b{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    b_level = rng.random(30)
    M = np.vstack([rng.poisson(20 * b_level), rng.poisson(20 * b_level), rng.poisson(5, 30),
                   rng.poisson(5, 30), rng.poisson(500, 30)])
    _h5(tmp_path / "m.h5", genes, bcs, M)
    s = vv.marker_scores(tmp_path / "m.h5", bcs[::-1] + ["missing"])
    assert s.index[0] == "b29" and s.loc["missing", "B"] == 0
    assert np.isnan(s["Plasma"]).all()                       # no plasma genes in matrix
    h2p = pd.DataFrame({"frac_B": np.r_[b_level[::-1], np.nan]}, index=s.index)
    tab = vv.agreement(h2p, s.rename(columns={"B": "frac_B"})[["frac_B"]])
    assert tab.spearman[0] > 0.6 and tab.n_spots[0] == 30


def test_annotation_report(tmp_path):
    pd.DataFrame({"Barcode": ["a", "b", "c", "d"], "Final_Annotations": ["GG4", "Benign", "Inflammation", "Exclude"]}).to_csv(
        tmp_path / "x.csv", index=False)
    lab = vv.read_annotations(tmp_path / "x.csv")
    reg = lab.map(vv.annotation_region)
    assert reg.tolist() == ["Tumor", "Benign_epithelium", "Immune", "Exclude"]
    assert vv.annotation_region("PIN") == "Precursor" and vv.annotation_region("Stroma") == "Stroma"
    h2p = pd.DataFrame({"barcode": list("abcd"), "n_nuclei": [5, 5, 5, 5]})
    for g in vv.COMPARTMENTS:
        h2p[f"frac_{g}"] = 0.0
    h2p.loc[0, "frac_Tumor"] = .9; h2p.loc[2, "frac_B"] = .7; h2p.loc[1, "frac_Epithelial"] = .8
    mean, auc = vv.annotation_report(h2p, reg)
    assert "Exclude" not in mean.index and mean.loc["Tumor", "Tumor"] == .9
    assert auc.set_index("region").auroc["Immune"] == 1.0
