# 2026.11.17, centroid axis check against the instance mask
import os
import time
import numpy as np
import pandas as pd
import anndata as ad
import pytest

import prepare_spora_sections as P


def _setup(tmp_path, monkeypatch, swapped, square=True):
    monkeypatch.setattr(P, "DATA_ROOT", tmp_path / "spora")
    monkeypatch.setattr(P, "PREPARED", tmp_path / "spora" / "prepared")
    ds, tissue = "schurch2020coordinated", "reg001_A"
    mask = np.zeros((60, 60 if square else 90), dtype=np.int32)
    rng = np.random.default_rng(0)
    ids = []
    for k in range(1, 41):
        r, c = rng.integers(3, 57), rng.integers(3, mask.shape[1] - 3)
        if mask[r - 1:r + 2, c - 1:c + 2].any():
            continue
        mask[r - 1:r + 2, c - 1:c + 2] = k
        ids.append((k, r, c))
    mdir = tmp_path / "spora" / ds / ds / "segmentations" / "1_0mpp" / "cell_masks" / "instances"
    mdir.mkdir(parents=True)
    np.savez(mdir / f"{tissue}.npz", mask=mask)
    out = tmp_path / "spora" / "prepared" / ds / tissue
    out.mkdir(parents=True)
    k, r, c = map(np.asarray, zip(*ids))
    x, y = (r, c) if swapped else (c, r)
    frame = pd.DataFrame({"cell_id": k, "centroid_x": x.astype(float), "centroid_y": y.astype(float), "final_CT": "B cells"})
    frame.to_csv(out / f"{tissue}_cells_with_pixel.csv", index=False)
    a = ad.AnnData(obs=frame.set_index(frame.cell_id.astype(str), drop=False))
    a.obs.index.name = None
    a.obsm["spatial_HE"] = frame[["centroid_x", "centroid_y"]].to_numpy(float)
    a.write_h5ad(out / f"{tissue}_matched_features.h5ad")
    store = out / f"{tissue}_project_all_UNI2" / "ImgEmbeddings_all" / "sc_pth_14_14"
    store.mkdir(parents=True)
    (store / "x.h5ad").write_text("old")
    old = time.time() - 3600
    for p in (store / "x.h5ad", store, store.parent, store.parent.parent):
        os.utime(p, (old, old))
    return ds, tissue, out, frame


def test_square_tissue_swapped_is_fixed_and_store_retired(tmp_path, monkeypatch):
    ds, tissue, out, frame = _setup(tmp_path, monkeypatch, swapped=True, square=True)
    msg = P.fix_centroid_axis(ds)
    assert "swapped 1" in msg and "retired 1" in msg
    new = pd.read_csv(out / f"{tissue}_cells_with_pixel.csv")
    assert np.allclose(new.centroid_x, frame.centroid_y) and np.allclose(new.centroid_y, frame.centroid_x)
    a = ad.read_h5ad(out / f"{tissue}_matched_features.h5ad")
    assert np.allclose(a.obsm["spatial_HE"][:, 0], new.centroid_x)
    assert not (out / f"{tissue}_project_all_UNI2").exists()
    # idempotent: second run swaps nothing
    assert "swapped 0, already correct 1" in P.fix_centroid_axis(ds)


def test_correct_table_untouched(tmp_path, monkeypatch):
    ds, tissue, out, frame = _setup(tmp_path, monkeypatch, swapped=False, square=False)
    before = (out / f"{tissue}_cells_with_pixel.csv").read_text()
    os.utime(out / f"{tissue}_cells_with_pixel.csv", (time.time() - 7200,) * 2)
    msg = P.fix_centroid_axis(ds)
    assert "swapped 0, already correct 1" in msg and "retired 0" in msg
    assert (out / f"{tissue}_cells_with_pixel.csv").read_text() == before
    assert (out / f"{tissue}_project_all_UNI2").exists()


def test_dry_run_changes_nothing(tmp_path, monkeypatch):
    ds, tissue, out, frame = _setup(tmp_path, monkeypatch, swapped=True)
    before = (out / f"{tissue}_cells_with_pixel.csv").read_text()
    msg = P.fix_centroid_axis(ds, dry_run=True)
    assert "would swap 1" in msg and "would retire 1" in msg
    assert (out / f"{tissue}_cells_with_pixel.csv").read_text() == before


def test_dry_run_flags_stale_store_on_correct_table(tmp_path, monkeypatch):
    # table already column-first but rewritten after the UNI2 store was built
    ds, tissue, out, frame = _setup(tmp_path, monkeypatch, swapped=False, square=False)
    msg = P.fix_centroid_axis(ds, dry_run=True)
    assert "would swap 0, already correct 1" in msg and "would retire 1" in msg
    assert (out / f"{tissue}_project_all_UNI2").exists()       # dry-run renames nothing
    assert "retired 1" in P.fix_centroid_axis(ds)
