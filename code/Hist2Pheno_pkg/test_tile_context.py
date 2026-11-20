"""2026.11.19 tile_context: geometry, sampling, extraction with a stub encoder, training access."""
import json
from pathlib import Path

import numpy as np
import pytest

import tile_context as tc


def test_image_wh_reads_bigtiff_header(tmp_path):
    """Pillow cannot identify BigTIFF; the CRC add-on H&E is that format."""
    import tifffile
    arr = np.zeros((8, 10, 3), np.uint8)
    path = tmp_path / "slide.ome.tif"
    tifffile.imwrite(path, arr, bigtiff=True)
    assert tc.image_wh(path) == (10, 8)


def test_covering_tiles_every_window_contains_point():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (500, 2))
    pairs = tc.covering_tiles(xy)
    o = pairs[:, 1:] * tc.STRIDE
    p = xy[pairs[:, 0]]
    assert ((p >= o) & (p < o + tc.TILE)).all()
    counts = np.bincount(pairs[:, 0], minlength=len(xy))
    assert (counts == 4).all()                    # stride = tile/2: every point lies in 2 x 2 tiles


def test_centre_weight_and_token_coords():
    o = np.array([[0.0, 0.0]])
    w = tc.centre_weight(np.array([[112.0, 112.0], [0.0, 112.0]]), o[0])
    assert np.isclose(w[0], 1.0) and np.isclose(w[1], np.exp(-1 / (2 * 0.25)))
    uv = tc.token_coords(np.array([[7.0, 7.0], [21.0, 7.0], [223.9, 0.0]]), o[0])
    assert np.allclose(uv[0], [0, 0]) and np.allclose(uv[1], [1, 0]) and np.isclose(uv[2, 0], 15.0)


def test_bilinear_reproduces_linear_field():
    g = np.zeros((16, 16, 2))
    vv, uu = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    g[..., 0], g[..., 1] = uu, vv
    uv = np.array([[3.25, 7.5], [0.0, 0.0], [15.0, 14.2]])
    assert np.allclose(tc.bilinear(g, uv), uv)


class _StubViT:
    """forward_features -> [cls, 8 registers, 16x16 patches]; patch value = mean RGB of its 14 px block."""
    num_prefix_tokens = 9

    def forward_features(self, x):
        import torch
        b = x.shape[0]
        pooled = torch.nn.functional.avg_pool2d(x, 14).mean(1)          # (B,16,16)
        patches = pooled.reshape(b, 256, 1).repeat(1, 1, tc.DIM)
        cls = x.mean(dim=(1, 2, 3)).reshape(b, 1, 1).repeat(1, 1, tc.DIM)
        regs = torch.zeros(b, 8, tc.DIM)
        return torch.cat([cls, regs, patches], 1)

    def forward_head(self, feats, pre_logits=True):
        return feats[:, 0]


def _section(tmp_path, monkeypatch, n=60, mpp=0.5):
    import anndata as ad
    from PIL import Image

    rng = np.random.default_rng(1)
    img = np.full((600, 800, 3), 255, np.uint8)
    img[:, :400] = 40                                                  # dark left half, white right half
    path = tmp_path / "he.png"
    Image.fromarray(img).save(path)
    xy = np.c_[rng.uniform(5, 795, n), rng.uniform(5, 595, n)]
    a = ad.AnnData(np.zeros((n, 1), np.float32))
    a.obsm["spatial_HE"] = xy
    h5 = tmp_path / "sec" / "x.h5ad"
    h5.parent.mkdir()
    a.write_h5ad(h5)
    monkeypatch.setattr(tc, "source_image", lambda h, u: (path, {"scale_image": False, "scale": 1.0}))
    return h5, xy


def test_extract_aligned_protocol_and_pca(tmp_path, monkeypatch):
    import anndata as ad
    import torch

    h5, xy = _section(tmp_path, monkeypatch)
    root = tmp_path / "tile_context"
    tc.extract_section(h5, 0.5, root, _StubViT(), torch.device("cpu"), batch_size=4, amp="fp32", encoder_sha="abc")
    out = tc.section_dir(root, h5)
    local = np.load(out / "local.npy").astype(np.float32)
    assert local.shape == (len(xy), tc.DIM) and np.isfinite(local).all()
    # local feature follows the pixel under the nucleus: dark (normalised < 0) left, white right
    left, right = xy[:, 0] < 380, xy[:, 0] > 420
    assert (local[left, 0] < 0).all() and (local[right, 0] > 0).all()
    adata = ad.read_h5ad(h5)
    A = tc.aligned(h5, adata, root, "uni2_tile_local")
    assert A.shape == (len(xy), tc.DIM)
    # reuse: same stamp -> no rewrite
    m = (out / "local.npy").stat().st_mtime_ns
    tc.extract_section(h5, 0.5, root, _StubViT(), torch.device("cpu"), batch_size=4, amp="fp32", encoder_sha="abc")
    assert (out / "local.npy").stat().st_mtime_ns == m
    # global PCA and the B matrix
    np.savez(tc.pca_path(root, 4), mean=np.zeros(tc.DIM, np.float32),
             components=np.eye(4, tc.DIM, dtype=np.float32), explained=np.ones(4, np.float32), n_rows=np.int64(1))
    monkeypatch.setenv("HIST2PHENO_TILE_GLOBAL_DIM", "4")
    monkeypatch.setenv("HIST2PHENO_TILE_PCA", "global")  # 2026.11.20, the default is now fold PCA
    B = tc.aligned(h5, adata, root, "uni2_tile_lg")
    assert B.shape == (len(xy), tc.DIM + 4)
    pr = tc.protocol(root, "uni2_tile_lg", h5)
    assert pr["dimension"] == tc.DIM + 4 and pr["encoder_sha256"] == "abc" and "global_pca_sha256" in pr
    # stale h5ad (coordinates moved) is refused
    adata.obsm["spatial_HE"] = xy + 1
    with pytest.raises(RuntimeError):
        tc.aligned(h5, adata, root, "uni2_tile_local")


def test_resize_factor_for_1um_sections(tmp_path, monkeypatch):
    import torch

    h5, xy = _section(tmp_path, monkeypatch)
    root = tmp_path / "tc"
    tc.extract_section(h5, 1.0, root, _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")
    stamp = json.loads((tc.section_dir(root, h5) / "stamp.json").read_text())
    assert stamp["resize_factor"] == 2.0 and stamp["n_finite"] == len(xy)


# ---------------------------------------------------------------------------
# 2026.11.20 fold PCA, lg_full, scale checks
# ---------------------------------------------------------------------------
def test_fold_mode_and_full_return_raw_3072(tmp_path, monkeypatch):
    import anndata as ad
    import torch

    h5, xy = _section(tmp_path, monkeypatch)
    root = tmp_path / "tile_context"
    tc.extract_section(h5, 0.5, root, _StubViT(), torch.device("cpu"), batch_size=4, amp="fp32", encoder_sha="abc")
    adata = ad.read_h5ad(h5)
    monkeypatch.delenv("HIST2PHENO_TILE_PCA", raising=False)
    monkeypatch.setenv("HIST2PHENO_TILE_GLOBAL_DIM", "256")
    assert tc.feature_projection("uni2_tile_lg") == f"split={tc.DIM},dim=256"
    assert tc.feature_projection("uni2_tile_lg_full") is None and tc.feature_projection("uni2_tile_local") is None
    for emb in ("uni2_tile_lg", "uni2_tile_lg_full"):
        X = tc.aligned(h5, adata, root, emb)
        assert X.shape == (len(xy), 2 * tc.DIM) and X.dtype == np.float16
        pr = tc.protocol(root, emb, h5)
        assert pr["dimension"] == 2 * tc.DIM and pr["global_pca"] == ("fold" if emb == "uni2_tile_lg" else "none")
    assert tc.protocol(root, "uni2_tile_lg", h5) != tc.protocol(root, "uni2_tile_lg_full", h5)


def test_scale_sources_mismatch_stops_extraction(tmp_path, monkeypatch):
    import anndata as ad
    import torch

    h5, xy = _section(tmp_path, monkeypatch)
    a = ad.read_h5ad(h5)
    a.uns["mpp"] = 1.0                          # SPORA-style statement in the h5ad
    a.write_h5ad(h5)
    root = tmp_path / "tc"
    with pytest.raises(ValueError, match="disagree"):
        tc.extract_section(h5, 0.5, root, _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")
    out = tc.extract_section(h5, 1.0, root, _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")
    stamp = json.loads((out / "stamp.json").read_text())
    assert stamp["scale_sources"]["consistent"] and stamp["resize_factor"] == 2.0
    # UNI2 manifest that resized by 2 agrees with 1.0 µm/px; one that did not resize is not a source
    assert tc.scale_sources(h5, 1.0, {"mpp": 1.0}, {"scale_image": True, "scale": 2.0})["consistent"]
    assert not tc.scale_sources(h5, 1.0, {}, {"scale_image": True, "scale": 1.0})["consistent"]
    assert tc.scale_sources(h5, 1.0, {}, {"scale_image": False, "scale": 1.0})["consistent"]


def test_wrong_pixel_grid_is_refused(tmp_path, monkeypatch):
    import torch

    import anndata as ad

    h5, xy = _section(tmp_path, monkeypatch)
    # spatial_HE on a 2x finer grid than the H&E file (e.g. 0.5 µm coordinates, 1 µm image):
    # image and coordinates are resized together, so most nuclei stay off the image
    a = ad.read_h5ad(h5)
    a.obsm["spatial_HE"] = xy * 2
    a.write_h5ad(h5)
    with pytest.raises(ValueError, match="inside"):
        tc.extract_section(h5, 0.5, tmp_path / "tc", _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")


def test_new_crop_manifest_does_not_force_new_tile_pass(tmp_path, monkeypatch):
    import torch

    h5, xy = _section(tmp_path, monkeypatch)
    root = tmp_path / "tc"
    out = tc.extract_section(h5, 1.0, root, _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")
    m = (out / "local.npy").stat().st_mtime_ns
    image = tmp_path / "he.png"
    monkeypatch.setattr(tc, "source_image", lambda h, u: (image, {"scale_image": True, "scale": 2.0}))
    tc.extract_section(h5, 1.0, root, _StubViT(), torch.device("cpu"), batch_size=8, amp="fp32")
    assert (out / "local.npy").stat().st_mtime_ns == m
