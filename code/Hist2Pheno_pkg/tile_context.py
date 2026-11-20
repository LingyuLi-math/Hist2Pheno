#!/usr/bin/env python3
"""2026.11.19 Cell-centred multi-scale UNI2-h features from one tile-level forward.
以细胞为中心的多尺度 UNI2-h 特征：每个 224 px tile 只做一次前向，同时得到局部和全局特征。

Why (见 code/PanCancer/README.md "2026-11-19 Tile-context features"):
the existing UNI2 token (``sc_pth_14_14``) is the class token of a 14 px crop
fed to UNI2-h on its own. At 0.5 µm/px that crop is 7 µm wide, a single ViT
patch: the encoder sees part of one nucleus and no tissue architecture
(ducts / myoepithelial rings, follicles, tumour-stroma borders).
旧特征 = 单独送入的 14 px 小块的 class token（7 µm，只有一个 patch，看不到组织结构）。

Here every section is brought to 0.5 µm/px and cut into 224 px tiles
(112 µm) with stride 112 (each point lies in up to 2x2 tiles). One UNI2-h
forward per tile gives a 16x16 grid of patch tokens (7 µm each, every one of
them mixed with the whole tile by 24 attention layers) and a class token.
For each nucleus:

  local   bilinear sample of the patch-token grid at the centroid
          (same 7 µm footprint as before, but context-aware), 1536-d
  global  class token of the tile (112 µm view), 1536-d

and both are averaged over the covering tiles with the Gaussian centre
weight exp(-(dx^2+dy^2)/(2*0.5^2)), dx, dy in [-1, 1] from the tile centre
(the overlap weighting of MeowCat's public code, get_center_weights), so the
tile in which the nucleus is most central dominates.
每个核：local = 质心处 patch token 的双线性插值；global = tile 的 class token；
覆盖它的 tile 按离 tile 中心的高斯权重加权平均。

Embedders used by PanCancer_train_validate_cv_UNIlabel.py ``--embedder``:
  uni2_tile_local   local only                          (ablation A)
  uni2_tile_lg      local + global, global PCA-reduced  (ablation B; + k=8 = C)
                    to HIST2PHENO_TILE_GLOBAL_DIM dims (default 256, 0 = full 1536)
  uni2_tile_lg_full local + full 1536-d global, no PCA  (2026.11.20, control for B)

2026.11.20 PCA of the global block (HIST2PHENO_TILE_PCA):
  fold    (default) fitted on the outer-train rows of each CV fold only, inside the
          fold scaler (projected_scaler.ProjectedScaler; saved with the checkpoint).
          The pooled matrix keeps the full 3072-d features (float16, lossless).
  global  one PCA over nuclei of all sections (``fit-pca``), applied before pooling
          (the 2026.11.19 behaviour; unsupervised but has seen the test-fold images).
按折 PCA（默认）：只用该折训练细胞拟合；global = 11.19 的全体 section PCA。

2026.11.20 scale check: the native µm/px from mpp_map.csv must agree with the
h5ad ``uns['mpp']`` (SPORA writes 1.0) and with the UNI2 manifest scale when
that extraction resized the image; ``audit-mpp`` lists every section. After the
resize, >= 98 % of the nuclei must fall inside the image (wrong pixel grid
otherwise). Any failure stops the section instead of a warning.
尺度检查：mpp_map 与 h5ad 的 uns['mpp']、UNI2 manifest 不一致即报错（不再只是警告）。

Physical scale: native µm/px per section comes from ``data/PanCancer/mpp_map.csv``
(``make_mpp_map.py``); the image is resized by mpp / 0.5. SPORA tissues
(1 µm/px) are upsampled 2x, so their tiles also cover 112 µm.
物理尺度统一到 0.5 µm/px；SPORA（1 µm/px）放大 2 倍。

Output per section (aligned to the h5ad rows, so no coordinate matching):
  data/PanCancer/tile_context/<sha(h5ad path)[:32]>/{local.npy, global.npy, stamp.json}

    python code/Hist2Pheno_pkg/tile_context.py extract [--datasets BRCA GBM] [--shard 0/4]
    python code/Hist2Pheno_pkg/tile_context.py fit-pca --dim 256     # only for HIST2PHENO_TILE_PCA=global
    python code/Hist2Pheno_pkg/tile_context.py audit-mpp              # 2026.11.20, scale of every section
    python code/Hist2Pheno_pkg/tile_context.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
REPO = _PKG.parents[1]
for _p in (_PKG, REPO / "code", REPO / "code" / "PanCancer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VERSION = 1
TARGET_MPP = 0.5
TILE = 224
STRIDE = 112
TOKEN = 14
GRID = TILE // TOKEN          # 16 patch tokens per side for UNI2-h ViT-H/14
SIGMA = 0.5                   # Gaussian centre weight, normalised tile coordinates
DIM = 1536
EMBEDDERS = ("uni2_tile_local", "uni2_tile_lg", "uni2_tile_lg_full")  # 2026.11.20, + lg_full
MPP_TOL = 0.02                # relative tolerance between scale sources
MIN_INSIDE = 0.98             # fraction of nuclei that must fall inside the resized image
WHITE = 255                   # padding outside the slide: H&E background, not black


# ---------------------------------------------------------------------------
# geometry (no torch; unit-tested)
# ---------------------------------------------------------------------------
def covering_tiles(xy: np.ndarray, *, tile: int = TILE, stride: int = STRIDE) -> np.ndarray:
    """(n_pairs, 3) int array of (cell row, tile ix, tile iy): every tile whose
    window [i*stride, i*stride + tile) contains the point, in scaled pixels."""
    xy = np.asarray(xy, dtype=np.float64)
    rows, ixs, iys = [], [], []
    n_per_axis = int(np.ceil(tile / stride))
    base = np.floor(xy / stride).astype(np.int64)
    for dx in range(n_per_axis):
        for dy in range(n_per_axis):
            ix = base[:, 0] - dx
            iy = base[:, 1] - dy
            ok = (xy[:, 0] < ix * stride + tile) & (xy[:, 1] < iy * stride + tile)
            ok &= (xy[:, 0] >= ix * stride) & (xy[:, 1] >= iy * stride)
            r = np.flatnonzero(ok)
            rows.append(r); ixs.append(ix[r]); iys.append(iy[r])
    return np.stack([np.concatenate(rows), np.concatenate(ixs), np.concatenate(iys)], axis=1)


def centre_weight(xy: np.ndarray, origin: np.ndarray, *, tile: int = TILE, sigma: float = SIGMA) -> np.ndarray:
    """MeowCat-style Gaussian weight: 1 at the tile centre, exp(-1/(2 sigma^2)) at an edge midpoint."""
    half = tile / 2.0
    d = (np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64) - half) / half
    return np.exp(-(d ** 2).sum(axis=1) / (2.0 * sigma ** 2))


def token_coords(xy: np.ndarray, origin: np.ndarray, *, token: int = TOKEN, grid: int = GRID) -> np.ndarray:
    """Continuous token-grid coordinates (u, v) of points inside a tile; token j covers
    [j*token, (j+1)*token) so its centre is j + 0.5 in token units. Clamped to the grid."""
    uv = (np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)) / token - 0.5
    return np.clip(uv, 0.0, grid - 1.0)


def bilinear(grid_tokens: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """grid_tokens (G, G, D) indexed [row v, col u]; uv (n, 2) -> (n, D)."""
    g = np.asarray(grid_tokens)
    u, v = uv[:, 0], uv[:, 1]
    u0 = np.floor(u).astype(np.int64); v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, g.shape[1] - 1); v1 = np.minimum(v0 + 1, g.shape[0] - 1)
    fu = (u - u0)[:, None]; fv = (v - v0)[:, None]
    return ((1 - fv) * ((1 - fu) * g[v0, u0] + fu * g[v0, u1])
            + fv * ((1 - fu) * g[v1, u0] + fu * g[v1, u1]))


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
def uni2_weight_file() -> Path:
    return REPO / "code" / "UNI2" / "assets" / "ckpts" / "uni2-h" / "pytorch_model.bin"


def load_uni2(device):
    """Same UNI2-h definition as Image_feature_extraction.py (method UNI2)."""
    import timm
    import torch

    weight = uni2_weight_file()
    if not weight.is_file() or weight.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"UNI2-h weights missing or incomplete: {weight}")
    model = timm.create_model(
        "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True, pretrained=False,
    )
    model.load_state_dict(torch.load(weight, map_location="cpu"), strict=True)
    return model.to(device).eval()


def encode_tiles(model, batch, *, amp: str = "bf16"):
    """batch (B,3,224,224) normalised -> (class token (B,D), patch grid (B,G,G,D)) as float32 numpy."""
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    with torch.inference_mode():
        ctx = (torch.autocast(device_type=batch.device.type, dtype=dtype)
               if dtype is not None and batch.device.type == "cuda" else _null())
        with ctx:
            feats = model.forward_features(batch)                  # (B, prefix + G*G, D), normed
            cls = model.forward_head(feats, pre_logits=True)       # class token (global_pool='token')
        prefix = int(getattr(model, "num_prefix_tokens", 1))
        patches = feats[:, prefix:].float()
        b, n, d = patches.shape
        g = int(round(n ** 0.5))
        if g * g != n:
            raise ValueError(f"expected a square token grid, got {n} patch tokens")
        return cls.float().cpu().numpy(), patches.reshape(b, g, g, d).cpu().numpy()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# per-section extraction
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def section_dir(root: Path, h5ad_path: Path) -> Path:
    return Path(root) / _sha(str(Path(h5ad_path).resolve()))[:32]


def xy_digest(xy: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(xy, dtype=np.float64)).tobytes()).hexdigest()


def h5ad_stat(h5ad_path: Path) -> dict:
    st = Path(h5ad_path).stat()
    return {"path": str(Path(h5ad_path).resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def source_image(h5ad_path: Path, uns) -> tuple[Path, dict]:
    """H&E path recorded by the UNI2 extraction of this section (embedding_manifest.json)."""
    from embedding_backend import embedding_dir
    from embedding_store import extraction_metadata

    meta = extraction_metadata(embedding_dir(h5ad_path, uns, "uni2"))
    image = Path(((meta.get("inputs") or {}).get("image") or {}).get("path", ""))
    if not image.is_file():
        raise FileNotFoundError(f"UNI2 manifest of {h5ad_path} names no readable H&E image: {image}")
    return image, meta


def image_wh(image_path: Path) -> tuple[int, int]:
    """Width and height from the header only.

    CRC add-on H&E is BigTIFF (``.ome.tif``, magic ``MM\\x00+``). Pillow raises
    ``UnidentifiedImageError`` on those files; tifffile reads the full-resolution
    page that ``load_he_image_pil`` later decodes. No pixels are loaded here.
    """
    path = Path(image_path)
    with open(path, "rb") as handle:
        bigtiff = handle.read(4) in (b"II+\x00", b"MM\x00+")
    if not bigtiff and not path.name.lower().endswith((".ome.tif", ".ome.tiff", ".btf")):
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as im:
                return int(im.size[0]), int(im.size[1])
        except (OSError, ValueError):
            pass
    import tifffile
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape
        axes = str(getattr(series, "axes", "") or "")
    if "Y" in axes and "X" in axes:
        return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
    if len(shape) >= 2 and shape[-1] in (1, 3, 4):
        return int(shape[-2]), int(shape[-3])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    raise ValueError(f"cannot read width/height of {path}: shape {shape} axes {axes!r}")


def load_scaled_image(image_path: Path, factor: float):
    """RGB uint8 PIL image resized by ``factor`` (scaled px = native px * factor)."""
    import logging
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:  # same reader as the UNI2 crop extraction (OME-TIFF / BigTIFF fall back to tifffile)
        from Image_feature_extraction import load_he_image_pil
        img = load_he_image_pil(str(image_path), logging.getLogger("tile_context"))
    except ImportError:  # extraction script deps (torchvision / skimage) missing: plain PIL
        img = Image.open(image_path)
        img.load()
    if abs(factor - 1.0) > 1e-6:
        w, h = img.size
        img = img.resize((max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                         resample=Image.BILINEAR)
    return img.convert("RGB")


def _crop(img, ix: int, iy: int, *, tile=TILE, stride=STRIDE):
    """Tile at origin (ix*stride, iy*stride); parts outside the slide are white."""
    from PIL import Image

    left, top = ix * stride, iy * stride
    w, h = img.size
    if left >= 0 and top >= 0 and left + tile <= w and top + tile <= h:
        return img.crop((left, top, left + tile, top + tile))
    canvas = Image.new("RGB", (tile, tile), (WHITE, WHITE, WHITE))
    box = (max(0, left), max(0, top), min(w, left + tile), min(h, top + tile))
    if box[2] > box[0] and box[3] > box[1]:
        canvas.paste(img.crop(box), (box[0] - left, box[1] - top))
    return canvas


def scale_sources(h5ad_path: Path, mpp: float, uns, manifest: dict) -> dict:
    """2026.11.20, every independent statement of the native µm/px of one section.
    同一 section 的原始 µm/px 的各个来源；不一致即报错。"""
    rec = {"mpp_map": float(mpp), "h5ad_uns_mpp": None, "uni2_manifest_mpp": None}
    for key in ("mpp", "microns_per_pixel", "HE_micron_per_pixel_size"):
        if key in uns:
            try:
                rec["h5ad_uns_mpp"] = float(np.asarray(uns[key]).reshape(-1)[0])
                break
            except (TypeError, ValueError):
                pass
    # the UNI2 crop extraction resized by scale = native / 0.5 when scale_image was True
    if manifest.get("scale_image") in (True, "True", "true") and manifest.get("scale"):
        rec["uni2_manifest_mpp"] = 0.5 * float(manifest["scale"])
    bad = [k for k in ("h5ad_uns_mpp", "uni2_manifest_mpp")
           if rec[k] is not None and abs(rec[k] - rec["mpp_map"]) > MPP_TOL * rec["mpp_map"]]
    rec["consistent"] = not bad
    rec["disagree"] = bad
    return rec


def extract_section(h5ad_path: Path, mpp: float, root: Path, model, device, *,
                    batch_size: int = 16, amp: str = "bf16", encoder_sha: str = "", force: bool = False) -> Path:
    import anndata as ad
    import torch

    out = section_dir(root, h5ad_path)
    stamp_path = out / "stamp.json"
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        xy_native = np.asarray(adata.obsm["spatial_HE"] if "spatial_HE" in adata.obsm else adata.obsm["spatial"],
                               dtype=np.float64)
        uns = dict(adata.uns)
        n_obs = int(adata.n_obs)
    finally:
        adata.file.close()
    image_path, manifest = source_image(h5ad_path, uns)
    factor = float(mpp) / TARGET_MPP
    # 2026.11.20, scale sources must agree (was a warning on the UNI2 manifest only)
    scale_rec = scale_sources(h5ad_path, mpp, uns, manifest)
    if not scale_rec["consistent"] and os.environ.get("HIST2PHENO_TILE_ALLOW_MPP_MISMATCH") != "1":
        raise ValueError(f"{Path(h5ad_path).parent.name}: µm/px disagree {scale_rec} "
                         "(fix data/PanCancer/mpp_map.csv; HIST2PHENO_TILE_ALLOW_MPP_MISMATCH=1 overrides)")
    stamp = {
        "version": VERSION, "h5ad": h5ad_stat(h5ad_path), "n_obs": n_obs, "xy_sha256": xy_digest(xy_native),
        "image": {"path": str(image_path.resolve()), "size": int(image_path.stat().st_size),
                  "mtime_ns": int(image_path.stat().st_mtime_ns)},
        "native_mpp": float(mpp), "target_mpp": TARGET_MPP, "resize_factor": factor,
        "tile": TILE, "stride": STRIDE, "token": TOKEN, "sigma": SIGMA, "padding": "white",
        "encoder": "MahmoodLab/UNI2-h vit_giant_patch14_224 reg8", "encoder_sha256": encoder_sha,
        "amp": amp, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "uni2_manifest_scale": manifest.get("scale"), "uni2_manifest_scale_image": manifest.get("scale_image"),
        "scale_sources": scale_rec,  # 2026.11.20
    }
    if not force and stamp_path.is_file():
        old = json.loads(stamp_path.read_text())
        # same = {k: v for k, v in old.items() if k not in ("finished", "seconds", "n_tiles", "n_finite")}
        # 2026.11.20, the UNI2 manifest (crop extraction) does not change the tile features:
        # re-extracting the SPORA crops at 0.5 µm/px must not force a new tile pass.
        volatile = ("finished", "seconds", "n_tiles", "n_finite", "inside_fraction",
                    "uni2_manifest_scale", "uni2_manifest_scale_image", "scale_sources")
        same = {k: v for k, v in old.items() if k not in volatile}
        new = {k: v for k, v in stamp.items() if k not in volatile}
        if same == new and (out / "local.npy").is_file() and (out / "global.npy").is_file():
            return out
    # 2026.11.19, the UNI2 crop extraction resized by manifest 'scale' (native/0.5); warn if mpp_map disagrees
    # (2026.11.20: kept; scale_sources above already raised on a disagreement)
    if manifest.get("scale_image") and manifest.get("scale") and abs(float(manifest["scale"]) - factor) > 0.02 * factor:
        print(f"  WARNING {Path(h5ad_path).parent.name}: mpp_map resize {factor:.4f} != UNI2 manifest scale "
              f"{float(manifest['scale']):.4f}", flush=True)

    t0 = time.time()
    xy = xy_native * factor
    finite = np.isfinite(xy).all(axis=1)
    rows_all = np.flatnonzero(finite)
    pairs = covering_tiles(xy[rows_all])
    pairs[:, 0] = rows_all[pairs[:, 0]]
    order = np.lexsort((pairs[:, 2], pairs[:, 1]))
    pairs = pairs[order]
    tiles, starts = np.unique(pairs[:, 1:], axis=0, return_index=True)
    ends = np.append(starts[1:], len(pairs))
    img = load_scaled_image(image_path, factor)
    w_img, h_img = img.size
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < w_img) & (xy[:, 1] < h_img)
    # 2026.11.20, coordinates on another pixel grid (wrong µm/px) push nuclei off the image
    inside_fraction = float(inside.sum()) / max(1, int(finite.sum()))
    if inside_fraction < MIN_INSIDE and os.environ.get("HIST2PHENO_TILE_ALLOW_MPP_MISMATCH") != "1":
        raise ValueError(f"{Path(h5ad_path).parent.name}: only {inside_fraction:.1%} of nuclei inside the "
                         f"x{factor:.3f} image {w_img}x{h_img}; µm/px or spatial_HE grid is wrong")

    local_sum = np.zeros((n_obs, DIM), dtype=np.float32)
    global_sum = np.zeros((n_obs, DIM), dtype=np.float32)
    wsum = np.zeros(n_obs, dtype=np.float64)
    # ToTensor + ImageNet Normalize, as eval_transforms() of the UNI2 crop extraction
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def norm(pil):
        t = torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return (t - mean) / std
    n_tiles = len(tiles)
    print(f"  {Path(h5ad_path).parent.name}: {n_obs:,} nuclei, {n_tiles:,} tiles, image {w_img}x{h_img} "
          f"(x{factor:.3f})", flush=True)
    for b0 in range(0, n_tiles, batch_size):
        b1 = min(n_tiles, b0 + batch_size)
        batch = torch.stack([norm(_crop(img, int(t[0]), int(t[1]))) for t in tiles[b0:b1]]).to(device)
        cls, grid = encode_tiles(model, batch, amp=amp)
        for j, t in enumerate(range(b0, b1)):
            rows = pairs[starts[t]:ends[t], 0]
            origin = tiles[t] * STRIDE
            w = centre_weight(xy[rows], origin)
            loc = bilinear(grid[j], token_coords(xy[rows], origin))
            local_sum[rows] += (w[:, None] * loc).astype(np.float32)
            global_sum[rows] += (w[:, None] * cls[j][None, :]).astype(np.float32)
            wsum[rows] += w
        if (b0 // batch_size) % 200 == 0:
            print(f"    tiles {b1:,}/{n_tiles:,}  {time.time() - t0:.0f}s", flush=True)
    ok = inside & (wsum > 0)
    local = np.full((n_obs, DIM), np.nan, dtype=np.float16)
    glob = np.full((n_obs, DIM), np.nan, dtype=np.float16)
    local[ok] = (local_sum[ok] / wsum[ok, None]).astype(np.float16)
    glob[ok] = (global_sum[ok] / wsum[ok, None]).astype(np.float16)
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in (("local", local), ("global", glob)):
        tmp = out / f".{name}.{os.getpid()}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{name}.npy")
    stamp.update(finished=time.strftime("%F %T"), seconds=round(time.time() - t0, 1),
                 n_tiles=int(n_tiles), n_finite=int(ok.sum()), inside_fraction=round(inside_fraction, 5))
    tmp = out / f".stamp.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True))
    os.replace(tmp, stamp_path)
    print(f"  wrote {out}  ({ok.sum():,}/{n_obs:,} nuclei, {time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# training-side access (PanCancer_train_validate_cv_UNIlabel.py --embedder uni2_tile_*)
# ---------------------------------------------------------------------------
def global_dim() -> int:
    return int(os.environ.get("HIST2PHENO_TILE_GLOBAL_DIM", "256"))


def pca_mode() -> str:
    """2026.11.20, 'fold' (default): PCA per CV fold in the scaler; 'global': fit-pca file."""
    mode = os.environ.get("HIST2PHENO_TILE_PCA", "fold").strip().lower()
    if mode not in ("fold", "global"):
        raise ValueError("HIST2PHENO_TILE_PCA must be fold or global")
    return mode


def feature_projection(embedder: str) -> str | None:
    """2026.11.20, value for projected_scaler.ENV, or None (no fold PCA for this embedder)."""
    if embedder == "uni2_tile_lg" and pca_mode() == "fold" and global_dim() > 0:
        return f"split={DIM},dim={global_dim()}"
    return None


def pca_path(root: Path, dim: int) -> Path:
    return Path(root) / f"global_pca_{dim}.npz"


def _check_stamp(root: Path, h5ad_path: Path, adata) -> dict:
    out = section_dir(root, h5ad_path)
    try:
        stamp = json.loads((out / "stamp.json").read_text())
    except (OSError, ValueError):
        raise FileNotFoundError(
            f"No tile-context features for {h5ad_path} in {out}. "
            "Run: python code/Hist2Pheno_pkg/tile_context.py extract") from None
    if stamp.get("h5ad") != h5ad_stat(h5ad_path) or int(stamp.get("n_obs", -1)) != int(adata.n_obs):
        raise RuntimeError(f"Tile-context features of {h5ad_path} are stale (h5ad changed); re-run extract")
    xy = adata.obsm["spatial_HE"] if "spatial_HE" in adata.obsm else adata.obsm["spatial"]
    if stamp.get("xy_sha256") != xy_digest(np.asarray(xy, dtype=np.float64)):
        raise RuntimeError(f"Tile-context features of {h5ad_path} were cut at other coordinates; re-run extract")
    return stamp


def aligned(h5ad_path: Path, adata, root: Path, embedder: str) -> np.ndarray:
    """h5ad-row-aligned float32 matrix; NaN rows have no feature (dropped by the caller)."""
    if embedder not in EMBEDDERS:
        raise ValueError(f"embedder must be one of {EMBEDDERS}")
    _check_stamp(root, h5ad_path, adata)
    out = section_dir(root, h5ad_path)
    local = np.load(out / "local.npy", mmap_mode="r")
    if embedder == "uni2_tile_local":
        return np.asarray(local, dtype=np.float32)
    # 2026.11.20, full 3072-d (lg_full, or lg with the fold PCA done in the scaler): float16 as
    # stored (lossless), so the pooled matrix is no larger than the 1536-d float32 crop features.
    # 3072 维原样（float16 无损），按折 PCA 在 scaler 中完成。
    if embedder == "uni2_tile_lg_full" or feature_projection(embedder) or global_dim() <= 0:
        return np.hstack([np.asarray(local, dtype=np.float16),
                          np.asarray(np.load(out / "global.npy", mmap_mode="r"), dtype=np.float16)])
    glob = np.asarray(np.load(out / "global.npy", mmap_mode="r"), dtype=np.float32)
    dim = global_dim()
    if dim > 0:
        pca = np.load(pca_path(root, dim))
        glob = (glob - pca["mean"]) @ pca["components"].T
    return np.hstack([np.asarray(local, dtype=np.float32), glob.astype(np.float32)])


def protocol(root: Path, embedder: str, h5ad_path: Path) -> dict:
    """Must be identical for every training section (checked by prepare_pooled_cv_data)."""
    stamp = json.loads((section_dir(root, h5ad_path) / "stamp.json").read_text())
    rec = {"method": "UNI2-tile", "embedder": embedder, "encoder_sha256": stamp.get("encoder_sha256"),
           "preprocessing": {k: stamp.get(k) for k in ("target_mpp", "tile", "stride", "token", "sigma", "padding",
                                                       "amp", "version")},
           "dimension": DIM}
    if embedder == "uni2_tile_lg":
        dim = global_dim()
        rec["global_dim"] = dim
        rec["dimension"] = DIM + (dim if dim > 0 else DIM)
        # 2026.11.20, fold PCA: pooled width is 3072; the scaler reduces it per fold
        if feature_projection(embedder):
            rec["global_pca"] = "fold"
            rec["dimension"] = 2 * DIM
        elif dim > 0:
            rec["global_pca"] = "global"
            rec["global_pca_sha256"] = hashlib.sha256(pca_path(root, dim).read_bytes()).hexdigest()
    if embedder == "uni2_tile_lg_full":  # 2026.11.20
        rec["global_dim"] = DIM
        rec["global_pca"] = "none"
        rec["dimension"] = 2 * DIM
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _sections(args):
    import pandas as pd
    from pancancer_cv import ok_mask

    index = pd.read_csv(args.index)
    index = index.loc[ok_mask(index)].copy()
    if args.datasets:
        index = index[index["dataset"].isin(args.datasets)]
    mpp = pd.read_csv(args.mpp_map).set_index("section_id")["mpp_um"].astype(float)
    missing = sorted(set(index.section_id.astype(str)) - set(mpp.index.astype(str)))
    if missing:
        raise SystemExit(f"{len(missing)} sections have no mpp in {args.mpp_map} (e.g. {missing[:3]}); "
                         "run code/PanCancer/make_mpp_map.py")
    index["mpp_um"] = index.section_id.astype(str).map(mpp)
    return index.reset_index(drop=True)


def cmd_extract(args):
    import torch
    import pipeline_provenance as pp

    index = _sections(args)
    k, n = (int(x) for x in args.shard.split("/"))
    index = index.iloc[k::n]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enc = pp.file_record(uni2_weight_file(), Path(args.root) / ".provenance")["sha256"]
    model = load_uni2(device)
    print(f"tile-context extract: {len(index)} sections (shard {args.shard}) on {device}, encoder {enc[:12]}",
          flush=True)
    failed = []
    for rec in index.itertuples(index=False):
        try:
            extract_section(Path(rec.h5ad), float(rec.mpp_um), Path(args.root), model, device,
                            batch_size=args.batch, amp=args.amp, encoder_sha=enc, force=args.force)
        except Exception as exc:  # keep going; report at the end
            failed.append((rec.section_id, repr(exc)))
            print(f"  FAILED {rec.section_id}: {exc!r}", flush=True)
    if failed:
        print(f"{len(failed)} sections failed", flush=True)
        return 1
    return 0


def cmd_fit_pca(args):
    """PCA of the per-nucleus global features (unsupervised; no labels, no fold use).
    2026.11.20: only needed for HIST2PHENO_TILE_PCA=global; the default fold PCA is fitted in training."""
    index = _sections(args)
    rng = np.random.default_rng(0)
    per = max(1, args.sample // max(1, len(index)))
    parts = []
    for rec in index.itertuples(index=False):
        f = section_dir(Path(args.root), Path(rec.h5ad)) / "global.npy"
        if not f.is_file():
            raise SystemExit(f"missing {f}; run extract first")
        g = np.load(f, mmap_mode="r")
        ok = np.flatnonzero(np.isfinite(np.asarray(g[:, 0], dtype=np.float32)))
        pick = np.sort(rng.choice(ok, size=min(per, len(ok)), replace=False)) if len(ok) else ok
        parts.append(np.asarray(g[pick], dtype=np.float32))
    X = np.vstack(parts)
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    comp = vt[:args.dim].astype(np.float32)
    ev = (s ** 2) / (s ** 2).sum()
    out = pca_path(Path(args.root), args.dim)
    np.savez(out, mean=mean.astype(np.float32), components=comp,
             explained=ev[:args.dim].astype(np.float32), n_rows=np.int64(len(X)))
    print(f"PCA {args.dim} on {len(X):,} nuclei: explained {ev[:args.dim].sum():.3f} -> {out}", flush=True)
    return 0


def cmd_audit_mpp(args):
    """2026.11.20, native µm/px per section from every source, resize factor, coordinate extent."""
    import anndata as ad
    import pandas as pd

    index = _sections(args)
    rows = []
    for rec in index.itertuples(index=False):
        row = {"section_id": rec.section_id, "dataset": rec.dataset,
               "pan_organ": getattr(rec, "pan_organ", ""), "mpp_map": float(rec.mpp_um)}
        try:
            a = ad.read_h5ad(rec.h5ad, backed="r")
            try:
                xy = np.asarray(a.obsm["spatial_HE"] if "spatial_HE" in a.obsm else a.obsm["spatial"], np.float64)
                uns = dict(a.uns)
            finally:
                a.file.close()
            image, manifest = source_image(Path(rec.h5ad), uns)
            src = scale_sources(Path(rec.h5ad), rec.mpp_um, uns, manifest)
            w, h = image_wh(image)  # 2026.11.20, BigTIFF / OME-TIFF (CRC add-on) is not a PIL file
            f = float(rec.mpp_um) / TARGET_MPP
            fin = np.isfinite(xy).all(1)
            inside = fin & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < w) & (xy[:, 1] < h)
            row.update(h5ad_uns_mpp=src["h5ad_uns_mpp"], uni2_manifest_mpp=src["uni2_manifest_mpp"],
                       uni2_scale_image=manifest.get("scale_image"), uni2_scale=manifest.get("scale"),
                       consistent=src["consistent"], resize_factor=f,
                       crop_um_old=14 * float(rec.mpp_um) / (float(manifest["scale"])
                                   if manifest.get("scale_image") in (True, "True", "true") and manifest.get("scale")
                                   else 1.0),
                       tile_um=TILE * TARGET_MPP, image_w=w, image_h=h,
                       x_max=float(np.nanmax(xy[:, 0])), y_max=float(np.nanmax(xy[:, 1])),
                       inside_fraction=float(inside.sum()) / max(1, int(fin.sum())), error="")
        except Exception as exc:  # report, do not stop the audit
            row.update(consistent=False, error=repr(exc))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(args.root) / "mpp_audit.csv"
    df.to_csv(out, index=False)
    group = df.assign(cohort=df.dataset.astype(str) + "/" + df.pan_organ.astype(str))
    cols = [c for c in ("mpp_map", "h5ad_uns_mpp", "uni2_manifest_mpp", "resize_factor", "crop_um_old",
                        "tile_um", "inside_fraction") if c in group]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(group.groupby("cohort")[cols].agg(["min", "max"]).round(4))
    bad = df[~df.consistent.astype(bool) | (df.get("inside_fraction", 1.0) < MIN_INSIDE)]
    print(f"{len(df)} sections; {len(bad)} with a scale problem -> {out}", flush=True)
    if len(bad):
        print(bad[["section_id", "mpp_map", "h5ad_uns_mpp", "uni2_manifest_mpp", "inside_fraction", "error"]]
              .head(20).to_string(index=False), flush=True)
    return 1 if len(bad) else 0


def cmd_status(args):
    index = _sections(args)
    done = 0
    for rec in index.itertuples(index=False):
        if (section_dir(Path(args.root), Path(rec.h5ad)) / "stamp.json").is_file():
            done += 1
    print(f"{done}/{len(index)} sections extracted under {args.root}", flush=True)
    for dim in sorted({global_dim(), 256}):
        p = pca_path(Path(args.root), dim)
        print(f"  {p.name}: {'present' if p.is_file() else 'missing'}")
    return 0 if done == len(index) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=REPO / "data/PanCancer/section_index.csv")
    ap.add_argument("--mpp-map", type=Path, default=REPO / "data/PanCancer/mpp_map.csv")
    ap.add_argument("--root", type=Path, default=REPO / "data/PanCancer/tile_context")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # --datasets lives on the subcommand: on the parent, nargs='*' swallows the subcommand name
    e = sub.add_parser("extract")
    e.add_argument("--datasets", nargs="*", default=None)
    e.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th section from k")
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    e.add_argument("--force", action="store_true")
    p = sub.add_parser("fit-pca")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--sample", type=int, default=200_000)
    st = sub.add_parser("status")
    st.add_argument("--datasets", nargs="*", default=None)
    au = sub.add_parser("audit-mpp")  # 2026.11.20
    au.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args(argv)
    Path(a.root).mkdir(parents=True, exist_ok=True)
    return {"extract": cmd_extract, "fit-pca": cmd_fit_pca, "status": cmd_status,
            "audit-mpp": cmd_audit_mpp}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
