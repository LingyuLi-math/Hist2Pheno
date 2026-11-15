"""2026.11.15 tests for stain_normalize.py on synthetic H&E images."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stain_normalize as sn  # noqa: E402


def _synthetic(stains, shape=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    c = np.stack([rng.gamma(2.0, 0.35, shape), rng.gamma(2.0, 0.25, shape)], -1)
    c[:40] = 0                                             # background band
    od = c.reshape(-1, 2) @ np.asarray(stains).T
    return np.clip(sn.IO * np.exp(-od), 0, 255).reshape(*shape, 3).astype(np.uint8)


REF = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]])
SRC = np.array([[0.55, 0.20], [0.80, 0.90], [0.25, 0.40]])
REF /= np.linalg.norm(REF, axis=0); SRC /= np.linalg.norm(SRC, axis=0)


def test_estimate_recovers_stain_vectors():
    s, _ = sn.estimate_stain(_synthetic(REF))
    cos = np.abs((s * REF).sum(0))
    assert (cos > 0.97).all(), cos


def test_normalisation_moves_colour_toward_reference():
    ref_img, src_img = _synthetic(REF, seed=1), _synthetic(SRC, seed=1)
    rs, rm = sn.estimate_stain(ref_img); ss, sm = sn.estimate_stain(src_img)
    out = sn.normalise(src_img, ss, sm, rs, rm)
    tissue = slice(40, None)
    before = np.abs(src_img[tissue].mean((0, 1)) - ref_img[tissue].mean((0, 1))).mean()
    after = np.abs(out[tissue].mean((0, 1)) - ref_img[tissue].mean((0, 1))).mean()
    assert after < 0.5 * before, (before, after)


def test_cli_fit_apply_roundtrip():
    import tifffile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tifffile.imwrite(d / "ref.tif", _synthetic(REF, (512, 512), 2))
        tifffile.imwrite(d / "src.tif", _synthetic(SRC, (700, 600), 3), resolution=(2.15, 2.15), resolutionunit="MICROMETER")
        sn.main(["fit", "--images", str(d / "ref.tif"), "--out", str(d / "ref.json")])
        assert json.loads((d / "ref.json").read_text())["method"] == "macenko"
        sn.main(["apply", "--ref", str(d / "ref.json"), "--slide", str(d / "src.tif"),
                 "--out", str(d / "out.tif"), "--tile", "256"])
        out = tifffile.imread(d / "out.tif")
        assert out.shape == (700, 600, 3) and out.dtype == np.uint8
