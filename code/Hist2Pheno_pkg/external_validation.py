"""StarDist HE external validation with the pan-cancer cross-section checkpoint.

Cohorts that have an HE slide and StarDist nuclei, and no 10x JSON or
cell-type ground truth, share this pipeline:

1. Read OME PhysicalSize and rescale onto 0.5 µm/px so a 16 px UNI crop is 8 µm.
2. Match the ``.pth`` embeddings to StarDist centroids and write a feature h5ad.
3. Predict L1 / L12 / L2 with spatial k=8.
4. Cache L2 softmax for histology niche indices, and plot spatial maps.

NPC, BRCA, CRC, and HCC are one row in ``_SIMPLE_COHORTS`` and are started with

``python code/Hist2Pheno_pkg/external_validation.py BRCA dry-run``.

Prostate and HER2ST stay in their own scripts: sample ids and µm/px are not
a single folder plus one HE suffix.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_UM_PER_PX = 0.5
PATCH_SIZE = 16
PATCH_UM = PATCH_SIZE * TARGET_UM_PER_PX  # 8 µm, same as pan-cancer UNI crops
SPATIAL_K = 8
HIDDEN_DIMS = (768, 384, 192)
HEADS = ("L1", "L12", "L2")
OBS_COL = {"L1": "pred_lineage", "L12": "pred_sublineage", "L2": "pred_CT"}
PALETTE_TIER = {"L1": "coarse", "L12": "intermediate", "L2": "fine"}

# name -> (data path under the repo, HE suffix, whether MPP_UM is required).
# JPEG/BigTIFF cohorts that still have one flat StarDist folder.
_SIMPLE_COHORTS: dict[str, tuple[str, str, bool]] = {
    "NPC": ("data/Visium/NPC", ".tif", False),
    "BRCA": ("data/Visium/BRCA", ".tif", False),
    "CRC": ("data/Visium/CRC", ".btf", False),
    "HCC": ("data/Visium/HCC", ".jpg", True),
}


def _mpp_from_env(name: str) -> float:
    raw = os.environ.get("MPP_UM", "").strip()
    if not raw:
        raise SystemExit(
            f"Set MPP_UM to the scanner µm/px before running {name}.\n"
            "The HE files have no OME PhysicalSize, and a JPEG dpi tag is not a pixel size.\n"
            f"Example: MPP_UM=0.5 bash code/External_val/{name}/demo.sh dry-run"
        )
    value = float(raw)
    if value <= 0:
        raise SystemExit(f"MPP_UM must be positive, got {value}")
    return value


def simple_cohort(name: str, repo: Path | None = None) -> ExternalCohort:
    """Build NPC, BRCA, CRC, or HCC. Other cohorts use their own subclass."""
    key = name.upper()
    if key not in _SIMPLE_COHORTS:
        known = ", ".join(sorted(_SIMPLE_COHORTS))
        raise SystemExit(f"unknown cohort {name!r}. Choose one of: {known}")
    rel, suffix, need_mpp = _SIMPLE_COHORTS[key]
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    return ExternalCohort(
        name=key,
        data_root=root / rel,
        repo=root,
        he_suffix=suffix,
        mpp_um=_mpp_from_env(key) if need_mpp else None,
    )


_CKPT_REL = Path(
    "data/PanCancer/result_all_spatial/cross_dataset_cv"
    "/D_emph_L2_spatial_pancancer/best_mlp_gpu.pt"
)
_SCALER_REL = Path("data/PanCancer/result_all_spatial/scaler.joblib")


def _import_sibling(name: str):
    """Import a Hist2Pheno_pkg module whether ``code`` or the package dir is on the path."""
    try:
        return importlib.import_module(f"Hist2Pheno_pkg.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(name)


def ome_um_per_px(tif_path: Path) -> tuple[float, float]:
    """OME PhysicalSizeX/Y in µm from the first page ImageDescription.

    TIFF XResolution/YResolution at 96 dpi is a placeholder and is not used.
    """
    import re
    import tifffile

    with tifffile.TiffFile(tif_path) as tif:
        desc = tif.pages[0].description or ""
    mx = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', desc)
    my = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', desc)
    if not mx or not my:
        raise RuntimeError(f"no OME PhysicalSize in {tif_path}")
    return float(mx.group(1)), float(my.group(1))


def scale_for_mpp(mpp: float, target_um_per_px: float = TARGET_UM_PER_PX) -> float:
    """Factor that maps native pixels onto the training grid. 1.0 means already there."""
    return float(mpp) / float(target_um_per_px)


def n_csv_rows(csv_path: Path) -> int:
    with csv_path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def n_pth(emb_dir: Path) -> int:
    if not emb_dir.is_dir():
        return 0
    return sum(1 for p in emb_dir.glob("*.pth"))


def _labels_from_index(idx: np.ndarray, names: list[str]) -> list[str]:
    out = []
    n = len(names)
    for i in idx:
        j = int(i)
        out.append(names[j] if 0 <= j < n else "")
    return out


@dataclass
class ExternalCohort:
    """One external cancer: HE + StarDist under ``data_root``, no cell-type GT."""

    name: str
    data_root: Path
    repo: Path
    target_um_per_px: float = TARGET_UM_PER_PX
    patch_size: int = PATCH_SIZE
    spatial_k: int = SPATIAL_K
    hidden_dims: tuple[int, ...] = HIDDEN_DIMS
    csv_template: str = "{sample}_Float_prob0.01_nms_0.3.csv"
    he_suffix: str = ".tif"
    mpp_um: float | None = None
    checkpoint: Path | None = None
    scaler: Path | None = None
    dataset_tag: str = "PanCancer"
    project_tag: str = "project_all_UNI"

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.repo = Path(self.repo)
        if self.checkpoint is None:
            self.checkpoint = self.repo / _CKPT_REL
        if self.scaler is None:
            self.scaler = self.repo / _SCALER_REL
        code = Path(__file__).resolve().parent.parent
        for folder in (code, code / "Hist2Pheno_pkg"):
            text = str(folder)
            if text not in sys.path:
                sys.path.insert(0, text)

    @property
    def patch_um(self) -> float:
        return self.patch_size * self.target_um_per_px

    @property
    def he_dir(self) -> Path:
        return self.data_root / "HE_images"

    @property
    def stardist_root(self) -> Path:
        return self.data_root / "StarDist_Segment"

    @property
    def uni_root(self) -> Path:
        return self.data_root / "project_all_UNI"

    @property
    def pred_root(self) -> Path:
        return self.data_root / "pred_pancancer_section"

    @property
    def feature_script(self) -> Path:
        return self.repo / "code" / "Image_feature_extraction.py"

    @property
    def _uns_key(self) -> str:
        return f"{self.name.lower()}_uni"

    def discover_samples(self, only: str | None = None) -> list[str]:
        root = self.stardist_root
        names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if only:
            if only not in names:
                raise SystemExit(f"sample {only!r} not under {root}")
            return [only]
        if not names:
            raise SystemExit(f"no StarDist sample folders in {root}")
        return names

    def sample_paths(self, sample: str) -> dict[str, Path]:
        emb = (
            self.uni_root
            / sample
            / "ImgEmbeddings_all_stardist"
            / f"sc_pth_{self.patch_size}_{self.patch_size}"
        )
        return {
            "he": self.he_dir / f"{sample}{self.he_suffix}",
            "csv": self.stardist_root / sample / self.csv_template.format(sample=sample),
            "emb": emb,
            "feat": self.pred_root / sample / f"{sample}_stardist_features.h5ad",
            "pred": self.pred_root / sample / f"{sample}_stardist_pred.h5ad",
            "log": self.uni_root / sample / "logs",
        }

    def geometry(self, sample: str) -> dict:
        paths = self.sample_paths(sample)
        if not paths["he"].is_file():
            raise FileNotFoundError(paths["he"])
        if self.mpp_um is not None:
            mpp_x = float(self.mpp_um)
        else:
            mpp_x, mpp_y = ome_um_per_px(paths["he"])
            if abs(mpp_x - mpp_y) > 1e-3:
                raise RuntimeError(f"{sample} anisotropic pixel size {mpp_x} vs {mpp_y}")
        scale = scale_for_mpp(mpp_x, self.target_um_per_px)
        return {
            "sample": sample,
            "mpp_um": mpp_x,
            "mpp_source": "cohort mpp_um" if self.mpp_um is not None else "OME PhysicalSize",
            "scale": scale,
            "scale_image": abs(scale - 1.0) > 1e-3,
            "patch_size_px": self.patch_size,
            "patch_um_after_rescale": self.patch_um,
            "patch_um_if_native_16px": self.patch_size * mpp_x,
            "he": str(paths["he"]),
            "csv": str(paths["csv"]),
            "embeddings": str(paths["emb"]),
        }

    def cmd_dry_run(self, samples: list[str]) -> None:
        print(
            f"target {self.target_um_per_px} µm/px, "
            f"patch {self.patch_size}px = {self.patch_um:.1f} µm"
        )
        print(f"checkpoint {self.checkpoint} exists={self.checkpoint.is_file()}")
        print(f"scaler     {self.scaler} exists={self.scaler.is_file()}")
        for sample in samples:
            geo = self.geometry(sample)
            paths = self.sample_paths(sample)
            n = n_csv_rows(paths["csv"]) if paths["csv"].is_file() else -1
            print(
                f"\n{sample}\n"
                f"  nuclei CSV rows : {n:,}\n"
                f"  µm/px           : {geo['mpp_um']:.6f}"
                f"  ({geo['mpp_source']})\n"
                f"  native 16px     : {geo['patch_um_if_native_16px']:.3f} µm"
                f"{'' if not geo['scale_image'] else '  (not the training field)'}\n"
                f"  scale_image     : {geo['scale_image']}  scale={geo['scale']:.6f}\n"
                f"  crop after scale: {geo['patch_um_after_rescale']:.1f} µm\n"
                f"  embeddings      : {n_pth(paths['emb']):,} .pth\n"
                f"  features h5ad   : {paths['feat'].is_file()}\n"
                f"  pred h5ad       : {paths['pred'].is_file()}"
            )

    def cmd_features(self, samples: list[str], *, force: bool) -> None:
        for sample in samples:
            paths = self.sample_paths(sample)
            geo = self.geometry(sample)
            if not paths["csv"].is_file():
                raise FileNotFoundError(paths["csv"])
            n_rows = n_csv_rows(paths["csv"])
            have = n_pth(paths["emb"])
            if not force and have >= int(0.9 * n_rows):
                print(f"[features] skip {sample}: {have:,} .pth already (FORCE=1 to redo)", flush=True)
                continue
            paths["emb"].mkdir(parents=True, exist_ok=True)
            paths["log"].mkdir(parents=True, exist_ok=True)
            dataset = f"sc_{sample}"
            cmd = [
                sys.executable,
                "-u",
                str(self.feature_script),
                "--dataset",
                dataset,
                "--position_path",
                str(paths["csv"]),
                "--rawimage_path",
                str(paths["he"]),
                "--scale_image",
                "True" if geo["scale_image"] else "False",
                "--scale",
                f"{geo['scale']:.8f}",
                "--method",
                "UNI",
                "--patch_size",
                str(self.patch_size),
                "--output_img",
                str(paths["emb"].parent / f"sc_pth_{self.patch_size}_{self.patch_size}_image"),
                "--output_pth",
                str(paths["emb"]),
                "--logging",
                str(paths["log"]) + "/",
                "--no_save_patch_images",
            ]
            print(
                f"[features] {sample}  mpp={geo['mpp_um']:.4f}  "
                f"scale={geo['scale']:.4f}  patch={self.patch_um:.1f}µm  nuclei={n_rows:,}",
                flush=True,
            )
            subprocess.run(cmd, cwd=str(self.repo), check=True)
            meta = {
                **geo,
                "n_csv": n_rows,
                "n_pth": n_pth(paths["emb"]),
                "pth_prefix": dataset,
            }
            (paths["emb"].parent / "uni_geometry.json").write_text(json.dumps(meta, indent=2))
            print(f"[features] {sample} wrote {meta['n_pth']:,} embeddings", flush=True)

    def _build_one_h5ad(self, sample: str, *, force: bool) -> Path:
        paths = self.sample_paths(sample)
        if paths["feat"].is_file() and not force:
            print(f"[h5ad] skip {sample}: {paths['feat']}", flush=True)
            return paths["feat"]
        if n_pth(paths["emb"]) == 0:
            raise FileNotFoundError(
                f"no UNI .pth under {paths['emb']}. "
                f"Run: bash code/External_val/{self.name}/demo.sh features"
            )
        base = _import_sibling("base")
        ad = importlib.import_module("anndata")

        raw = pd.read_csv(paths["csv"])
        if "centroid_x" not in raw.columns or "centroid_y" not in raw.columns:
            raise KeyError(f"{paths['csv']} needs centroid_x, centroid_y")
        width = max(8, len(str(len(raw))))
        raw.insert(0, "cell_id", [f"{sample}-stardist-{i:0{width}d}" for i in range(len(raw))])
        matching = raw.copy()
        matching["celltype"] = "unlabeled"
        prefix = f"sc_{sample}"
        print(f"[h5ad] load embeddings {paths['emb']}", flush=True)
        embeddings = base.load_hist_embeddings(
            paths["emb"],
            celltype_df=matching,
            coord_cols=("centroid_x", "centroid_y"),
            tolerance=1.0,
            pth_prefix=prefix,
        )
        matched = base.match_embeddings(
            matching,
            embeddings,
            tolerance=1.0,
            coord_cols=("centroid_x", "centroid_y"),
        )
        if not matched:
            raise RuntimeError(f"matched 0 embeddings for {sample}; check pth prefix {prefix}")
        idx = np.asarray([int(item["idx"]) for item in matched], dtype=np.int64)
        obs = raw.iloc[idx].copy().reset_index(drop=True)
        obs.index = pd.Index(obs["cell_id"].astype(str), name="cell_id")
        X = np.vstack([np.asarray(item["embedding"], dtype=np.float32).reshape(-1) for item in matched])
        adata = ad.AnnData(X=X, obs=obs)
        xy = obs[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        adata.obsm["spatial"] = xy
        adata.obsm["spatial_HE"] = xy.copy()
        geo = self.geometry(sample)
        uns = {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in geo.items()
            if k != "scale_image"
        }
        uns["scale_image"] = bool(geo["scale_image"])
        uns["n_matched"] = int(adata.n_obs)
        uns["n_csv"] = int(len(raw))
        adata.uns[self._uns_key] = uns
        paths["feat"].parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(paths["feat"])
        print(f"[h5ad] {sample}: {adata.n_obs:,}/{len(raw):,} nuclei, {adata.n_vars} features", flush=True)
        return paths["feat"]

    def cmd_h5ad(self, samples: list[str], *, force: bool) -> None:
        for sample in samples:
            self._build_one_h5ad(sample, force=force)

    def _load_model(self, device):
        import joblib

        model_mod = _import_sibling("model")
        labels = _import_sibling("pancancer_labels")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.scaler.is_file():
            raise FileNotFoundError(self.scaler)
        classes = labels.ontology_class_names()
        model = model_mod.load_model_for_predict(
            str(self.repo),
            self.dataset_tag,
            self.project_tag,
            checkpoint_path=str(self.checkpoint),
            device=device,
            hidden_dims=self.hidden_dims,
            parent_dir=True,
        )
        scaler = joblib.load(self.scaler)
        return model, scaler, classes

    def _device(self, allow_cpu: bool):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if allow_cpu:
            print("WARNING: CUDA unavailable; predicting on CPU.", flush=True)
            return torch.device("cpu")
        raise RuntimeError("CUDA unavailable. Pass --allow-cpu.")

    def cmd_predict(self, samples: list[str], *, force: bool, allow_cpu: bool) -> None:
        ad = importlib.import_module("anndata")
        base = _import_sibling("base")
        plot = _import_sibling("plot")

        device = self._device(allow_cpu)
        model, scaler, classes = self._load_model(device)
        print(
            f"[predict] device={device}  ckpt={self.checkpoint.name}  spatial_k={self.spatial_k}",
            flush=True,
        )
        prob_key = {"L1": "l1", "L12": "l12", "L2": "l2"}

        for sample in samples:
            paths = self.sample_paths(sample)
            if paths["pred"].is_file() and not force:
                print(f"[predict] skip {sample}: {paths['pred']}", flush=True)
                continue
            feat_path = paths["feat"] if paths["feat"].is_file() else self._build_one_h5ad(sample, force=False)
            adata = ad.read_h5ad(feat_path)
            X = np.asarray(base.adata_X_to_dense(adata.X), dtype=np.float32)
            xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
            Xs = np.asarray(scaler.transform(X), dtype=np.float32)
            nbr = base.build_spatial_neighbor_index(xy, k_neighbors=self.spatial_k)
            probs = plot.mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=nbr)
            obs = adata.obs.copy()
            for head in HEADS:
                pr = np.asarray(probs[prob_key[head]], dtype=np.float32)
                pred_i = np.argmax(pr, axis=1).astype(np.int64)
                obs[OBS_COL[head]] = _labels_from_index(pred_i, classes[head])
                obs[f"conf_{head}"] = pr.max(axis=1).astype(np.float32)
            pred = ad.AnnData(obs=obs)
            pred.obsm["spatial_HE"] = xy
            pred.uns["checkpoint"] = str(self.checkpoint)
            pred.uns["scaler"] = str(self.scaler)
            pred.uns["spatial_k"] = self.spatial_k
            pred.uns["classes"] = classes
            pred.uns["note"] = (
                f"No {self.name} ground truth. Labels are pan-cancer L1/L12/L2 from the "
                "cross-section checkpoint applied to StarDist nuclei."
            )
            paths["pred"].parent.mkdir(parents=True, exist_ok=True)
            pred.write_h5ad(paths["pred"])
            self._write_l2_probs(paths["pred"], probs["l2"], classes["L2"])
            print(f"[predict] {sample}: n={pred.n_obs:,} -> {paths['pred']}", flush=True)

    def l2_probs_path(self, sample: str) -> Path:
        return self.sample_paths(sample)["pred"].with_name(f"{sample}_l2_probs.npz")

    def _write_l2_probs(self, pred_h5ad: Path, probs_l2, class_names) -> Path:
        path = pred_h5ad.with_name(pred_h5ad.name.replace("_stardist_pred.h5ad", "_l2_probs.npz"))
        np.savez_compressed(
            path,
            probs=np.asarray(probs_l2, dtype=np.float32),
            class_names=np.asarray(list(class_names), dtype=object),
        )
        print(f"[probs] {path}  shape={tuple(np.asarray(probs_l2).shape)}", flush=True)
        return path

    def ensure_l2_probs(self, sample: str, *, force: bool = False, allow_cpu: bool = False):
        """L2 softmax used by TLS / SRI / TNI. Cached next to the pred h5ad."""
        path = self.l2_probs_path(sample)
        if path.is_file() and not force:
            z = np.load(path, allow_pickle=True)
            names = [str(x) for x in z["class_names"].tolist()]
            print(f"[probs] cache {path}  {z['probs'].shape}", flush=True)
            return np.asarray(z["probs"], dtype=np.float32), names

        ad = importlib.import_module("anndata")
        base = _import_sibling("base")
        plot = _import_sibling("plot")

        paths = self.sample_paths(sample)
        if not paths["feat"].is_file():
            raise FileNotFoundError(paths["feat"])
        device = self._device(allow_cpu)
        model, scaler, classes = self._load_model(device)
        adata = ad.read_h5ad(paths["feat"])
        X = np.asarray(base.adata_X_to_dense(adata.X), dtype=np.float32)
        xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        Xs = np.asarray(scaler.transform(X), dtype=np.float32)
        nbr = base.build_spatial_neighbor_index(xy, k_neighbors=self.spatial_k)
        got = plot.mlp_collect_five_head_softmax_probs(model, Xs, device, neighbor_index=nbr)
        probs = np.asarray(got["l2"], dtype=np.float32)
        if paths["pred"].is_file():
            pred = ad.read_h5ad(paths["pred"])
            if pred.n_obs != probs.shape[0]:
                raise RuntimeError(f"{sample}: probs {probs.shape[0]} != pred {pred.n_obs}")
            pred_names = [str(x) for x in pred.uns["classes"]["L2"]]
            if pred_names != list(classes["L2"]):
                raise RuntimeError(f"{sample}: L2 class order drifted from the pred h5ad")
            agree = float(
                np.mean(
                    np.array(pred.obs["pred_CT"].astype(str))
                    == np.array(classes["L2"])[probs.argmax(1)]
                )
            )
            print(f"[probs] {sample} argmax agrees with pred_CT: {agree:.4f}", flush=True)
        self._write_l2_probs(paths["pred"], probs, classes["L2"])
        return probs, list(classes["L2"])

    def cmd_probs(self, samples: list[str], *, force: bool, allow_cpu: bool) -> None:
        for sample in samples:
            self.ensure_l2_probs(sample, force=force, allow_cpu=allow_cpu)

    def _palette(self):
        palettes = _import_sibling("plotting_palettes")
        return {head: palettes.get_palette("pancancer", PALETTE_TIER[head]) for head in HEADS}

    def _he_thumbnail(self, he_path: Path, max_side: int = 1800) -> tuple[np.ndarray, int, int]:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        suffix = he_path.suffix.lower()
        if suffix in {".tif", ".tiff", ".btf"}:
            import tifffile

            with tifffile.TiffFile(he_path) as tif:
                arr = tif.series[0].asarray()
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            im = Image.open(he_path).convert("RGB")
            arr = np.asarray(im)
        h, w = arr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            im = Image.fromarray(arr, mode="RGB")
            im = im.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.BILINEAR,
            )
            arr = np.asarray(im)
        return arr, w, h

    # Notebook overview: three panels on one screen, each nucleus a ~2 px dot.
    # The HE thumbnail is matched to that panel, not the full-resolution slide.
    _PLOT_DPI = 160
    _PLOT_FIGSIZE = (17.2, 6.6)
    _PLOT_HE_MAX_SIDE = 1400

    def _neighbor_spacing_px(self, xy: np.ndarray) -> float:
        """Median nearest-neighbor distance in HE pixels."""
        from scipy.spatial import cKDTree

        n = int(xy.shape[0])
        if n < 5:
            return 30.0
        rng = np.random.default_rng(0)
        sample = xy if n <= 8000 else xy[rng.choice(n, 8000, replace=False)]
        dist, _ = cKDTree(xy).query(sample, k=2)
        med = float(np.median(dist[:, 1]))
        if not np.isfinite(med) or med <= 0:
            return 30.0
        return med

    def _scatter_on_he(
        self,
        ax,
        thumb,
        full_w,
        full_h,
        xy,
        labels,
        palette,
        title: str,
        point_s: float,
    ) -> None:
        ax.imshow(thumb, extent=(0, full_w, full_h, 0), interpolation="bilinear")
        ax.set_xlim(0, full_w)
        ax.set_ylim(full_h, 0)
        colors = [palette.get(str(lab), (0.55, 0.55, 0.55, 1.0)) for lab in labels]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors,
            s=point_s,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")

    def composition_table(self, samples: list[str]) -> pd.DataFrame:
        ad = importlib.import_module("anndata")
        rows = []
        for sample in samples:
            path = self.sample_paths(sample)["pred"]
            if not path.is_file():
                raise FileNotFoundError(path)
            obs = ad.read_h5ad(path).obs
            n = len(obs)
            for head, col in OBS_COL.items():
                counts = obs[col].astype(str).value_counts()
                for name, k in counts.items():
                    rows.append(
                        {
                            "sample": sample,
                            "head": head,
                            "class": name,
                            "n": int(k),
                            "fraction": float(k) / float(n),
                            "mean_conf": float(obs.loc[obs[col].astype(str) == name, f"conf_{head}"].mean()),
                        }
                    )
        return pd.DataFrame(rows)

    def plot_spatial(
        self,
        samples: list[str],
        *,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        he_max_side: int | None = None,
        marker_px: float = 2.2,
        save: bool = False,
    ) -> list:
        """Draw one L1/L12/L2 panel per sample. ``save=False`` writes nothing."""
        ad = importlib.import_module("anndata")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        figsize = self._PLOT_FIGSIZE if figsize is None else tuple(figsize)
        dpi = self._PLOT_DPI if dpi is None else int(dpi)
        he_max_side = self._PLOT_HE_MAX_SIDE if he_max_side is None else int(he_max_side)
        point_s = (float(marker_px) * 72.0 / dpi) ** 2
        palettes = self._palette()
        fig_dir = self.pred_root / "figures"
        if save:
            fig_dir.mkdir(parents=True, exist_ok=True)
        figures = []
        for sample in samples:
            paths = self.sample_paths(sample)
            adata = ad.read_h5ad(paths["pred"])
            xy = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
            spacing = self._neighbor_spacing_px(xy)
            thumb, full_w, full_h = self._he_thumbnail(paths["he"], max_side=he_max_side)
            fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
            print(
                f"[plot] {sample}  nuclei={len(xy):,}  neighbor={spacing:.1f}px  "
                f"figsize={figsize}  dpi={dpi}  marker_px={marker_px}",
                flush=True,
            )
            for ax, head in zip(axes, HEADS):
                labels = adata.obs[OBS_COL[head]].astype(str).to_numpy()
                self._scatter_on_he(
                    ax,
                    thumb,
                    full_w,
                    full_h,
                    xy,
                    labels,
                    palettes[head],
                    f"{head}  n={adata.n_obs:,}",
                    point_s,
                )
                names = list(dict.fromkeys(labels.tolist()))
                if len(names) <= 5:
                    ncol = 1
                elif len(names) <= 10:
                    ncol = 2
                else:
                    ncol = 3
                handles = [
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        markersize=9,
                        markerfacecolor=palettes[head].get(n, (0.5, 0.5, 0.5, 1))[:3],
                        markeredgecolor="none",
                        label=n,
                    )
                    for n in names
                ]
                ax.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.04),
                    fontsize=10,
                    frameon=False,
                    ncol=ncol,
                    handletextpad=0.3,
                    columnspacing=0.8,
                    labelspacing=0.35,
                )
            fig.suptitle(
                f"{self.name} {sample}  pan-cancer cross-section Hist2Pheno  (no GT)",
                fontsize=12,
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22, top=0.90)
            if save:
                out = fig_dir / f"{sample}_spatial_L1_L12_L2.png"
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
                print(f"[plot] {out}", flush=True)
            figures.append(fig)
        return figures

    def plot_composition(self, samples: list[str], *, save: bool = False) -> list:
        """Bar charts from ``composition_table``. ``save=False`` writes nothing."""
        import matplotlib.pyplot as plt

        comp = self.composition_table(samples)
        labels = _import_sibling("pancancer_labels")
        class_order = labels.ontology_class_names()
        fig_dir = self.pred_root / "figures"
        if save:
            fig_dir.mkdir(parents=True, exist_ok=True)
        figures = []
        for head in HEADS:
            sub = comp[comp["head"] == head]
            present = set(sub["class"].astype(str))
            classes = [c for c in class_order[head] if c in present]
            classes += sorted(present - set(classes))
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(classes) + 2), 4.2))
            x = np.arange(len(classes))
            width = 0.8 / max(len(samples), 1)
            for i, sample in enumerate(samples):
                ss = sub[sub["sample"] == sample].set_index("class")
                vals = [float(ss.loc[c, "fraction"]) if c in ss.index else 0.0 for c in classes]
                ax.bar(x + i * width, vals, width=width, label=sample)
            ax.set_xticks(x + width * (len(samples) - 1) / 2)
            ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8)
            ax.set_ylabel("fraction of StarDist nuclei")
            ax.set_title(f"{self.name} predicted composition  {head}")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            if save:
                out = fig_dir / f"composition_{head}.png"
                fig.savefig(out, dpi=140, bbox_inches="tight")
                print(f"[plot] {out}", flush=True)
            figures.append(fig)
        return figures

    def cmd_plot(self, samples: list[str]) -> None:
        """Optional batch export. ``demo.sh all`` does not call this."""
        import matplotlib.pyplot as plt

        for fig in self.plot_spatial(samples, save=True):
            plt.close(fig)
        for fig in self.plot_composition(samples, save=True):
            plt.close(fig)

    def main(self, argv: list[str] | None = None) -> int:
        p = argparse.ArgumentParser(
            description=f"{self.name} external validation (pan-cancer cross-section Hist2Pheno)"
        )
        p.add_argument("step", choices=("dry-run", "features", "h5ad", "predict", "probs", "plot"))
        p.add_argument("--sample", default="", help="one sample id; default all StarDist folders")
        p.add_argument("--force", action="store_true")
        p.add_argument("--allow-cpu", action="store_true")
        args = p.parse_args(argv)
        samples = self.discover_samples(args.sample or None)
        if args.step == "dry-run":
            self.cmd_dry_run(samples)
        elif args.step == "features":
            self.cmd_features(samples, force=args.force)
        elif args.step == "h5ad":
            self.cmd_h5ad(samples, force=args.force)
        elif args.step == "predict":
            self.cmd_predict(samples, force=args.force, allow_cpu=args.allow_cpu)
        elif args.step == "probs":
            self.cmd_probs(samples, force=args.force, allow_cpu=args.allow_cpu)
        elif args.step == "plot":
            self.cmd_plot(samples)
        return 0


def main(argv: list[str] | None = None) -> int:
    """``python external_validation.py BRCA dry-run [--sample ID]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    known = ", ".join(sorted(_SIMPLE_COHORTS))
    if not argv or argv[0].upper() not in _SIMPLE_COHORTS:
        raise SystemExit(
            "usage: python code/Hist2Pheno_pkg/external_validation.py "
            f"{{{known}}} {{dry-run|features|h5ad|predict|probs|plot}}"
        )
    name = argv.pop(0)
    return simple_cohort(name).main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
