## 2026.11.20, lingyu, add patient1_axisfix
"""Patient1 figures for ``Pred_visual_npc_uni2_ens5_ivsafe_seed42.ipynb``.

The notebook only constructs :class:`Patient1Run` and calls its methods.
Full-resolution PDF and PNG files go to the prediction ``figures/`` directory.
A notebook cell shows a reduced JPEG preview so the ipynb does not store the
full raster.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATIENT1 = "20210809-C-AH4199551"
PRED_DIRNAME = "pred_uni2_ens5_l12_links_1_ivsafe_seed42"
OOF_RELATIVE = (
    "data/PanCancer/result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42"
    "/oof_predictions.npz"
)
FEAT_RELATIVE = (
    "data/Visium/NPC/pred_uni2_fold0_l12_links_1_ivsafe_seed42"
    "/20210809-C-AH4199551/20210809-C-AH4199551_stardist_features.h5ad"
)
SAMPLE_TO_VISIUM = {
    "20210809-C-AH4199551": "patient1",
    "20210809-D-AJ3037946": "patient2",
}
SAMPLE_TO_C2L = {
    "20210809-C-AH4199551": "AH4199551",
    "20210809-D-AJ3037946": "AJ3037946",
}
ABUNDANCE_PANELS = (
    ("B cell", ("B_cell", "Plasma")),
    ("T cell", ("CD4_T", "CD8_T", "Tregs")),
    ("DC", ("DC",)),
    ("Tumor", ("Tumor",)),
)
TLS_RADIUS_UM = 55.0
UM_PER_HE_PIXEL = 0.4650793650793653
PREVIEW_MAX_PX = 1400


def resolve_repo(repo=None) -> Path:
    if repo is not None:
        root = Path(repo).expanduser().resolve()
    elif os.environ.get("HIST2PHENO_REPO"):
        root = Path(os.environ["HIST2PHENO_REPO"]).expanduser().resolve()
    else:
        root = Path("/home/lingyu/ssd2/Python/Hist2Pheno")
    if not (root / "code").is_dir():
        root = Path("/nobackup2/users/lingyu/Python/Hist2Pheno")
    return root


def _display(obj):
    try:
        from IPython.display import display
    except ImportError:
        print(obj)
        return
    display(obj)


def _preview_png(png: Path):
    """Embed a small JPEG. The full PNG and PDF stay on disk."""
    try:
        from IPython.display import Image, display
        from PIL import Image as PILImage
    except ImportError:
        print(png)
        return
    resample = getattr(getattr(PILImage, "Resampling", PILImage), "LANCZOS")
    image = PILImage.open(png).convert("RGB")
    image.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX), resample)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72)
    display(Image(data=buffer.getvalue()))


def _preview_figure(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    preview = Path(os.environ.get("TMPDIR", "/tmp")) / "_h2p_preview.png"
    preview.write_bytes(buffer.getvalue())
    _preview_png(preview)


def _present(fig, path, *, save: bool):
    from npc_figures import save_figure

    if save and path is not None:
        pdf = save_figure(fig, path)
        plt.close(fig)
        _preview_png(pdf.with_suffix(".png"))
        return
    _preview_figure(fig)
    plt.close(fig)


def _read_ensemble(path):
    return importlib.import_module("anndata").read_h5ad(path)


def _ensemble_probs(cohort, sample, *, force=False, allow_cpu=False):
    del force, allow_cpu
    path = cohort.l2_probs_path(sample)
    with np.load(path, allow_pickle=True) as z:
        probs = np.asarray(z["probs"], dtype="float32").copy()
        names = [str(x) for x in z["class_names"]]
    return probs, names


def _install_ensemble_reader():
    """Read the five-fold mean. It has no single-checkpoint ``.pipeline.json``."""
    provenance = importlib.import_module("Hist2Pheno_pkg.pipeline_provenance")
    consistency = importlib.import_module("Hist2Pheno_pkg.external_consistency")
    provenance.read_verified_prediction = _read_ensemble
    consistency.preflight = lambda cohort, samples, *, allow_cpu=False: None
    consistency.probabilities = _ensemble_probs


class Patient1Run:
    """One patient1 five-fold mean, with the figure calls used by the notebook."""

    def __init__(self, repo=None):
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        self.repo = resolve_repo(repo)
        npc_dir = self.repo / "code" / "External_val" / "NPC"
        for path in (self.repo / "code", npc_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.environ["STAR_PROB_MIN"] = "0"
        from Hist2Pheno_pkg.external_validation import ExternalCohort
        from npc_figures import configure_figure_dpi

        _install_ensemble_reader()
        self.patient = PATIENT1
        self.oof_npz = self.repo / OOF_RELATIVE
        self.feat_h5ad = self.repo / FEAT_RELATIVE
        self.save = True
        self.figsize = (17.2, 6.6)
        self.dpi = 200
        self.save_dpi = 300
        self.he_max_side = 1400
        self.marker_px = 1.2
        self.star_prob_min = 0.0
        configure_figure_dpi(save_dpi=self.save_dpi, plot_dpi=self.dpi)
        self.cohort = ExternalCohort(
            name="NPC",
            data_root=self.repo / "data" / "Visium" / "NPC",
            repo=self.repo,
            patch_size=14,
            method="UNI2",
            embedding_dirname="project_all_UNI2",
            pred_dirname=PRED_DIRNAME,
            project_tag="project_all_UNI2",
        )
        self.samples = [
            sample for sample in self.cohort.discover_samples() if sample == PATIENT1
        ]
        if not self.samples:
            raise RuntimeError(f"patient1 {PATIENT1} was not discovered")
        self.spot_maps = None
        self.c2l_color = None
        self._tls = None
        self.niche_frames = None
        self.recompute_niche = True
        self._report_setup()

    def _report_setup(self):
        print("notebook shows patient1 only:", self.samples)
        print(f"StarDist probability >= {self.star_prob_min:.2f}")
        print(f"OOF exists={self.oof_npz.is_file()}  {self.oof_npz.name}")
        print(f"pred root {self.cohort.pred_root}")
        for sample in self.samples:
            geometry = self.cohort.geometry(sample)
            print(
                f"{sample}  mpp={geometry['mpp_um']:.4f} µm/px  "
                f"native14={geometry['patch_um_if_native_16px']:.2f} µm  "
                f"after rescale={geometry['patch_um_after_rescale']:.1f} µm  "
                f"scale={geometry['scale']:.4f}"
            )
            paths = self.cohort.sample_paths(sample)
            print(f"  embeddings: {paths['emb']}")
            print(f"  pred exists: {paths['pred'].is_file()}  {paths['pred']}")

    def _require_predictions(self):
        missing = [
            sample for sample in self.samples
            if not self.cohort.sample_paths(sample)["pred"].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "No UNI2 predictions yet for "
                + ", ".join(missing)
                + ". Run: RUNS=data/PanCancer/"
                "result_uni2_ablation_l12_links_1_new1066axisfix_ivsafe_paired_v2_seed42 "
                "CUDA_DEVICE=2 bash code/External_val/predict_fold_ensemble.sh NPC"
            )

    def figure_dir(self) -> Path:
        path = self.cohort.pred_root / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def composition(self):
        """L1 / L12 / L2 composition of the five-fold mean."""
        self._require_predictions()
        table = self.cohort.composition_table(self.samples)
        for head in ("L1", "L12", "L2"):
            sub = table.loc[table["head"] == head].copy()
            order = sub.groupby("class")["fraction"].max().sort_values(ascending=False).index
            wide = (
                sub.pivot_table(
                    index="class", columns="sample", values=["fraction", "n", "mean_conf"],
                ).reindex(order)
            )
            print(f"\n===== {head} =====")
            _display(wide.round(3))
        return table

    def spatial_maps(self):
        """L1 / L12 / L2 on the H&E and as points."""
        self._require_predictions()
        figures = self.cohort.plot_spatial(
            self.samples,
            figsize=self.figsize,
            dpi=self.dpi,
            he_max_side=self.he_max_side,
            marker_px=self.marker_px,
            save=self.save,
            save_dpi=self.save_dpi,
        )
        for sample, fig in zip(self.samples, figures):
            png = self.figure_dir() / f"{sample}_spatial_L1_L12_L2.png"
            if self.save and png.is_file():
                plt.close(fig)
                _preview_png(png)
            else:
                _present(fig, None, save=False)

    def fold_l2_maps(self):
        """L2 argmax of each fold's own softmax, one panel per weight.

        The color is ``argmax`` of that fold's L2 head probabilities in
        ``{sample}_l2_probs.npz``. It is not the five-fold mean, the decision
        rule, or prior shift. Fold 4 is the inner-validation winner on the
        training cohorts; this slide was not part of that choice.
        """
        from matplotlib.lines import Line2D
        from npc_figures import load_aligned_l2_probs

        self._require_predictions()
        sample = self.patient
        paths = self.cohort.sample_paths(sample)
        thumb, full_w, full_h = self.cohort._he_thumbnail(paths["he"], max_side=self.he_max_side)
        palette = self.cohort._palette()["L2"]
        point_s = (float(self.marker_px) * 72.0 / float(self.dpi)) ** 2
        aspect = full_h / max(full_w, 1)
        panel_w = 3.35
        fig, axes = plt.subplots(
            1, 5, figsize=(panel_w * 5 + 0.4, panel_w * aspect + 1.35), dpi=self.dpi,
        )
        columns = []
        present = set()
        for fold, ax in enumerate(axes):
            folder = (
                self.repo / "data" / "Visium" / "NPC"
                / f"pred_uni2_fold{fold}_l12_links_1_ivsafe_seed42" / sample
            )
            pred = folder / f"{sample}_stardist_pred.h5ad"
            npz = folder / f"{sample}_l2_probs.npz"
            if not pred.is_file() or not npz.is_file():
                raise FileNotFoundError(f"fold {fold} prediction is missing: {folder}")
            xy, probs, names = load_aligned_l2_probs(pred, npz)
            labels = np.asarray(names, dtype=object)[np.argmax(probs, axis=1)]
            counts = pd.Series(labels).value_counts(normalize=True)
            present.update(counts.index.astype(str))
            columns.append(counts.rename(f"fold{fold}"))
            top = f"{counts.index[0]} {counts.iloc[0]:.0%}"
            self.cohort._scatter_on_he(
                ax, thumb, full_w, full_h, xy, labels, palette,
                f"fold {fold}\n{top}", point_s, show_he=True,
            )
        ordered = [name for name in palette if name in present]
        ordered += sorted(present - set(ordered))
        fig.legend(
            handles=[
                Line2D(
                    [0], [0], linestyle="none", marker="o", markersize=8,
                    markerfacecolor=palette.get(name, (0.5, 0.5, 0.5, 1))[:3],
                    markeredgecolor="none", label=name,
                )
                for name in ordered
            ],
            loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False,
            fontsize=9, ncol=min(8, len(ordered)),
        )
        fig.suptitle(
            f"{sample}  L2 argmax of each fold's softmax",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        table = pd.concat(columns, axis=1).fillna(0.0)
        table = table.loc[table.max(axis=1).sort_values(ascending=False).index]
        table.index.name = "L2"
        fig_dir = self.figure_dir()
        if self.save:
            table.to_csv(fig_dir / f"{sample}_fold_L2_argmax_fraction.csv")
        _present(fig, fig_dir / f"{sample}_fold_L2_argmax.pdf", save=self.save)
        _display(table.round(3))
        return table

    def bcell_presence(self):
        """Argmax B/Plasma beside the FPR-gated presence score."""
        from Hist2Pheno_pkg.prediction_confidence import plot_bcell_presence_figure

        self._require_predictions()
        for sample in self.samples:
            fig = plot_bcell_presence_figure(
                self.cohort,
                sample,
                oof_npz=self.oof_npz,
                figsize=(self.figsize[0] * 4 / 3, self.figsize[1] * 1.9),
                dpi=self.dpi,
                marker_px=self.marker_px,
                he_max_side=self.he_max_side,
                save=self.save,
                save_dpi=self.save_dpi,
            )
            png = self.figure_dir() / f"{sample}_spatial_B_confidence.png"
            if self.save and png.is_file():
                plt.close(fig)
                _preview_png(png)
            else:
                _present(fig, None, save=False)

    def _c2l_table(self) -> pd.DataFrame:
        from npc_figures import C2L_TYPES

        npc_data = self.repo / "data" / "Visium" / "NPC"
        files = [
            npc_data / "Annotation" / "sp.obs.csv",
            npc_data / "Annotation" / "AJ_sp.obs.csv",
        ]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing)
        table = pd.concat([pd.read_csv(path, index_col=0) for path in files], axis=0)
        table.index = table.index.astype(str)
        self._c2l_types = list(C2L_TYPES)
        print(
            "cell2location",
            [path.name for path in files],
            "rows",
            len(table),
            "samples",
            sorted(table["sample"].astype(str).unique()),
        )
        return table

    def _in_tissue_positions(self, visium_id: str):
        from scipy.spatial import cKDTree

        npc_dir = self.repo / "data" / "Visium" / "NPC" / visium_id
        scale = json.loads((npc_dir / "scalefactors_json.json").read_text())
        radius_px = float(scale["spot_diameter_fullres"]) / 2
        positions = pd.read_csv(
            npc_dir / "tissue_positions_list.csv",
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
        )
        positions = positions.loc[positions["in_tissue"] == 1].reset_index(drop=True)
        positions["barcode"] = positions["barcode"].astype(str)
        spot_xy = positions[["pxl_col", "pxl_row"]].to_numpy(dtype=np.float64)
        pitch_px = float(np.median(cKDTree(spot_xy).query(spot_xy, k=2)[0][:, 1]))
        return positions, spot_xy, radius_px, pitch_px

    def visium_resolution(self):
        """Single-nucleus L2, pseudo-spot majority, and cell2location argmax."""
        import anndata as ad
        from Hist2Pheno_pkg.nuclei_to_spots import nuclei_to_spots
        from Hist2Pheno_pkg.plotting_palettes import get_palette
        from npc_figures import (
            STACK_ORDER,
            c2l_palette,
            draw_spot_circles,
            group_fraction,
            legend_handles,
            map_l2,
        )

        self._require_predictions()
        palette = get_palette("pancancer", "fine")
        l2_names = list(palette)
        c2l_color = c2l_palette(palette)
        c2l_obs = self._c2l_table()
        nucleus_s = max(0.12, (float(self.marker_px) * 72.0 / float(self.dpi)) ** 2 * 0.35)
        spot_maps = {}
        for sample in self.samples:
            visium_id = SAMPLE_TO_VISIUM.get(sample)
            if visium_id is None:
                print(f"[skip] {sample}: no patient1/patient2 Visium folder mapping")
                continue
            pred_path = self.cohort.sample_paths(sample)["pred"]
            positions, spot_xy, radius_px, pitch_px = self._in_tissue_positions(visium_id)
            npc = ad.read_h5ad(pred_path)
            npc_xy = np.asarray(npc.obsm["spatial_HE"], dtype=np.float64)
            npc_lab = npc.obs["pred_CT"].astype(str).to_numpy()
            captured = nuclei_to_spots(
                npc_xy, spot_xy, radius_px, shape="circle",
                labels=npc_lab, class_names=l2_names,
            )
            counts = captured["n_nuclei"]
            used = counts > 0
            npc_spots = positions.loc[used].copy()
            npc_spots["pixel_x"] = spot_xy[used, 0]
            npc_spots["pixel_y"] = spot_xy[used, 1]
            npc_spots["n_nuclei"] = counts[used]
            npc_spots["dominant_l2"] = captured["dominant"][used]
            npc_spots["dominant_group"] = map_l2(npc_spots["dominant_l2"])
            for i, name in enumerate(l2_names):
                npc_spots[f"n_{name}"] = captured["class_counts"][used, i]
                npc_spots[f"fraction_{name}"] = captured["class_counts"][used, i] / counts[used]
            fig_dir = self.figure_dir()
            npc_spots.to_csv(fig_dir / f"{sample}_visium_composition.csv", index=False)
            h2p_frac = group_fraction(map_l2(npc_lab))
            pseudo_frac = group_fraction(npc_spots["dominant_group"].to_numpy())
            c2l_id = SAMPLE_TO_C2L.get(sample)
            c2l_xy = c2l_lab = c2l_frac = None
            if c2l_id is not None:
                sub = c2l_obs.loc[c2l_obs["sample"].astype(str) == str(c2l_id)].copy()
                hit = positions["barcode"].isin(sub.index)
                aligned = sub.reindex(positions.loc[hit, "barcode"])
                raw = aligned[self._c2l_types].to_numpy(dtype=np.float64)
                ok = np.isfinite(raw).all(axis=1) & (raw.sum(axis=1) > 0)
                raw = raw[ok]
                c2l_xy = spot_xy[hit.to_numpy()][ok]
                c2l_lab = np.array(self._c2l_types)[np.argmax(raw, axis=1)]
                c2l_frac = group_fraction(c2l_lab)
                print(
                    f"cell2location {c2l_id}: {int(ok.sum()):,} / {int(hit.sum()):,} "
                    "in-tissue spots matched"
                )
            spot_maps[sample] = {
                "h2p": h2p_frac,
                "pseudo": pseudo_frac,
                "c2l": c2l_frac,
                "n_nuc": len(npc_lab),
                "n_spot": int(used.sum()),
                "n_c2l": 0 if c2l_lab is None else int(len(c2l_lab)),
            }
            fraction = pd.DataFrame({
                "group": list(STACK_ORDER),
                "Hist2Pheno": [h2p_frac[key] for key in STACK_ORDER],
                "pseudo_spot": [pseudo_frac[key] for key in STACK_ORDER],
                "cell2location": [
                    float(c2l_frac[key]) if c2l_frac is not None else np.nan
                    for key in STACK_ORDER
                ],
            })
            fraction.to_csv(fig_dir / f"{sample}_nuclei_vs_visium_fraction.csv", index=False)
            _display(fraction.round(4))
            print(
                f"NPC {sample} ({visium_id}): nuclei {len(npc_xy):,}, "
                f"in-tissue spots {len(spot_xy):,}, pitch {pitch_px:.1f} px, "
                f"capture radius {radius_px:.1f} px, "
                f"nuclei in a spot {int(captured['inside'].sum()):,} / {len(npc_xy):,}, "
                f"spots with nuclei {int(used.sum()):,}"
            )
            spot_draw_radius = 0.50 * pitch_px
            fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), dpi=self.dpi)
            axes[0].set_title("Hist2Pheno  single nuclei")
            for name in l2_names:
                mask = npc_lab == name
                if np.any(mask):
                    axes[0].scatter(
                        npc_xy[mask, 0], npc_xy[mask, 1], s=nucleus_s,
                        c=[palette[name]], linewidths=0, rasterized=True,
                    )
            draw_spot_circles(
                axes[1], spot_xy[used], npc_spots["dominant_l2"].to_numpy(),
                spot_draw_radius, palette, l2_names,
            )
            axes[1].set_title("Pseudo-spot  (majority L2)")
            if c2l_xy is None:
                axes[2].text(0.5, 0.5, "no cell2location\nfor this slide", ha="center", va="center")
                axes[2].set_title("cell2location")
            else:
                draw_spot_circles(
                    axes[2], c2l_xy, c2l_lab, spot_draw_radius, c2l_color, self._c2l_types,
                )
                axes[2].set_title("cell2location  (argmax abundance)")
            pad = pitch_px
            for ax in axes:
                ax.set_xlim(npc_xy[:, 0].min() - pad, npc_xy[:, 0].max() + pad)
                ax.set_ylim(npc_xy[:, 1].max() + pad, npc_xy[:, 1].min() - pad)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            present_l2 = [
                name for name in l2_names
                if name in set(npc_lab) or name in set(npc_spots["dominant_l2"].astype(str))
            ]
            axes[1].legend(
                handles=legend_handles(present_l2, palette),
                frameon=False, fontsize=9, loc="upper center",
                bbox_to_anchor=(0.5, -0.04), ncol=4,
            )
            if c2l_lab is not None:
                present_c2l = [name for name in self._c2l_types if name in set(c2l_lab)]
                axes[2].legend(
                    handles=legend_handles(present_c2l, c2l_color, markersize=9),
                    frameon=False, fontsize=10, loc="upper center",
                    bbox_to_anchor=(0.5, -0.04), ncol=4,
                )
            fig.suptitle(f"NPC {sample}  nuclei / pseudo-spot / cell2location", fontsize=12, y=1.02)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.16)
            _present(fig, fig_dir / f"{sample}_nuclei_vs_visium.pdf", save=self.save)
        self.spot_maps = spot_maps
        self.c2l_color = c2l_color
        return spot_maps

    def abundance_vs_cell2location(self):
        """Per-type mean softmax against cell2location counts. Each panel has its own scale."""
        from Hist2Pheno_pkg.nuclei_to_spots import nuclei_to_spots
        from npc_figures import C2L_TYPES, l2_group_means, load_aligned_l2_probs, plot_abundance_maps

        self._require_predictions()
        c2l_obs = self._c2l_table()
        fig_dir = self.figure_dir()
        for sample in self.samples:
            visium_id = SAMPLE_TO_VISIUM.get(sample)
            c2l_id = SAMPLE_TO_C2L.get(sample)
            if visium_id is None or c2l_id is None:
                print(f"[skip] {sample}: no Visium or cell2location id")
                continue
            pred_path = self.cohort.sample_paths(sample)["pred"]
            npc_xy, probs, class_names = load_aligned_l2_probs(
                pred_path, pred_path.with_name(f"{sample}_l2_probs.npz"),
            )
            positions, spot_xy, radius_px, pitch_px = self._in_tissue_positions(visium_id)
            captured = nuclei_to_spots(npc_xy, spot_xy, radius_px, shape="circle", values=probs)
            h2p = l2_group_means(captured["mean"], class_names)
            h2p[captured["n_nuclei"] == 0] = np.nan
            sub = c2l_obs.loc[c2l_obs["sample"].astype(str) == str(c2l_id)].copy()
            aligned = sub.reindex(positions["barcode"])
            c2l = aligned[list(C2L_TYPES)].to_numpy(dtype=np.float64)
            c2l[~np.isfinite(c2l).all(axis=1) | ~(np.nansum(c2l, axis=1) > 0)] = np.nan
            fig, table = plot_abundance_maps(
                spot_xy, h2p, c2l, list(C2L_TYPES),
                radius=0.50 * pitch_px, sample=sample, dpi=self.dpi,
            )
            table.to_csv(fig_dir / f"{sample}_abundance_spearman.csv", index=False)
            _display(table.round(3))
            print(
                f"NPC {sample}: spots {len(spot_xy):,}, "
                f"with nuclei {int((captured['n_nuclei'] > 0).sum()):,}, "
                f"with cell2location {int(np.isfinite(c2l).all(axis=1).sum()):,}"
            )
            _present(fig, fig_dir / f"{sample}_h2p_vs_c2l_abundance.pdf", save=self.save)

    def nucleus_confidence(self):
        """Per-L2-type softmax and cluster conformal confidence. Left off in the notebook."""
        from Hist2Pheno_pkg.prediction_confidence import ensure_cluster_conformal_npz
        from npc_figures import load_aligned_l2_probs, plot_nucleus_abundance_confidence

        self._require_predictions()
        pred_path = self.cohort.sample_paths(PATIENT1)["pred"]
        probs_path = pred_path.with_name(f"{PATIENT1}_l2_probs.npz")
        xy, probs, class_names = load_aligned_l2_probs(pred_path, probs_path)
        sidecar = pred_path.with_name(f"{PATIENT1}_confidence_he_cluster.npz")
        ensure_cluster_conformal_npz(
            probs_path, self.oof_npz, self.feat_h5ad, sidecar, class_names, n_clusters=20,
        )
        with np.load(sidecar) as z:
            confidence = np.column_stack([
                np.asarray(z[f"confidence_{name}"], dtype=np.float32) for name in class_names
            ])
        nucleus_s = max(0.12, (float(self.marker_px) * 72.0 / float(self.dpi)) ** 2 * 0.35)
        figures = plot_nucleus_abundance_confidence(
            xy, probs, confidence, class_names,
            sample=f"NPC {PATIENT1} (patient1)", dpi=self.dpi, marker_s=nucleus_s,
        )
        fig_dir = self.figure_dir()
        for page, fig in enumerate(figures, start=1):
            _present(
                fig,
                fig_dir / f"{PATIENT1}_nuclei_abundance_confidence_p{page}.pdf",
                save=self.save,
            )

    def stacked_composition(self):
        """100% bars: Hist2Pheno nuclei, pseudo-spot majority, cell2location argmax."""
        from npc_figures import STACK_ORDER

        if not self.spot_maps or self.c2l_color is None:
            raise RuntimeError("run visium_resolution() before stacked_composition()")
        methods = ("Hist2Pheno", "pseudo-spot", "cell2location")
        method_key = {"Hist2Pheno": "h2p", "pseudo-spot": "pseudo", "cell2location": "c2l"}
        samples_plot = [sample for sample in self.samples if sample in self.spot_maps]
        n_panel = max(len(samples_plot), 1)
        fig, axes = plt.subplots(
            1, n_panel, figsize=(1.2 * n_panel + 1.8, 3.0), dpi=self.dpi, squeeze=False,
        )
        axes = axes[0]
        for ax, sample in zip(axes, samples_plot):
            record = self.spot_maps[sample]
            x = np.arange(len(methods))
            bottom = np.zeros(len(methods))
            for group in STACK_ORDER:
                values = []
                for method in methods:
                    series = record[method_key[method]]
                    values.append(float(series[group]) if series is not None else 0.0)
                values = np.asarray(values, dtype=float)
                if np.allclose(values, 0.0):
                    continue
                ax.bar(
                    x, values, bottom=bottom, width=0.62,
                    color=self.c2l_color[group][:3], edgecolor="white", linewidth=0.4, label=group,
                )
                bottom = bottom + values
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=20, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("fraction")
            extra = ""
            if record["c2l"] is None:
                extra = "  (no cell2location)"
                ax.text(2, 0.5, "n/a", ha="center", va="center", fontsize=9, color="#888888")
            ax.set_title(f"{sample}{extra}", fontsize=10)
            ax.set_xlim(-0.55, 2.55)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False, fontsize=10, title="mapped type",
        )
        fig.suptitle("NPC  Hist2Pheno vs pseudo-spot vs cell2location", fontsize=12, y=1.02)
        fig.tight_layout()
        fig_dir = self.figure_dir()
        _present(
            fig,
            fig_dir / "NPC_hist2pheno_pseudospot_cell2location_stacked.pdf",
            save=self.save,
        )
        rows = []
        for sample, record in self.spot_maps.items():
            for method, key in method_key.items():
                series = record[key]
                if series is None:
                    continue
                for group in STACK_ORDER:
                    rows.append({
                        "sample": sample,
                        "method": method,
                        "group": group,
                        "fraction": float(series[group]),
                    })
        table = pd.DataFrame(rows)
        path = fig_dir / "NPC_hist2pheno_pseudospot_cell2location_stacked.csv"
        table.to_csv(path, index=False)
        print(path)
        _display(table.pivot_table(
            index="group", columns=["sample", "method"], values="fraction",
        ).round(3))
        return table

    def _ensure_tls(self):
        if self._tls is not None:
            return self._tls
        import anndata as ad
        from Hist2Pheno_pkg.panel_figures import crop_he_rectangle, nuclei_in_rectangle
        from npc_figures import keep_leftmost_foci, load_tls_xy, nuclei_in_contours, tls_smooth_contours

        sample = PATIENT1
        npc_data = self.repo / "data" / "Visium" / "NPC"
        tls_xy, tls_bc, pitch_px, spot_r_px = load_tls_xy(
            npc_data / "Annotation" / "TLS_spot_label.csv", npc_data / "patient1",
        )
        keep = keep_leftmost_foci(tls_xy, pitch_px, n_keep=3)
        tls_xy, tls_bc = tls_xy[keep], tls_bc[keep]
        contours = tls_smooth_contours(tls_xy, disk_r=0.70 * pitch_px, sigma=0.45 * pitch_px)
        paths = self.cohort.sample_paths(sample)
        geometry = self.cohort.geometry(sample)
        pad = 2.6 * pitch_px
        x0 = max(0.0, float(tls_xy[:, 0].min() - pad))
        y0 = max(0.0, float(tls_xy[:, 1].min() - pad))
        thumb, full_w, full_h = self.cohort._he_thumbnail(paths["he"], max_side=self.he_max_side)
        x1 = min(float(full_w), float(tls_xy[:, 0].max() + pad))
        y1 = min(float(full_h), float(tls_xy[:, 1].max() + pad))
        adata = ad.read_h5ad(paths["pred"])
        xy_he = np.asarray(adata.obsm["spatial_HE"], dtype=np.float64)
        in_roi = nuclei_in_rectangle(xy_he, x0, y0, x1, y1)
        in_outline = nuclei_in_contours(xy_he, contours) & in_roi
        self._tls = {
            "sample": sample,
            "tls_xy": tls_xy,
            "tls_bc": tls_bc,
            "pitch_px": pitch_px,
            "spot_r_px": spot_r_px,
            "contours": contours,
            "paths": paths,
            "w_um": (x1 - x0) * geometry["mpp_um"],
            "h_um": (y1 - y0) * geometry["mpp_um"],
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "thumb": thumb,
            "full_w": full_w,
            "full_h": full_h,
            "he_roi": crop_he_rectangle(paths["he"], x0, y0, x1, y1),
            "adata": adata,
            "xy_he": xy_he,
            "in_roi": in_roi,
            "in_outline": in_outline,
        }
        return self._tls

    def tls_annotation(self):
        """Expert outline, L1/L12/L2 in the zoom, and the stacked bar inside the outline."""
        from Hist2Pheno_pkg.panel_figures import plot_pred_composition_stacked, plot_roi_pred_spatial
        from matplotlib.patches import Rectangle
        from npc_figures import draw_tls

        self._require_predictions()
        tls = self._ensure_tls()
        sample = tls["sample"]
        print(
            f"TLS {sample}: {len(tls['tls_bc'])} spots, {len(tls['contours'])} smooth foci, "
            f"pitch {tls['pitch_px']:.1f} px, "
            f"zoom [{tls['x0']:.0f}:{tls['x1']:.0f}]×[{tls['y0']:.0f}:{tls['y1']:.0f}]  "
            f"{tls['w_um']:.0f}×{tls['h_um']:.0f} µm"
        )
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), dpi=self.dpi)
        axes[0].imshow(
            tls["thumb"], extent=(0, tls["full_w"], tls["full_h"], 0), interpolation="bilinear",
        )
        draw_tls(axes[0], tls["contours"], fill=False, lw=1.6)
        axes[0].add_patch(Rectangle(
            (tls["x0"], tls["y0"]), tls["x1"] - tls["x0"], tls["y1"] - tls["y0"],
            fill=False, ec="#E6B800", lw=1.7, zorder=7,
        ))
        axes[0].set_title("patient1  HE + expert TLS")
        axes[0].set_xlim(0, tls["full_w"])
        axes[0].set_ylim(tls["full_h"], 0)
        he_roi = tls["he_roi"]
        axes[1].imshow(
            he_roi, extent=(0, he_roi.shape[1], he_roi.shape[0], 0), interpolation="bilinear",
        )
        draw_tls(
            axes[1], tls["contours"], spots=tls["tls_xy"], spot_r=tls["spot_r_px"],
            local=(tls["x0"], tls["y0"]), fill=True, lw=2.2,
        )
        axes[1].set_title(f"TLS zoom  {tls['w_um']:.0f}×{tls['h_um']:.0f} µm")
        axes[1].set_xlim(0, he_roi.shape[1])
        axes[1].set_ylim(he_roi.shape[0], 0)
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig_dir = self.figure_dir()
        _present(fig, fig_dir / f"{sample}_TLS_HE_overview_zoom.pdf", save=self.save)

        print(
            f"nuclei inside black TLS outline: {int(tls['in_outline'].sum()):,} "
            f"/ zoom {int(tls['in_roi'].sum()):,}"
        )
        fig_z, _in_roi = plot_roi_pred_spatial(
            tls["adata"],
            tls["paths"]["he"],
            x0=tls["x0"], y0=tls["y0"], x1=tls["x1"], y1=tls["y1"],
            palettes=self.cohort._palette(),
            in_roi=tls["in_roi"],
            legend_mask=tls["in_outline"],
            marker_px=3.2,
            dpi=self.dpi,
            figsize=(16.8, 11.2),
            title=f"{sample}  Hist2Pheno in expert TLS zoom  n={int(tls['in_roi'].sum()):,}",
            save_path=None,
            save_dpi=self.save_dpi,
        )
        for ax in fig_z.axes[:6]:
            draw_tls(ax, tls["contours"], local=(tls["x0"], tls["y0"]), fill=False, lw=1.6)
        _present(fig_z, fig_dir / f"{sample}_TLS_ROI_spatial_L1_L12_L2.pdf", save=self.save)

        fig_bar, composition = plot_pred_composition_stacked(
            tls["adata"].obs.loc[tls["in_outline"]],
            self.cohort._palette(),
            dpi=self.dpi,
            figsize=(12.6, 4.6),
            title=(
                f"{sample}  Hist2Pheno inside expert TLS outline  "
                f"n={int(tls['in_outline'].sum()):,}"
            ),
        )
        if self.save:
            composition.to_csv(fig_dir / f"{sample}_TLS_outline_stacked.csv", index=False)
        _present(fig_bar, fig_dir / f"{sample}_TLS_outline_stacked.pdf", save=self.save)
        _display(composition)
        return composition

    def _abundance_table(self, tls):
        from npc_figures import load_aligned_l2_probs, nuclei_in_contours

        xy, probs, names = load_aligned_l2_probs(
            tls["paths"]["pred"],
            tls["paths"]["pred"].with_name(f"{tls['sample']}_l2_probs.npz"),
        )
        if len(xy) != len(tls["xy_he"]):
            raise RuntimeError(f"probs {len(xy)} != prediction {len(tls['xy_he'])}")
        inside = nuclei_in_contours(xy, tls["contours"])
        in_zoom = (
            (xy[:, 0] >= tls["x0"]) & (xy[:, 0] < tls["x1"])
            & (xy[:, 1] >= tls["y0"]) & (xy[:, 1] < tls["y1"])
        )
        scores = {}
        rows = []
        for label, members in ABUNDANCE_PANELS:
            missing = [name for name in members if name not in names]
            if missing:
                raise KeyError(f"{label}: {missing} not in {names}")
            score = probs[:, [names.index(name) for name in members]].sum(axis=1)
            scores[label] = score
            inn = float(score[inside].mean())
            out = float(score[~inside].mean())
            rows.append({
                "panel": label,
                "columns": "+".join(members),
                "inside": inn,
                "outside": out,
                "inside / outside": inn / out,
                "p99": float(np.quantile(score, 0.99)),
            })
        table = pd.DataFrame(rows)
        return xy, scores, inside, in_zoom, table

    def tls_softmax_abundance(self, *, gray_outside=False):
        """L2 softmax of B, T, DC, and Tumor on the expert-TLS crop.

        ``gray_outside=True`` colors only nuclei inside the black outline.
        Nuclei in the crop but outside the outline are light gray.
        """
        from npc_figures import draw_tls

        self._require_predictions()
        tls = self._ensure_tls()
        xy, scores, inside, in_zoom, table = self._abundance_table(tls)
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        he_roi = tls["he_roi"]
        aspect = he_roi.shape[0] / he_roi.shape[1]
        panel_w = 3.7
        fig, axes = plt.subplots(
            1, 4,
            figsize=(panel_w * 4 + 2.6, panel_w * aspect + 1.5),
            dpi=self.dpi,
            layout="constrained",
        )
        colored = in_zoom & inside
        outside = in_zoom & ~inside
        for ax, (label, members) in zip(axes, ABUNDANCE_PANELS):
            score = scores[label]
            vmax = float(table.loc[table["panel"] == label, "p99"].iloc[0])
            ax.imshow(
                he_roi, extent=(0, he_roi.shape[1], he_roi.shape[0], 0), interpolation="bilinear",
            )
            if gray_outside:
                ax.scatter(
                    xy[outside, 0] - tls["x0"], xy[outside, 1] - tls["y0"],
                    c="#d0d0d0", s=4.0, linewidths=0, rasterized=True, zorder=2,
                )
                points = colored
            else:
                points = in_zoom
            scatter = ax.scatter(
                xy[points, 0] - tls["x0"], xy[points, 1] - tls["y0"],
                c=score[points], s=4.0, cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6),
                linewidths=0, rasterized=True, zorder=3,
            )
            draw_tls(ax, tls["contours"], local=(tls["x0"], tls["y0"]), fill=False, lw=1.2)
            ax.set_xlim(0, he_roi.shape[1])
            ax.set_ylim(he_roi.shape[0], 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            inn = float(score[inside].mean())
            if gray_outside:
                ax.set_title(f"{label}\ninside {inn:.3f}", fontsize=10)
            else:
                out = float(score[~inside].mean())
                ax.set_title(f"{label}\n{inn:.3f} inside / {out:.3f} outside", fontsize=10)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02, label="+".join(members))
        if gray_outside:
            fig.suptitle(
                f"{tls['sample']}  expert TLS   L2 softmax inside the outline, gray outside",
                fontsize=12,
            )
            stem = f"{tls['sample']}_TLS_abundance_B_T_DC_Tumor_inside"
        else:
            fig.suptitle(
                f"{tls['sample']}  expert TLS   L2 softmax abundance, own 99th percentile",
                fontsize=12,
            )
            stem = f"{tls['sample']}_TLS_abundance_B_T_DC_Tumor"
        fig_dir = self.figure_dir()
        if self.save and not gray_outside:
            table.to_csv(fig_dir / f"{stem}.csv", index=False)
        _present(fig, fig_dir / f"{stem}.pdf", save=self.save)
        _display(table)
        return table

    def _niche_config(self):
        for path in (self.repo / "code" / "Hist2Pheno_pkg", self.repo / "code" / "Xenium_lung"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        niche = importlib.import_module("histology_niche_index")
        derived = importlib.import_module("histology_derived_niche_index")
        importlib.reload(niche)
        importlib.reload(derived)
        config = niche.OrganNicheConfig(
            pan_organ="pancancer_npc",
            tls_b=("B_cell", "Plasma"),
            tls_t=("CD4_T", "CD8_T", "Tregs"),
            tls_dc=("DC",),
            epithelium=(
                "Epithelial", "Tumor", "Alveolar",
                "Injury_epithelial", "DCIS", "Myoepithelial",
            ),
            fibroblast=("Fibroblast", "Myofibroblast", "Stromal"),
            endothelial=("Endothelial",),
            macrophage=("Macrophage",),
            um_per_he_pixel=UM_PER_HE_PIXEL,
            spatial_radius_um=TLS_RADIUS_UM,
            cohort="NPC",
        )
        return niche, derived, config

    def niche_formulas(self):
        niche, _derived, config = self._niche_config()
        self._niche = niche
        self._niche_derived = _derived
        self._niche_config_value = config
        _display(niche.index_formula_table(config))
        return config

    def niche_indices(self, *, recompute=None):
        """TLS / SRI / TNI from the stored five-fold L2 softmax."""
        import anndata as ad

        self._require_predictions()
        if not hasattr(self, "_niche"):
            self.niche_formulas()
        niche = self._niche
        config = self._niche_config_value
        recompute = self.recompute_niche if recompute is None else recompute
        frames = {}
        for sample in self.samples:
            paths = self.cohort.sample_paths(sample)
            niche_path = paths["pred"].with_name(f"{sample}_niche_indices.h5ad")
            mpp = float(self.cohort.geometry(sample)["mpp_um"])
            if niche_path.is_file() and not recompute:
                cached = ad.read_h5ad(niche_path)
                frame = cached.obs.copy()
                xy = np.asarray(cached.obsm["spatial_HE"], dtype=np.float64)
                frame["coord_x"] = xy[:, 0]
                frame["coord_y"] = xy[:, 1]
                print(f"[niche] cache {niche_path.name}  n={len(frame):,}  µm/px={mpp:.4f}")
            else:
                probs, class_names = self.cohort.ensure_l2_probs(sample)
                missing = niche.missing_index_cell_types(class_names, config)
                if missing:
                    raise RuntimeError(f"{sample} missing niche compartments: {missing}")
                pred = ad.read_h5ad(paths["pred"])
                coords = np.asarray(pred.obsm["spatial_HE"], dtype=np.float64)
                frame = pred.obs.copy()
                frame["coord_x"] = coords[:, 0]
                frame["coord_y"] = coords[:, 1]
                frame = niche.add_abundance_indices(frame, probs, class_names, config)
                frame = niche.add_spatial_indices(
                    frame, probs, class_names, coords, config,
                    radius_um=TLS_RADIUS_UM, um_per_pixel=mpp,
                )
                out = ad.AnnData(obs=frame.drop(columns=["coord_x", "coord_y"]))
                out.obsm["spatial_HE"] = coords
                out.uns["tls_radius_um"] = TLS_RADIUS_UM
                out.uns["um_per_px"] = mpp
                out.write_h5ad(niche_path)
                print(f"[niche] wrote {niche_path}  n={len(frame):,}")
            frames[sample] = frame
            print(f"  TLS hotspot: {niche.summarize_tls_hotspots(frame)}")
            print(
                f"  mean TLS={frame['idx_TLS'].mean():.3f}  "
                f"TLS_spatial={frame['idx_TLS_spatial'].mean():.3f}  "
                f"SRI={frame['idx_SRI_ratio'].mean():.3f}  "
                f"TNI={frame['idx_TNI_ratio'].mean():.3f}"
            )
        self.niche_frames = frames
        return frames

    def niche_maps(self):
        from npc_figures import tls_score_row

        if not self.niche_frames:
            raise RuntimeError("run niche_indices() before niche_maps()")
        if not hasattr(self, "_niche_derived"):
            self.niche_formulas()
        derived = self._niche_derived
        fig_dir = self.figure_dir()
        table = pd.DataFrame([
            tls_score_row(frame, sample) for sample, frame in self.niche_frames.items()
        ])
        path = fig_dir / "tls_score_summary.csv"
        table.to_csv(path, index=False)
        print(path)
        _display(table.round(3))
        for sample, frame in self.niche_frames.items():
            fig, _axes = derived.plot_tls_abundance_vs_spatial(
                frame, sample, radius_um=TLS_RADIUS_UM, use_entropy=False,
                split_spatial_candidates=True, dpi=self.dpi, save_dpi=self.save_dpi,
                point_size=0.15, candidate_size=1.2, save_path=None,
            )
            _present(fig, fig_dir / f"{sample}_TLS_abundance_vs_spatial.png", save=self.save)
            for abundance_col, spatial_col, candidate_col, tag in (
                ("idx_SRI_ratio", "idx_SRI_spatial", "sri_candidate", "SRI"),
                ("idx_TNI_ratio", "idx_TNI_spatial", "tni_candidate", "TNI"),
            ):
                fig, _axes = derived.plot_tls_abundance_vs_spatial(
                    frame, sample, radius_um=TLS_RADIUS_UM,
                    abundance_col=abundance_col, spatial_col=spatial_col,
                    candidate_col=candidate_col, use_entropy=False,
                    split_spatial_candidates=True, score_name=tag,
                    dpi=self.dpi, save_dpi=self.save_dpi,
                    point_size=0.15, candidate_size=1.2, save_path=None,
                )
                _present(fig, fig_dir / f"{sample}_{tag}_abundance_vs_spatial.png", save=self.save)
        return table


def open_patient1(repo=None) -> Patient1Run:
    return Patient1Run(repo)
