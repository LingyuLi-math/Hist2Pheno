"""Unit tests for UTOPIA-style per-nucleus confidence."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from prediction_confidence import (
    annotate_nuclei,
    attach_to_obs,
    bh_adjust,
    cluster_adjusted_pvalues,
    confidence_from_p,
    conformal_from_oof,
    conformal_pvalues,
    correct_predictions_by_he,
    ensure_cluster_conformal_npz,
    gated_from_npz,
    highlight_labels,
    histology_tolerance,
    load_l2_probs,
    softmax_entropy,
    softmax_margin,
    trusted_argmax,
    write_confidence_npz,
)


class SoftmaxDiagnosticsTests(unittest.TestCase):
    def test_entropy_onehot_is_zero(self):
        p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
        ent = softmax_entropy(p)
        self.assertAlmostEqual(float(ent[0]), 0.0, places=5)
        self.assertGreater(float(ent[1]), 0.3)

    def test_margin_and_family_collapse(self):
        names = ["B_cell", "Plasma", "Tumor"]
        probs = np.array(
            [
                [0.48, 0.46, 0.06],
                [0.05, 0.10, 0.85],
                [0.34, 0.33, 0.33],
            ],
            dtype=np.float32,
        )
        top1, p1, margin = softmax_margin(probs)
        self.assertEqual(int(top1[0]), 0)
        self.assertLess(float(margin[0]), 0.05)
        self.assertGreater(float(margin[1]), 0.7)
        table = annotate_nuclei(probs, names, margin_min=0.08, max_prob_min=0.25)
        self.assertEqual(table.loc[0, "label_plot"], "B_Plasma")
        self.assertEqual(table.loc[1, "label_plot"], "Tumor")
        self.assertEqual(table.loc[2, "label_plot"], "low_confidence")


class ConformalTests(unittest.TestCase):
    def test_bh_monotonic_and_clip(self):
        adj = bh_adjust(np.array([0.001, 0.02, 0.8]))
        self.assertTrue(np.all(np.diff(adj) >= -1e-12))
        self.assertLessEqual(float(adj[-1]), 1.0)
        self.assertLess(float(adj[0]), float(adj[-1]))

    def test_confidence_stretch(self):
        self.assertAlmostEqual(float(confidence_from_p(np.array([0.01]))[0]), 1.0, places=5)
        self.assertAlmostEqual(float(confidence_from_p(np.array([0.5]))[0]), 0.0, places=5)

    def test_high_score_beats_null_calibration(self):
        rng = np.random.default_rng(0)
        calib_y = np.zeros(2000)
        calib_yhat = rng.beta(1, 20, size=2000)
        test = np.array([0.9, 0.01])
        p = conformal_pvalues(calib_y, calib_yhat, test)
        self.assertLess(float(p[0]), 0.02)
        self.assertGreater(float(p[1]), 0.2)

    def test_oof_roundtrip(self):
        names = ["B_cell", "Tumor"]
        rng = np.random.default_rng(1)
        labels = np.array([0] * 80 + [1] * 420)
        oof = np.zeros((500, 2), dtype=np.float32)
        oof[labels == 0, 0] = rng.uniform(0.6, 0.95, size=80)
        oof[labels == 0, 1] = 1.0 - oof[labels == 0, 0]
        oof[labels == 1, 1] = rng.uniform(0.6, 0.95, size=420)
        oof[labels == 1, 0] = 1.0 - oof[labels == 1, 1]
        test = np.array([[0.88, 0.12], [0.04, 0.96]], dtype=np.float32)
        with TemporaryDirectory() as tmp:
            oof_path = Path(tmp) / "oof_predictions.npz"
            np.savez(oof_path, l2_labels=labels, l2_probs=oof)
            conf = conformal_from_oof(
                test,
                names,
                oof_path,
                classes=["B_cell"],
                groups={"B_Plasma": ("B_cell",)},
                max_calib=0,
            )
            self.assertIn("B_cell", conf)
            self.assertTrue(bool(conf["B_cell"]["present"][0]))
            self.assertFalse(bool(conf["B_cell"]["present"][1]))
            self.assertGreater(float(conf["B_cell"]["confidence"][0]), 0.5)
            pred_path = Path(tmp) / "sample_l2_probs.npz"
            np.savez(pred_path, probs=test, class_names=np.array(names, dtype=object))
            loaded_names, loaded_p = load_l2_probs(pred_path)
            self.assertEqual(loaded_names, names)
            table = annotate_nuclei(loaded_p, loaded_names)
            obs = attach_to_obs(pd.DataFrame({"id": [0, 1]}), table, conformal=conf)
            self.assertIn("label_plot", obs.columns)
            self.assertTrue(bool(obs.loc[0, "present_B_cell_fpr0.10"]))
            out = write_confidence_npz(Path(tmp) / "sample_confidence.npz", table, conf)
            self.assertTrue(out.exists())

    def test_wsi_bh_hides_tail_that_fpr_keeps(self):
        names = ["B_cell", "Tumor"]
        rng = np.random.default_rng(2)
        labels = np.array([0] * 40 + [1] * 400)
        oof = np.zeros((440, 2), dtype=np.float32)
        oof[labels == 0, 0] = rng.uniform(0.55, 0.9, size=40)
        oof[labels == 0, 1] = 1.0 - oof[labels == 0, 0]
        oof[labels == 1, 1] = rng.uniform(0.7, 0.98, size=400)
        oof[labels == 1, 0] = 1.0 - oof[labels == 1, 1]
        test = np.zeros((5000, 2), dtype=np.float32)
        test[:, 1] = 0.97
        test[:, 0] = 0.03
        test[0] = (0.85, 0.15)
        with TemporaryDirectory() as tmp:
            oof_path = Path(tmp) / "oof_predictions.npz"
            np.savez(oof_path, l2_labels=labels, l2_probs=oof)
            conf = conformal_from_oof(
                test, names, oof_path, classes=["B_cell"], max_calib=0, fpr=0.10
            )
            self.assertTrue(bool(conf["B_cell"]["present"][0]))
            self.assertEqual(int(conf["B_cell"]["present"][1:].sum()), 0)
            self.assertEqual(int((conf["B_cell"]["p_adj"] <= 0.10).sum()), 0)

    def test_highlight_and_gated_from_npz(self):
        pred = np.array(["B_cell", "Tumor", "Plasma"])
        lab = highlight_labels(pred, ("B_cell", "Plasma"))
        self.assertEqual(list(lab), ["B_cell", "other", "Plasma"])
        z = {
            "present_B_cell": np.array([True, False, True]),
            "confidence_B_cell": np.array([0.8, 0.2, 0.4], dtype=np.float32),
        }
        present, conf, gated = gated_from_npz(z, "B_cell")
        self.assertTrue(bool(present[0]))
        self.assertAlmostEqual(float(gated[0]), 0.8, places=5)
        self.assertEqual(float(gated[1]), 0.0)

    def test_trusted_argmax_unlabeled_when_nothing_present(self):
        n = 4
        conf = {
            "B_cell": {
                "confidence": np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32),
                "present": np.array([True, False, False, False]),
            },
            "Tumor": {
                "confidence": np.array([0.3, 0.7, 0.95, 0.2], dtype=np.float32),
                "present": np.array([True, True, False, False]),
            },
        }
        labels, peak = trusted_argmax(conf)
        self.assertEqual(list(labels), ["B_cell", "Tumor", "low_confidence", "low_confidence"])
        self.assertAlmostEqual(float(peak[0]), 0.9, places=5)
        self.assertEqual(float(peak[2]), 0.0)


class HistologyClusterTests(unittest.TestCase):
    def test_tolerance_is_negative_when_the_test_slide_scores_higher(self):
        rng = np.random.default_rng(0)
        calib_y = np.zeros(800)
        calib_yhat = rng.uniform(0.0, 0.2, size=800)
        test_yhat = rng.uniform(0.25, 0.45, size=400)
        tol = histology_tolerance(calib_yhat, calib_y, test_yhat, 0.0, "medium")
        self.assertLess(tol, 0.0)
        flat = histology_tolerance(calib_yhat, calib_y, calib_yhat, 0.0, "medium")
        self.assertEqual(flat, 0.0)

    def test_cluster_bh_separates_a_high_score_group(self):
        rng = np.random.default_rng(0)
        calib_y = np.zeros(500)
        calib_yhat = rng.uniform(0.0, 0.05, size=500)
        test_yhat = np.concatenate([np.full(30, 0.9), np.full(30, 0.02)])
        clusters = np.array([0] * 30 + [1] * 30)
        adj = cluster_adjusted_pvalues(
            calib_y, calib_yhat, test_yhat, clusters, correct_batch_effect=False
        )
        self.assertLess(float(adj[:30].max()), 0.05)
        self.assertGreater(float(adj[30:].min()), 0.2)

    def test_small_calibration_cluster_stays_at_one(self):
        calib_y = np.zeros(10)
        calib_yhat = np.full(10, 0.01)
        test_yhat = np.full(40, 0.9)
        clusters = np.zeros(40, dtype=np.int32)
        adj = cluster_adjusted_pvalues(
            calib_y, calib_yhat, test_yhat, clusters, correct_batch_effect=False
        )
        self.assertTrue(np.all(adj == 1.0))

    def test_embedding_clusters_roundtrip(self):
        import h5py

        rng = np.random.default_rng(1)
        n, d = 60, 12
        names = ["B_cell", "Tumor"]
        labels = np.array([0] * 80 + [1] * 420)
        oof = np.zeros((500, 2), dtype=np.float32)
        oof[labels == 0, 0] = rng.uniform(0.6, 0.95, size=80)
        oof[labels == 0, 1] = 1.0 - oof[labels == 0, 0]
        oof[labels == 1, 1] = rng.uniform(0.6, 0.95, size=420)
        oof[labels == 1, 0] = 1.0 - oof[labels == 1, 1]
        probs = np.zeros((n, 2), dtype=np.float32)
        probs[:, 0] = 0.02
        probs[:, 1] = 0.98
        probs[:8, 0] = 0.95
        probs[:8, 1] = 0.05
        obs = np.array([f"n{i:03d}" for i in range(n)], dtype=object)
        emb = np.vstack(
            [
                rng.normal(loc=0.0, scale=0.1, size=(30, d)),
                rng.normal(loc=4.0, scale=0.1, size=(30, d)),
            ]
        ).astype(np.float32)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat = root / "features.h5ad"
            with h5py.File(feat, "w") as handle:
                handle.create_dataset("X", data=emb)
                handle.create_dataset("obs/cell_id", data=obs.astype("S"))
            oof_path = root / "oof_predictions.npz"
            np.savez(oof_path, l2_labels=labels, l2_probs=oof)
            probs_path = root / "sample_l2_probs.npz"
            np.savez(
                probs_path,
                probs=probs,
                class_names=np.array(names, dtype=object),
                obs_names=obs,
            )
            sidecar = root / "confidence_he_cluster.npz"
            ensure_cluster_conformal_npz(
                probs_path,
                oof_path,
                feat,
                sidecar,
                names,
                n_clusters=2,
                pca_dim=4,
                seed=0,
            )
            with np.load(sidecar) as stored:
                clusters = np.asarray(stored["he_cluster"])
                conf = np.asarray(stored["confidence_B_cell"])
            self.assertEqual(len(np.unique(clusters)), 2)
            self.assertGreater(float(conf[:8].mean()), float(conf[30:].mean()))
            again = ensure_cluster_conformal_npz(
                probs_path, oof_path, feat, sidecar, names, n_clusters=2, pca_dim=4, seed=0
            )
            self.assertEqual(again, sidecar)

    def test_he_match_drops_a_score_that_is_normal_for_that_morphology(self):
        rng = np.random.default_rng(0)
        names = ["B_cell", "Tumor"]
        n_null, n_pos, n_test = 220, 30, 24

        def blob(center, n):
            return rng.normal(loc=center, scale=0.05, size=(n, 8)).astype(np.float32)

        calib_emb = np.vstack([blob(0.0, n_null + n_pos), blob(4.0, n_null + n_pos)])
        test_emb = np.vstack([blob(0.0, n_test), blob(4.0, n_test)])
        calib_y = np.array(
            [0] * n_null + [0] * n_pos + [1] * n_null + [1] * n_pos, dtype=np.int64
        )
        # Positives are the first class in blob 0 and the second class in blob 1.
        calib_y[: n_null + n_pos] = np.array([1] * n_null + [0] * n_pos)
        calib_p = np.zeros((len(calib_y), 2), dtype=np.float32)
        calib_p[:n_null, 0] = 0.02
        calib_p[n_null : n_null + n_pos, 0] = 0.85
        calib_p[n_null + n_pos : n_null + n_pos + n_null, 0] = rng.uniform(
            0.75, 0.98, size=n_null
        )
        calib_p[:, 1] = 1.0 - calib_p[:, 0]
        test_p = np.zeros((2 * n_test, 2), dtype=np.float32)
        test_p[:, 0] = 0.90
        out = correct_predictions_by_he(
            test_p,
            names,
            test_emb,
            calib_p,
            calib_y,
            calib_emb,
            n_clusters=2,
            pca_dim=4,
            seed=0,
            max_calib=0,
            correct_batch_effect=False,
        )
        kept = out["corrected_probs"][:, 0] > 0
        self.assertGreater(int(kept[:n_test].sum()), n_test // 2)
        self.assertEqual(int(kept[n_test:].sum()), 0)
        self.assertTrue(np.all(out["corrected_label"][n_test:] == "low_confidence"))


if __name__ == "__main__":
    unittest.main()


def test_ensure_conformal_npz_family_after_partial_sidecar(tmp_path):
    # 2026.11.18, a sidecar with B_cell / Plasma only, then a call that also asks for B_Plasma
    import numpy as np
    from prediction_confidence import ensure_conformal_npz

    rng = np.random.default_rng(0)
    names = ["Tumor", "B_cell", "Plasma", "DC"]
    def soft(n):
        z = rng.normal(0, 1, (n, len(names))); e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    oof = tmp_path / "oof.npz"
    np.savez(oof, l2_labels=rng.integers(0, 4, 3000), l2_probs=soft(3000).astype(np.float32),
             l2_class_names=np.array(names))
    probs = tmp_path / "s_l2_probs.npz"
    np.savez(probs, probs=soft(500).astype(np.float32), class_names=np.array(names))
    side = tmp_path / "s_confidence.npz"
    ensure_conformal_npz(probs, oof, side, ("B_cell", "Plasma"))
    ensure_conformal_npz(probs, oof, side, ("B_cell", "Plasma", "B_Plasma"))
    with np.load(side) as z:
        assert {"present_B_cell", "present_Plasma", "present_B_Plasma"} <= set(z.files)
        assert (z["score_B_Plasma"] >= z["score_B_cell"] - 1e-6).all()
