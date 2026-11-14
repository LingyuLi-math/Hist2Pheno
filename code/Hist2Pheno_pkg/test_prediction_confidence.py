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
    confidence_from_p,
    conformal_from_oof,
    conformal_pvalues,
    gated_from_npz,
    highlight_labels,
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


if __name__ == "__main__":
    unittest.main()
