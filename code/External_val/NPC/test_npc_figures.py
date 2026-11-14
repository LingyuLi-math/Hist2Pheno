"""Checks for the NPC figure helpers moved out of the Pred_visual notebooks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from npc_figures import (
    group_fraction,
    keep_leftmost_foci,
    load_tls_xy,
    map_l2,
    nuclei_in_contours,
    spline_closed,
    tls_score_row,
)


class NpcFigureHelperTests(unittest.TestCase):
    def test_map_l2_and_fraction(self):
        labels = map_l2(["Tumor", "Plasma", "Endothelial", "B_cell"])
        self.assertEqual(labels.tolist(), ["tumor", "B", "other", "B"])
        frac = group_fraction(labels)
        self.assertAlmostEqual(float(frac.sum()), 1.0)
        self.assertAlmostEqual(float(frac["tumor"]), 0.25)
        self.assertAlmostEqual(float(frac["B"]), 0.5)
        self.assertEqual(list(frac.index[:2]), ["tumor", "normal"])

    def test_keep_leftmost_foci(self):
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0], [21.0, 0.0]])
        mask = keep_leftmost_foci(xy, pitch=1.0, n_keep=1, link=2.2)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_spline_and_contours(self):
        square = np.array([[0.0, 0.0], [3.0, 0.2], [3.1, 3.0], [0.2, 2.8]])
        smooth = spline_closed(square, n=40, s_scale=0.1)
        self.assertEqual(smooth.shape, (40, 2))
        self.assertTrue(np.isfinite(smooth).all())
        inside = nuclei_in_contours(
            np.array([[1.5, 1.5], [10.0, 10.0]]),
            [np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])],
        )
        self.assertEqual(inside.tolist(), [True, False])

    def test_tls_score_row_identical_scores(self):
        values = np.arange(20, dtype=float)
        df = pd.DataFrame(
            {
                "idx_TLS": values,
                "idx_TLS_spatial": values,
                "tls_candidate": values >= np.percentile(values, 95),
            }
        )
        row = tls_score_row(df, "slide")
        self.assertEqual(row["sample"], "slide")
        self.assertEqual(row["n"], 20)
        self.assertAlmostEqual(row["pearson"], 1.0)
        self.assertAlmostEqual(row["top5_jaccard"], 1.0)

    def test_load_tls_xy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"barcode": ["a", "b"]}).to_csv(root / "tls.csv", index=False)
            pd.DataFrame(
                [
                    ["a", 1, 0, 0, 10, 0],
                    ["b", 1, 0, 1, 10, 100],
                    ["c", 1, 0, 2, 10, 200],
                ]
            ).to_csv(root / "tissue_positions_list.csv", header=False, index=False)
            (root / "scalefactors_json.json").write_text(
                json.dumps({"spot_diameter_fullres": 40.0})
            )
            xy, barcodes, pitch, spot_r = load_tls_xy(root / "tls.csv", root)
        self.assertEqual(barcodes.tolist(), ["a", "b"])
        self.assertEqual(xy.tolist(), [[0.0, 10.0], [100.0, 10.0]])
        self.assertAlmostEqual(pitch, 100.0)
        self.assertAlmostEqual(spot_r, 20.0)


if __name__ == "__main__":
    unittest.main()
