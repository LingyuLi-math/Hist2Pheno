"""Focused tests for dataset-neutral plotting utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plotting_utils import (
    celltype_counts_from_df,
    plot_celltype_count_bars,
    plot_celltype_proportions_stacked,
    plot_final_ct_by_lineage,
    pooled_celltype_counts,
)


class PlottingUtilsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_celltype_counts_excludes_unknown(self):
        frame = pd.DataFrame({"label": ["B", "A", "B", "Unknown"]})
        counts = celltype_counts_from_df(frame, celltype_col="label")
        self.assertEqual(counts.to_dict(), {"B": 2, "A": 1})

    def test_pooled_counts_sum_samples(self):
        frames = {
            "s1": pd.DataFrame({"celltype": ["A", "B", "Unknown"]}),
            "s2": pd.DataFrame({"celltype": ["A", "A", "C"]}),
        }
        counts = pooled_celltype_counts(frames)
        self.assertEqual(counts.to_dict(), {"A": 3, "B": 1, "C": 1})

    def test_count_bar_returns_figure_and_saves(self):
        counts = pd.Series({"A": 5, "B": 2})
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "counts.png"
            fig = plot_celltype_count_bars(
                counts,
                celltype_colors={"A": "#ff0000", "B": "#0000ff"},
                save_path=output,
                show=False,
            )
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(output.is_file())
            self.assertEqual(len(fig.axes[0].patches), 2)

    def test_stacked_proportions_with_generic_palette(self):
        counts = pd.DataFrame(
            {"sample-a": [3, 1], "sample-b": [1, 3]},
            index=["Type A", "Type B"],
        )
        fig = plot_celltype_proportions_stacked(
            counts,
            celltype_colors={"Type A": "#123456", "Type B": "#abcdef"},
            show_shading=False,
            show=False,
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.axes[0].get_ylim(), (0.0, 100.0))
        self.assertEqual(len(fig.axes[0].patches), 4)

    def test_final_ct_by_lineage_returns_figure_and_saves(self):
        frame = pd.DataFrame(
            {
                "lineage": ["Immune", "Immune", "Epithelial", "Epithelial"],
                "cell_type": ["T", "B", "AT1", "AT1"],
            }
        )
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "lineage.png"
            fig = plot_final_ct_by_lineage(
                frame,
                lineage_col="lineage",
                ct_col="cell_type",
                lineage_order=["Immune", "Epithelial"],
                lineage_cmaps={"Immune": "Blues", "Epithelial": "Greens"},
                save_path=output,
                show=False,
            )
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(output.is_file())
            self.assertEqual(
                [axis.get_title() for axis in fig.axes],
                ["Immune", "Epithelial"],
            )


if __name__ == "__main__":
    unittest.main()
