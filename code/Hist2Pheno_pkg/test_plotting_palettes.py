"""Focused smoke tests for the dataset-aware palette registry."""

import inspect
import pickle
import unittest

from plot import (
    _roc_colors_for_class_names,
    build_xenium_spatial_color_overrides,
    plot_celltype_spatial_distribution,
    plot_level1_roc_from_level1_head,
    plot_level1_roc_from_level2_scores,
    plot_multiclass_roc_curves,
)
from plotting_palettes import (
    Cell_Type_COLORS,
    CNICHE_COLORS,
    GRAY_RGBA,
    LINEAGE_COLORS,
    SUBLINEAGE_COLORS,
    TNICHE_COLORS,
    _LEVEL1_SPATIAL_DISTINCT_RGBA,
    build_ncrt_spatial_color_map,
    get_palette,
    ncrt_roc_color_overrides,
    normalize_dataset_id,
    normalize_tier,
    resolve_palette,
    resolve_roc_colors,
    resolve_tier,
    clinical_columns_for_pan_organ,
    clinical_group_order,
)
from uni_label_cv_helpers import (
    STARDIST_AUROC_TIER_SPECS,
    _extra_tier_color_scheme,
)


class PlottingPaletteTests(unittest.TestCase):
    def test_codex_escc_is_canonical_dataset(self):
        for dataset in ("codex_escc", "escc", "ncrt"):
            self.assertEqual(normalize_dataset_id(dataset), "codex_escc")
        for scheme in ("codex_escc", "ncrt"):
            self.assertEqual(normalize_dataset_id(scheme=scheme), "codex_escc")

    def test_ncrt_alias_is_palette_and_roc_identical(self):
        labels = ["CD8+T", "Immune", "unknown"]
        canonical = resolve_palette(
            labels, scheme="codex_escc", tier="celltype_level1",
        )
        legacy = resolve_palette(labels, scheme="ncrt", tier="celltype_level1")
        self.assertEqual(canonical, legacy)
        self.assertEqual(pickle.dumps(canonical), pickle.dumps(legacy))

        canonical_roc = resolve_roc_colors(
            labels, scheme="codex_escc", tier="celltype_level1",
        )
        legacy_roc = resolve_roc_colors(
            labels, scheme="ncrt", tier="celltype_level1",
        )
        self.assertEqual(canonical_roc, legacy_roc)
        self.assertEqual(pickle.dumps(canonical_roc), pickle.dumps(legacy_roc))

    ## 2026.08.13, add test_codex_escc_five_tier_palettes_and_ncrt_alias for CODEX ESCC dataset
    def test_codex_escc_five_tier_palettes_and_ncrt_alias(self):
        tier_labels = {
            "celltype": ["Treg", "Tumor"],
            "celltype_level1": ["CD8+T", "B"],
            "celltype_level12": ["B_other", "Macrophage"],
            "celltype_level0": ["Immune", "Stromal"],
            "celltype_level01": ["B", "Myeloid"],
        }
        for tier, labels in tier_labels.items():
            with self.subTest(tier=tier):
                canonical = resolve_palette(
                    labels, scheme="codex_escc", tier=tier,
                )
                legacy = resolve_palette(labels, scheme="ncrt", tier=tier)
                self.assertEqual(canonical, legacy)
                self.assertEqual(
                    canonical,
                    {label: get_palette("codex_escc", tier)[label] for label in labels},
                )

    def test_hcc_coarse_colors_are_exact(self):
        palette = get_palette("codex_hcc", "coarse", rgba=False)
        self.assertEqual(
            palette,
            {
                "Stromal": "#98df8a",
                "Immune": "#d62728",
                "Endothelial": "#2ca02c",
                "Epithelial": "#1f77b4",
            },
        )

    def test_xenium_lineage_colors_are_exact(self):
        palette = get_palette("xenium_lung", "lineage", rgba=False)
        self.assertEqual(palette["Epithelial"], "#8103fb")
        self.assertEqual(palette["Immune"], "#2adddc")
        self.assertEqual(palette["Endothelial"], "#d4df8a")
        self.assertEqual(palette["Mesenchymal"], "#f80505")

    def test_xenium_five_tier_canonical_schemes(self):
        exact_palettes = {
            "fine": Cell_Type_COLORS,
            "intermediate": SUBLINEAGE_COLORS,
            "coarse": LINEAGE_COLORS,
            "cniche": CNICHE_COLORS,
            "tniche": TNICHE_COLORS,
        }
        for tier, expected in exact_palettes.items():
            self.assertEqual(
                get_palette("xenium_lung", tier, rgba=False), expected
            )

        cases = {
            "xenium_lung_fine": ("AT1", "#1f77b4", "fine"),
            "xenium_lung_intermediate": (
                "Alveolar", SUBLINEAGE_COLORS["Alveolar"], "intermediate"
            ),
            "xenium_lung_coarse": ("Immune", "#2adddc", "coarse"),
            "xenium_lung_CNiche": ("C1", CNICHE_COLORS["C1"], "cniche"),
            "xenium_lung_TNiche": ("T1", TNICHE_COLORS["T1"], "tniche"),
        }
        for scheme, (label, expected_hex, tier) in cases.items():
            with self.subTest(scheme=scheme):
                self.assertEqual(
                    normalize_dataset_id(scheme=scheme), "xenium_lung"
                )
                self.assertEqual(resolve_tier([label], scheme=scheme), tier)
                self.assertEqual(
                    resolve_palette([label], scheme=scheme)[label],
                    resolve_palette(
                        [label], dataset="xenium_lung", tier=tier
                    )[label],
                )
                self.assertEqual(
                    get_palette("xenium_lung", tier, rgba=False)[label],
                    expected_hex,
                )
                expected_rgba = resolve_palette([label], scheme=scheme)[label]
                self.assertEqual(
                    build_xenium_spatial_color_overrides(
                        [label], spatial_color_scheme=scheme
                    )[label],
                    expected_rgba,
                )
                self.assertEqual(
                    _roc_colors_for_class_names(
                        [label], roc_color_scheme=scheme
                    )[0],
                    expected_rgba,
                )

    def test_ncrt_legacy_builder_is_compatible(self):
        palette = build_ncrt_spatial_color_map("celltype_level1")
        self.assertEqual(palette["CD8+T"], _LEVEL1_SPATIAL_DISTINCT_RGBA["CD8+T"])
        self.assertEqual(palette["Immune"], (82 / 255, 182 / 255, 174 / 255, 1.0))
        self.assertEqual(
            ncrt_roc_color_overrides(["CD8+T"], "celltype_level1"),
            resolve_palette(
                ["CD8+T"], dataset="codex_escc", tier="celltype_level1",
            ),
        )

    def test_plot_defaults_use_codex_escc_scheme(self):
        self.assertEqual(
            inspect.signature(_roc_colors_for_class_names)
            .parameters["roc_color_scheme"].default,
            "codex_escc",
        )
        self.assertEqual(
            inspect.signature(plot_celltype_spatial_distribution)
            .parameters["spatial_color_scheme"].default,
            "codex_escc",
        )
        self.assertEqual(
            inspect.signature(plot_multiclass_roc_curves)
            .parameters["roc_color_scheme"].default,
            "codex_escc",
        )
        for helper in (
            plot_level1_roc_from_level2_scores,
            plot_level1_roc_from_level1_head,
        ):
            self.assertEqual(
                inspect.signature(helper).parameters["roc_color_scheme"].default,
                "codex_escc",
            )
        labels = ["CD8+T", "Immune"]
        self.assertEqual(
            _roc_colors_for_class_names(
                labels, roc_color_scheme="codex_escc",
                ncrt_color_tier="celltype_level1",
            ),
            _roc_colors_for_class_names(
                labels, roc_color_scheme="ncrt",
                ncrt_color_tier="celltype_level1",
            ),
        )
        
    ## 2026.08.13, add test_l1_roc_explicit_dataset_scheme_overrides_resolve for HCC dataset
    def test_l1_roc_explicit_dataset_scheme_overrides_resolve(self):
        cases = {
            "xenium_lung_coarse": ["Immune", "Epithelial"],
            "codex_hcc_coarse": ["Immune", "Stromal"],
        }
        for scheme, labels in cases.items():
            with self.subTest(scheme=scheme):
                self.assertEqual(
                    _roc_colors_for_class_names(
                        labels, roc_color_scheme=scheme,
                    ),
                    resolve_roc_colors(labels, scheme=scheme),
                )
    ###############################################################

    def test_hcc_model_level1_resolves_to_coarse(self):
        self.assertEqual(
            normalize_tier("celltype_level1", dataset="codex_hcc"),
            "coarse",
        )
        resolved = resolve_palette(
            ["Immune"], dataset="codex_hcc", tier="celltype_level1",
        )
        self.assertEqual(resolved["Immune"], (214 / 255, 39 / 255, 40 / 255, 1.0))

    def test_unknown_non_fine_label_is_gray(self):
        resolved = resolve_palette(
            ["not-a-lineage"], dataset="xenium_lung", tier="lineage",
        )
        self.assertEqual(resolved["not-a-lineage"], GRAY_RGBA)

    def test_explicit_override_has_precedence(self):
        resolved = resolve_palette(
            ["Immune"],
            scheme="codex_hcc_coarse",
            color_overrides={"Immune": "#010203"},
        )
        self.assertEqual(resolved["Immune"], (1 / 255, 2 / 255, 3 / 255, 1.0))

    def test_old_xenium_scheme_aliases(self):
        lineage = resolve_palette(["Immune"], scheme="xenium_lineage")
        fine = resolve_palette(["AT1"], scheme="xenium_ct")
        auto = resolve_palette(["Immune"], scheme="xenium")
        self.assertEqual(lineage["Immune"], (42 / 255, 221 / 255, 220 / 255, 1.0))
        self.assertEqual(fine["AT1"], (31 / 255, 119 / 255, 180 / 255, 1.0))
        self.assertEqual(auto["Immune"], lineage["Immune"])
        self.assertEqual(
            fine,
            resolve_palette(["AT1"], scheme="xenium_lung_fine"),
        )
        self.assertEqual(
            lineage,
            resolve_palette(["Immune"], scheme="xenium_lung_coarse"),
        )

    def test_extra_tier_scheme_selection(self):
        expected = {
            "l12": "xenium_lung_intermediate",
            "l3": "xenium_lung_CNiche",
            "l4": "xenium_lung_TNiche",
        }
        for tier, scheme in expected.items():
            self.assertEqual(_extra_tier_color_scheme(None, tier), scheme)
            self.assertEqual(
                STARDIST_AUROC_TIER_SPECS[tier]["roc_color_scheme"], scheme
            )
        self.assertEqual(
            _extra_tier_color_scheme("codex_hcc_intermediate", "l3"),
            "codex_hcc_intermediate",
        )
        self.assertEqual(
            _extra_tier_color_scheme({"l3": "custom"}, "l3"), "custom"
        )

    def test_pan_organ_routes_palettes_and_roc(self):
        # Palette routing: pan_organ overrides the default dataset when scheme is omitted.
        labels = ["Immune", "Stromal"]
        via_pan = resolve_palette(labels, pan_organ="codex_hcc", tier="coarse")
        via_scheme = resolve_palette(labels, scheme="codex_hcc_coarse")
        self.assertEqual(via_pan, via_scheme)

        # ROC routing: pan_organ + color tier chooses the correct coarse scheme.
        roc = _roc_colors_for_class_names(
            labels,
            pan_organ="codex_hcc",
            color_tier="celltype_level1",
        )
        self.assertEqual(roc, resolve_roc_colors(labels, scheme="codex_hcc_coarse"))

        # Extra tiers: default mapping becomes organ-specific when pan_organ is provided.
        self.assertEqual(
            _extra_tier_color_scheme(None, "l3", pan_organ="codex_hcc"),
            "codex_hcc_intermediate",
        )

    ## 2026.08.14, add test_clinical_group_order_by_pan_organ for CODEX HCC dataset
    def test_clinical_group_order_by_pan_organ(self):
        lung = clinical_group_order("xenium_lung")
        self.assertEqual(lung["Status"], ("Control", "Disease"))
        self.assertEqual(
            lung["Sample_Affect_Pairing"],
            ("Unaffected", "Less_Affected", "More_Affected"),
        )
        self.assertNotIn("Response", lung)

        hcc = clinical_group_order("codex_hcc")
        self.assertEqual(hcc["Response"], ("Responder", "Non_Responder"))
        self.assertEqual(hcc["diagnosis"], ("Pre", "Post"))
        self.assertNotIn("Status", hcc)
        self.assertEqual(clinical_group_order("hcc"), hcc)

        merged = clinical_group_order(
            "codex_hcc", extra={"treatment": ("Atezo", "Atezo+Bev")}
        )
        self.assertEqual(merged["treatment"], ("Atezo", "Atezo+Bev"))
        self.assertEqual(
            clinical_columns_for_pan_organ("xenium_lung"),
            ("Status", "Sample_Affect_Pairing"),
        )
        self.assertEqual(
            clinical_columns_for_pan_organ("codex_hcc"),
            ("Response", "diagnosis", "treatment"),
        )


if __name__ == "__main__":
    unittest.main()
