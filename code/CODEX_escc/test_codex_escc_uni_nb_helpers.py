## 2026.08.13, add test_codex_escc_uni_nb_helpers for CODEX ESCC dataset

"""Import compatibility tests for the CODEX ESCC notebook helper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ESCC_DIR = Path(__file__).resolve().parent
if str(ESCC_DIR) not in sys.path:
    sys.path.insert(0, str(ESCC_DIR))

import codex_escc_uni_nb_helpers as canonical
import ncrt_uni_nb_helpers as legacy


class CodexEsccHelperCompatibilityTests(unittest.TestCase):
    def test_public_function_is_reexported(self):
        self.assertIs(legacy.plot_he_roc_extra, canonical.plot_he_roc_extra)

    def test_private_helper_is_forwarded_and_discoverable(self):
        name = "_ncrt_tier_tag_from_title"
        self.assertIs(getattr(legacy, name), getattr(canonical, name))
        self.assertIn(name, dir(legacy))


if __name__ == "__main__":
    unittest.main()
