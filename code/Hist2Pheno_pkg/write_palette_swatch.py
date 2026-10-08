#!/usr/bin/env python3
"""Write the pan-cancer L1 / L12 / L2 color swatch used by the plots.

    python code/Hist2Pheno_pkg/write_palette_swatch.py

The PNG is ``Hist2Pheno_pkg/figures/pancancer_L1_L12_L2_swatch.png``.
Colors come from ``plotting_palettes.palette_for``, the same map as the
spatial plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from Hist2Pheno_pkg.plotting_palettes import write_pancancer_swatch


def main() -> int:
    path = write_pancancer_swatch()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
