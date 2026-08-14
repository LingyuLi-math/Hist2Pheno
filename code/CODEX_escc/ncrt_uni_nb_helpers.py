## 2026.08.13, add ncrt_uni_nb_helpers for NCRT dataset, learn from HCC and Lung dataset
## 2026.08.13, retain ncrt_uni_nb_helpers as a compatibility shim

"""Deprecated compatibility shim for :mod:`codex_escc_uni_nb_helpers`.

``NCRT`` remains the cohort name, but new code should import the canonical
CODEX ESCC helper module. All public exports and module attributes are forwarded.
"""

from __future__ import annotations

import codex_escc_uni_nb_helpers as _canonical
from codex_escc_uni_nb_helpers import *  # noqa: F401,F403

__all__ = getattr(
    _canonical,
    "__all__",
    tuple(name for name in vars(_canonical) if not name.startswith("_")),
)


def __getattr__(name):
    """Forward private and future attributes to the canonical module."""
    return getattr(_canonical, name)


def __dir__():
    """Expose the canonical module namespace during introspection."""
    return sorted(set(globals()) | set(dir(_canonical)))
