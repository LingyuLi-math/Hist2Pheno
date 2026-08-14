"""Deprecated compatibility import for the shared UNI-label CV helpers.

New code should import public helpers directly from ``uni_label_cv_helpers``.
This module intentionally emits no runtime warning so existing notebooks stay
quiet.
"""

from uni_label_cv_helpers import *  # noqa: F401,F403
from uni_label_cv_helpers import _auroc_csv_path_from_metrics  # noqa: F401
