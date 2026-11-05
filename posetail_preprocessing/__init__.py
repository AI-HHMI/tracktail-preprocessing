"""Deprecated alias for :mod:`tracktail_preprocessing`.

The package was renamed to ``tracktail_preprocessing``. This shim keeps old
imports such as ``from posetail_preprocessing.datasets import Rat7MDataset``
working; it will be removed in a future release.
"""

import importlib
import sys
import warnings

warnings.warn(
    "posetail_preprocessing is deprecated; import tracktail_preprocessing instead",
    DeprecationWarning, stacklevel=2)

import tracktail_preprocessing as _new
from tracktail_preprocessing import *  # noqa: F401,F403

for _sub in ("datasets", "utils", "scripts"):
    try:
        _mod = importlib.import_module(f"tracktail_preprocessing.{_sub}")
    except ImportError:  # optional heavy deps missing
        continue
    sys.modules[f"{__name__}.{_sub}"] = _mod
    globals()[_sub] = _mod
