"""Error measures in percent (Chapra & Canale's definitions)."""

from __future__ import annotations

import numpy as np


def true_relative_error(true_value, approx_value):
    """True percent relative error,
    ``|(true - approx) / true| * 100``. Works on scalars and arrays."""
    return np.abs((true_value - approx_value) / true_value) * 100


def approx_error(present, previous):
    """Approximate relative error in percent,
    ``|(present - previous) / present| * 100``.

    Returns None when ``previous`` is None (no earlier estimate).
    """
    if previous is None:
        return None
    return float(abs((present - previous) / present) * 100)
