"""Bracketing and open root-finding methods.

Each method runs a fixed number of iterations and returns one record per
iterate: ``{"iteration", "x", "approx_error", "true_error"}`` with errors in
percent (``approx_error`` is None for the first record).
"""

from __future__ import annotations

from .errors import approx_error, true_relative_error


def _record(i, x, previous, root):
    return {"iteration": i, "x": float(x),
            "approx_error": approx_error(x, previous),
            "true_error": float(true_relative_error(root, x))}


def bisection(f, lower, upper, root, iterations):
    """Bisection on ``[lower, upper]``. Records the midpoints of iterations
    1..``iterations``. Keeps the half whose endpoints change sign; if
    ``f(lower) * f(mid) > 0`` the root is in ``[mid, upper]``."""
    out, previous = [], None
    for i in range(1, iterations + 1):
        mid = (lower + upper) / 2
        out.append(_record(i, mid, previous, root))
        if f(lower) * f(mid) > 0:
            lower = mid
        else:
            upper = mid
        previous = mid
    return out


def fixed_point(g, x0, root, iterations):
    """Fixed-point iteration ``x_{i+1} = g(x_i)``. Records ``x_0`` to
    ``x_{iterations-1}`` (iteration 0 is the starting value)."""
    out, x, previous = [], x0, None
    for i in range(iterations):
        out.append(_record(i, x, previous, root))
        previous, x = x, g(x)
    return out


def newton_raphson(f, df, x0, root, iterations):
    """Newton-Raphson ``x_{i+1} = x_i - f(x_i) / f'(x_i)``. Records ``x_0`` to
    ``x_{iterations-1}`` (iteration 0 is the starting value)."""
    out, x, previous = [], x0, None
    for i in range(iterations):
        out.append(_record(i, x, previous, root))
        previous, x = x, x - f(x) / df(x)
    return out
