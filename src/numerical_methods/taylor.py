"""Taylor polynomials; sin(x) about x0 = pi/6.

The derivatives of sin at pi/6 repeat with period 4:
``1/2, sqrt(3)/2, -1/2, -sqrt(3)/2``.
"""

from __future__ import annotations

import math

import numpy as np

SIN_DERIVS_AT_PI_6 = (0.5, math.sqrt(3) / 2, -0.5, -math.sqrt(3) / 2)


def taylor_poly(n, x, x0, deriv_at_x0):
    """Degree-``n`` Taylor polynomial about ``x0``,
    ``sum_{i=0}^{n} f^(i)(x0) (x - x0)^i / i!``.

    ``deriv_at_x0(i)`` returns the i-th derivative at ``x0``.
    """
    h = x - x0
    return sum(deriv_at_x0(i) * h ** i / math.factorial(i)
               for i in range(n + 1))


def sin_taylor(n, x):
    """Degree-``n`` Taylor polynomial of ``sin`` about ``pi/6``."""
    return taylor_poly(n, x, np.pi / 6, lambda i: SIN_DERIVS_AT_PI_6[i % 4])
