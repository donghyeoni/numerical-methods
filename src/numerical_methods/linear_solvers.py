"""Direct and iterative solvers for the 4x4 test system ``A x = b``.

``round_digits`` of the elimination solvers simulates arithmetic with ``k``
decimal places: the elimination multipliers, the updated matrix entries and
the back-substitution results are rounded to ``k`` decimals with
``numpy.round``. The products and inner sums between them and the input
system are not rounded. ``None`` means full float64 arithmetic.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def build_system(delta):
    """The test system: ``delta`` on the diagonal of ``A``."""
    A = np.array([[delta, 3, 2, 1],
                  [4, delta, 7, 5],
                  [8, 2, delta, 2],
                  [0, 1, 2, delta]], dtype=float)
    b = np.array([-3, 2, -2, -5], dtype=float)
    return A, b


def exact_solution(A, b):
    """Exact solution of the float system ``A x = b``, by Gaussian
    elimination with partial pivoting in rational arithmetic (every float
    entry converted exactly with ``Fraction``), returned as float64."""
    n = len(b)
    M = [[Fraction(float(v)) for v in row] + [Fraction(float(b[i]))]
         for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for j in range(i + 1, n):
            m = M[j][i] / M[i][i]
            M[j] = [a - m * c for a, c in zip(M[j], M[i])]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return np.array([float(v) for v in x])


def _r(v, k):
    return v if k is None else np.round(v, k)


def _eliminate(A, b, pivoting, k):
    n = len(b)
    M = np.hstack([A, b[:, None]]).astype(float)
    for i in range(n - 1):
        if pivoting:
            p = i + int(np.argmax(np.abs(M[i:, i])))
            M[[i, p]] = M[[p, i]]
        for j in range(i + 1, n):
            m = _r(M[j, i] / M[i, i], k)
            M[j, i:] = _r(M[j, i:] - m * M[i, i:], k)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = _r((M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i], k)
    return x


def gauss_naive(A, b, round_digits=None):
    """Gaussian elimination without pivoting, then back substitution."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _eliminate(A, b, False, round_digits)


def gauss_pivoting(A, b, round_digits=None):
    """Gaussian elimination with partial (row) pivoting."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _eliminate(A, b, True, round_digits)


def gauss_seidel(A, b, *, relaxation=1.0, tol=1.0, max_iter=1000):
    """Gauss-Seidel iteration from ``x = 0``.

    Each component is updated in turn with the newest values,
    ``x_i <- (b_i - sum_{j != i} a_ij x_j) / a_ii``, then relaxed,
    ``x_i <- w * x_i + (1 - w) * x_i_old`` (``w = relaxation``; 1 = none).
    Stops when the largest approximate error ``|(x_i - x_i_old) / x_i| * 100``
    of a sweep is below ``tol`` percent, or after ``max_iter`` sweeps.

    Returns ``(x, sweeps, converged)``; ``converged`` is True if the stop
    was by ``tol``.
    """
    n = len(b)
    C = A / np.diag(A)[:, None]
    np.fill_diagonal(C, 0.0)
    d = b / np.diag(A)
    x = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for sweep in range(1, max_iter + 1):
            err = np.zeros(n)
            for i in range(n):
                old = x[i]
                new = d[i] - C[i] @ x
                x[i] = relaxation * new + (1 - relaxation) * old
                err[i] = abs((x[i] - old) / x[i]) * 100
            if np.max(err) < tol:
                return x, sweep, True
    return x, max_iter, False


def iteration_spectral_radius(A, relaxation=1.0):
    """Spectral radius of the Gauss-Seidel iteration matrix with relaxation
    ``w``: ``T = (D + wL)^-1 ((1 - w) D - wU)``. The iteration converges for
    every start if and only if the radius is below 1. The triangular solve
    is done by forward substitution, so a tiny diagonal is not rejected."""
    D = np.diag(np.diag(A))
    L, U = np.tril(A, -1), np.triu(A, 1)
    lower = D + relaxation * L
    rhs = (1 - relaxation) * D - relaxation * U
    n = len(A)
    T = np.zeros_like(rhs)
    for i in range(n):
        T[i] = (rhs[i] - lower[i, :i] @ T[:i]) / lower[i, i]
    return float(np.max(np.abs(np.linalg.eigvals(T))))
