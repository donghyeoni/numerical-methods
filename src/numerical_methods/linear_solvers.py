from fractions import Fraction

import numpy as np


def build_system(delta):
    A = np.array([[delta, 3, 2, 1],
                  [4, delta, 7, 5],
                  [8, 2, delta, 2],
                  [0, 1, 2, delta]], dtype=float)
    b = np.array([-3, 2, -2, -5], dtype=float)
    return A, b


def exact_solution(A, b):
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


@np.errstate(divide="ignore", invalid="ignore", over="ignore")
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
    return _eliminate(A, b, False, round_digits)


def gauss_pivoting(A, b, round_digits=None):
    return _eliminate(A, b, True, round_digits)


@np.errstate(divide="ignore", invalid="ignore", over="ignore")
def gauss_seidel(A, b, *, relaxation=1.0, tol=1.0, max_iter=1000):
    n = len(b)
    C = A / np.diag(A)[:, None]
    np.fill_diagonal(C, 0.0)
    d = b / np.diag(A)
    x = np.zeros(n)
    for sweep in range(1, max_iter + 1):
        err = np.zeros(n)
        for i in range(n):
            old = x[i]
            x[i] = relaxation * (d[i] - C[i] @ x) + (1 - relaxation) * old
            err[i] = abs((x[i] - old) / x[i]) * 100
        if np.max(err) < tol:
            return x, sweep, True
    return x, max_iter, False


def iteration_spectral_radius(A, relaxation=1.0):
    D = np.diag(np.diag(A))
    lower = D + relaxation * np.tril(A, -1)
    rhs = (1 - relaxation) * D - relaxation * np.triu(A, 1)
    T = np.zeros_like(rhs)
    for i in range(len(A)):
        T[i] = (rhs[i] - lower[i, :i] @ T[:i]) / lower[i, i]
    return float(np.max(np.abs(np.linalg.eigvals(T))))
