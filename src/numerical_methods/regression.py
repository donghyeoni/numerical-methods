import numpy as np


def empirical_cdf(sample, x):
    sample = np.asarray(sample)
    return np.count_nonzero(sample <= x) / len(sample)


def _lstsq(Z, t):
    return np.linalg.solve(Z.T @ Z, Z.T @ t)


def _powers(x, order):
    return np.vander(x, order + 1, increasing=True)


def fit_linear(x, y):
    return _lstsq(_powers(x, 1), y)


def fit_polynomial(x, y, order):
    return _lstsq(_powers(x, order), y)


def eval_polynomial(a, x):
    return _powers(np.atleast_1d(x), len(a) - 1) @ a


def fit_exponential(x, y, order):
    m = y > 0
    a = _lstsq(_powers(x[m], order), np.log(y[m]))
    a[0] = np.exp(a[0])
    return a


def eval_exponential(a, x):
    x = np.atleast_1d(x)
    return a[0] * np.exp(_powers(x, len(a) - 1)[:, 1:] @ a[1:])


def fit_power(x, y):
    m = (x > 0) & (y > 0)
    a = _lstsq(_powers(np.log10(x[m]), 1), np.log10(y[m]))
    a[0] = 10 ** a[0]
    return a


def eval_power(a, x):
    return a[0] * np.asarray(x, dtype=float) ** a[1]


def fit_sigmoid(x, y):
    m = (y > 0) & (y < 1)
    return _lstsq(_powers(x[m], 1), np.log(1 / y[m] - 1))


def eval_sigmoid(a, x):
    return 1 / (1 + np.exp(a[0] + a[1] * np.asarray(x, dtype=float)))


def r_squared(y, y_fit):
    y = np.asarray(y, dtype=float)
    s_t = np.sum((y - y.mean()) ** 2)
    s_r = np.sum((y - np.asarray(y_fit, dtype=float)) ** 2)
    return float((s_t - s_r) / s_t)
