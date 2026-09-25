import numpy as np


def empirical_cdf(sample, x):
    return np.count_nonzero(sample <= x) / len(sample)


def _lstsq(Z, t):
    if Z.shape[0] < Z.shape[1]:
        raise np.linalg.LinAlgError("fewer points than coefficients")
    return np.linalg.solve(Z.T @ Z, Z.T @ t)


def _powers(x, order):
    return np.vander(x, order + 1, increasing=True)


def fit_linear(x, y):
    return _lstsq(_powers(x, 1), y)


def fit_polynomial(x, y, order):
    return _lstsq(_powers(x, order), y)


def eval_polynomial(a, x):
    return _powers(x, len(a) - 1) @ a


def fit_exponential(x, y, order):
    m = y > 0
    a = _lstsq(_powers(x[m], order), np.log(y[m]))
    a[0] = np.exp(a[0])
    return a


def eval_exponential(a, x):
    return a[0] * np.exp(_powers(x, len(a) - 1)[:, 1:] @ a[1:])


def fit_power(x, y):
    m = (x > 0) & (y > 0)
    a = _lstsq(_powers(np.log10(x[m]), 1), np.log10(y[m]))
    a[0] = 10 ** a[0]
    return a


def eval_power(a, x):
    return a[0] * x ** a[1]


def fit_sigmoid(x, y):
    m = (y > 0) & (y < 1)
    return _lstsq(_powers(x[m], 1), np.log(1 / y[m] - 1))


def eval_sigmoid(a, x):
    return 1 / (1 + np.exp(a[0] + a[1] * x))


def r_squared(y, y_fit):
    s_t = np.sum((y - y.mean()) ** 2)
    if s_t == 0:
        return float("nan")
    s_r = np.sum((y - y_fit) ** 2)
    return float((s_t - s_r) / s_t)
