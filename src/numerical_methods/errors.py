import numpy as np


def true_relative_error(true_value, approx_value):
    return np.abs((true_value - approx_value) / true_value) * 100


def approx_error(present, previous):
    if previous is None or present == 0:
        return None
    return float(abs((present - previous) / present) * 100)
