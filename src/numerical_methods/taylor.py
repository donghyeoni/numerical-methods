import math

import numpy as np

SIN_DERIVS_AT_PI_6 = (0.5, math.sqrt(3) / 2, -0.5, -math.sqrt(3) / 2)


def taylor_poly(n, x, x0, deriv_at_x0):
    h = x - x0
    return sum(deriv_at_x0(i) * h ** i / math.factorial(i)
               for i in range(n + 1))


def sin_taylor(n, x):
    return taylor_poly(n, x, np.pi / 6, lambda i: SIN_DERIVS_AT_PI_6[i % 4])
