from .errors import approx_error, true_relative_error


def _record(i, x, previous, root):
    return {"iteration": i, "x": float(x),
            "approx_error": approx_error(x, previous),
            "true_error": float(true_relative_error(root, x))}


def bisection(f, lower, upper, root, iterations):
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


def _iterate(step, x0, root, iterations):
    out, x, previous = [], x0, None
    for i in range(iterations):
        out.append(_record(i, x, previous, root))
        if i < iterations - 1:
            previous, x = x, step(x)
    return out


def fixed_point(g, x0, root, iterations):
    return _iterate(g, x0, root, iterations)


def newton_raphson(f, df, x0, root, iterations):
    return _iterate(lambda x: x - f(x) / df(x), x0, root, iterations)
