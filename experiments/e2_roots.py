import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import md_table, out_dir, s4, vfull, write_csv, write_tables
from numerical_methods.roots import bisection, fixed_point, newton_raphson

ROOT = 3.0
ITERATIONS = 15
NEWTON_C = 0.045


def f(x):
    return (x - 1) * (x - 2) * (x - 3)


def df(x):
    return 3 * x ** 2 - 12 * x + 11


def g(x):
    return 6 / 11 + 6 / 11 * x ** 2 - 1 / 11 * x ** 3


METHODS = [
    ("bisection", "bisection on [2.5, 4]",
     lambda: bisection(f, 2.5, 4.0, ROOT, ITERATIONS)),
    ("fixed-point", "fixed-point iteration, g(x) = 6/11 + 6/11·x² − 1/11·x³, "
     "x0 = 2.5", lambda: fixed_point(g, 2.5, ROOT, ITERATIONS)),
    ("newton", "Newton-Raphson, x0 = 4",
     lambda: newton_raphson(f, df, 4.0, ROOT, ITERATIONS)),
]


def main():
    out = out_dir("e2_roots")
    rows, sections, curves = [], [], {}
    for tag, (key, title, run) in zip("abc", METHODS):
        hist = run()
        rows += [{"method": key, **h} for h in hist]
        curves[key] = hist
        headers = ["iteration", "x", "ε_a (%)", "ε_t (%)"]
        cells = [[h["iteration"], vfull(h["x"]), s4(h["approx_error"]),
                  s4(h["true_error"])] for h in hist]
        prev = [None] + [h["true_error"] for h in hist[:-1]]
        if key == "fixed-point":
            headers.append("ε_t(i) / ε_t(i−1)")
            for c, h, p in zip(cells, hist, prev):
                c.append(s4(None if p is None else h["true_error"] / p))
        if key == "newton":
            headers.append(f"{NEWTON_C}·ε_t(i−1)²")
            for c, p in zip(cells, prev):
                c.append(s4(None if p is None else NEWTON_C * p ** 2))
        sections.append((f"E2-{tag} {title}", md_table(headers, cells)))
    write_csv(os.path.join(out, "iterations.csv"), rows)
    write_tables(os.path.join(out, "tables.md"), sections)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for key, color in (("bisection", "#4C78A8"), ("fixed-point", "#F58518"),
                       ("newton", "#54A24B")):
        pts = [(h["iteration"], h["true_error"]) for h in curves[key]
               if h["true_error"] > 0]
        ax.plot(*zip(*pts), marker="o", ms=3, color=color, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("true percent relative error ε_t (%)")
    ax.set_title("Root finding for f(x) = (x−1)(x−2)(x−3), root 3",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "errors.png"), dpi=150)
    plt.close(fig)
    print(open(os.path.join(out, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
