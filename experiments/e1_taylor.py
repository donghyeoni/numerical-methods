import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import md_table, out_dir, s4, vfull, write_csv, write_tables
from numerical_methods.errors import approx_error, true_relative_error
from numerical_methods.taylor import sin_taylor

DELTA = 0.005
MAX_ORDER = 10


def main():
    out = out_dir("e1_taylor")
    x = np.pi / 6 + DELTA
    true = np.sin(x)
    rows, previous = [], None
    for n in range(MAX_ORDER + 1):
        p = sin_taylor(n, x)
        rows.append({"order": n, "value": p,
                     "true_error": float(true_relative_error(true, p)),
                     "approx_error": approx_error(p, previous)})
        previous = p
    write_csv(os.path.join(out, "orders.csv"), rows)

    t = md_table(["order n", "P_n(x)", "ε_t (%)", "ε_a (%)"],
                 [[r["order"], vfull(r["value"]), s4(r["true_error"]),
                   s4(r["approx_error"])] for r in rows])
    write_tables(os.path.join(out, "tables.md"), [
        (f"E1-a sin(x) about π/6 at x = π/6 + {DELTA} "
         f"(sin(x) = {vfull(true)})", t)])

    grid = np.arange(-np.pi / 2, np.pi / 2, 0.001)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for n, c in ((1, "#E45756"), (2, "#54A24B"), (4, "#4C78A8")):
        ax.plot(grid, sin_taylor(n, grid), color=c, label=f"P{n}")
    ax.plot(grid, np.sin(grid), "k--", label="sin(x)")
    ax.axvline(np.pi / 6, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Taylor polynomials of sin(x) about π/6", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "polynomials.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, "tables.md"), encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
