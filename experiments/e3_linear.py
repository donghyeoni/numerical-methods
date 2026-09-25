import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import md_table, out_dir, s4, vfull, write_csv, write_tables
from numerical_methods.errors import true_relative_error
from numerical_methods.linear_solvers import (
    build_system, exact_solution, gauss_naive, gauss_pivoting, gauss_seidel,
    iteration_spectral_radius)

NS = tuple(range(1, 17))
ROUND_DIGITS = (1, 2, 3, 4, 6, 8)
ROUND_NS = (1, 2, 4, 8)
RELAX = 0.9


def max_error(x_exact, x):
    e = true_relative_error(x_exact, x)
    return float(np.nan) if np.any(np.isnan(e)) else float(np.max(e))


def main():
    out = out_dir("e3_linear")
    rows, solutions = [], {}
    for n in NS:
        A, b = build_system(10.0 ** -n)
        x_ex = exact_solution(A, b)
        with np.errstate(all="ignore"):
            x_inv = np.linalg.inv(A) @ b
        x_nv = gauss_naive(A, b)
        x_pv = gauss_pivoting(A, b)
        x_gs, it_gs, ok_gs = gauss_seidel(A, b)
        x_rx, it_rx, ok_rx = gauss_seidel(A, b, relaxation=RELAX)
        rows.append({
            "n": n, "cond": float(np.linalg.cond(A)),
            "inverse": max_error(x_ex, x_inv),
            "naive": max_error(x_ex, x_nv),
            "pivoting": max_error(x_ex, x_pv),
            "gs": max_error(x_ex, x_gs), "gs_sweeps": it_gs,
            "gs_converged": ok_gs,
            "gs_finite": bool(np.all(np.isfinite(x_gs))),
            "relax": max_error(x_ex, x_rx), "relax_sweeps": it_rx,
            "relax_converged": ok_rx,
            "relax_finite": bool(np.all(np.isfinite(x_rx))),
            "rho_gs": iteration_spectral_radius(A, 1.0),
            "rho_relax": iteration_spectral_radius(A, RELAX),
        })
        if n == 1:
            solutions = {"exact": x_ex, "inverse": x_inv, "naive": x_nv,
                         "pivoting": x_pv, "Gauss-Seidel": x_gs,
                         f"Gauss-Seidel, relaxation {RELAX}": x_rx}
    write_csv(os.path.join(out, "solvers.csv"), rows)

    rrows = []
    for n in ROUND_NS:
        A, b = build_system(10.0 ** -n)
        x_ex = exact_solution(A, b)
        for k in ROUND_DIGITS:
            rrows.append({"n": n, "digits": k,
                          "naive": max_error(x_ex, gauss_naive(A, b, k)),
                          "pivoting": max_error(x_ex, gauss_pivoting(A, b, k))})
    write_csv(os.path.join(out, "rounding.csv"), rrows)

    t_sol = md_table(
        ["method", "x1", "x2", "x3", "x4"],
        [[m] + [vfull(v) for v in x] for m, x in solutions.items()])
    t_err = md_table(
        ["n (δ = 10⁻ⁿ)", "cond₂(A)", "inverse", "naive", "pivoting",
         "Gauss-Seidel", f"Gauss-Seidel, relaxation {RELAX}"],
        [[r["n"], s4(r["cond"]), s4(r["inverse"]), s4(r["naive"]),
          s4(r["pivoting"]), s4(r["gs"]), s4(r["relax"])] for r in rows])
    t_gs = md_table(
        ["n (δ = 10⁻ⁿ)", "ρ (relaxation 1)", f"ρ (relaxation {RELAX})",
         "Gauss-Seidel: sweeps, stopped by tolerance",
         f"relaxation {RELAX}: sweeps, stopped by tolerance"],
        [[r["n"], s4(r["rho_gs"]), s4(r["rho_relax"]),
          f"{r['gs_sweeps']}, {'yes' if r['gs_converged'] else 'no'}",
          f"{r['relax_sweeps']}, {'yes' if r['relax_converged'] else 'no'}"]
         for r in rows])

    def t_round(method):
        return md_table(
            ["decimals k"] + [f"n = {n}" for n in ROUND_NS],
            [[k] + [s4(next(r[method] for r in rrows
                            if r["n"] == n and r["digits"] == k))
                    for n in ROUND_NS] for k in ROUND_DIGITS])

    write_tables(os.path.join(out, "tables.md"), [
        ("E3-a solutions at δ = 0.1 (n = 1)", t_sol),
        ("E3-b largest component true percent relative error ε_t (%) against the "
         "exact solution", t_err),
        ("E3-c Gauss-Seidel: spectral radius ρ of the iteration matrix and "
         "stopping (tolerance 1%, at most 1000 sweeps)", t_gs),
        ("E3-d naive elimination with k-decimal rounding: largest ε_t (%)",
         t_round("naive")),
        ("E3-e partial pivoting with k-decimal rounding: largest ε_t (%)",
         t_round("pivoting")),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for key, color in (("inverse", "#B279A2"), ("naive", "#E45756"),
                       ("pivoting", "#4C78A8")):
        pts = [(r["n"], r[key]) for r in rows if r[key] > 0]
        ax.plot(*zip(*pts), marker="o", ms=3, color=color, label=key)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 17, 3))
    ax.set_xlabel("n (δ = 10⁻ⁿ)")
    ax.set_ylabel("largest ε_t (%)")
    ax.set_title("full float64", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for n, ls in ((1, "-"), (4, "--")):
        for key, color in (("naive", "#E45756"), ("pivoting", "#4C78A8")):
            pts = [(r["digits"], r[key]) for r in rrows
                   if r["n"] == n and np.isfinite(r[key]) and r[key] > 0]
            ax.plot(*zip(*pts), ls=ls, marker="o", ms=3, color=color,
                    label=f"{key}, n = {n}")
    ax.set_yscale("log")
    ax.set_xlabel("decimals k")
    ax.set_title("k-decimal rounding", fontsize=10)
    ax.legend(fontsize=8)
    for a in axes:
        a.grid(True, which="both", color="#e5e5e2", lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "errors.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, "tables.md"), encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
