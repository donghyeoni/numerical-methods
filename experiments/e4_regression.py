import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import md_table, out_dir, s4, write_csv, write_tables
from numerical_methods import regression as rg

SEEDS = tuple(range(100))
N, MEAN, SD, POINTS, ORDER, SHIFT = 100, 0.0, 4.0, 15, 2, 10.0

MODELS = {
    "linear": (lambda x, y: rg.fit_linear(x, y), rg.eval_polynomial),
    "polynomial (order 2)": (lambda x, y: rg.fit_polynomial(x, y, ORDER),
                             rg.eval_polynomial),
    "exponential (order 2)": (lambda x, y: rg.fit_exponential(x, y, ORDER),
                              rg.eval_exponential),
    f"power (x + {SHIFT:g})": (lambda x, y: rg.fit_power(x + SHIFT, y),
                               lambda a, x: rg.eval_power(a, x + SHIFT)),
    "sigmoid": (lambda x, y: rg.fit_sigmoid(x, y), rg.eval_sigmoid),
}


def dataset(seed):
    np.random.seed(seed)
    sample = np.random.normal(MEAN, SD, N)
    x = np.random.uniform(-10, 10, POINTS)
    y = np.array([rg.empirical_cdf(sample, v) for v in x])
    return x, y


def fit_all(x, y):
    fits = {}
    for name, (fit, _) in MODELS.items():
        try:
            fits[name] = fit(x, y)
        except np.linalg.LinAlgError:
            fits[name] = None
    return fits


def main():
    out = out_dir("e4_regression")
    rows = []
    for seed in SEEDS:
        x, y = dataset(seed)
        fits = fit_all(x, y)
        row = {"seed": seed}
        for name, (_, ev) in MODELS.items():
            a = fits[name]
            row[name] = float("nan") if a is None else rg.r_squared(y, ev(a, x))
        rows.append(row)
    write_csv(os.path.join(out, "r2.csv"), rows)

    names = list(MODELS)
    R = np.array([[r[m] for m in names] for r in rows])
    valid = ~np.all(np.isnan(R), axis=1)
    best = np.full(len(R), -1)
    best[valid] = np.nanargmax(R[valid], axis=1)
    t_seed0 = md_table(["model", "R²"],
                       [[m, s4(rows[0][m])] for m in names])
    t_sum = md_table(
        ["model", "R² mean ± SD", "median", "min–max", "highest R²",
         "failed fits"],
        [[m, f"{s4(np.nanmean(R[:, j]))} ± {s4(np.nanstd(R[:, j]))}",
          s4(np.nanmedian(R[:, j])),
          f"{s4(np.nanmin(R[:, j]))} – {s4(np.nanmax(R[:, j]))}",
          f"{int(np.sum(best == j))} / {len(SEEDS)}",
          int(np.sum(np.isnan(R[:, j])))] for j, m in enumerate(names)])
    write_tables(os.path.join(out, "tables.md"), [
        ("E4-a R² at seed 0", t_seed0),
        (f"E4-b R² over {len(SEEDS)} seeds (0–{SEEDS[-1]}), SD with ddof = 0",
         t_sum),
    ])

    colors = ["#E45756", "#54A24B", "#F58518", "#B279A2", "#4C78A8"]
    x, y = dataset(0)
    fits = fit_all(x, y)
    grid = np.linspace(-10, 10, 1000)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(x, y, color="k", s=16, zorder=3, label="empirical CDF points")
    for (name, (_, ev)), c in zip(MODELS.items(), colors):
        if fits[name] is None:
            continue
        ax.plot(grid, ev(fits[name], grid), color=c, lw=1.3, label=name)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title("Fits at seed 0", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, color="#e5e5e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fits_seed0.png"), dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for j, (name, c) in enumerate(zip(names, colors)):
        v = R[:, j][~np.isnan(R[:, j])]
        ax.scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=c,
                   edgecolor="white", linewidth=0.6, zorder=3)
        med = float(np.median(v))
        ax.hlines(med, j - 0.25, j + 0.25, color="#0b0b0b", lw=2, zorder=4)
        ax.annotate(s4(med), (j + 0.27, med), va="center", fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title(f"R² over {len(SEEDS)} seeds (bars: medians)", fontsize=10)
    ax.grid(True, axis="y", color="#e5e5e2", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r2.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, "tables.md"), encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
