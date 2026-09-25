# Experiment Log

The plan, the settings, the full result tables and the observations of the
four experiments, which are independent of each other. The
[README](../README.md) summarises the results; every result value in it
appears in the tables below.

## Conventions

- **Errors** are in percent (Chapra & Canale):
  - true percent relative error `ε_t = |(true − approx) / true| · 100`;
  - approximate percent relative error
    `ε_a = |(present − previous) / present| · 100`, not defined for the
    first estimate or when the present estimate is 0 (shown as `–`).
- **Precision.** Errors and other measured values are shown with 4
  significant digits (`format(x, "#.4g")`, for example `0.9643`, `14.94`,
  `3.575e-06`). Approximate values (polynomial values, iterates, solution
  components) are shown in full float64 precision (the shortest decimal that
  converts back to the same float). `0` means exactly zero in float64;
  `nan` means the computation gave NaN.
- **Arithmetic.** float64 (NumPy) unless a section says otherwise.
- **Result tables.** Every result table (titled with its id and file) is
  written by the script named in its section to a `tables.md` file and
  copied here unchanged.

## Plan (written before any run)

1. **E1 — Taylor series.** `sin(x)` about `x0 = π/6`: values and errors of
   the degree-0 to 10 polynomials at `x = π/6 + 0.005`; figure of `P1`, `P2`,
   `P4` and `sin(x)` on `[−π/2, π/2)`.
2. **E2 — Root finding.** `f(x) = (x − 1)(x − 2)(x − 3)`, root 3: bisection on
   `[2.5, 4]`, fixed-point iteration with
   `g(x) = 6/11 + 6/11·x² − 1/11·x³` from 2.5, Newton-Raphson from 4, each for
   15 iterations.
3. **E3 — Linear systems.** The 4×4 system with `δ` on the diagonal, for
   `δ = 10⁻ⁿ`, `n = 1–16`: matrix inverse, naive Gaussian elimination,
   partial pivoting, Gauss-Seidel and Gauss-Seidel with relaxation 0.9,
   against the exact solution; then the same direct solvers with every
   intermediate value rounded to `k` = 1, 2, 3, 4, 6, 8 decimals.
4. **E4 — Distribution regression.** 100 normal samples (mean 0, SD 4), their
   empirical CDF at 15 uniform points on `[−10, 10)`, fitted by 5 models; `R²`
   over 100 random seeds (0–99) instead of one.

## E1 — Taylor series

**Setting.** `experiments/e1_taylor.py` → `results/e1_taylor/`.
`P_n(x) = Σ_{i=0}^{n} sin⁽ⁱ⁾(π/6)·(x − π/6)ⁱ / i!`, with the derivatives of
`sin` at `π/6` repeating as `1/2, √3/2, −1/2, −√3/2`
(`src/numerical_methods/taylor.py`). `ε_t` is against `numpy.sin(x)`; `ε_a`
of order `n` compares `P_n` with `P_{n−1}`.

**E1-a sin(x) about π/6 at x = π/6 + 0.005 (sin(x) = 0.5043238589897696)** (`results/e1_taylor/tables.md`)

| order n | P_n(x) | ε_t (%) | ε_a (%) |
| --- | --- | --- | --- |
| 0 | 0.5 | 0.8574 | – |
| 1 | 0.5043301270189222 | 0.001243 | 0.8586 |
| 2 | 0.5043238770189222 | 3.575e-06 | 0.001239 |
| 3 | 0.5043238589767263 | 2.586e-09 | 3.578e-06 |
| 4 | 0.5043238589897471 | 4.469e-12 | 2.582e-09 |
| 5 | 0.5043238589897697 | 2.201e-14 | 4.491e-12 |
| 6 | 0.5043238589897696 | 0 | 2.201e-14 |
| 7 | 0.5043238589897696 | 0 | 0 |
| 8 | 0.5043238589897696 | 0 | 0 |
| 9 | 0.5043238589897696 | 0 | 0 |
| 10 | 0.5043238589897696 | 0 | 0 |

Figure `results/e1_taylor/polynomials.png`: `P1`, `P2`, `P4` and `sin(x)` on
`[−π/2, π/2)` (step 0.001); the dotted line marks `x = π/6`.

**Observations.**

- `ε_t` falls from 0.8574% at order 0 to 2.201e-14% at order 5. From order 6
  on, `P_n(x)` equals `numpy.sin(x)` in float64 (`ε_t = 0`).
- For orders 0–5, `ε_a` of order `n + 1` agrees with `ε_t` of order `n` to 2
  significant digits (for example 0.001239 and 0.001243).

## E2 — Root finding

**Setting.** `experiments/e2_roots.py` → `results/e2_roots/`. 15 records per
method (`src/numerical_methods/roots.py`). Bisection records the midpoints of
iterations 1–15 and keeps `[mid, upper]` when `f(lower)·f(mid) > 0`, otherwise
`[lower, mid]`. Fixed-point iteration and Newton-Raphson record `x_0` (the
starting value, iteration 0) to `x_14`. `ε_t` is against the root 3.

**E2-a bisection on [2.5, 4]** (`results/e2_roots/tables.md`)

| iteration | x | ε_a (%) | ε_t (%) |
| --- | --- | --- | --- |
| 1 | 3.25 | – | 8.333 |
| 2 | 2.875 | 13.04 | 4.167 |
| 3 | 3.0625 | 6.122 | 2.083 |
| 4 | 2.96875 | 3.158 | 1.042 |
| 5 | 3.015625 | 1.554 | 0.5208 |
| 6 | 2.9921875 | 0.7833 | 0.2604 |
| 7 | 3.00390625 | 0.3901 | 0.1302 |
| 8 | 2.998046875 | 0.1954 | 0.06510 |
| 9 | 3.0009765625 | 0.09762 | 0.03255 |
| 10 | 2.99951171875 | 0.04884 | 0.01628 |
| 11 | 3.000244140625 | 0.02441 | 0.008138 |
| 12 | 2.9998779296875 | 0.01221 | 0.004069 |
| 13 | 3.00006103515625 | 0.006103 | 0.002035 |
| 14 | 2.999969482421875 | 0.003052 | 0.001017 |
| 15 | 3.0000152587890625 | 0.001526 | 0.0005086 |

**E2-b fixed-point iteration, g(x) = 6/11 + 6/11·x² − 1/11·x³, x0 = 2.5** (`results/e2_roots/tables.md`)

| iteration | x | ε_a (%) | ε_t (%) | ε_t(i) / ε_t(i−1) |
| --- | --- | --- | --- | --- |
| 0 | 2.5 | – | 16.67 | – |
| 1 | 2.5340909090909087 | 1.345 | 15.53 | 0.9318 |
| 2 | 2.568794529275663 | 1.351 | 14.37 | 0.9255 |
| 3 | 2.6037739854974244 | 1.343 | 13.21 | 0.9189 |
| 4 | 2.6386532936853966 | 1.322 | 12.04 | 0.9120 |
| 5 | 2.673031534294562 | 1.286 | 10.90 | 0.9049 |
| 6 | 2.7065013040124994 | 1.237 | 9.783 | 0.8976 |
| 7 | 2.738669972350745 | 1.175 | 8.711 | 0.8904 |
| 8 | 2.7691815193511484 | 1.102 | 7.694 | 0.8832 |
| 9 | 2.7977363197089797 | 1.021 | 6.742 | 0.8763 |
| 10 | 2.8241063469154426 | 0.9337 | 5.863 | 0.8696 |
| 11 | 2.848143935456076 | 0.8440 | 5.062 | 0.8633 |
| 12 | 2.8697833151889274 | 0.7540 | 4.341 | 0.8575 |
| 13 | 2.8890353350847535 | 0.6664 | 3.699 | 0.8522 |
| 14 | 2.905976806409865 | 0.5830 | 3.134 | 0.8473 |

**E2-c Newton-Raphson, x0 = 4** (`results/e2_roots/tables.md`)

| iteration | x | ε_a (%) | ε_t (%) | 0.045·ε_t(i−1)² |
| --- | --- | --- | --- | --- |
| 0 | 4.0 | – | 33.33 | – |
| 1 | 3.4545454545454546 | 15.79 | 15.15 | 50.00 |
| 2 | 3.151046789377547 | 9.632 | 5.035 | 10.33 |
| 3 | 3.0253259289766983 | 4.156 | 0.8442 | 1.141 |
| 4 | 3.0009084519430513 | 0.8137 | 0.03028 | 0.03207 |
| 5 | 3.0000012353089454 | 0.03024 | 4.118e-05 | 4.126e-05 |
| 6 | 3.000000000002289 | 4.118e-05 | 7.629e-11 | 7.630e-11 |
| 7 | 3.0 | 7.629e-11 | 0 | 2.619e-22 |
| 8 | 3.0 | 0 | 0 | 0 |
| 9 | 3.0 | 0 | 0 | 0 |
| 10 | 3.0 | 0 | 0 | 0 |
| 11 | 3.0 | 0 | 0 | 0 |
| 12 | 3.0 | 0 | 0 | 0 |
| 13 | 3.0 | 0 | 0 | 0 |
| 14 | 3.0 | 0 | 0 | 0 |

Figure `results/e2_roots/errors.png`: `ε_t` per iteration on a log scale;
records with `ε_t = 0` (Newton-Raphson from iteration 7) are not drawn.

**Observations.**

- **Bisection:** `ε_t` halves at every iteration, from 8.333% to 0.0005086%
  at iteration 15.
- **Fixed-point iteration:** `ε_t` is still 3.134% at `x_14`. The ratio of
  successive `ε_t` falls from 0.9318 to 0.8473 over iterations 1–14; the
  linear rate at the root is `g′(3) = 9/11 = 0.8182`.
- **Newton-Raphson:** `x = 3.0` exactly from iteration 7. At iterations 5 and
  6, `ε_t` (4.118e-05% and 7.629e-11%) matches the quadratic estimate
  `0.045·ε_t(i−1)²` (4.126e-05% and 7.630e-11%). Here
  `0.045 = f″(3) / (2 f′(3)) · 3 / 100` converts the absolute-error rate
  `f″(3) / (2 f′(3)) = 1.5` to percent of the root.

## E3 — Linear systems

**Setting.** `experiments/e3_linear.py` → `results/e3_linear/`, solvers in
`src/numerical_methods/linear_solvers.py`.

- `A = [[δ, 3, 2, 1], [4, δ, 7, 5], [8, 2, δ, 2], [0, 1, 2, δ]]`,
  `b = [−3, 2, −2, −5]`, `δ = 10⁻ⁿ` (the float64 value of `10.0**-n`).
- **Exact solution:** partial-pivoting elimination in rational arithmetic
  (`fractions.Fraction`) on the float64 entries, converted to float64.
  `ε_t` below is the largest of the four component errors.
- **Solvers:** `numpy.linalg.inv(A) @ b`; naive Gaussian elimination; partial
  (row) pivoting; Gauss-Seidel from `x = 0`, each component updated with the
  newest values and relaxed right after its update,
  `x_i ← w·x_i + (1 − w)·x_i,old` (`w = 1` and `w = 0.9`), stopping when the
  largest `ε_a` of a sweep is below 1% or after 1000 sweeps.
- **Rounding:** in the elimination solvers, the multipliers, the updated
  matrix entries and the back-substitution results are rounded to `k`
  decimals with `numpy.round`; the products and inner sums between them and
  the input system are not rounded.
- `ρ` is the spectral radius of the Gauss-Seidel iteration matrix
  `(D + wL)⁻¹((1 − w)D − wU)`; the iteration converges from every start if and
  only if `ρ < 1`.

**E3-a solutions at δ = 0.1 (n = 1)** (`results/e3_linear/tables.md`)

| method | x1 | x2 | x3 | x4 |
| --- | --- | --- | --- | --- |
| exact | -1.0968655406403376 | -0.9456855558177555 | -2.2494382035188747 | 4.4456196285550496 |
| inverse | -1.0968655406403383 | -0.9456855558177547 | -2.249438203518876 | 4.44561962855505 |
| naive | -1.0968655406403371 | -0.9456855558177553 | -2.249438203518875 | 4.4456196285550496 |
| pivoting | -1.0968655406403376 | -0.9456855558177558 | -2.2494382035188742 | 4.4456196285550496 |
| Gauss-Seidel | nan | nan | nan | nan |
| Gauss-Seidel, relaxation 0.9 | nan | nan | nan | nan |

**E3-b largest component true percent relative error ε_t (%) against the exact solution** (`results/e3_linear/tables.md`)

| n (δ = 10⁻ⁿ) | cond₂(A) | inverse | naive | pivoting | Gauss-Seidel | Gauss-Seidel, relaxation 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.94 | 8.218e-14 | 4.049e-14 | 3.522e-14 | nan | nan |
| 2 | 13.55 | 2.201e-14 | 4.952e-12 | 3.347e-14 | nan | nan |
| 3 | 13.43 | 4.436e-14 | 2.183e-11 | 4.443e-14 | nan | nan |
| 4 | 13.42 | 2.221e-14 | 3.110e-10 | 2.221e-14 | nan | nan |
| 5 | 13.42 | 2.220e-14 | 4.202e-09 | 2.220e-14 | nan | nan |
| 6 | 13.42 | 2.220e-14 | 3.238e-08 | 3.331e-14 | nan | nan |
| 7 | 13.42 | 4.441e-14 | 1.250e-06 | 2.220e-14 | nan | nan |
| 8 | 13.42 | 2.220e-14 | 1.488e-06 | 2.220e-14 | nan | nan |
| 9 | 13.42 | 2.220e-14 | 8.186e-06 | 3.331e-14 | nan | nan |
| 10 | 13.42 | 0 | 0.0008799 | 2.220e-14 | nan | nan |
| 11 | 13.42 | 2.220e-14 | 8.273e-06 | 2.220e-14 | nan | nan |
| 12 | 13.42 | 3.331e-14 | 0.09771 | 2.220e-14 | nan | nan |
| 13 | 13.42 | 2.220e-14 | 0.3642 | 3.331e-14 | nan | nan |
| 14 | 13.42 | 2.220e-14 | 2.300 | 2.220e-14 | nan | nan |
| 15 | 13.42 | 2.220e-14 | 11.18 | 2.220e-14 | nan | nan |
| 16 | 13.42 | 4.441e-14 | nan | 3.331e-14 | nan | nan |

**E3-c Gauss-Seidel: spectral radius ρ of the iteration matrix and stopping (tolerance 1%, at most 1000 sweeps)** (`results/e3_linear/tables.md`)

| n (δ = 10⁻ⁿ) | ρ (relaxation 1) | ρ (relaxation 0.9) | Gauss-Seidel: sweeps, stopped by tolerance | relaxation 0.9: sweeps, stopped by tolerance |
| --- | --- | --- | --- | --- |
| 1 | 1.091e+05 | 6.830e+04 | 1000, no | 1000, no |
| 2 | 1.545e+09 | 1.009e+09 | 1000, no | 1000, no |
| 3 | 1.594e+13 | 1.046e+13 | 1000, no | 1000, no |
| 4 | 1.599e+17 | 1.049e+17 | 1000, no | 1000, no |
| 5 | 1.600e+21 | 1.050e+21 | 1000, no | 1000, no |
| 6 | 1.600e+25 | 1.050e+25 | 1000, no | 1000, no |
| 7 | 1.600e+29 | 1.050e+29 | 1000, no | 1000, no |
| 8 | 1.600e+33 | 1.050e+33 | 1000, no | 1000, no |
| 9 | 1.600e+37 | 1.050e+37 | 1000, no | 1000, no |
| 10 | 1.600e+41 | 1.050e+41 | 1000, no | 1000, no |
| 11 | 1.600e+45 | 1.050e+45 | 1000, no | 1000, no |
| 12 | 1.600e+49 | 1.050e+49 | 1000, no | 1000, no |
| 13 | 1.600e+53 | 1.050e+53 | 1000, no | 1000, no |
| 14 | 1.600e+57 | 1.050e+57 | 1000, no | 1000, no |
| 15 | 1.600e+61 | 1.050e+61 | 1000, no | 1000, no |
| 16 | 1.600e+65 | 1.050e+65 | 1000, no | 1000, no |

**E3-d naive elimination with k-decimal rounding: largest ε_t (%)** (`results/e3_linear/tables.md`)

| decimals k | n = 1 | n = 2 | n = 4 | n = 8 |
| --- | --- | --- | --- | --- |
| 1 | 86.71 | 891.2 | 9.989e+04 | 1.000e+09 |
| 2 | 8.831 | 30.17 | 1.010e+04 | 1.000e+08 |
| 3 | 1.403 | 20.70 | 1100. | 1.000e+07 |
| 4 | 0.09166 | 1.111 | 200.0 | 100.0 |
| 6 | 0.001280 | 0.02166 | 1.076 | 1.010e+04 |
| 8 | 6.067e-06 | 5.473e-05 | 0.01120 | 28.13 |

**E3-e partial pivoting with k-decimal rounding: largest ε_t (%)** (`results/e3_linear/tables.md`)

| decimals k | n = 1 | n = 2 | n = 4 | n = 8 |
| --- | --- | --- | --- | --- |
| 1 | 11.09 | 10.79 | 10.01 | 10.00 |
| 2 | 2.109 | 1.847 | 1.991 | 2.000 |
| 3 | 0.1701 | 0.08619 | 0.01490 | 0.02500 |
| 4 | 0.01305 | 0.01799 | 0.004899 | 0.009999 |
| 6 | 0.0001424 | 8.616e-05 | 0.0001011 | 9.912e-05 |
| 8 | 1.765e-06 | 1.422e-06 | 8.072e-07 | 2.380e-06 |

Figure `results/e3_linear/errors.png`: left, E3-b for the inverse, naive and
pivoting solvers; right, E3-d and E3-e at `n = 1` and `n = 4`. Log scales;
values of exactly 0 or nan are not drawn.

**Observations.**

- `cond₂(A)` stays between 13.42 and 14.94 for all `n`.
- The inverse and pivoting solvers have `ε_t` of at most 8.218e-14% for every
  `n`.
- Naive elimination divides by the pivot `δ`. Its `ε_t` grows from 4.049e-14%
  at `n = 1` to 11.18% at `n = 15`, not monotonically (0.0008799% at `n = 10`,
  8.273e-06% at `n = 11`), and is nan at `n = 16`.
- Gauss-Seidel: `ρ` is at least 1.091e+05 (`w = 1`) and 6.830e+04
  (`w = 0.9`) for all `n`. Every run used all 1000 sweeps and ended with nan
  components.
- With rounding, the pivoting error falls as `k` grows at each tested `δ`
  (`n` = 1, 2, 4, 8), from
  10.00–11.09% at `k = 1` to 8.072e-07–2.380e-06% at `k = 8`. Naive
  elimination also falls with `k` at `n` = 1, 2 and 4, but at `n = 8` it is
  28.13% or more for every `k`.

## E4 — Distribution regression

**Setting.** `experiments/e4_regression.py` → `results/e4_regression/`,
models in `src/numerical_methods/regression.py`. For each seed 0–99:
`numpy.random.seed(seed)`; 100 samples from `N(0, 4²)`; 15 points
`x ~ U[−10, 10)`; `y = F(x)`, the fraction of samples `≤ x`. Models, fitted by
the normal equations:

- linear `a0 + a1·x`; polynomial of order 2;
- exponential `a0·exp(a1·x + a2·x²)`, fitted on `ln y` (points with `y = 0`
  left out);
- power `a0·(x + 10)^a1`, fitted on `log10` (points with `y = 0` or
  `x + 10 ≤ 0` left out; no drawn point has `x + 10 ≤ 0`);
- sigmoid `1 / (1 + exp(a0 + a1·x))`, fitted on `ln(1/y − 1)` (only
  `0 < y < 1`).

`R² = (S_t − S_r) / S_t` on all 15 points. "Highest R²" counts the seeds where
a model has the largest `R²` of the five.

**E4-a R² at seed 0** (`results/e4_regression/tables.md`)

| model | R² |
| --- | --- |
| linear | 0.9429 |
| polynomial (order 2) | 0.9923 |
| exponential (order 2) | 0.9969 |
| power (x + 10) | 0.8799 |
| sigmoid | 0.9940 |

**E4-b R² over 100 seeds (0–99), SD with ddof = 0** (`results/e4_regression/tables.md`)

| model | R² mean ± SD | median | min–max | highest R² | failed fits |
| --- | --- | --- | --- | --- | --- |
| linear | 0.9643 ± 0.01392 | 0.9654 | 0.9205 – 0.9910 | 0 / 100 | 0 |
| polynomial (order 2) | 0.9724 ± 0.01162 | 0.9734 | 0.9377 – 0.9923 | 6 / 100 | 0 |
| exponential (order 2) | 0.9774 ± 0.02529 | 0.9847 | 0.8331 – 0.9989 | 15 / 100 | 0 |
| power (x + 10) | 0.8609 ± 0.1239 | 0.8906 | 0.07798 – 0.9810 | 0 / 100 | 0 |
| sigmoid | 0.9934 ± 0.005766 | 0.9954 | 0.9697 – 0.9992 | 79 / 100 | 0 |

Figures: `results/e4_regression/fits_seed0.png` (the five fits at seed 0) and
`results/e4_regression/r2.png` (`R²` of every seed, bars at the medians).

**Observations.**

- At seed 0 the exponential model has the highest `R²` (0.9969).
- Over the 100 seeds, the sigmoid model has the highest mean (0.9934) and the
  highest `R²` at 79 of them; exponential 15, polynomial 6, linear and power 0.
- The power model has the lowest mean (0.8609) and the widest range
  (0.07798–0.9810).
- No fit failed.

## Changes made after a run

1. **Number format.** After the first runs of E1 and E4, measured values
   changed from scientific notation with 4 significant digits (`3.575e-06`,
   `9.643e-01`) to `format(x, "#.4g")`, and approximate values from 10
   significant digits to full float64 precision, because 10 digits showed
   different `P_n(x)` as equal. The computed values did not change.
2. **E3 figure.** After the first run, the x axis of the left panel got
   integer ticks. The data did not change.
3. **E2 tables.** After the first run, the fixed-point table got the column
   `ε_t(i) / ε_t(i−1)` and the Newton-Raphson table the column
   `0.045·ε_t(i−1)²`, so that the convergence checks in the observations are
   read from the tables. The iterates did not change.
4. **Code cleanup.** After the E1–E4 runs above, the code was tidied: the
   last, unrecorded update of fixed-point iteration and Newton-Raphson is no
   longer computed; the power fit also leaves out points with `x + 10 ≤ 0`;
   seeds where every fit fails would no longer count toward "highest R²"; the
   CSV writing moved to one helper. The E2 figure's y label and the E3-b title
   now say "true percent relative error". A rerun gave the same values in
   every CSV and table.
5. **Script output.** After the E1–E4 runs above, the scripts stopped
   printing their tables to the console, so that a run with redirected output
   does not fail on characters outside the console encoding; CSV files are
   written as UTF-8; `ε_a` and the E2 ratio column are `–` when their divisor
   is 0 (it is not 0 in any run). A rerun gave byte-identical results.

**Differences between the plan and the runs** (the plan above is kept as
written):

- E2: "15 iterations" is 15 records per method. Bisection records iterations
  1–15; fixed-point iteration and Newton-Raphson record the starting value
  `x_0` and 14 iterations (`x_1`–`x_14`), as in the original scripts.
- E3: the rounding study covers naive elimination and partial pivoting only,
  at `n` = 1, 2, 4 and 8. The matrix inverse (`numpy.linalg.inv`) is a library
  routine whose intermediate values cannot be rounded. Not every intermediate
  value is rounded: only the multipliers, the updated matrix entries and the
  back-substitution results (see the E3 setting).

## Environment

Python 3.12.10, NumPy 2.5.3, matplotlib 3.11.2, Windows 11, CPU only. With
the package installed (`pip install -e .`), `python experiments/run_all.py`
runs E1–E4 in order and rewrites `results/`.
