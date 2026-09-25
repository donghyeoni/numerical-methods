"""Run the four experiments and regenerate ``results/``.

    python experiments/run_all.py
"""

from __future__ import annotations

import e1_taylor
import e2_roots
import e3_linear
import e4_regression

STEPS = [("E1 Taylor series", e1_taylor), ("E2 root finding", e2_roots),
         ("E3 linear systems", e3_linear),
         ("E4 distribution regression", e4_regression)]


def main():
    for i, (name, module) in enumerate(STEPS, 1):
        print(f"\n##### {i}/{len(STEPS)} {name}")
        module.main()


if __name__ == "__main__":
    main()
