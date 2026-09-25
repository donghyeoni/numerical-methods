"""Shared paths and table helpers for the experiment scripts."""

from __future__ import annotations

import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")


def out_dir(name):
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


def s4(x):
    """A measured value to 4 significant digits (``format(x, "#.4g")``:
    e.g. ``0.9643``, ``14.94``, ``3.575e-06``); ``0`` for exactly zero,
    ``nan`` / ``inf`` as is, ``–`` for None."""
    if x is None:
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    return format(x, "#.4g")


def vfull(x):
    """An approximate value (iterate, polynomial value, solution component)
    in full float64 precision: the shortest decimal string that converts
    back to the same float (``repr``)."""
    return repr(float(x))


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(path, sections):
    """Write ``[(title, table_markdown), ...]`` to a ``tables.md`` file."""
    with open(path, "w", encoding="utf-8") as f:
        for title, table in sections:
            f.write(f"### {title}\n\n{table}\n\n")
