import csv
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")


def out_dir(name):
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


def s4(x):
    if x is None:
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    return format(x, "#.4g")


def vfull(x):
    return repr(float(x))


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(path, sections):
    with open(path, "w", encoding="utf-8") as f:
        for title, table in sections:
            f.write(f"### {title}\n\n{table}\n\n")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
