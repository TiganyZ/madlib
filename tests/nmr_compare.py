#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Two one-value-per-atom dumps agree. The ghost-cluster route against the cell one.

    nmr_compare.py <a.dump> <b.dump> <tolerance>

The value column is whichever one is not the atom id, so it does not matter
what the compute in the deck was called.
"""
import sys

import numpy as np


def read(path):
    txt = open(path).read()
    head = txt.split("ITEM: ATOMS")[1].split("\n")[0].split()
    col = {k: i for i, k in enumerate(head)}
    rows = [l.split() for l in txt.split("ITEM: ATOMS")[1].split("\n")[1:] if l.strip()]
    rows.sort(key=lambda r: int(r[col["id"]]))
    value = [k for k in head if k != "id"]
    if len(value) != 1:
        raise SystemExit(f"{path}: expected one value column beside id, found {value}")
    return np.array([float(r[col[value[0]]]) for r in rows])


a, b, tol = read(sys.argv[1]), read(sys.argv[2]), float(sys.argv[3])
d = np.abs(a - b)
print("    %d shieldings: max|d| = %.3e ppm  rms = %.3e" %
      (len(a), d.max(), np.sqrt((d ** 2).mean())))
sys.exit(0 if d.max() < tol else 1)
