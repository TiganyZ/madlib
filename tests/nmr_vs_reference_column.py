#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""LAMMPS's shieldings against a cs_iso column produced outside this code.

The succinic acid frame arrived with ShiftML4's own prediction attached, which
makes it the one oracle here that owes nothing to MADLIB: agreeing with it means
the compute, the module, the cluster-or-cell choice and the type mapping are all
right at once, in a way agreeing with a second run of the same code does not.

    nmr_vs_reference_column.py <dump> <reference column> <tolerance ppm>
"""
import sys

import numpy as np


def read_dump(path):
    txt = open(path).read()
    head = txt.split("ITEM: ATOMS")[1].split("\n")[0].split()
    col = {k: i for i, k in enumerate(head)}
    rows = [l.split() for l in txt.split("ITEM: ATOMS")[1].split("\n")[1:] if l.strip()]
    rows.sort(key=lambda r: int(r[col["id"]]))
    return np.array([float(r[col["c_cs"]]) for r in rows])


got = read_dump(sys.argv[1])
ref = np.loadtxt(sys.argv[2])[:, 2]
tol = float(sys.argv[3])
d = np.abs(got - ref)
print("    %d shieldings vs the column in the source xyz: max|d| = %.3e ppm  rms = %.3e"
      % (len(got), d.max(), np.sqrt((d ** 2).mean())))
sys.exit(0 if d.max() < tol else 1)
