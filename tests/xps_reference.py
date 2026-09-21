#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""An independent smooth coordination number and XPS spectrum.

Shares nothing with the C++: brute-force periodic images, the raised-cosine
cutoff written out, and the Gaussian broadening summed with no truncation.

    q_i  = sum_j fc(r_ij),  fc = 1 below r1, raised cosine to r2, 0 beyond
    u_l  = sum_i exp( -(x_l - q_i)^2 / 2 sigma^2 )
    y    = u / ||u||
"""
import argparse
import itertools
import sys

import numpy as np

from pdf_reference import read_extxyz


def coordination(pos, cell, r1, r2):
    n = len(pos)
    rec = np.linalg.inv(cell).T
    reps = [int(np.ceil(r2 * np.linalg.norm(rec[k]))) + 1 for k in range(3)]
    shifts = np.array([i * cell[0] + j * cell[1] + k * cell[2]
                       for i, j, k in itertools.product(*[range(-m, m + 1) for m in reps])])
    q = np.zeros(n)
    for i in range(n):
        d = pos[None, :, :] + shifts[:, None, :] - pos[i]
        r = np.sqrt((d * d).sum(-1)).ravel()
        r = r[(r < r2) & (r > 1e-12)]
        fc = np.where(r <= r1, 1.0, 0.5 * (1.0 + np.cos(np.pi * (r - r1) / (r2 - r1))))
        q[i] = fc.sum()
    return q


def spectrum(q, emin, emax, nk, sigma, norm):
    x = np.linspace(emin, emax, nk)
    u = np.exp(-(x[None, :] - q[:, None]) ** 2 / (2 * sigma ** 2)).sum(0)
    if norm == "l2":
        s = np.sqrt((u * u).sum())
    elif norm == "area":
        s = u.sum() * (x[1] - x[0])
    else:
        s = 1.0
    return x, u / (s if s > 0 else 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xyz")
    p.add_argument("dump", help="LAMMPS dump of id type c_cn")
    p.add_argument("spec", help="fix mad/xps output file")
    p.add_argument("--r1", type=float, default=1.7)
    p.add_argument("--r2", type=float, default=2.1)
    p.add_argument("--emin", type=float, default=0.0)
    p.add_argument("--emax", type=float, default=5.0)
    p.add_argument("--nk", type=int, default=201)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--norm", default="l2")
    p.add_argument("--tol", type=float, default=1e-11)
    a = p.parse_args()

    sp, pos, cell = read_extxyz(a.xyz)
    q = coordination(pos, cell, a.r1, a.r2)

    body = open(a.dump).read().split('ITEM: ATOMS')[-1].split('\n')[1:]
    got = np.array([float(l.split()[2]) for l in body if len(l.split()) == 3])
    dq = np.abs(got - q).max()
    print(f"    coordination:  max|dq| = {dq:.3e}   (mean {q.mean():.4f})")

    x, y = spectrum(q, a.emin, a.emax, a.nk, a.sigma, a.norm)
    rows = [l.split() for l in open(a.spec) if not l.startswith('#')]
    gy = np.array([float(r[1]) for r in rows])
    gx = np.array([float(r[0]) for r in rows])
    dx = np.abs(gx - x).max()
    dy = np.abs(gy - y).max()
    print(f"    spectrum:      max|dx| = {dx:.3e}  max|dy| = {dy:.3e}  (peak {y.max():.6g})")

    return 0 if (dq < a.tol and dy < a.tol and dx < 1e-12) else 1


if __name__ == "__main__":
    sys.exit(main())
