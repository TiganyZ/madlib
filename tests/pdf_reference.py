#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""An independent partial pair distribution, for checking compute mad/pdf.

Deliberately naive: build every periodic image inside the cutoff by brute force
and evaluate the Gaussian on the whole grid, sharing nothing with the C++.

    g_ab(r_l) = V/(N_a N_b f_ab) 1/(sqrt(2pi) sigma 4 pi r_l^2)
                sum_(i,j) exp( -(r_l - r_ij)^2 / 2 sigma^2 )

over ordered pairs of the channel, f_ab = 1 if a == b else 2.
"""
import argparse
import itertools
import re
import sys

import numpy as np


def read_extxyz(path):
    L = open(path).read().split('\n')
    n = int(L[0].split()[0])
    cell = np.array([float(v) for v in
                     re.search(r'Lattice="([^"]+)"', L[1]).group(1).split()]).reshape(3, 3)
    sp, pos = [], []
    for line in L[2:2 + n]:
        w = line.split()
        sp.append(w[0]); pos.append([float(v) for v in w[1:4]])
    return sp, np.array(pos), cell


def partial_g(sp, pos, cell, rmin, rmax, nbins, rcut, sigma):
    kinds = sorted(set(sp), key=sp.index)
    tid = np.array([kinds.index(s) for s in sp])
    n = len(sp)
    vol = abs(np.linalg.det(cell))
    counts = np.array([(tid == a).sum() for a in range(len(kinds))], dtype=float)

    delr = (rmax - rmin) / nbins
    r = rmin + (np.arange(nbins) + 0.5) * delr

    # enough images to cover rcut in every direction
    rec = np.linalg.inv(cell).T
    reps = [int(np.ceil(rcut * np.linalg.norm(rec[k]))) + 1 for k in range(3)]
    shifts = np.array([i * cell[0] + j * cell[1] + k * cell[2]
                       for i, j, k in itertools.product(*[range(-m, m + 1) for m in reps])])

    nk = len(kinds)
    chan = {}
    c = 0
    for a in range(nk):
        for b in range(a, nk):
            chan[(a, b)] = chan[(b, a)] = c
            c += 1
    acc = np.zeros((c, nbins))

    for i in range(n):
        d = pos[None, :, :] + shifts[:, None, :] - pos[i]
        dist = np.sqrt((d * d).sum(-1)).ravel()
        who = np.tile(tid, len(shifts))
        keep = (dist < rcut) & (dist > 1e-12)
        for rij, j in zip(dist[keep], who[keep]):
            acc[chan[(tid[i], j)]] += np.exp(-(r - rij) ** 2 / (2 * sigma ** 2))

    g = np.zeros_like(acc)
    for a in range(nk):
        for b in range(a, nk):
            cc = chan[(a, b)]
            fab = 1.0 if a == b else 2.0
            if counts[a] == 0 or counts[b] == 0:
                continue
            g[cc] = acc[cc] * (vol / (counts[a] * counts[b] * fab)) \
                    / (np.sqrt(2 * np.pi) * sigma * 4 * np.pi * r ** 2)
    return r, g, kinds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xyz")
    p.add_argument("lammps")
    p.add_argument("--rmin", type=float, default=0.1)
    p.add_argument("--rmax", type=float, default=6.0)
    p.add_argument("--nbins", type=int, default=201)
    p.add_argument("--rcut", type=float, default=6.0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--rtol", type=float, default=1e-10)
    a = p.parse_args()

    sp, pos, cell = read_extxyz(a.xyz)
    r, g, kinds = partial_g(sp, pos, cell, a.rmin, a.rmax, a.nbins, a.rcut, a.sigma)

    got = np.loadtxt(a.lammps, skiprows=4)
    if got.shape[0] != a.nbins:
        print(f"FAIL: {a.lammps} has {got.shape[0]} rows, expected {a.nbins}")
        return 1

    dr = np.abs(got[:, 1] - r).max()
    bad = dr > 1e-10
    print(f"    grid  max|dr| = {dr:.2e}")
    for c in range(g.shape[0]):
        scale = max(float(np.abs(g[c]).max()), 1e-30)
        d = float(np.abs(got[:, 2 + c] - g[c]).max())
        tol = a.rtol * scale
        print(f"    channel {c}  max|g| = {scale:.6e}  max|dg| = {d:.2e}  tol = {tol:.2e}")
        if d > tol:
            bad = True
            k = int(np.argmax(np.abs(got[:, 2 + c] - g[c])))
            print(f"      worst bin {k}: r = {r[k]:.6f}  ref = {g[c][k]:.12e}  lammps = {got[k, 2+c]:.12e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
