#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Recompute a fix mad/nmr/aniso spectrum from the tensors it was built from.

    nmr_aniso_reference.py <tensor dump> <spectrum file> <sigma> <norient> \
                           <reference> <slope> [tolerance]

The dump holds the six Voigt components per atom and the spectrum file is what
the fix wrote. Everything between the two -- the orientation average, the
Gaussian broadening, the shielding-to-shift map and the normalisation -- is
redone here in numpy from the formula in the documentation, so a factor of two
in the off-diagonal components, a hemisphere mistaken for a sphere or a
normalisation applied twice shows up as a disagreement rather than as a
spectrum that merely looks plausible.
"""
import sys

import numpy as np


def read_tensors(path):
    body = open(path).read().split("ITEM: ATOMS")[-1].split("\n")[1:]
    rows = [r.split() for r in body if len(r.split()) == 7]
    rows.sort(key=lambda r: int(r[0]))
    return np.array([[float(v) for v in r[1:]] for r in rows])


def read_spectrum(path):
    data = np.array([[float(v) for v in line.split()]
                     for line in open(path) if line.strip() and not line.startswith("#")])
    return data[:, 0], data[:, 1]


def orientations(n):
    """The golden-spiral hemisphere grid, as v(n) = (nx^2, ny^2, nz^2, 2nxny, 2nxnz, 2nynz)."""
    m = np.arange(n)
    z = (m + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * m
    nx, ny, nz = r * np.cos(phi), r * np.sin(phi), z
    return np.stack([nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * nx * nz, 2 * ny * nz], axis=1)


def spectrum(tensors, grid, sigma, norient, reference, slope):
    v = orientations(norient)
    omega = tensors @ v.T                      # (natoms, norient)
    x = reference - slope * omega
    d = grid[None, None, :] - x[:, :, None]
    u = np.exp(-0.5 * (d / sigma) ** 2).sum(axis=(0, 1)) / norient
    return u / np.linalg.norm(u)


def main():
    dump, specfile, sigma, norient, reference, slope = sys.argv[1:7]
    tol = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-10

    tensors = read_tensors(dump)
    grid, y_lammps = read_spectrum(specfile)
    y_ref = spectrum(tensors, grid, float(sigma), int(norient), float(reference), float(slope))

    err = np.abs(y_ref - y_lammps).max()
    print(f"    {len(tensors)} tensors, {len(grid)} points, peak {y_ref.max():.6f}: "
          f"max |reference - LAMMPS| = {err:.3e}")
    if not np.isfinite(err) or err > tol:
        worst = int(np.argmax(np.abs(y_ref - y_lammps)))
        print(f"    worst at x = {grid[worst]:.4f}: {y_ref[worst]:.12g} vs {y_lammps[worst]:.12g}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
