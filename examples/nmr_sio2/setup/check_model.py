#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The tensors MADLIB read, against the ones the published file already holds.

    check_model.py <ms_tensors.dump> <aSiO2_models.xyz> [frame] [tolerance ppm]

ML_ISD_ms is the same model's prediction for the same frame, made by its
authors with nequip-evaluate, so comparing against it tests the graph, the
edge convention, the per-species de-standardisation, the spherical-to-Cartesian
transform and the Voigt ordering at once, against an oracle that owes nothing
to this code. Only the symmetric part: that is all a spectrum sees, and all
quantity=tensor6 returns.
"""

import sys

import numpy as np
from ase.io import read

VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def read_dump(path):
    body = open(path).read().split("ITEM: ATOMS")[-1].split("\n")[1:]
    rows = [r.split() for r in body if len(r.split()) == 8]
    rows.sort(key=lambda r: int(r[0]))
    return np.array([[float(v) for v in r[2:]] for r in rows])


def voigt(nine):
    t = nine.reshape(-1, 3, 3)
    t = 0.5 * (t + np.transpose(t, (0, 2, 1)))
    return np.stack([t[:, a, b] for a, b in VOIGT], axis=1)


def main():
    dump, xyz = sys.argv[1:3]
    frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

    atoms = read(xyz, index=frame)
    if "ML_ISD_ms" not in atoms.arrays:
        raise SystemExit(f"{xyz} frame {frame} has no ML_ISD_ms column to compare with")
    reference = voigt(atoms.arrays["ML_ISD_ms"])
    mine = read_dump(dump)
    if mine.shape != reference.shape:
        raise SystemExit(f"{dump} holds {mine.shape} where the file has {reference.shape}")

    d = np.abs(mine - reference)
    iso_ref = reference[:, :3].sum(axis=1) / 3.0
    iso_mine = mine[:, :3].sum(axis=1) / 3.0
    print(f"    {len(mine)} atoms, isotropic shielding {iso_ref.min():.2f} to "
          f"{iso_ref.max():.2f} ppm")
    print(f"    max |MADLIB - file| = {d.max():.4f} ppm   rms = {np.sqrt((d ** 2).mean()):.4f} ppm")
    print(f"    isotropic part:       max {np.abs(iso_mine - iso_ref).max():.4f} ppm")
    if not np.isfinite(d.max()) or d.max() > tol:
        worst = int(np.unravel_index(np.argmax(d), d.shape)[0])
        print(f"    worst atom {worst + 1}: {mine[worst]} vs {reference[worst]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
