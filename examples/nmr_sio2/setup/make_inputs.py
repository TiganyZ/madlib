#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Turn one downloaded silica model into the deck's inputs.

    make_inputs.py <aSiO2_models.xyz> <outdir> [frame] [reference ppm]

    silica.data           type 1 = Si, 2 = O
    silica_rattled.data   every atom displaced: what the biased run starts from
    si29_dft.dat          the 29Si powder lineshape of the file's DFT tensors,
                          which is the 'experiment'
    si29_ml_in_file.dat   the same from the file's own ML prediction, which is
                          as close as the bias can get

The lineshape is the one fix mad/nmr/aniso predicts: a Gaussian per
crystallite orientation at omega = n . T . n over a golden-spiral hemisphere
grid, on a shift axis, delta = sigma_ref - sigma.
"""

import sys

import numpy as np
from ase.io import read

GRID = (-140.0, -60.0, 201)      # 29Si shift window for amorphous silica, ppm
SIGMA = 2.0                      # ppm, instrumental broadening
ORIENTATIONS = 4096              # the target is not where to economise
RATTLE = 0.12                    # Angstrom, the distortion the bias has to undo
SEED = 20261107
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def voigt(tensors):
    """(n, 9) rows -> the six components of the symmetric part, Voigt order."""
    t = tensors.reshape(-1, 3, 3)
    t = 0.5 * (t + np.transpose(t, (0, 2, 1)))
    return np.stack([t[:, a, b] for a, b in VOIGT], axis=1)


def orientations(n):
    m = np.arange(n)
    z = (m + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * m
    nx, ny, nz = r * np.cos(phi), r * np.sin(phi), z
    return np.stack([nx * nx, ny * ny, nz * nz, 2 * nx * ny, 2 * nx * nz, 2 * ny * nz], axis=1)


def powder(tensors, reference, grid):
    """The static powder lineshape of a set of shielding tensors, l2-normalised."""
    v = orientations(ORIENTATIONS)
    shift = reference - voigt(tensors) @ v.T
    d = grid[None, None, :] - shift[:, :, None]
    u = np.exp(-0.5 * (d / SIGMA) ** 2).sum(axis=(0, 1)) / ORIENTATIONS
    return u / np.linalg.norm(u)


def write_spectrum(path, grid, y, origin):
    with open(path, "w") as fh:
        fh.write(f"# shift/ppm  intensity  ({origin})\n")
        for x, v in zip(grid, y):
            fh.write(f"{x:.6f} {v:.12e}\n")


def write_data(path, atoms, origin):
    cell = atoms.get_cell()
    if not np.allclose(cell, np.diag(np.diagonal(cell))):
        raise SystemExit("this script writes orthorhombic cells only")
    kinds = {"Si": 1, "O": 2}
    with open(path, "w") as fh:
        fh.write(f"# {origin}\n\n{len(atoms)} atoms\n2 atom types\n\n")
        for k, length in zip("xyz", np.diagonal(cell)):
            fh.write(f"0.0 {length:.12f} {k}lo {k}hi\n")
        fh.write("\nMasses\n\n1 28.0855\n2 15.9994\n\nAtoms # atomic\n\n")
        for i, (sym, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), 1):
            fh.write(f"{i} {kinds[sym]} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}\n")


def main():
    xyz, outdir = sys.argv[1:3]
    frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    reference = float(sys.argv[4]) if len(sys.argv) > 4 else 328.4

    atoms = read(xyz, index=frame)
    symbols = np.array(atoms.get_chemical_symbols())
    si = symbols == "Si"
    grid = np.linspace(*GRID[:2], GRID[2])
    origin = f"{xyz}, frame {frame}, sigma_ref = {reference} ppm"

    write_data(f"{outdir}/silica.data", atoms, origin)
    rattled = atoms.copy()
    rattled.rattle(stdev=RATTLE, seed=SEED)
    write_data(f"{outdir}/silica_rattled.data",
               rattled, f"{origin}, every atom rattled by {RATTLE} A (seed {SEED})")

    for key, name, what in (("QM_ms", "si29_dft.dat", "DFT shielding tensors"),
                            ("ML_ISD_ms", "si29_ml_in_file.dat", "the paper's own prediction")):
        if key not in atoms.arrays:
            print(f"  (no {key} in this frame, skipping {name})")
            continue
        y = powder(atoms.arrays[key][si], reference, grid)
        write_spectrum(f"{outdir}/{name}", grid, y,
                       f"29Si static powder lineshape from {what}; {origin}")
        print(f"  {name}: peak {y.max():.4f} at {grid[int(np.argmax(y))]:.1f} ppm")

    print(f"  {len(atoms)} atoms, {si.sum()} Si, cell {np.diagonal(atoms.get_cell())[0]:.3f} A")


if __name__ == "__main__":
    main()
