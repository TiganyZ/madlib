"""The synthetic experiment: MD with the labeller, frames written as a LAMMPS dump.

    python reference_md.py START MODEL TEMP EQUIL SAMPLE EVERY SEED OUT
"""

import sys

import numpy as np
from ase import units
from ase.data import atomic_numbers
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from tensorpotential.calculator import TPCalculator

TIMESTEP_FS = 0.5
FRICTION = 0.01  # 1/fs
SPECIES = ["Li", "C", "O", "F"]


def load(path):
    z = {i + 1: atomic_numbers[s] for i, s in enumerate(SPECIES)}
    return read(path, format="lammps-data", atom_style="atomic", Z_of_type=z)


def write_frame(f, step, atoms):
    cell = atoms.cell.array
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-8:
        raise ValueError("orthogonal cells only")
    pos = atoms.get_positions(wrap=True)
    types = [SPECIES.index(s) + 1 for s in atoms.get_chemical_symbols()]
    f.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
    f.write("ITEM: BOX BOUNDS pp pp pp\n")
    for k in range(3):
        f.write(f"0.0 {cell[k, k]:.10f}\n")
    f.write("ITEM: ATOMS id type element x y z\n")
    for i, (t, s, p) in enumerate(zip(types, atoms.get_chemical_symbols(), pos)):
        f.write(f"{i + 1} {t} {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def run(start, model, temp, equil, sample, every, seed, out):
    rng = np.random.default_rng(seed)
    atoms = load(start)
    atoms.calc = TPCalculator(model)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, TIMESTEP_FS * units.fs, temperature_K=temp,
                   friction=FRICTION / units.fs, rng=rng)
    dyn.run(equil)
    with open(out, "w") as f:
        for step in range(0, sample + 1, every):
            if step:
                dyn.run(every)
            write_frame(f, step, atoms)
            print(f"step {step} T {atoms.get_temperature():.1f} K "
                  f"E {atoms.get_potential_energy():.4f} eV", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], float(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), a[7])
