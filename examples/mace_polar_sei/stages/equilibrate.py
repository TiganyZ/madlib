"""Relax one graphite | SEI | electrolyte slab, then NPT MD, with the fine-tuned PolarMACE + D3(BJ).

    python equilibrate.py STRUCTURE.xyz MODEL OUT_DIR MD_STEPS [DEVICE]

As the GRACE runs in ~/claude/graphite-sei-structures: x (the interface normal)
is held, y and z relax to zero stress, then NPT on y and z.
"""

import sys
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.md import MDLogger
from ase.md.melchionna import MelchionnaNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from ase.units import GPa, fs

from calc import polar_d3

DTYPE = "float32"
FMAX = 0.05             # eV/A
RELAX_STEPS = 2000
CELL_MASK = [False, True, True, False, False, False]   # xx yy zz yz xz xy
TEMP = 300.0            # K
TIMESTEP = 1.0          # fs
TTIME = 100.0           # fs
PTIME = 1000.0          # fs
BULK_MODULUS = 30.0     # GPa, only sets the barostat's response time
NPT_MASK = (0, 1, 1)
DUMP_EVERY = 100
SEED = 20261027


def relax(atoms, out):
    done = out / "relaxed.xyz"
    if done.is_file():
        return read(done)
    opt = FIRE(FrechetCellFilter(atoms, mask=CELL_MASK), logfile=str(out / "relax.log"))
    opt.run(fmax=FMAX, steps=RELAX_STEPS)
    write(done, atoms)
    return atoms


def npt(atoms, out, steps):
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    dyn = MelchionnaNPT(atoms, timestep=TIMESTEP * fs, temperature_K=TEMP, externalstress=0.0,
                        ttime=TTIME * fs, pfactor=(PTIME * fs) ** 2 * BULK_MODULUS * GPa, mask=NPT_MASK)
    dyn.attach(MDLogger(dyn, atoms, str(out / "md.log"), stress=True, peratom=True), interval=DUMP_EVERY)
    dyn.attach(lambda: write(out / "md.xyz", atoms, append=True), interval=DUMP_EVERY)
    dyn.run(steps)
    write(out / "final.xyz", atoms)


def main(structure, model, out_dir, md_steps, device="cuda"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.xyz").is_file():
        print(f"[cached] {out / 'final.xyz'}")
        return
    atoms = read(structure)
    atoms.calc = polar_d3(model, device, DTYPE)
    start = time.time()
    atoms = relax(atoms, out)
    atoms.calc = polar_d3(model, device, DTYPE)
    print(f"relaxed in {time.time() - start:.0f} s; cell {atoms.cell.lengths().round(2)}")
    start = time.time()
    npt(atoms, out, int(md_steps))
    print(f"{md_steps} MD steps in {time.time() - start:.0f} s; cell {atoms.cell.lengths().round(2)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
