"""Starting structures: a random LiF + Li2CO3 mixture, and strained/rattled crystals of both.

    python structures.py OUTDIR SEED
"""

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.spacegroup import crystal

N_CO3 = 16
N_LIF = 48
PACKING_DENSITY = 1.6  # g/cm3; NPT compresses it
MIN_DIST = 1.9
CO_BOND = 1.29
SPECIES = ["Li", "C", "O", "F"]

STRAINS = (-0.03, 0.0, 0.03)
RATTLES = (0.05, 0.10)


def li2co3():
    """Zabuyelite, C2/c."""
    return crystal(["Li", "C", "O", "O"],
                   basis=[(0.1965, 0.4484, 0.8350), (0.0, 0.0663, 0.25),
                          (0.0, 0.3218, 0.25), (0.1462, -0.0636, 0.3135)],
                   spacegroup=15, cellpar=[8.359, 4.973, 6.198, 90.0, 114.83, 90.0])


def lif():
    return bulk("LiF", "rocksalt", a=4.03, cubic=True).repeat(2)


def carbonate(rng):
    angles = np.radians([0.0, 120.0, 240.0])
    o = CO_BOND * np.stack([np.cos(angles), np.sin(angles), np.zeros(3)], axis=1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return np.vstack([[0.0, 0.0, 0.0], o @ q.T])


def box_length():
    amu_g = 1.66053907e-24
    mass = N_CO3 * 73.89 + N_LIF * 25.94
    return (mass * amu_g / PACKING_DENSITY * 1e24) ** (1 / 3)


def fits(trial, placed, cell):
    if not placed:
        return True
    d = trial[:, None, :] - np.asarray(placed)[None, :, :]
    d -= cell * np.round(d / cell)
    return np.linalg.norm(d, axis=2).min() > MIN_DIST


def place(units, cell, rng):
    symbols, positions = [], []
    for unit_symbols, make in units:
        for _ in range(10000):
            trial = make(rng) + rng.uniform(0, cell, size=3)
            if fits(trial, positions, cell):
                symbols += unit_symbols
                positions += list(trial)
                break
        else:
            raise RuntimeError("packing failed; lower PACKING_DENSITY")
    return symbols, positions


def mixture(rng):
    cell = box_length()
    units = [(["C", "O", "O", "O"], carbonate)] * N_CO3
    units += [(["Li"], lambda r: np.zeros((1, 3)))] * (2 * N_CO3 + N_LIF)
    units += [(["F"], lambda r: np.zeros((1, 3)))] * N_LIF
    symbols, positions = place(units, cell, rng)
    atoms = Atoms(symbols, positions=positions, cell=[cell] * 3, pbc=True)
    atoms.wrap()
    return atoms[np.argsort([SPECIES.index(s) for s in atoms.get_chemical_symbols()], kind="stable")]


def variants(base, name, rng):
    out = []
    for s in STRAINS:
        a = base.copy()
        a.set_cell(base.cell * (1 + s), scale_atoms=True)
        a.info["config_type"] = f"{name}_strain{s:+.2f}"
        out.append(a)
    for r in RATTLES:
        a = base.copy()
        a.positions += rng.normal(scale=r, size=a.positions.shape)
        a.info["config_type"] = f"{name}_rattle{r:.2f}"
        out.append(a)
    return out


def main(outdir, seed):
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    mix = mixture(rng)
    write(outdir / "mixture.data", mix, format="lammps-data", specorder=SPECIES, masses=True)
    crystals = variants(li2co3(), "Li2CO3", rng) + variants(lif(), "LiF", rng)
    write(outdir / "crystals.xyz", crystals)
    print(f"mixture {len(mix)} atoms, L = {mix.cell[0, 0]:.3f} A; {len(crystals)} crystal cells")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
