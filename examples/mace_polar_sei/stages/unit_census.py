"""Covalent units before and after an MD stage, so a molecule that broke up is visible at once.

    python unit_census.py BEFORE.data AFTER.data "SPECIES"
"""

import sys
from collections import Counter
from pathlib import Path

from ase.data import atomic_numbers
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carve_clusters import CATIONS, unit_formulas, units_of  # noqa: E402

TOP = 10


def census(path, species):
    z = {i + 1: atomic_numbers[s] for i, s in enumerate(species.split())}
    atoms = read(path, format="lammps-data", atom_style="atomic", Z_of_type=z)
    formulas = unit_formulas(atoms, units_of(atoms)).values()
    return Counter(f for f in formulas if f not in CATIONS)


def main(before, after, species):
    a, b = census(before, species), census(after, species)
    print(f"units before: {dict(a.most_common(TOP))}")
    print(f"units after : {dict(b.most_common(TOP))}")
    lost = {f: n - b[f] for f, n in a.items() if b[f] < n}
    print(f"lost: {lost}" if lost else "every unit intact")


if __name__ == "__main__":
    main(*sys.argv[1:])
