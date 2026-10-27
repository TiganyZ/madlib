"""Per-atom formal charges for LSC-MACE (mace-scf's LocalSplitCharges), as the array `formal_charges`.

    python formal_charges.py IN.xyz OUT.xyz

Each covalent unit gets carve_clusters' formal charge by composition (Li+/Na+, F-, O2-, CO3 2-,
PF6-, EDC 2-, MC-, neutral EC/DMC), spread evenly over its most electronegative element (the O
of a carbonate, the F of PF6). Graphite and other unknown units are taken as neutral. A frame
whose formal charges do not sum to zero is dropped, since LSC conserves that total.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carve_clusters import unit_charges, unit_formulas, units_of  # noqa: E402

# Pauling electronegativity of the elements in these systems
ELECTRONEGATIVITY = {"H": 2.20, "Li": 0.98, "C": 2.55, "O": 3.44, "F": 3.98, "Na": 0.93, "P": 2.19}


def formal_charges(atoms):
    labels = units_of(atoms)
    charges = unit_charges(unit_formulas(atoms, labels))
    symbols = np.array(atoms.get_chemical_symbols())
    q = np.zeros(len(atoms))
    for unit, charge in charges.items():
        if not charge:
            continue
        members = np.flatnonzero(labels == unit)
        top = max(ELECTRONEGATIVITY[s] for s in symbols[members])
        carriers = members[[ELECTRONEGATIVITY[s] == top for s in symbols[members]]]
        q[carriers] = charge / len(carriers)
    return q


def main(src, dst):
    kept, dropped = [], 0
    for atoms in read(src, ":"):
        q = formal_charges(atoms)
        if abs(q.sum()) > 1e-6:
            dropped += 1
            continue
        atoms.new_array("formal_charges", q)
        kept.append(atoms)
    write(dst, kept, format="extxyz")
    print(f"{len(kept)} frames with formal charges -> {dst}; {dropped} dropped (formal charges do not sum to 0)")


if __name__ == "__main__":
    main(*sys.argv[1:])
