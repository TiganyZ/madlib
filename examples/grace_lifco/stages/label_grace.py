"""Label frames with a GRACE SavedModel: the stand-in for DFT.

    python label_grace.py MODEL IN.pkl.gz OUT.pkl.gz
"""

import sys

import numpy as np
import pandas as pd
from tensorpotential.calculator import TPCalculator


def label(model, src, dst):
    df = pd.read_pickle(src, compression="gzip")
    calc = TPCalculator(model)
    energy, forces, stress = [], [], []
    for atoms in df["ase_atoms"]:
        atoms = atoms.copy()
        atoms.calc = calc
        energy.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        stress.append(atoms.get_stress(voigt=False))
    df["energy"] = energy
    df["forces"] = forces
    df["stress"] = stress
    df["ase_atoms"] = [a.copy() for a in df["ase_atoms"]]
    df.to_pickle(dst, compression="gzip")
    fmax = max(np.abs(f).max() for f in forces)
    print(f"labelled {len(df)} structures, |F|max {fmax:.2f} eV/A")


if __name__ == "__main__":
    label(*sys.argv[1:])
