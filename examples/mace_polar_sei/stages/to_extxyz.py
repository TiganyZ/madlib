"""grace_lifco's labelled dataframe (07_label/*.pkl.gz) as MACE extxyz.

    python to_extxyz.py IN.pkl.gz OUT.xyz

stress there is -virial/V in eV/A^3, which is ASE's sign; MACE reads it as REF_stress.
"""

import sys

import numpy as np
import pandas as pd
from ase.io import write


def frames(df):
    for _, row in df.iterrows():
        atoms = row["ase_atoms"].copy()
        atoms.calc = None
        atoms.info = {
            "name": row["name"],
            "config_type": str(row["name"]).split("-")[0],
            "REF_energy": float(row["energy"]),
            "REF_stress": np.asarray(row["stress"], dtype=float).reshape(9),
            "charge": 0,
            "spin": 1,
        }
        atoms.arrays["REF_forces"] = np.asarray(row["forces"], dtype=float)
        yield atoms


def main(src, dst):
    df = pd.read_pickle(src, compression="gzip")
    out = list(frames(df))
    write(dst, out, format="extxyz")
    print(f"{len(out)} frames -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
