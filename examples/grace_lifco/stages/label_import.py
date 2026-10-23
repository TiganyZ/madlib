"""Attach VASP labels harvested by autoplex_soap_turbo (workflows/lifco_mad) to the selected frames.

    python label_import.py LABELLED.xyz IN.pkl.gz OUT.pkl.gz
"""

import sys

import numpy as np
import pandas as pd
from ase.io import read


def labels_by_name(path):
    out = {}
    for atoms in read(path, index=":"):
        virial = np.asarray(atoms.info["REF_virial"], dtype=float).reshape(3, 3)
        out[atoms.info["name"]] = {
            "energy": float(atoms.info["REF_energy"]),
            "forces": np.asarray(atoms.arrays["REF_forces"]),
            "stress": -virial / atoms.get_volume(),
        }
    return out


def attach(labelled, src, dst):
    labels = labels_by_name(labelled)
    df = pd.read_pickle(src, compression="gzip")
    missing = [n for n in df["name"] if n not in labels]
    if missing:
        print(f"dropping {len(missing)} frames without a VASP label: {missing[:5]}")
    df = df[df["name"].isin(labels)].reset_index(drop=True)
    for key in ("energy", "forces", "stress"):
        df[key] = [labels[n][key] for n in df["name"]]
    df.to_pickle(dst, compression="gzip")
    print(f"labelled {len(df)} structures from {labelled}")


if __name__ == "__main__":
    attach(*sys.argv[1:])
