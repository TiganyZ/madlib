"""Errors of the foundation and the finetuned model against the labels.

    python validate.py LABELS.pkl.gz OUT.json NAME=MODEL.yaml ...
"""

import json
import sys

import numpy as np
import pandas as pd
from pyace.asecalc import PyGRACEFSCalculator

EV_A3_TO_GPA = 160.21766


def predict(df, model):
    calc = PyGRACEFSCalculator(model)
    e, f, s = [], [], []
    for atoms in df["ase_atoms"]:
        atoms = atoms.copy()
        atoms.calc = calc
        e.append(atoms.get_potential_energy())
        f.append(atoms.get_forces())
        s.append(atoms.get_stress(voigt=False))
    return np.array(e), f, np.array(s)


def element_shift(df, de):
    """Per-element energy offsets fitted by least squares: the model and the labels may use different zeros."""
    species = sorted({s for a in df["ase_atoms"] for s in a.get_chemical_symbols()})
    counts = np.array([[a.get_chemical_symbols().count(x) for x in species] for a in df["ase_atoms"]])
    mu, *_ = np.linalg.lstsq(counts, de, rcond=None)
    return de - counts @ mu


def errors(df, model):
    e, f, s = predict(df, model)
    n = df["ase_atoms"].map(len).values
    de = e - df["energy"].values
    df_ = np.concatenate([(p - r).ravel() for p, r in zip(f, df["forces"])])
    ds = (s - np.stack(df["stress"].values)) * EV_A3_TO_GPA
    return {
        "E_MAE_meV_atom": 1e3 * np.mean(np.abs(de / n)),
        "E_MAE_shifted_meV_atom": 1e3 * np.mean(np.abs(element_shift(df, de) / n)),
        "F_MAE_meV_A": 1e3 * np.mean(np.abs(df_)),
        "F_RMSE_meV_A": 1e3 * np.sqrt(np.mean(df_ ** 2)),
        "S_MAE_GPa": float(np.mean(np.abs(ds))),
    }


def main(labels, out, models):
    df = pd.read_pickle(labels, compression="gzip")
    result = {name: errors(df, path) for name, path in (m.split("=", 1) for m in models)}
    json.dump(result, open(out, "w"), indent=2)
    keys = list(next(iter(result.values())))
    print(f"{'':12s}" + "".join(f"{k:>24s}" for k in keys))
    for name, r in result.items():
        print(f"{name:12s}" + "".join(f"{r[k]:24.3f}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
