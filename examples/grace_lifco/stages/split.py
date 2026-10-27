"""Seeded train/test split of the selected frames, stratified by MAD temperature; crystals go to train.

    python split.py SELECTED CRYSTALS TEST_FRACTION SEED OUTDIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write


def selected_atoms(path):
    df = pd.read_pickle(path, compression="gzip")
    return [a.copy() for a in df["ase_atoms"]], list(df["name"])


def stratified(names, fraction, rng):
    temps = np.array([n.split("_")[0] for n in names])
    test = []
    for t in sorted(set(temps)):
        idx = rng.permutation(np.where(temps == t)[0])
        test += list(idx[: int(round(fraction * len(idx)))])
    return sorted(test), sorted(set(range(len(names))) - set(test))


def split(selected, crystals, fraction, seed, outdir):
    atoms, names = selected_atoms(selected)
    test_idx, train_idx = stratified(names, fraction, np.random.default_rng(seed))
    test = pd.DataFrame({"name": [names[i] for i in test_idx],
                         "ase_atoms": [atoms[i] for i in test_idx]})
    xtal = read(crystals, index=":") if Path(crystals).exists() else []
    train = pd.DataFrame({"name": [names[i] for i in train_idx] + [a.info["config_type"] for a in xtal],
                          "ase_atoms": [atoms[i] for i in train_idx] + xtal})
    outdir.mkdir(parents=True, exist_ok=True)
    train.to_pickle(outdir / "train_unlabelled.pkl.gz", compression="gzip")
    test.to_pickle(outdir / "test_unlabelled.pkl.gz", compression="gzip")
    to_label = []
    for name, atoms in zip(list(train["name"]) + list(test["name"]), list(train["ase_atoms"]) + list(test["ase_atoms"])):
        atoms = atoms.copy()
        atoms.info = {"name": name}
        to_label.append(atoms)
    write(outdir / "to_label.xyz", to_label, format="extxyz")
    print(f"train {len(train)} ({len(xtal)} crystals), test {len(test)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    split(a[0], a[1], float(a[2]), int(a[3]), Path(a[4]))
