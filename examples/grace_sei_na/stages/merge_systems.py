"""Every system's train/test selection as one, names prefixed by system.

    python merge_systems.py RUNDIR OUTDIR SYSTEM...
"""

import sys
from pathlib import Path

import pandas as pd
from ase.io import write


def merged(rundir, systems, split):
    parts = []
    for system in systems:
        df = pd.read_pickle(rundir / system / "06_select" / f"{split}_unlabelled.pkl.gz", compression="gzip")
        df["name"] = [f"{system}-{n}" for n in df["name"]]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main(rundir, outdir, systems):
    outdir.mkdir(parents=True, exist_ok=True)
    to_label = []
    for split in ("train", "test"):
        df = merged(rundir, systems, split)
        df.to_pickle(outdir / f"{split}_unlabelled.pkl.gz", compression="gzip")
        for name, atoms in zip(df["name"], df["ase_atoms"]):
            atoms = atoms.copy()
            atoms.info = {"name": name}
            to_label.append(atoms)
        print(f"{split}: {len(df)}")
    write(outdir / "to_label.xyz", to_label, format="extxyz")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
