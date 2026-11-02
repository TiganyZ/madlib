"""Every system's MAD frames (grace_lifco 06_select/pool*.pkl.gz) minus the periodic test set, as extxyz.

    python pool_frames.py GRACE_RUNDIR TEST_UNLABELLED.pkl.gz OUT.xyz SYSTEM...
"""

import sys
from pathlib import Path

import pandas as pd
from ase.io import write


def frames(rundir, systems, test_names):
    out = []
    for system in systems:
        for pool in sorted((rundir / system / "06_select").glob("pool*.pkl.gz")):
            df = pd.read_pickle(pool, compression="gzip")
            for name, atoms in zip(df["name"], df["ase_atoms"]):
                name = f"{system}-{name}"
                if name in test_names:
                    continue
                atoms = atoms.copy()
                atoms.calc = None
                atoms.info = {"name": name}
                out.append(atoms)
    return out


def main(rundir, test, dst, *systems):
    test_names = set(pd.read_pickle(test, compression="gzip")["name"])
    out = frames(Path(rundir), systems, test_names)
    write(dst, out, format="extxyz")
    print(f"{len(out)} frames from {len(systems)} systems ({len(test_names)} test frames left out) -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
