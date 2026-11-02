"""MAD frames of the runs matching a glob, as one dataframe (name, ase_atoms), named by run and frame.

    python pool.py MADDIR GLOB OUT
"""

import sys
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.io import read

SKIP_FIRST = 1  # the starting structure is the same in every run


def frames(maddir, pattern):
    for dump in sorted(Path(maddir).glob(f"{pattern}/mad.dump")):
        run = dump.parent.name
        for k, atoms in enumerate(read(dump, index=":", format="lammps-dump-text")):
            if k < SKIP_FIRST:
                continue
            clean = Atoms(atoms.symbols, positions=atoms.positions, cell=atoms.cell, pbc=True)
            clean.info = {"name": f"{run}_{k:03d}", "run": run}
            yield clean


if __name__ == "__main__":
    pool = list(frames(sys.argv[1], sys.argv[2]))
    df = pd.DataFrame({"name": [a.info["name"] for a in pool], "ase_atoms": pool})
    df.to_pickle(sys.argv[3], compression="gzip")
    print(f"{len(pool)} frames from {len({a.info['run'] for a in pool})} runs")
