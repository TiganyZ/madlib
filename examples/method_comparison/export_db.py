#!/usr/bin/env python3
"""Export every structure an EZGA run evaluated, as LAMMPS data files.

The per-individual work directory holds `start.data`, which is the structure
*before* its relaxation, and the runner reuses those directories. So the only
copy of a relaxed structure is the one in the SAGE database, and analyse.py
scores files. Without this step the GA legs contribute nothing to the
comparison -- or worse, whichever scratch file happened to exist.

Needs the environment the GA ran in: sage_lib and ase.
"""

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from sage_lib.partition.Partition import Partition


def export(db_path, out_dir):
    db = Partition(storage="hybrid", local_root=str(db_path), access="ro")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    try:
        for i in db.get_ids():
            manager = db.get_container(int(i)).AtomPositionManager
            positions = np.asarray(manager.atomPositions, dtype=float)
            cell = np.asarray(manager.latticeVectors, dtype=float)
            symbols = [str(s) for s in manager.atomLabelsList]
            if len(positions) == 0 or abs(np.linalg.det(cell)) < 1e-9:
                skipped += 1
                continue
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            write(str(out_dir / f"id_{int(i):06d}.data"), atoms,
                  format="lammps-data", masses=True, specorder=sorted(set(symbols)))
            written += 1
    finally:
        db.close()
    return written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    written, skipped = export(args.db, args.out)
    print(f"{written} structures -> {args.out}" + (f"  ({skipped} skipped)" if skipped else ""))


main()
