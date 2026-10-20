"""Pull the Pareto front out of a finished EZGA run.

EZGA keeps everything it evaluated in a SAGE database but writes no front of
its own -- the engine's final export is unreachable, because the flag it tests
is not a field its config model accepts. So the front is reconstructed here.
"""

import csv
from pathlib import Path

import numpy as np

from .features import _coordination

AMU_PER_CM3_TO_G = 1.66053906660


def non_dominated(objectives):
    """Indices of the non-dominated rows of an (N, K) array. Lower is better."""
    objectives = np.asarray(objectives, dtype=float)
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i when it is no worse everywhere and better somewhere
        no_worse = np.all(objectives <= objectives[i], axis=1)
        better = np.any(objectives < objectives[i], axis=1)
        dominated_by = no_worse & better
        if np.any(dominated_by):
            keep[i] = False
    return np.flatnonzero(keep)


def load_run(db_path, misfit_key="xrd_rel_error"):
    """Every evaluated structure as (ids, energies_per_atom, misfits, generations)."""
    from sage_lib.partition.Partition import Partition

    db = Partition(storage="hybrid", local_root=str(db_path), access="ro")
    try:
        ids = np.asarray(db.get_ids(), dtype=int)
        energies = np.asarray(db.get_all_energies(), dtype=float)
        misfits = np.asarray(
            [np.nan if v is None else float(v) for v in db.get_all_metadata(misfit_key)],
            dtype=float,
        )
        generations = db.get_all_metadata("generation")
        counts = np.asarray(
            [len(db.get_container(int(i)).AtomPositionManager.atomPositions) for i in ids],
            dtype=float,
        )
    finally:
        db.close()
    counts[counts == 0] = 1.0
    return ids, energies / counts, misfits, generations


def write_front(db_path, out_dir, misfit_key="xrd_rel_error"):
    """Write pareto.csv and pareto.xyz; return the front's (energy, misfit)."""
    from sage_lib.partition.Partition import Partition

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, energy_per_atom, misfits, generations = load_run(db_path, misfit_key)

    finite = np.isfinite(energy_per_atom) & np.isfinite(misfits)
    if not np.any(finite):
        raise ValueError("no structure has both an energy and a misfit")
    rows = np.flatnonzero(finite)
    front_local = non_dominated(np.column_stack([energy_per_atom[rows], misfits[rows]]))
    front = rows[front_local]
    order = np.argsort(misfits[front])
    front = front[order]

    # Coordination and density come from the structures themselves, in the
    # same pass that exports them: a front without them says which structures
    # trade off, not whether any of them is physically sensible -- which is
    # the question that separates a fit from a model.
    db = Partition(storage="hybrid", local_root=str(db_path), access="ro")
    rows = []
    try:
        path = str(out_dir / "pareto.xyz")
        for rank, i in enumerate(front):
            manager = db.get_container(int(ids[i])).AtomPositionManager
            positions = np.asarray(manager.atomPositions, dtype=float)
            cell = np.asarray(manager.latticeVectors, dtype=float)
            try:
                coordination = _coordination(positions, cell, 1.85)
                density = (len(positions) * 12.011 * AMU_PER_CM3_TO_G
                           / abs(np.linalg.det(cell)))
            except (ValueError, np.linalg.LinAlgError):
                coordination, density = float("nan"), float("nan")
            rows.append((ids[i], generations[i] if generations is not None else "",
                         energy_per_atom[i], misfits[i], coordination, density))
            manager.export_as_xyz(file_location=path, save_to_file="w" if rank == 0 else "a")
    finally:
        db.close()

    with open(out_dir / "pareto.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["db_id", "generation", "energy_per_atom_eV", misfit_key,
                         "coordination", "density"])
        for db_id, generation, energy, misfit, coordination, density in rows:
            writer.writerow([db_id, generation, f"{energy:.10g}", f"{misfit:.10g}",
                             f"{coordination:.6g}", f"{density:.6g}"])

    return np.column_stack([energy_per_atom[front], misfits[front]])
