"""Structural features for the selection's diversity term.

Features do not score anything. They are the space selection keeps
individuals apart in, which is what stops a Pareto run collapsing onto a few
near-identical structures -- the opposite of the point when the question is
how *many* structures fit one diffraction pattern.

EZGA's own `feature_species_pair_fraction` is trivially 1.0 for a
single-species cell, and `feature_average_interatomic_distance` and
`feature_radius_of_gyration` unpack the container as a tuple and raise. So
the one that matters for carbon is here: mean coordination separates an sp2
network from an sp3 one, and nothing in EZGA measures it.
"""

import numpy as np

SP2 = 3.0
SP3 = 4.0


def _frames(dataset):
    """(positions, cell) per structure, however the container offers them."""
    if hasattr(dataset, "get_all_positions") and hasattr(dataset, "get_all_lattice_vectors"):
        positions = dataset.get_all_positions()
        cells = dataset.get_all_lattice_vectors()
        return list(zip(positions, cells))
    out = []
    for structure in dataset:
        manager = getattr(structure, "AtomPositionManager", structure)
        out.append((np.asarray(manager.atomPositions, dtype=float),
                    np.asarray(manager.latticeVectors, dtype=float)))
    return out


def _coordination(positions, cell, cutoff):
    """Mean number of neighbours within cutoff, minimum image."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 2:
        return 0.0
    cell = np.asarray(cell, dtype=float)
    inv = np.linalg.inv(cell)
    delta = positions[:, None, :] - positions[None, :, :]
    fractional = delta @ inv
    fractional -= np.round(fractional)
    delta = fractional @ cell
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(np.count_nonzero(distance < cutoff) / n)


def feature_mean_coordination(cutoff=1.85):
    """Mean coordination number. ~3 for sp2, ~4 for sp3."""

    def compute(dataset):
        frames = _frames(dataset)
        out = np.zeros((len(frames), 1), dtype=float)
        for i, (positions, cell) in enumerate(frames):
            try:
                out[i, 0] = _coordination(positions, cell, cutoff)
            except (ValueError, np.linalg.LinAlgError):
                out[i, 0] = 0.0
        return out

    compute.get_feature_index_map = lambda: {"mean_coord": 0}
    compute.scaling_logic = "invariant"
    return compute


def feature_sp_fractions(cutoff=1.85):
    """Fraction of atoms with 3 and with 4 neighbours, as two columns."""

    def compute(dataset):
        frames = _frames(dataset)
        out = np.zeros((len(frames), 2), dtype=float)
        for i, (positions, cell) in enumerate(frames):
            positions = np.asarray(positions, dtype=float)
            n = len(positions)
            if n < 2:
                continue
            cell = np.asarray(cell, dtype=float)
            fractional = (positions[:, None, :] - positions[None, :, :]) @ np.linalg.inv(cell)
            fractional -= np.round(fractional)
            distance = np.linalg.norm(fractional @ cell, axis=-1)
            np.fill_diagonal(distance, np.inf)
            counts = np.count_nonzero(distance < cutoff, axis=1)
            out[i, 0] = np.count_nonzero(counts == SP2) / n
            out[i, 1] = np.count_nonzero(counts == SP3) / n
        return out

    compute.get_feature_index_map = lambda: {"sp2_frac": 0, "sp3_frac": 1}
    compute.scaling_logic = "invariant"
    return compute
