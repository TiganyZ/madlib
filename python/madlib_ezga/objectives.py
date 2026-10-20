"""EZGA objectives backed by MADlib.

An EZGA objective is a plain callable over the whole dataset returning one
number per structure, lower being better. These are factories: call the
factory, pass the result to `evaluator.objectives_funcs`.

The name matters. EZGA decides whether an objective must see the whole
population by looking for "hull", "formation", "novelty", "similarity",
"anomality", "fit" or "regression" in the function, module and class names.
These are per-structure, so none of those substrings may appear -- which is
why this is rel_error and not misfit.
"""

import numpy as np

MISSING = np.inf


def _metadata(dataset, key):
    """Per-structure metadata, by whichever access the container offers."""
    if hasattr(dataset, "get_all_metadata"):
        values = dataset.get_all_metadata(key)
        if values is not None:
            return list(values)
    out = []
    for structure in dataset:
        manager = getattr(structure, "AtomPositionManager", structure)
        meta = getattr(manager, "metadata", None) or {}
        out.append(meta.get(key))
    return out


def _n_atoms(dataset):
    if hasattr(dataset, "get_all_positions"):
        return np.array([len(p) for p in dataset.get_all_positions()], dtype=float)
    return np.array(
        [len(getattr(s, "AtomPositionManager", s).atomPositions) for s in dataset],
        dtype=float,
    )


def objective_xrd_rel_error(key="xrd_rel_error", missing=MISSING):
    """MADlib's dimensionless weighted-L2 residual against the target pattern.

    Read from the metadata the calculator already attached, so scoring costs
    nothing. A structure the calculator never saw scores `missing`, which
    keeps it out of the front rather than poisoning selection with NaN.
    """

    def compute(dataset):
        values = _metadata(dataset, key)
        out = np.full(len(values), float(missing))
        for i, value in enumerate(values):
            if value is None:
                continue
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                continue
        return np.nan_to_num(out, nan=float(missing), posinf=float(missing))

    compute.scaling_logic = "invariant"
    return compute


def objective_energy_per_atom(scale=1.0):
    """Energy per atom, so cells of different size stay comparable."""

    def compute(dataset):
        energies = np.asarray(dataset.get_all_energies(), dtype=float)
        counts = _n_atoms(dataset)
        counts[counts == 0] = 1.0
        return scale * np.nan_to_num(energies / counts, nan=0.0)

    compute.scaling_logic = "invariant"
    return compute
