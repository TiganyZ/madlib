"""Atom-level constraints for EZGA's variation operators.

A constraint is `(idx, structure) -> bool`. EZGA's `validate` applies it to
every atom and the operator touches only the ones that pass.
"""

import numpy as np


def random_subset(fraction, seed=0):
    """Pass a fresh random subset of the atoms, of expected size `fraction` * N.

    An all-atom rattle cannot pass a minimum-distance gate in a dense cell: the
    gate tests the closest pair out of N, so the chance that no pair falls under
    it goes to zero with N however small the displacement. Measured on the
    glassy-carbon population at a 0.912 A gate: 0.20 A on every atom is rejected
    every time, 0.20 A on 2% of them is accepted 92% of the time. Displacing a
    few atoms keeps the move big enough to matter and the child admissible.

    The draw is per atom and per call, so the subset is re-chosen every time the
    operator runs rather than fixed when it is built.
    """
    rng = np.random.default_rng(seed)

    def constraint(idx, structure):
        return bool(rng.random() < fraction)

    return constraint
