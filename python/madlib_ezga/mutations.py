"""Variation operators for EZGA that MADlib needs and EZGA does not have.

An EZGA mutation is `(structure, rng) -> structure`, mutating in place through
SAGE's setters so the cached inverse lattice and kdtree are invalidated.
"""

import numpy as np


def isotropic_strain(max_strain, seed=None):
    """Scale the cell and the positions by one random factor.

    EZGA's `mutation_random_strain` draws a full symmetric tensor, so it shears
    the cell triclinic. Two things follow. The quantity `compute mad/pdf` needs
    is then the distance between opposite faces, not the lattice-vector length,
    and it is smaller -- measured runs reached 17.5 A from an 18.4 A start,
    against the 16.0 A that an r_max of 8 requires. And the collision gate only
    rejects compression, so an anisotropic walk drifts towards expansion rather
    than sampling density evenly.

    One factor on all three axes keeps the cell cubic, which keeps density a
    single axis and matches the MAD and HRMC runs, whose cell never changes.
    """

    def mutate(structure, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        manager = structure.AtomPositionManager
        cell = np.asarray(manager.latticeVectors, dtype=float)
        positions = np.asarray(manager.atomPositions, dtype=float)
        if cell.shape != (3, 3) or positions.ndim != 2 or positions.shape[0] == 0:
            return structure

        factor = 1.0 + float(rng.uniform(-max_strain, max_strain))
        manager.set_latticeVectors(cell * factor, edit_positions=False)
        manager.set_atomPositions(positions * factor)
        return structure

    return mutate
