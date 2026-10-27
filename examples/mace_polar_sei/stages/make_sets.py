"""The training files: periodic frames + clusters for PolarMACE, clusters alone for the dipole and dielectric fits.

    python make_sets.py PERIODIC_TRAIN PERIODIC_TEST CLUSTERS OUT_DIR TEST_FRACTION CLUSTER_ENERGY_WEIGHT [SEED]

Writes OUT_DIR/{polar,clusters}_{train,test}.xyz.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write


def split(frames, test_fraction, seed):
    order = np.random.default_rng(seed).permutation(len(frames))
    n_test = int(round(test_fraction * len(frames)))
    test = set(order[:n_test].tolist())
    return ([f for i, f in enumerate(frames) if i not in test],
            [f for i, f in enumerate(frames) if i in test])


def as_cluster(atoms, energy_weight):
    atoms = atoms.copy()
    atoms.pbc = False
    atoms.info.update(config_type="cluster", charge=0, spin=1, config_energy_weight=energy_weight)
    return atoms


def main(periodic_train, periodic_test, clusters, out_dir, test_fraction, energy_weight, seed="0"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clusters = [as_cluster(c, float(energy_weight)) for c in read(clusters, ":")]
    c_train, c_test = split(clusters, float(test_fraction), int(seed))
    p_train, p_test = read(periodic_train, ":"), read(periodic_test, ":")
    sets = {
        "polar_train": p_train + c_train,
        "polar_test": p_test + c_test,
        "clusters_train": c_train,
        "clusters_test": c_test,
    }
    for name, frames in sets.items():
        write(out / f"{name}.xyz", frames, format="extxyz")
        print(f"{name:15s} {len(frames)} frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
