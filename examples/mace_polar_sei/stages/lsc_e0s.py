"""Per-element reference energies for mace-scf's --E0s, by least squares over the frames that have REF_energy.

    python lsc_e0s.py TRAIN.xyz        prints {Z: E0, ...}

mace-scf's own `average` fit includes frames without an energy and returns NaN. An element seen
only in such frames gets 0.
"""

import sys

import numpy as np
from ase.io import read


def main(path):
    frames = read(path, ":")
    elements = sorted({int(z) for a in frames for z in a.numbers})
    labelled = [a for a in frames if "REF_energy" in a.info]
    counts = np.array([[np.sum(a.numbers == z) for z in elements] for a in labelled], dtype=float)
    energies = np.array([a.info["REF_energy"] for a in labelled])
    seen = counts.sum(axis=0) > 0
    e0 = np.zeros(len(elements))
    e0[seen] = np.linalg.lstsq(counts[:, seen], energies, rcond=None)[0]
    print("{" + ", ".join(f"{z}: {e:.6f}" for z, e in zip(elements, e0)) + "}")


if __name__ == "__main__":
    main(*sys.argv[1:])
