#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Check mad_shiftml_span against ShiftML's own tensor API.

mad_shiftml reconstructs the Cartesian shielding tensor from the model's
irreducible components and mad_shiftml_span takes its largest principal value
minus its smallest. shiftml.ase exposes get_cs_tensor, which reaches the same
tensor by its own route. A wrong reconstruction still produces a perfectly
plausible spectrum, so this is checked rather than assumed.

    nmr_span_reference.py <span dump> <structure.data> <tol ppm>
"""

import os
import sys

import numpy as np


def read_dump(path):
    txt = open(path).read()
    rows = [l.split() for l in txt.split("ITEM: ATOMS")[1].split("\n")[1:] if l.strip()]
    rows.sort(key=lambda r: int(r[0]))
    return np.array([float(r[-1]) for r in rows])


def read_data(path, names):
    """(symbols, positions, cell) of a LAMMPS data file, in id order."""
    box, tilt, rows = {}, (0.0, 0.0, 0.0), []
    section = None
    for line in open(path):
        bare = line.split("#")[0].strip()
        w = bare.split()
        if len(w) == 4 and w[2:] == ["xlo", "xhi"]:
            box["x"] = float(w[1]) - float(w[0])
        elif len(w) == 4 and w[2:] == ["ylo", "yhi"]:
            box["y"] = float(w[1]) - float(w[0])
        elif len(w) == 4 and w[2:] == ["zlo", "zhi"]:
            box["z"] = float(w[1]) - float(w[0])
        elif len(w) == 6 and w[3:] == ["xy", "xz", "yz"]:
            tilt = tuple(float(v) for v in w[:3])
        elif bare in ("Atoms", "Masses", "Velocities"):
            section = bare
        elif section == "Atoms" and len(w) >= 5:
            rows.append((int(w[0]), int(w[1]), [float(v) for v in w[2:5]]))
    xy, xz, yz = tilt
    cell = np.array([[box["x"], 0, 0], [xy, box["y"], 0], [xz, yz, box["z"]]])
    rows.sort()
    return ([names[t - 1] for _, t, _ in rows],
            np.array([p for _, _, p in rows]), cell)


def reference(symbols, pos, cell):
    from ase import Atoms
    from shiftml.ase import ShiftML

    model = ShiftML(os.environ.get("MADLIB_SHIFTML_MODEL", "ShiftML41"),
                    device=os.environ.get("MADLIB_SHIFTML_DEVICE", "cpu"))
    t = model.get_cs_tensor(Atoms(symbols=symbols, positions=pos, cell=cell,
                                  pbc=True), return_symmetric=True).reshape(-1, 3, 3)
    lam = np.linalg.eigvalsh(0.5 * (t + t.transpose(0, 2, 1)))
    return lam[:, 2] - lam[:, 0]


def main(argv):
    dump, data, tol = argv[0], argv[1], float(argv[2])
    names = os.environ["MADLIB_SHIFTML_TYPES"].split()
    symbols, pos, cell = read_data(data, names)
    want = reference(symbols, pos, cell)
    got = read_dump(dump)
    d = np.abs(got - want)
    print("    %d spans vs shiftml's own get_cs_tensor: max|d| = %.3e ppm  "
          "over %.1f .. %.1f" % (len(got), d.max(), want.min(), want.max()))
    return 0 if d.max() < tol else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
