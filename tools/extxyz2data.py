#!/usr/bin/env python3
"""extended-xyz frame -> LAMMPS data file, keeping the triclinic cell."""
import re, sys
import numpy as np

def read(path):
    L = open(path).read().split('\n')
    n = int(L[0].split()[0])
    m = re.search(r'Lattice="([^"]+)"', L[1])
    cell = np.array([float(v) for v in m.group(1).split()]).reshape(3, 3)
    sp, pos = [], []
    for line in L[2:2 + n]:
        w = line.split()
        sp.append(w[0]); pos.append([float(v) for v in w[1:4]])
    return sp, np.array(pos), cell

def main():
    sp, pos, cell = read(sys.argv[1])
    kinds = sorted(set(sp), key=sp.index)
    a, b, c = cell
    xhi, xy, yhi, xz, yz, zhi = a[0], b[0], b[1], c[0], c[1], c[2]
    assert abs(a[1]) < 1e-10 and abs(a[2]) < 1e-10 and abs(b[2]) < 1e-10, "cell not lower-triangular"

    with open(sys.argv[2], 'w') as f:
        f.write("# from %s\n\n%d atoms\n%d atom types\n\n" % (sys.argv[1], len(sp), len(kinds)))
        f.write("0.0 %.12f xlo xhi\n0.0 %.12f ylo yhi\n0.0 %.12f zlo zhi\n" % (xhi, yhi, zhi))
        if abs(xy) > 1e-12 or abs(xz) > 1e-12 or abs(yz) > 1e-12:
            f.write("%.12f %.12f %.12f xy xz yz\n" % (xy, xz, yz))
        f.write("\nAtoms # atomic\n\n")
        for i, (s, p) in enumerate(zip(sp, pos)):
            f.write("%d %d %.12f %.12f %.12f\n" % (i + 1, kinds.index(s) + 1, p[0], p[1], p[2]))
    print(" ".join(kinds))

main()
