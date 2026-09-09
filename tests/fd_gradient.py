#!/usr/bin/env python3
"""Finite-difference check of a MADLIB bias: forces and virial from its own energy.

Displaces single coordinates and strains the cell, differences the reported
energy, and compares against the reported force and virial. The point is the
h-scan: a correct analytic gradient converges as h^2 until round-off, whereas a
wrong constant factor gives a flat curve that a single tolerance would pass.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

TEMPLATE = """units           metal
atom_style      atomic
boundary        p p p
read_data       {data}
mass            * 12.011
pair_style      zero {paircut}
pair_coeff      * *
neighbor        0.5 bin
{extra}
compute vir all pressure NULL virial
thermo_style    custom step f_mad c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6]
thermo_modify   format float %.17g
dump  d all custom 1 {dump} id fx fy fz
dump_modify d format float %.17g sort id
run 0
"""


def read_data(path):
    lines = open(path).read().split('\n')
    box, tilt, atoms, hdr = {}, None, [], []
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if len(w) >= 4 and w[2] in ('xlo', 'ylo', 'zlo'):
            box[w[2][0]] = (float(w[0]), float(w[1]))
        elif len(w) >= 6 and w[3] == 'xy':
            tilt = [float(w[0]), float(w[1]), float(w[2])]
        elif lines[i].startswith('Atoms'):
            i += 2
            while i < len(lines) and lines[i].strip():
                w = lines[i].split()
                atoms.append([int(w[0]), int(w[1]), float(w[2]), float(w[3]), float(w[4])])
                i += 1
            break
        i += 1
    return box, tilt, atoms


def write_data(path, box, tilt, atoms, ntypes):
    with open(path, 'w') as f:
        f.write("# fd\n\n%d atoms\n%d atom types\n\n" % (len(atoms), ntypes))
        for k in 'xyz':
            f.write("%.17g %.17g %slo %shi\n" % (box[k][0], box[k][1], k, k))
        if tilt:
            f.write("%.17g %.17g %.17g xy xz yz\n" % tuple(tilt))
        f.write("\nAtoms # atomic\n\n")
        for a in atoms:
            f.write("%d %d %.17g %.17g %.17g\n" % tuple(a))


def run(lmp, data, extra, paircut, tmp):
    inp = os.path.join(tmp, "in.fd")
    dmp = os.path.join(tmp, "fd.dump")
    open(inp, 'w').write(TEMPLATE.format(data=data, extra=extra, dump=dmp, paircut=paircut))
    out = subprocess.run([lmp, "-in", inp, "-log", "none"], capture_output=True, text=True,
                         cwd=os.getcwd())
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit("lammps failed")
    e, vir = None, None
    for ln in out.stdout.split('\n'):
        f = ln.split()
        if len(f) == 8 and f[0] == '0':
            e = float(f[1])
            # compute pressure returns stress; the virial is that times volume,
            # and LAMMPS reports pressure in bar under metal units.
            vir = [float(v) for v in f[2:]]
    forces = {}
    body = open(dmp).read().split('ITEM: ATOMS')[-1].split('\n')[1:]
    for ln in body:
        f = ln.split()
        if len(f) == 4:
            forces[int(f[0])] = [float(v) for v in f[1:]]
    return e, forces, vir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("extra", help="file holding the compute/fix lines under test")
    p.add_argument("--lmp", default="/home/tigany/Documents/lammps/lammps/build/lmp")
    p.add_argument("--paircut", type=float, default=6.5)
    p.add_argument("--atoms", type=int, default=5)
    p.add_argument("--hs", default="1e-3,3e-4,1e-4,3e-5")
    p.add_argument("--tol", type=float, default=2e-5)
    a = p.parse_args()

    extra = open(a.extra).read()
    box, tilt, atoms = read_data(a.data)
    ntypes = max(x[1] for x in atoms)
    tmp = tempfile.mkdtemp()
    base = os.path.join(os.path.dirname(os.path.abspath(a.data)) or ".", "_fd.data")

    write_data(base, box, tilt, atoms, ntypes)
    e0, f0, v0 = run(a.lmp, base, extra, a.paircut, tmp)
    vol0 = (box['x'][1]-box['x'][0])*(box['y'][1]-box['y'][0])*(box['z'][1]-box['z'][0])
    print(f"  E = {e0:.12g}")

    rng = np.random.default_rng(7)
    sel = sorted(rng.choice(len(atoms), size=min(a.atoms, len(atoms)), replace=False).tolist())
    hs = [float(v) for v in a.hs.split(',')]

    print("\n  forces:  max|analytic - FD| over %d atoms" % len(sel))
    prev, ok, scan = None, True, []
    for h in hs:
        worst = 0.0
        for idx in sel:
            for d in range(3):
                num = []
                for s in (+1, -1):
                    pert = [list(x) for x in atoms]
                    pert[idx][2 + d] += s * h
                    write_data(base, box, tilt, pert, ntypes)
                    num.append(run(a.lmp, base, extra, a.paircut, tmp)[0])
                fd = -(num[0] - num[1]) / (2 * h)
                worst = max(worst, abs(fd - f0[atoms[idx][0]][d]))
        rate = "" if prev is None else "  ratio %.2f" % (prev / worst if worst else float('inf'))
        print(f"    h = {h:8.1e}   max|df| = {worst:.3e}{rate}")
        prev = worst
        scan.append((h, worst))

    # The criterion is the shape of the scan, not a tolerance at every step: a
    # large h legitimately has a large truncation error, and demanding it be
    # small would only reward a gradient that is wrong in a way that happens to
    # cancel. What a correct gradient must do is converge -- order ~2 -- or
    # already sit at the round-off floor.
    h0, e0 = scan[0]
    h1, e1 = scan[-1]
    order = (math.log(e0 / e1) / math.log(h0 / h1)) if (e1 > 0 and e0 > 0 and h0 != h1) else 0.0
    at_floor = e1 < 1e-9
    good = e1 < a.tol and (order > 1.5 or at_floor)
    print(f"\n  observed order {order:.2f} over h {h0:.1e} -> {h1:.1e}"
          f"{'  (at the round-off floor)' if at_floor else ''}")
    print("  (h^2 is order 2; a wrong constant factor gives order 0 and any single"
          " tolerance passes it)")
    ok = ok and good

    # virial, by straining the cell and every position affinely.
    # F = I + e S, so dE/de = -sum_ab W_ab S_ab. Only strains that keep the
    # cell lower-triangular are used, so S is diagonal or the xy shear.
    BAR = 1.0 / 1.6021765e6   # bar -> eV/A^3, LAMMPS metal units
    print("\n  virial:  analytic vs FD")
    for name, comp in (("xx", 0), ("yy", 1), ("zz", 2), ("xy", 3)):
        Wan = v0[comp] * BAR * vol0
        row = []
        for h in hs:
            num = []
            for sgn in (+1, -1):
                e = sgn * h
                bx = {k: list(v) for k, v in box.items()}
                tl = list(tilt) if tilt else [0.0, 0.0, 0.0]
                pert = [list(x) for x in atoms]
                if comp < 3:
                    k = 'xyz'[comp]
                    bx[k] = (bx[k][0], bx[k][0] + (bx[k][1] - bx[k][0]) * (1 + e))
                    if comp == 0: tl[0] *= (1 + e); tl[1] *= (1 + e)
                    if comp == 1: tl[2] *= (1 + e)
                    for x in pert: x[2 + comp] *= (1 + e)
                else:
                    yprd = bx['y'][1] - bx['y'][0]
                    tl[0] += e * yprd
                    tl[1] += e * tl[2]
                    for x in pert: x[2] += e * x[3]
                write_data(base, bx, tl, pert, ntypes)
                num.append(run(a.lmp, base, extra, a.paircut, tmp)[0])
            fd = -(num[0] - num[1]) / (2 * h)
            row.append((h, fd))
        best = row[-1][1]
        rel = abs(best - Wan) / max(abs(Wan), 1e-30)
        print(f"    {name}: analytic W = {Wan: .8e}   FD = {best: .8e}   rel = {rel:.2e}"
              f"   {'ok' if rel < 1e-3 else 'MISMATCH'}")
        ok = ok and rel < 1e-3
        for h, fd in row: print(f"         h = {h:8.1e}  FD = {fd: .8e}")

    write_data(base, box, tilt, atoms, ntypes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
