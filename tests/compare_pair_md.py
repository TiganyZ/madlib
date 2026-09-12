#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Trajectories after a few MD steps, against pair_style turbogap with newton on.

Why steps and not `run 0`. LAMMPS's Verlet::force_clear() zeroes the ghost
range of f only when newton is on. A style that writes ghost forces and folds
them back itself therefore sees a clean array on the first step and last
step's leftovers on every one after -- exact at step 0, and wrong from step 1
by as much as 16 eV/A. Every `run 0` check in this suite passed throughout.

usage: compare_pair_md.py <ref_dump> <ref_log> <label> <dump> <log> [...]
"""
import sys

F_TOL = 1e-6     # eV/A, absolute. The bug this guards against is ~1e+01.
E_TOL = 1e-6     # eV, on the final potential energy.


def final_forces(path):
    """Forces from the LAST frame of a dump."""
    body = open(path).read().split('ITEM: ATOMS')[-1].split('\n')[1:]
    return {int(l.split()[0]): [float(v) for v in l.split()[1:4]]
            for l in body if len(l.split()) == 4}


def final_energy(path):
    """The last step's potential energy from the thermo table."""
    pe = None
    for line in open(path):
        w = line.split()
        if len(w) == 2:
            try:
                step, val = int(w[0]), float(w[1])
            except ValueError:
                continue
            pe = val
    return pe


def compare(label, ref_f, ref_e, dump, log):
    f, e = final_forces(dump), final_energy(log)
    if not f or e is None:
        print(f"    {label}: no final frame -- the run did not finish")
        return False
    if set(f) != set(ref_f):
        print(f"    {label}: {len(f)} atoms against {len(ref_f)}")
        return False
    df = max(max(abs(f[i][c] - ref_f[i][c]) for c in range(3)) for i in ref_f)
    de = abs(e - ref_e)
    ok = df < F_TOL and de < E_TOL
    print(f"    {label}: max|dF| = {df:.2e} eV/A, dE = {de:.2e} eV   "
          f"{'ok' if ok else 'DIFFERS'}")
    return ok


def main():
    a = sys.argv[1:]
    ref_f, ref_e = final_forces(a[0]), final_energy(a[1])
    rest = a[2:]
    good = True
    for i in range(0, len(rest), 3):
        good &= compare(rest[i], ref_f, ref_e, rest[i + 1], rest[i + 2])
    sys.exit(0 if good else 1)


main()
