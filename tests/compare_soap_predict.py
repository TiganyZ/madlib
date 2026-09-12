#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""SOAP energies and forces from the host path against the device one.

Energies and forces are compared separately and never pooled. They differ in
magnitude by a couple of orders here, so a pooled maximum would let a real
force error sit under the energy scale unnoticed -- the same trap that let a
deleted core_pot term pass in compare_pair_kk.py.

The file is: a count line "n_energies n_forces", then the energies, then the
force components.
"""
import sys

TOL = 1e-9


def load(path):
    lines = open(path).read().split('\n')
    n_e, n_f = (int(x) for x in lines[0].split())
    vals = [float(x) for x in lines[1:1 + n_e + n_f] if x.strip()]
    if len(vals) != n_e + n_f:
        raise SystemExit(f"    {path}: expected {n_e + n_f} values, got {len(vals)}")
    return vals[:n_e], vals[n_e:]


def check(label, a, b):
    scale = max(abs(x) for x in a) if a else 0.0
    if scale < 1e-12:
        print(f"    {label}: the host side is all but zero ({scale:.2e}) -- nothing compared")
        return False
    worst = max(abs(x - y) for x, y in zip(a, b))
    nbad = sum(1 for x, y in zip(a, b) if abs(x - y) > TOL)
    print(f"    {label}: {len(a)} values, max|host| = {scale:.6f}, "
          f"max|device - host| = {worst:.3e} ({worst / scale:.1e} rel), over {TOL:g}: {nbad}")
    return nbad == 0


def main():
    e_a, f_a = load(sys.argv[1])
    e_b, f_b = load(sys.argv[2])
    if (len(e_a), len(f_a)) != (len(e_b), len(f_b)):
        print("    the two dumps have different shapes")
        sys.exit(1)
    ok = check("energies", e_a, e_b)
    ok = check("forces  ", f_a, f_b) and ok
    sys.exit(0 if ok else 1)


main()
