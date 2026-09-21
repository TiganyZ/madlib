#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Two SOAP descriptor dumps, from the CPU and GPU soap_turbo.

The two come from separate binaries because the implementations share a module
name and entry point. Same source, same configuration, so any difference here
is the implementation.

Tolerance: the observed agreement is ~1e-12 absolute on components of order
0.3. 1e-9 leaves three orders over that and is far below anything meaningful --
a 0.1% error in a single device hyperparameter moves every component by 6e-3,
which is what the control for this test does.
"""
import sys

TOL = 1e-9


def load(path):
    lines = open(path).read().split('\n')
    n_sites, dim = (int(x) for x in lines[0].split())
    vals = [float(x) for x in lines[1:1 + n_sites * dim]]
    if len(vals) != n_sites * dim:
        raise SystemExit(f"    {path}: expected {n_sites * dim} values, got {len(vals)}")
    return n_sites, dim, vals


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    ns, dim, a = load(a_path)
    ns2, dim2, b = load(b_path)
    if (ns, dim) != (ns2, dim2):
        print(f"    shapes differ: {ns}x{dim} against {ns2}x{dim2}")
        sys.exit(1)

    amax = max(abs(x) for x in a)
    # A dump of zeros would agree with another dump of zeros. Descriptors are
    # unit length per site, so this is a real floor, not a formality.
    if amax < 1e-6:
        print(f"    the host descriptor is all but zero (max {amax:.2e}) -- nothing compared")
        sys.exit(1)

    worst = max(abs(x - y) for x, y in zip(a, b))
    nbad = sum(1 for x, y in zip(a, b) if abs(x - y) > TOL)
    print(f"    {ns} sites x {dim} components, max|component| = {amax:.6f}")
    print(f"    max |device - host| = {worst:.3e}   ({worst / amax:.1e} relative)")
    print(f"    components over {TOL:g}: {nbad} of {len(a)}")
    sys.exit(0 if nbad == 0 else 1)


main()
