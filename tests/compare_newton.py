# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
# One thermo line per newton mode; the two must agree.
import sys

NAMES = ["pe", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]


def read(path):
    return open(path).read().split()


def worst_difference(on, off):
    worst = 0.0
    for i in range(1, len(NAMES) + 1):
        a, b = float(on[i]), float(off[i])
        worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1.0))
    return worst


def main():
    on, off = read(sys.argv[1]), read(sys.argv[2])
    if len(on) < 8 or len(off) < 8:
        print("    a run produced no thermo line")
        return 1
    worst = worst_difference(on, off)
    print(f"    pxx  newton on {float(on[2]):.6g}   newton off {float(off[2]):.6g}")
    print(f"    worst relative difference over {len(NAMES)} quantities: {worst:.2e}")
    # A virial that is zero in both modes agrees with itself, and is exactly
    # the failure this leg exists to catch.
    if max(abs(float(on[2])), abs(float(off[2]))) == 0.0:
        print("    the virial is zero in both modes")
        return 1
    return 0 if worst < 1e-12 else 1


sys.exit(main())
