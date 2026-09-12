#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""turbogap/kk against turbogap: same configuration, same answer.

This is what establishes stage one of docs/TURBOGAP-KK.md. The Kokkos variant
replaces the neighbour build and nothing else, so with the compute held fixed
any difference here IS the neighbour list.
"""
import sys


def forces(path):
    body = open(path).read().split('ITEM: ATOMS')[-1].split('\n')[1:]
    return {int(l.split()[0]): [float(v) for v in l.split()[1:4]]
            for l in body if len(l.split()) == 4}


def energy(path):
    pe = None
    for line in open(path):
        w = line.split()
        if len(w) == 2 and w[0] == '0':
            try:
                pe = float(w[1])
            except ValueError:
                pass
    return pe


host, dev = forces('pair_turbogap.dump'), forces('pair_turbogap_kk.dump')
if set(host) != set(dev):
    print(f"    different atom sets: {len(host)} against {len(dev)}")
    sys.exit(1)

fmax = max(max(abs(v) for v in f) for f in host.values())
dF = max(max(abs(dev[i][k] - host[i][k]) for k in range(3)) for i in host)
eh, ed = energy('/tmp/pair_tg.log'), energy('/tmp/pair_tg_kk.log')

print(f"    {len(host)} atoms, max |F| = {fmax:.4f} eV/A")
print(f"    max |F_kk - F_host| = {dF:.2e} eV/A   ({dF / fmax:.1e} relative)")

# NOT bitwise, and it cannot be. The two runs build their neighbour lists
# differently -- LAMMPS's CPU binning against Kokkos's -- so a site's
# neighbours are enumerated in a different order, and the SOAP sum over them
# reassociates. The observed difference is ~2e-09 relative, the same level at
# which the device and host TurboGAP builds differ from each other, and four
# orders below anything that changes a trajectory.
#
# The energies DO agree exactly, because the energy is a sum over sites and
# each site's own sum is ordered the same way; it is the per-pair force
# adjoint that reorders.
good = dF / fmax < 1e-8
if eh is None or ed is None:
    print("    could not read one of the energies back")
    good = False
else:
    dE = abs(ed - eh)
    print(f"    energy {ed:.8f} against {eh:.8f}   (difference {dE:.2e})")
    good = good and dE < 1e-8

sys.exit(0 if good else 1)
