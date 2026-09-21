#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""pair_style turbogap against TurboGAP's own energy and forces."""
import sys

ref, etot = {}, None
for line in open('data/pair_turbogap_ref.dat'):
    if line.startswith('#'):
        continue
    w = line.split()
    if w[0] == 'energy':
        etot = float(w[1])
        continue
    ref[int(w[0])] = [float(v) for v in w[1:4]]

body = open('pair_turbogap.dump').read().split('ITEM: ATOMS')[-1].split('\n')[1:]
got = {int(l.split()[0]): [float(v) for v in l.split()[1:4]] for l in body if len(l.split()) == 4}

missing = set(ref) - set(got)
if missing:
    print(f"    {len(missing)} atoms missing from the dump")
    sys.exit(1)

pe = None
for line in open('/tmp/pair_tg.log'):
    w = line.split()
    if len(w) == 2 and w[0] == '0':
        try:
            pe = float(w[1])
        except ValueError:
            pass

fmax = max(max(abs(v) for v in f) for f in ref.values())
dF = max(max(abs(got[i][k] - ref[i][k]) for k in range(3)) for i in ref)
print(f"    {len(ref)} atoms, max |F| = {fmax:.4f} eV/A")
print(f"    max |F_LAMMPS - F_TurboGAP| = {dF:.2e} eV/A   ({dF / fmax:.1e} relative)")

good = dF / fmax < 1e-7
if pe is None:
    print("    could not read the energy back from the log")
    good = False
else:
    dE = abs(pe - etot)
    print(f"    energy {pe:.8f} against {etot:.8f}   ({dE / abs(etot):.1e} relative)")
    good = good and dE / abs(etot) < 1e-7

sys.exit(0 if good else 1)
