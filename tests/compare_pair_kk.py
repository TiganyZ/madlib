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


def term(path, label):
    """(blocks, running energy) after a term, from the TURBOGAP_TERMS report.

    The line is "  after <label>: blocks=N E=<x>".
    """
    for line in open(path):
        if label in line and 'blocks=' in line:
            try:
                blocks = int(line.split('blocks=')[1].split()[0])
                return blocks, float(line.rsplit('=', 1)[1])
            except (ValueError, IndexError):
                pass
    return None, None


# The four paths, so the same comparison can run against a second
# configuration. Defaults are the glassy-carbon cell.
argv = sys.argv[1:]
h_dump, k_dump, h_log, k_log = (argv + ['pair_turbogap.dump', 'pair_turbogap_kk.dump',
                                        '/tmp/pair_tg.log', '/tmp/pair_tg_kk.log'][len(argv):])

host, dev = forces(h_dump), forces(k_dump)
if set(host) != set(dev):
    print(f"    different atom sets: {len(host)} against {len(dev)}")
    sys.exit(1)

fmax = max(max(abs(v) for v in f) for f in host.values())
dF = max(max(abs(dev[i][k] - host[i][k]) for k in range(3)) for i in host)
eh, ed = energy(h_log), energy(k_log)

print(f"    {len(host)} atoms, max |F| = {fmax:.4f} eV/A")
print(f"    max |F_kk - F_host| = {dF:.2e} eV/A   ({dF / fmax:.1e} relative)")

# NOT bitwise, and it cannot be. The two runs build their neighbour lists
# differently -- LAMMPS's CPU binning against Kokkos's -- so a site's
# neighbours are enumerated in a different order, and the SOAP sum over them
# reassociates. The observed difference is ~2e-09 relative, the same level at
# which the device and host TurboGAP builds differ from each other, and four
# orders below anything that changes a trajectory.
#
# Terms are compared at %.17g, and separately. Both matter:
#
#  - At %.8f a term near 100 eV resolves to 1e-8, the tolerance itself, and the
#    check reported an exact match where there is one of 2e-11.
#  - Lumped as one sum, deleting the device core_pot loop outright still
#    passed. core_pot is small here and the sum hid it.
#
# The 2e-11 is atomic ordering: three consecutive device runs of the same input
# gave ...537444588, ...537277738, ...537750832. The kernels atomic_add into
# the site energies and the hardware does not order those, so the device does
# not reproduce itself. Hence no fixed reference number -- the comparison is
# against the host in the same session, at 1e-9, two orders above that noise
# and far below a term that is actually wrong.
term_ok = True
for name in ('2b', 'core_pot'):
    hb, hv = term(h_log, f'after {name}:')
    db, dv = term(k_log, f'after {name} (device):')
    if hv is None or dv is None:
        print(f"    {name}: could not read the term back from both runs")
        term_ok = False
        continue
    if hb != db:
        print(f"    {name}: host ran {hb} blocks, device ran {db}")
        term_ok = False
        continue
    if hb == 0:
        # Nothing to compare. Agreement here would mean only that both sides
        # did nothing, which is exactly the vacuous pass this guards against.
        print(f"    {name}: NO BLOCKS in this potential -- nothing verified")
        term_ok = False
        continue
    rel = abs(dv - hv) / max(abs(hv), 1e-30)
    print(f"    {name}: {hb} blocks, device {dv!r} host {hv!r}   ({rel:.1e} relative)")
    term_ok = term_ok and rel < 1e-9

good = term_ok and dF / fmax < 1e-8
if eh is None or ed is None:
    print("    could not read one of the energies back")
    good = False
else:
    dE = abs(ed - eh)
    print(f"    energy {ed:.8f} against {eh:.8f}   (difference {dE:.2e})")
    good = good and dE < 1e-8

sys.exit(0 if good else 1)
