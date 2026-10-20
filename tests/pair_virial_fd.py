#!/usr/bin/env python3
"""pair_style turbogap's virial against central differences of its own energy.

The gap this closes: the `newton` leg compares the two newton modes to each
other, so a virial that is simply wrong agrees with itself and passes. This
compares it to something outside the style.

Deform by F = I + eps E_ab, affinely. Moving an atom by dr_a = eps r_b changes
the energy by -eps sum_i f_ia r_ib, so W_ab = -dE/deps, component by
component. That pins sign, factor and off-diagonals at once.

    pair_virial_fd.py <data-file> --lmp <binary> --gap <file> --species C [...]

eps defaults to 1e-3. At 1e-4 the off-diagonal signal is ~100x weaker than
the diagonal and sits in the noise of a four-figure total energy; at 1e-2 the
higher-order terms show.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EV_A3_TO_GPA = 160.21766208
BAR_TO_GPA = 1.0e-4
COMPONENTS = [(0, 0, "xx"), (1, 1, "yy"), (2, 2, "zz"),
              (0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]

DECK = """units           metal
atom_style      atomic
boundary        p p p
atom_modify     map yes
read_data       {data}
{masses}
neighbor        0.5 bin
pair_style      turbogap {e0}
pair_coeff      * * {gap} {species}
compute pepair all pe pair
compute pv     all pressure NULL virial
thermo_style custom step c_pepair c_pv[1] c_pv[2] c_pv[3] c_pv[4] c_pv[5] c_pv[6] vol
thermo_modify format float %.14g
run 0
"""


def read_data_cell(path):
    """(cell 3x3, text) of a LAMMPS data file, orthogonal or triclinic."""
    text = Path(path).read_text()
    def bound(tag):
        m = re.search(rf"^\s*(\S+)\s+(\S+)\s+{tag}lo {tag}hi", text, re.M)
        return (float(m.group(1)), float(m.group(2)))
    xlo, xhi = bound("x")
    ylo, yhi = bound("y")
    zlo, zhi = bound("z")
    tilt = re.search(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+xy xz yz", text, re.M)
    xy, xz, yz = (float(g) for g in tilt.groups()) if tilt else (0.0, 0.0, 0.0)
    return [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]], text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--gap", required=True)
    ap.add_argument("--species", nargs="+", default=["C"])
    ap.add_argument("--mass", nargs="+", default=["12.011"])
    ap.add_argument("--e0", nargs="*", default=["C", "-.16138053"])
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=0.05, help="GPa")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        from ase.io import read, write
    except ImportError:
        # 2, not 0: a skip that exits 0 is reported as a pass, which is a
        # worse failure mode than the bug this leg exists to catch.
        print("    SKIP: needs numpy and ase (set PYTHON)")
        return 2

    atoms = read(args.data, format="lammps-data")
    masses = "\n".join(f"mass            {i + 1} {m}"
                       for i, m in enumerate(args.mass))
    e0 = " ".join(f"e0 {args.e0[i]} {args.e0[i + 1]}"
                  for i in range(0, len(args.e0), 2)) if args.e0 else ""

    work = Path(tempfile.mkdtemp(prefix="mad_virial_fd_"))
    for name in ("gap_files",):
        src = Path(args.gap).parent
        if src.name == name and not (work / name).exists():
            (work / name).symlink_to(src.resolve())
    gap_ref = f"gap_files/{Path(args.gap).name}" if (work / "gap_files").exists() \
        else str(Path(args.gap).resolve())

    def evaluate(cell_atoms, tag):
        data = work / f"{tag}.data"
        write(str(data), cell_atoms, format="lammps-data", masses=True)
        deck = work / f"in.{tag}"
        deck.write_text(DECK.format(data=data.name, masses=masses, e0=e0,
                                    gap=gap_ref, species=" ".join(args.species)))
        proc = subprocess.run([args.lmp, "-in", deck.name, "-log", f"log.{tag}"],
                              cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout[-800:] + proc.stderr[-800:])
        log = (work / f"log.{tag}").read_text()
        rows = [l.split() for l in log.splitlines()
                if re.match(r"^\s*0\s", l) and len(l.split()) == 9]
        row = [float(v) for v in rows[-1]]
        return row[1], np.array(row[2:8]) * BAR_TO_GPA, row[8]

    energy, analytic, volume = evaluate(atoms, "ref")
    if not args.quiet:
        print(f"    {len(atoms)} atoms, V = {volume:.3f} A^3, eps = {args.eps}")

    worst = 0.0
    for index, (a, b, name) in enumerate(COMPONENTS):
        values = []
        for sign in (+1, -1):
            gradient = np.eye(3)
            gradient[a, b] += sign * args.eps
            case = atoms.copy()
            case.set_cell(np.array(atoms.get_cell()) @ gradient.T, scale_atoms=True)
            values.append(evaluate(case, f"{name}{'p' if sign > 0 else 'm'}")[0])
        numeric = -(values[0] - values[1]) / (2 * args.eps) / volume * EV_A3_TO_GPA
        diff = abs(analytic[index] - numeric)
        worst = max(worst, diff)
        if not args.quiet:
            print(f"    {name}  analytic {analytic[index]:10.4f}  "
                  f"-dE/deps/V {numeric:10.4f}  diff {diff:8.4f} GPa")

    print(f"    worst |analytic - finite difference| = {worst:.4f} GPa "
          f"(tolerance {args.tol})")
    return 0 if worst < args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
