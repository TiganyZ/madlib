"""Integration checks: the calculator's contract against a real LAMMPS.

Skips unless MADLIB_LMP, MADLIB_GAP_DIR and MADLIB_XRD_TARGET are set.

    MADLIB_LMP=/path/to/lmp \
    MADLIB_GAP_DIR=/path/to/gap_files \
    MADLIB_XRD_TARGET=/path/to/xrd_glassy_carbon_zeng_2017.fq \
    python3 tests/test_roundtrip.py <structure.data>

What it protects. A structure makes the round trip out to a dump and back
once per generation, so any loss there is noise the GA accumulates; and
mutation_random_strain makes cells triclinic, where ASE rotates into LAMMPS's
prism, so coordinates legitimately change while the physics must not.
"""

import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

from madlib_ezga import Pdf, Potential, Xrd, madlib_runner

FAILURES = []

STRAINS = {
    "cubic": np.zeros((3, 3)),
    "isotropic": np.eye(3) * 0.02,
    "uniaxial": np.diag([0.03, 0.0, -0.01]),
    "shear": np.array([[0.0, 0.025, 0.0], [0.0, 0.0, 0.018], [0.0, 0.0, 0.0]]),
}


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {name} {detail}")
    if not condition:
        FAILURES.append(name)


def main(structure, workroot):
    lmp = os.environ["MADLIB_LMP"]
    gap_dir = Path(os.environ["MADLIB_GAP_DIR"])
    target = Path(os.environ["MADLIB_XRD_TARGET"])

    atoms = read(structure, format="lammps-data")
    atoms.symbols = ["C"] * len(atoms)

    runner = madlib_runner(
        potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
        pdf=Pdf(rmax=8.0), xrd=Xrd(target=target.name), gamma_ladder=(0.0,),
        relax_steps=0, ranks=1, lmp=lmp, links=[gap_dir, target],
        workroot=workroot,
    )

    def call(positions, cell, symbols):
        return runner(symbols=symbols, positions=positions, cell=cell, seed=1)

    for name, strain in STRAINS.items():
        case = atoms.copy()
        case.set_cell(np.array(atoms.get_cell()) @ (np.eye(3) + strain), scale_atoms=True)
        symbols = [str(s) for s in case.symbols]

        pos, sym, cell, energy, meta = call(case.get_positions(),
                                            np.array(case.get_cell()), symbols)
        if not meta.get("madlib_ok"):
            check(f"{name} runs", False, meta.get("madlib_error", "")[:100])
            continue
        _, _, _, energy2, meta2 = call(pos, cell, [str(s) for s in sym])

        n = len(case)
        check(f"{name}: energy survives the round trip",
              abs(energy - energy2) / n < 1e-8, f"dE={abs(energy - energy2) / n:.1e}")
        check(f"{name}: misfit survives the round trip",
              abs(meta["xrd_rel_error"] - meta2["xrd_rel_error"]) < 1e-8)
        check(f"{name}: volume survives the round trip",
              abs(np.linalg.det(np.array(case.get_cell())) - np.linalg.det(cell)) < 1e-6)

    # The bias must not leak into the reported energy, whatever gamma is.
    biased = madlib_runner(
        potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
        pdf=Pdf(rmax=8.0), xrd=Xrd(target=target.name), gamma_ladder=(50.0,),
        relax_steps=0, ranks=1, lmp=lmp, links=[gap_dir, target],
        workroot=Path(workroot) / "biased",
    )
    symbols = [str(s) for s in atoms.symbols]
    _, _, _, e_plain, _ = call(atoms.get_positions(), np.array(atoms.get_cell()), symbols)
    _, _, _, e_biased, meta_b = biased(symbols=symbols, positions=atoms.get_positions(),
                                       cell=np.array(atoms.get_cell()), seed=1)
    check("energy is the potential alone, not potential+bias",
          abs(e_plain - e_biased) < 1e-8,
          f"gamma=0 {e_plain:.6f} vs gamma=50 {e_biased:.6f}, bias={meta_b['bias_energy']:.3f}")


if __name__ == "__main__":
    needed = ["MADLIB_LMP", "MADLIB_GAP_DIR", "MADLIB_XRD_TARGET"]
    if any(v not in os.environ for v in needed) or len(sys.argv) < 2:
        print(f"skipped: needs {', '.join(needed)} and a structure argument")
        raise SystemExit(0)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tmp/madlib_ezga_roundtrip")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all passed")
