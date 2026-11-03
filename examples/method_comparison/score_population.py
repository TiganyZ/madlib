#!/usr/bin/env python3
"""Score every seed structure before the GA starts.

EZGA only ever simulates offspring, so a seed read from a file keeps whatever
energy the file gave it. With none, `objective_energy_per_atom` turns the
missing value into 0.0 -- the worst value a minimisation can see -- and the
seeds sit in the parent pool at a fake objective for the whole run. A measured
run had 24 of 24 parents like that at generation 0 and still 6 of 24 at
generation 24.

This writes the same frames back out with `energy` and `xrd_rel_error` in the
comment line, which sage_lib reads into `AtomPositionManager.E` and
`.metadata`. It is `run 0`, not a relaxation: scoring tells selection what the
seeds are, relaxing them would hand the GA a head start the other methods do
not get.
"""

import argparse
import json
from pathlib import Path

from ase.io import read, write

from madlib_ezga import Pdf, Potential, Xrd, deck as deck_mod
from madlib_ezga.lammps_run import find_lmp, run


def score_one(atoms, workdir, potential, pdf, xrd, lmp, ranks, links, mass):
    workdir.mkdir(parents=True, exist_ok=True)
    data_file = workdir / "start.data"
    write(str(data_file), atoms, format="lammps-data", masses=True,
          specorder=sorted(set(atoms.get_chemical_symbols())))
    text = deck_mod.render(
        data_file.name, potential, pdf=pdf, xrd=xrd, gamma=0.0,
        moves=deck_mod.score(),
        masses={i + 1: mass for i in range(len(set(atoms.get_chemical_symbols())))},
        dump_file=None)
    thermo, _ = run(workdir, text, lmp, ranks=ranks, links=links)
    return {"energy": float(thermo["c_pepair"][-1]),
            "xrd_rel_error": float(thermo["f_mad[2]"][-1])}


def score_all(frames, work, potential, pdf, xrd, lmp, ranks, links, mass):
    scored = []
    for i, atoms in enumerate(frames):
        values = score_one(atoms, work / f"seed_{i:03d}", potential, pdf, xrd,
                           lmp, ranks, links, mass)
        atoms.info.update(values)
        scored.append(atoms)
        print(f"  seed {i:3d}  E {values['energy'] / len(atoms):9.4f} eV/atom"
              f"  rel_error {values['xrd_rel_error']:.4f}")
    return scored


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--nbins", type=int, default=201)
    ap.add_argument("--kde", type=float, default=0.1)
    ap.add_argument("--qmin", type=float, default=1.04525)
    ap.add_argument("--qmax", type=float, default=9.97006)
    ap.add_argument("--nq", type=int, default=201)
    ap.add_argument("--xrd-output", default="qFq")
    ap.add_argument("--weighting", default="xray")
    ap.add_argument("--mass", type=float, default=12.011)
    ap.add_argument("--ranks", type=int, default=4)
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "score_work"

    frames = read(args.population, index=":")
    scored = score_all(
        frames, work,
        Potential(style="turbogap", model="gap_files/carbon.gap"),
        Pdf(rmin=0.1, rmax=args.rmax, nbins=args.nbins, kde=args.kde),
        Xrd(target=args.target, qmin=args.qmin, qmax=args.qmax, nq=args.nq,
            output=args.xrd_output, weighting=args.weighting),
        find_lmp(args.lmp), args.ranks,
        [data / "gap_files", data / args.target], args.mass)

    write(str(out), scored, format="extxyz")
    # One evaluation each, charged to the GA's budget: the seeds are part of
    # what the run spends.
    (out.parent / "seed_spend.json").write_text(json.dumps(
        {"seeds": len(scored), "evaluations": len(scored)}, indent=2))
    print(f"\n{len(scored)} seeds scored -> {out}")


main()
