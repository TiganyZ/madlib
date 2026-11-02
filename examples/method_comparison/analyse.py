#!/usr/bin/env python3
"""Score every structure the four methods produced, identically, and front them.

The methods report different things natively -- HRMC's own energy is the
biased one, EZGA's is already the pure potential, MAD writes a final
structure and nothing else -- so none of their logs are comparable. Every
structure is re-run here at gamma 0 and read off the same two numbers.

`--lmp` is required because the scoring is a LAMMPS run, not a parse.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

DECK = """units           metal
atom_style      atomic
boundary        p p p
atom_modify     map yes
read_data       {struct} nocoeff
mass            * {mass}
comm_modify     cutoff {comm}
neighbor        0.5 bin
include         {potential}
compute pepair all pe pair
compute gr     all mad/pdf r 0.1 {rmax} {nbins} kde {kde}
fix     mad all mad/xrd c_gr target {target} &
        q {qmin} {qmax} {nq} output {output} species {species} &
        weighting {weighting} gamma 0.0
thermo_style custom step c_pepair f_mad[2] f_mad[3]
thermo_modify format float %.10g
run 0
"""


def natoms(path):
    for line in Path(path).read_text().splitlines():
        if line.strip().endswith("atoms"):
            return int(line.split()[0])
    raise ValueError(f"no atom count in {path}")


def score(struct, work, args):
    work.mkdir(parents=True, exist_ok=True)
    deck = work / "in.score"
    deck.write_text(DECK.format(
        struct=struct, mass=args.mass, comm=args.comm, potential=args.potential,
        rmax=args.rmax, nbins=args.nbins, kde=args.kde, target=args.target,
        qmin=args.qmin, qmax=args.qmax, nq=args.nq, output=args.output,
        species=args.species, weighting=args.weighting))
    r = subprocess.run(["mpirun", "-np", str(args.ranks), args.lmp,
                        "-in", "in.score", "-log", "log"],
                       cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    rows = (work / "log").read_text().splitlines()
    for i, line in enumerate(rows):
        if line.split()[:1] == ["Step"]:
            f = rows[i + 1].split()
            n = natoms(struct)
            return {"natoms": n, "energy_per_atom": float(f[1]) / n,
                    "rel_error": float(f[2]), "dissimilarity": float(f[3])}
    return None


def non_dominated(points):
    out = []
    for i, p in enumerate(points):
        if any(q["energy_per_atom"] <= p["energy_per_atom"]
               and q["rel_error"] <= p["rel_error"]
               and (q["energy_per_atom"] < p["energy_per_atom"]
                    or q["rel_error"] < p["rel_error"])
               for j, q in enumerate(points) if j != i):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p["rel_error"])


def collect(runs):
    """Every structure each method left behind, tagged with its method."""
    found = []
    for pattern, method in (("mad_gamma*/final.data", "mad"),
                            ("hrmc_gamma*/final.data", "hrmc"),
                            ("ezga_energy/**/*.data", "ezga"),
                            ("ezga_energy_exp/**/*.data", "ezga_exp")):
        for path in sorted(Path(runs).glob(pattern)):
            found.append((method, path))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--potential", default="potential.lmp")
    ap.add_argument("--species", default="C")
    ap.add_argument("--mass", default="12.011")
    ap.add_argument("--comm", default="9.3")
    ap.add_argument("--rmax", default="8.0")
    ap.add_argument("--nbins", default="201")
    ap.add_argument("--kde", default="0.1")
    ap.add_argument("--qmin", default="1.04525")
    ap.add_argument("--qmax", default="9.97006")
    ap.add_argument("--nq", default="201")
    ap.add_argument("--output", default="qFq")
    ap.add_argument("--weighting", default="xray")
    ap.add_argument("--ranks", type=int, default=4)
    args = ap.parse_args()

    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, path in collect(runs):
        res = score(path, out / "work" / method / path.parent.name, args)
        if res is None:
            print(f"  FAILED {path}")
            continue
        res.update({"method": method, "label": str(path.relative_to(runs))})
        rows.append(res)
        print(f"  {method:<10} {res['energy_per_atom']:9.4f} {res['rel_error']:8.4f}  "
              f"{res['label']}")

    fields = ["method", "label", "natoms", "energy_per_atom", "rel_error",
              "dissimilarity"]
    with (out / "scored.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    fronts = {}
    for method in sorted({r["method"] for r in rows}):
        pts = [r for r in rows if r["method"] == method]
        fronts[method] = non_dominated(pts)
    (out / "fronts.json").write_text(json.dumps(
        {k: [{kk: v[kk] for kk in ("label", "energy_per_atom", "rel_error")}
             for v in vs] for k, vs in fronts.items()}, indent=2))

    print(f"\n{len(rows)} structures scored -> {out / 'scored.csv'}")
    print(f"{'method':<12} {'points':>6} {'front':>6} {'best E':>9} {'best misfit':>12}")
    for method, front in fronts.items():
        pts = [r for r in rows if r["method"] == method]
        print(f"{method:<12} {len(pts):>6} {len(front):>6} "
              f"{min(p['energy_per_atom'] for p in pts):>9.4f} "
              f"{min(p['rel_error'] for p in pts):>12.4f}")


main()
