#!/usr/bin/env python3
"""EZGA with and without the experimental objective.

`--objectives energy` is the ordinary structure search: the GA sees only the
potential, and the measurement is used afterwards to score what it found.
`--objectives energy+exp` adds MADlib's misfit as a second objective, so the
GA is searching the trade-off rather than the energy landscape.

Running both at the same budget is the point. The first says what the
potential alone will give you; the second says what the measurement buys.
"""

import argparse
import json
import time
from pathlib import Path

from ezga.evaluator.features import feature_density, feature_minimum_interatomic_distance
from ezga.factory import build_default_engine, load_config
from ezga.variation.mutation import mutation_random_strain, mutation_rattle

from madlib_ezga import Pdf, Potential, Xrd, madlib_runner
from madlib_ezga.features import feature_mean_coordination, feature_sp_fractions
from madlib_ezga.front import write_front
from madlib_ezga.objectives import objective_energy_per_atom, objective_xrd_rel_error


def build_config(args, runner, objectives):
    return {
        "output_path": str(Path(args.out) / "engine"),
        "max_generations": args.generations,
        "resume": False,
        "rng": args.seed,
        "save_generations": True,
        "population": {
            "dataset_path": args.population,
            "db_path": str(Path(args.out) / "db"),
            "filter_duplicates": True,
            "collision_factor": 0.75,
            # deny, not the default remove: "remove" strips the overlapping
            # atoms and keeps the structure, which silently changes N and so
            # both objectives, and it strips the worst-coordinated sites,
            # which flatters them. Rejecting the child keeps N fixed.
            "collision_action": "deny",
        },
        "multiobjective": {
            "size": args.size,
            "selection_method": "nondominated_sorted",
        },
        "thermostat": {"initial_temperature": 0.05, "constant_temperature": True},
        "variation": {"initial_mutation_rate": 1.0, "crossover_probability": 0.0},
        "evaluator": {
            # Features are not objectives: they are the space selection keeps
            # individuals apart in. Without them the repulsion term is zero
            # and the run collapses onto a few near-identical structures.
            "features_funcs": [
                feature_density(),
                feature_minimum_interatomic_distance(),
                feature_mean_coordination(),
                feature_sp_fractions(),
            ],
            "objectives_funcs": objectives,
        },
        "mutation_funcs": [
            mutation_rattle(std=args.rattle, seed=args.seed),
            mutation_random_strain(max_strain=args.strain, seed=args.seed + 1),
        ],
        "simulator": {"calculator": runner, "seed": args.seed},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--objectives", choices=("energy", "energy+exp"), required=True)
    ap.add_argument("--generations", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--md-steps", type=int, required=True)
    ap.add_argument("--relax-steps", type=int, required=True)
    ap.add_argument("--gammas", default="0,1,3,10,30")
    ap.add_argument("--species", default="C")
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--nbins", type=int, default=201)
    ap.add_argument("--qmin", type=float, default=1.04525)
    ap.add_argument("--qmax", type=float, default=9.97006)
    ap.add_argument("--nq", type=int, default=201)
    ap.add_argument("--xrd-output", default="qFq")
    ap.add_argument("--weighting", default="xray")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--rattle", type=float, default=0.12)
    ap.add_argument("--strain", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # With energy as the only objective the bias must be off, or the
    # relaxation is still steered by the measurement and the two legs differ
    # in more than their objective list.
    gammas = ((0.0,) if args.objectives == "energy"
              else tuple(float(g) for g in args.gammas.split(",")))

    runner = madlib_runner(
        potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
        target=args.target,
        pdf=Pdf(rmin=0.1, rmax=args.rmax, nbins=args.nbins, kde=0.1),
        xrd=Xrd(target=args.target, qmin=args.qmin, qmax=args.qmax, nq=args.nq,
                output=args.xrd_output, weighting=args.weighting),
        gamma_ladder=gammas,
        nsteps=args.md_steps,
        relax_steps=args.relax_steps,
        lmp=args.lmp,
        ranks=args.ranks,
        links=[data / "gap_files", data / args.target],
        workroot=out / "work",
    )

    objectives = [objective_energy_per_atom()]
    if args.objectives == "energy+exp":
        objectives.append(objective_xrd_rel_error())

    started = time.time()
    build_default_engine(load_config(build_config(args, runner, objectives))).run()
    wall = time.time() - started

    front = write_front(out / "db", out)
    evals = args.generations * args.size * (args.md_steps + args.relax_steps)
    (out / "timing.json").write_text(json.dumps({
        "method": f"ezga_{args.objectives.replace('+', '_')}",
        "wall_seconds": round(wall, 1),
        "evaluations": evals,
        "seconds_per_evaluation": round(wall / evals, 5),
        "front_points": len(front),
    }, indent=2))
    print(f"\n{len(front)} front points, {wall:.0f} s, {evals} evaluations")


main()
