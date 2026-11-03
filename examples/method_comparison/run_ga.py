#!/usr/bin/env python3
"""EZGA with and without the experimental objective.

`--objectives energy` is the ordinary structure search: the GA sees only the
potential, and the measurement is used afterwards to score what it found.
`--objectives energy+exp` adds MADlib's misfit as a second objective, so the
GA is searching the trade-off rather than the energy landscape.

Running both at the same budget is the point. The first says what the
potential alone will give you; the second says what the measurement buys.

The operator settings are measured, not chosen: EZGA's collision gate sits
between the mutation and the relaxation, so a child that clashes is discarded
before LAMMPS can repair it, and an operator whose children always clash is
silently dead. See operator_probe.py for the acceptance rates these come from.
"""

import argparse
import json
import time
from pathlib import Path

from ezga.evaluator.features import feature_density, feature_minimum_interatomic_distance
from ezga.factory import build_default_engine, load_config
from ezga.variation.crossover import crossover_two_cut
from ezga.variation.mutation import mutation_random_strain, mutation_rattle

from madlib_ezga import (Pdf, Potential, Xrd, isotropic_strain, madlib_runner,
                         random_subset)
from madlib_ezga.features import feature_mean_coordination, feature_sp_fractions
from madlib_ezga.front import write_front
from madlib_ezga.objectives import objective_energy_per_atom, objective_xrd_rel_error


def strain(args):
    """Isotropic keeps the cell cubic; symmetric shears it triclinic.

    Isotropic is the default because MAD and HRMC never change their cell, so
    a cubic GA cell is the comparison those two can be read against, and
    because a sheared cell narrows the distance between opposite faces -- the
    quantity compute mad/pdf is limited by -- while the collision gate, which
    only rejects compression, biases an anisotropic walk towards expansion.
    """
    if args.strain_mode == "isotropic":
        return isotropic_strain(args.strain, seed=args.seed + 1)
    return mutation_random_strain(max_strain=args.strain, seed=args.seed + 1)


def mutations(args):
    """Two rattle scales and a strain, each on a measured acceptance rate.

    A rattle of every atom is rejected every time at this density -- the gate
    tests the closest pair out of N -- so both rattles move a random subset:
    0.20 A on 2% of atoms is accepted 92% of the time, 0.10 A on 10% of them
    always. The strain is the only operator that was ever alive in the previous
    run, and 0.02 exhausted its space by generation 10, hence 0.04.
    """
    return [
        mutation_rattle(std=args.rattle,
                        constraints=[random_subset(args.rattle_fraction, args.seed)],
                        seed=args.seed),
        mutation_rattle(std=args.rattle_broad,
                        constraints=[random_subset(args.rattle_broad_fraction,
                                                   args.seed + 3)],
                        seed=args.seed + 3),
        strain(args),
    ]


def build_config(args, runner, objectives):
    return {
        "output_path": str(Path(args.out) / "engine"),
        # A cap, not the plan: the budget is enforced in the calculator, which
        # declines to run once it is spent, so the remaining generations are
        # free and the spend lands on the budget instead of near it.
        "max_generations": args.generations,
        "resume": False,
        "rng": args.seed,
        "save_generations": True,
        "population": {
            "dataset_path": args.population,
            "db_path": str(Path(args.out) / "db"),
            "filter_duplicates": True,
            # 0.60 x 2 x the covalent radius = a 0.912 A gate. At the 0.75
            # default the gate is 1.140 A, and the seed population's own
            # closest pair is 1.178 A -- every move large enough to matter put
            # some pair under it, so every rattled child was discarded.
            "collision_factor": 0.60,
            # deny, not the default remove: "remove" strips the overlapping
            # atoms and keeps the structure, which silently changes N and so
            # both objectives, and it strips the worst-coordinated sites,
            # which flatters them. Rejecting the child keeps N fixed.
            "collision_action": "deny",
            # eV/atom, and it defaults to None: without it the children the
            # calculator declines once the budget is gone enter the archive at
            # their placeholder energy. A pilot accumulated 1900 of them. No
            # carbon structure worth keeping is above 0.
            "ef_bounds": (-12.0, 0.0),
        },
        "multiobjective": {
            "size": args.size,
            "selection_method": "nondominated_sorted",
        },
        # 0.05 is near-greedy on an energy spread of a few tenths of an eV: the
        # previous run's duplicate rate reached 0.69 and the median energy
        # converged onto the best. Note that `foreigners` is not the fix for
        # that -- it only acts through the generative path, which is off -- so
        # diversity here comes from the operators, the feature repulsion and
        # this temperature.
        "thermostat": {"initial_temperature": args.temperature,
                       "constant_temperature": True},
        "variation": {"initial_mutation_rate": 1.0,
                      "crossover_probability": args.crossover},
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
        "mutation_funcs": mutations(args),
        # A two-plane slab splice. Of the eight crossovers EZGA ships it is
        # the one that both recombines and survives: the Voronoi bisector puts
        # atoms 0.005 A apart and every child is rejected, and the fractional
        # mixing averages Cartesian positions, which across a periodic
        # boundary is not a structure at all.
        "crossover_funcs": [crossover_two_cut],
        # The calculator relaxes; declaring it says so. EZGA's per-individual
        # temperature is a selection temperature in objective units, not a
        # kelvin, so the runner ignores it and anneals at its own.
        "simulator": {"calculator": runner, "seed": args.seed,
                      "mode": "relaxation"},
    }


def seed_spend(population):
    """Evaluations already spent scoring the seeds, charged to this run."""
    path = Path(population).parent / "seed_spend.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text())["evaluations"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lmp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--objectives", choices=("energy", "energy+exp"), required=True)
    ap.add_argument("--budget", type=int, required=True,
                    help="potential evaluations; the calculator stops at it")
    ap.add_argument("--generations", type=int, required=True,
                    help="cap, reached only if the budget is not spent first")
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
    ap.add_argument("--rattle", type=float, default=0.20)
    ap.add_argument("--rattle-fraction", type=float, default=0.02)
    ap.add_argument("--rattle-broad", type=float, default=0.10)
    ap.add_argument("--rattle-broad-fraction", type=float, default=0.10)
    ap.add_argument("--strain", type=float, default=0.04)
    ap.add_argument("--strain-mode", choices=("isotropic", "symmetric"),
                    default="isotropic")
    ap.add_argument("--crossover", type=float, default=0.30)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # With energy as the only objective the bias must be off, or the
    # relaxation is still steered by the measurement and the two legs differ
    # in more than their objective list.
    gammas = ((0.0,) if args.objectives == "energy"
              else tuple(float(g) for g in args.gammas.split(",")))

    spent_on_seeds = seed_spend(args.population)
    runner = madlib_runner(
        potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
        target=args.target,
        pdf=Pdf(rmin=0.1, rmax=args.rmax, nbins=args.nbins, kde=0.1),
        xrd=Xrd(target=args.target, qmin=args.qmin, qmax=args.qmax, nq=args.nq,
                output=args.xrd_output, weighting=args.weighting),
        gamma_ladder=gammas,
        nsteps=args.md_steps,
        relax_steps=args.relax_steps,
        budget=args.budget - spent_on_seeds,
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
    spend = runner.spend
    evals = int(spend["steps"]) + spent_on_seeds
    (out / "timing.json").write_text(json.dumps({
        "method": f"ezga_{args.objectives.replace('+', '_')}",
        "strain_mode": args.strain_mode,
        "wall_seconds": round(wall, 1),
        "evaluations": evals,
        "budget": args.budget,
        "evaluations_on_seeds": spent_on_seeds,
        "individuals": spend["individuals"],
        "failed_individuals": spend["failed"],
        "declined_over_budget": spend["declined"],
        "seconds_per_evaluation": round(wall / evals, 5) if evals else None,
        "front_points": len(front),
    }, indent=2))
    print(f"\n{len(front)} front points, {wall:.0f} s, {evals} evaluations "
          f"of a {args.budget} budget over {spend['individuals']} individuals "
          f"({spend['declined']} declined once the budget was gone)")


main()
