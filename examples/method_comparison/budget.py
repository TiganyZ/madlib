#!/usr/bin/env python3
"""Turn one compute budget into parameters for each method.

Four ways of fitting a diffraction pattern, and they spend compute so
differently that comparing them at "the settings everyone uses" compares the
settings, not the methods. The one currency they share is the **potential
evaluation**: one call returning energy and forces for the whole cell. Every
method is given the same number of them.

    MAD     one evaluation per MD step and per minimiser iteration
    EZGA    one per MD step and per relaxation step, per individual
    HRMC    one per attempted move, because full_energy re-evaluates the cell

MAD and HRMC spend a fixed number of steps, so their budget is arithmetic.
EZGA's depends on how many children survive its collision and duplicate gates,
so its budget is enforced at run time and this only sets a generation ceiling.

MAD and HRMC produce a front by scanning gamma, so their budget is divided
across the ladder. EZGA produces a front from a single run, so it keeps the
whole budget -- that difference is a property of the methods and is what the
matching is meant to expose, not hide.

Run it to see the parameters, or source the shell it emits.
"""

import argparse
import json


def mad(budget, n_gamma, minimise_frac):
    """Per gamma: an MD anneal then a minimisation."""
    per = budget // n_gamma
    mini = int(per * minimise_frac)
    return {"n_gamma": n_gamma, "steps_per_gamma": per - mini,
            "minimise_iters": mini, "evals": per * n_gamma}


def hrmc(budget, n_gamma, moves_per_cycle):
    """Per gamma: a Metropolis chain, one evaluation per attempted move."""
    per = budget // n_gamma
    cycles = per // moves_per_cycle
    return {"n_gamma": n_gamma, "cycles": cycles,
            "moves_per_cycle": moves_per_cycle,
            "evals": cycles * moves_per_cycle * n_gamma}


def ezga(budget, population, md_steps, relax_steps, steps_per_individual,
         offspring_per_generation, cap_margin):
    """One run, stopped by its own spend rather than by a generation count.

    Only offspring are simulated -- survivors carry their evaluation forward,
    and collided or duplicate children are rejected before being evaluated --
    so how much a generation costs depends on how many children survive the
    gate, which is not known in advance. A generation count therefore cannot
    spend a budget exactly, and the first attempt at one overshot by a third.
    run_ga.py enforces the budget in the calculator instead; this only has to
    supply a cap high enough that the budget is what binds.

    The cap is a ceiling, not a target: it is the generations the measured
    yield needs, times a margin. Generations past the budget still cost a
    little in selection and gate checks, so a loose cap is waste; a run that
    reaches the cap without spending the budget says so in its timing.json.
    """
    per_generation = max(1, offspring_per_generation * steps_per_individual)
    needed = max(1, -(-budget // per_generation))
    return {"population": population,
            "generations": int(round(needed * cap_margin)),
            "md_steps": md_steps, "relax_steps": relax_steps,
            "steps_per_individual": steps_per_individual,
            "offspring_per_generation": offspring_per_generation,
            "budget": budget, "evals": budget}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=60000,
                    help="potential evaluations per method")
    ap.add_argument("--gammas", type=int, default=5,
                    help="ladder length for the methods that scan gamma")
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--offspring-per-generation", type=int, default=24,
                    help="measured children simulated per generation")
    ap.add_argument("--cap-margin", type=float, default=1.5,
                    help="slack on the generation ceiling")
    ap.add_argument("--md-steps", type=int, default=100)
    ap.add_argument("--relax-steps", type=int, default=150,
                    help="cap; the relaxation stops when it converges")
    ap.add_argument("--steps-per-individual", type=int, default=242,
                    help="measured MD + relaxation steps actually run")
    ap.add_argument("--moves-per-cycle", type=int, default=50)
    ap.add_argument("--minimise-frac", type=float, default=0.2)
    ap.add_argument("--md-frac", type=float, default=0.4)
    ap.add_argument("--shell", action="store_true", help="emit shell assignments")
    args = ap.parse_args()

    plan = {
        "budget_per_method": args.budget,
        "mad": mad(args.budget, args.gammas, args.minimise_frac),
        "hrmc": hrmc(args.budget, args.gammas, args.moves_per_cycle),
        "ezga": ezga(args.budget, args.population, args.md_steps,
                     args.relax_steps, args.steps_per_individual,
                     args.offspring_per_generation, args.cap_margin),
    }
    plan["ezga_exp"] = dict(plan["ezga"])

    if args.shell:
        for method in ("mad", "hrmc", "ezga"):
            for key, value in plan[method].items():
                print(f"{method.upper()}_{key.upper()}={value}")
        return

    print(json.dumps(plan, indent=2))
    print()
    print("evaluations actually spent, which is what has to match:")
    for method in ("mad", "hrmc", "ezga", "ezga_exp"):
        print(f"  {method:<10} {plan[method]['evals']:>8d}")


main()
