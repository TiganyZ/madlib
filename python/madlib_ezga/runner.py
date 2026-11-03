"""MADlib as an EZGA calculator.

EZGA's calculator slot is a structure transformer, not an energy function: it
is handed a structure and returns a possibly different one plus its energy.
A MAD relaxation is exactly that, which is what makes the bias an accelerator
rather than just another score -- children arrive already moved downhill on
E_pot + bias instead of being scattered at random.

Each individual is relaxed at its own gamma, drawn from a ladder. One gamma for
the whole population would pull every member to the same corner of the
(energy, misfit) plane; a ladder spreads them along the front.
"""

import itertools
import tempfile
import zlib
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from . import deck as deck_mod
from .lammps_run import LammpsError, find_lmp, run

FAILED_ENERGY = 1.0e6


def _gamma_for(parent_hash, ladder, fallback):
    """One gamma per family, not one per arrival.

    EZGA hands the calculator the parent's hash but not the parent's metadata,
    so a child cannot read the gamma its parent was relaxed at; deriving it from
    that hash is what the contract allows. Every child of one parent then gets
    the same bias strength, instead of a family being split across the ladder by
    the order its members happen to reach the queue -- which undoes in the child
    whatever fit the parent gained. crc32, not hash(), because hash() of a str
    is salted per process and the run would not reproduce.
    """
    if parent_hash is None:
        return next(fallback)
    return ladder[zlib.crc32(str(parent_hash).encode()) % len(ladder)]


def _moves(gamma, nsteps, temp, seed, relax_steps, box):
    # With a box relax, two passes: cg to settle the cell -- LAMMPS refuses a
    # damped dynamics style with fix box/relax -- then fire at fixed cell to
    # polish, because cg alone converges to a visibly worse point on the
    # combined landscape (at gamma 30, rel_error 0.443 against fire's 0.383).
    if box:
        relax = (deck_mod.minimise(maxiter=max(1, relax_steps // 2), box=box)
                 + deck_mod.minimise(maxiter=relax_steps))
    else:
        relax = deck_mod.minimise(maxiter=relax_steps)
    if nsteps:
        return deck_mod.anneal(nsteps, temp, temp * 0.1, seed) + relax
    return relax


def madlib_runner(
    potential=None,
    target="xrd_glassy_carbon_zeng_2017.fq",
    gamma_ladder=(0.0, 0.3, 1.0, 3.0, 10.0),
    nsteps=0,
    relax_steps=200,
    temperature=1000.0,
    box_relax=None,
    budget=None,
    pdf=None,
    xrd=None,
    lmp=None,
    ranks=1,
    links=(),
    workroot=None,
    keep_runs=False,
    mass=12.011,
):
    """Return the callable EZGA calls once per individual."""
    potential = potential or deck_mod.Potential()
    pdf = pdf or deck_mod.Pdf()
    xrd = xrd or deck_mod.Xrd(target=target)
    lmp = find_lmp(lmp)
    workroot = Path(workroot or tempfile.mkdtemp(prefix="madlib_ezga_"))
    workroot.mkdir(parents=True, exist_ok=True)
    ladder = tuple(gamma_ladder)
    gammas = itertools.cycle(ladder)
    counter = itertools.count()
    # What the run actually spent, and with `budget` set also its stop
    # condition. EZGA stops on a generation count, and the children a generation
    # yields are not known in advance, so a generation count cannot spend a
    # budget exactly; declining to run once the budget is gone can. The
    # generation cap is set generously and the remaining ones cost nothing.
    spend = {"individuals": 0, "steps": 0.0, "failed": 0, "declined": 0}

    def calculate(symbols, positions, cell, seed=0, output_filename=None,
                  parent_hash=None, **_kwargs):
        if budget is not None and spend["steps"] >= budget:
            spend["declined"] += 1
            return (np.asarray(positions, float), list(symbols),
                    np.asarray(cell, float), FAILED_ENERGY,
                    {"madlib_ok": False, "madlib_error": "budget exhausted"})

        index = next(counter)
        gamma = _gamma_for(parent_hash, ladder, gammas)
        workdir = workroot / f"ind_{index:06d}"
        atoms = Atoms(symbols=list(symbols), positions=np.asarray(positions, float),
                      cell=np.asarray(cell, float), pbc=True)

        workdir.mkdir(parents=True, exist_ok=True)
        data_file = workdir / "start.data"
        write(str(data_file), atoms, format="lammps-data", masses=True,
              specorder=sorted(set(symbols)))

        text = deck_mod.render(
            data_file.name,
            potential,
            pdf=pdf,
            xrd=xrd,
            gamma=gamma,
            moves=_moves(gamma, nsteps, temperature, int(seed) + index, relax_steps,
                         box_relax),
            masses={i + 1: mass for i in range(len(set(symbols)))},
        )

        try:
            thermo, out = run(workdir, text, lmp, ranks=ranks, links=links)
        except (LammpsError, OSError) as exc:
            # A failed individual must not stop the run; give it an energy no
            # selection will keep and say why in the metadata.
            spend["individuals"] += 1
            spend["failed"] += 1
            return (np.asarray(positions, float), list(symbols),
                    np.asarray(cell, float), FAILED_ENERGY,
                    {"madlib_ok": False, "madlib_error": str(exc)[:400], "gamma": gamma})

        steps = float(thermo["Step"][-1]) if "Step" in thermo else 0.0
        spend["individuals"] += 1
        spend["steps"] += steps

        meta = {
            "madlib_ok": True,
            "gamma": gamma,
            "steps": steps,
            "xrd_rel_error": float(thermo["f_mad[2]"][-1]),
            "xrd_dissimilarity": float(thermo["f_mad[3]"][-1]),
            "bias_energy": float(thermo["f_mad"][-1]),
        }
        energy = float(thermo["c_pepair"][-1])

        if out is None:
            return (np.asarray(positions, float), list(symbols),
                    np.asarray(cell, float), energy, meta)

        final_positions = out.get_positions()
        final_cell = np.asarray(out.get_cell())
        if not keep_runs:
            for leftover in workdir.glob("*"):
                if leftover.is_file() and leftover.name != "log.lammps":
                    leftover.unlink()

        return final_positions, list(symbols), final_cell, energy, meta

    calculate.spend = spend
    return calculate
