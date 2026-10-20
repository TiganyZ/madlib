"""MADlib as an EZGA calculator.

EZGA's calculator slot is a structure transformer, not an energy function: it
is handed a structure and returns a possibly different one plus its energy.
A MAD relaxation is exactly that, which is what makes the bias an accelerator
rather than just another score -- children arrive already moved downhill on
E_pot + bias instead of being scattered at random.

Each individual is relaxed at its own gamma, drawn in turn from a ladder. One
gamma for the whole population would pull every member to the same corner of
the (energy, misfit) plane; a ladder spreads them along the front.
"""

import itertools
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from . import deck as deck_mod
from .lammps_run import LammpsError, find_lmp, run

FAILED_ENERGY = 1.0e6


def _moves(gamma, nsteps, temp, seed, relax_steps):
    if nsteps:
        return deck_mod.anneal(nsteps, temp, temp * 0.1, seed) + deck_mod.minimise(
            maxiter=relax_steps
        )
    return deck_mod.minimise(maxiter=relax_steps)


def madlib_runner(
    potential=None,
    target="xrd_glassy_carbon_zeng_2017.fq",
    gamma_ladder=(0.0, 0.3, 1.0, 3.0, 10.0),
    nsteps=0,
    relax_steps=200,
    temperature=1000.0,
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
    gammas = itertools.cycle(gamma_ladder)
    counter = itertools.count()

    def calculate(symbols, positions, cell, seed=0, output_filename=None, **_kwargs):
        index = next(counter)
        gamma = next(gammas)
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
            moves=_moves(gamma, nsteps, temperature, int(seed) + index, relax_steps),
            masses={i + 1: mass for i in range(len(set(symbols)))},
        )

        try:
            thermo, out = run(workdir, text, lmp, ranks=ranks, links=links)
        except (LammpsError, OSError) as exc:
            # A failed individual must not stop the run; give it an energy no
            # selection will keep and say why in the metadata.
            return (np.asarray(positions, float), list(symbols),
                    np.asarray(cell, float), FAILED_ENERGY,
                    {"madlib_ok": False, "madlib_error": str(exc)[:400], "gamma": gamma})

        meta = {
            "madlib_ok": True,
            "gamma": gamma,
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

    return calculate
