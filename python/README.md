# madlib_ezga -- MADlib as an EZGA plugin

Puts an experimental diffraction pattern into a multi-objective genetic
algorithm, and uses the MAD bias as the local optimiser that reaches it.

XRD is degenerate: many structures fit one pattern. The useful answer is not
one structure but the Pareto front of (potential energy, experimental misfit).

## How it attaches

EZGA has no entry-point plugin system and needs none: its config loader
resolves `"madlib_ezga.runner:madlib_runner"`, or a bare path to a `.py` file.
EZGA is an unmodified dependency.

Two slots are filled:

| EZGA slot | What MADlib does |
|---|---|
| `simulator.calculator` | **MAD relaxation.** EZGA's calculator is a structure *transformer*: given a structure it returns a possibly different one plus its energy. A short LAMMPS run under `pair_style turbogap` (or MACE) with `fix mad/xrd` is exactly that, so children arrive already moved downhill on `E_pot + bias`. |
| `evaluator.objectives_funcs` | **The misfit**, `f_mad[2]`. The calculator leaves it in the individual's metadata, so scoring is a dictionary lookup, not a second LAMMPS call. |

The energy axis is EZGA's own `objective_energy()`, or
`objective_energy_per_atom()` from here when cells differ in size.

## Relaxing the cell

`madlib_runner(box_relax="iso")` lets each individual relax its cell as well as
its positions, under `E_pot + bias`. Use it when the energy and the experiment
prefer different densities, so that density becomes an axis of the front.

## Things that are easy to get wrong

- **Report `c_pepair`, never `pe`.** `fix mad/*` sets `thermo_energy` on by
  default, unlike most LAMMPS fixes, so `pe` silently contains the bias. An
  energy axis contaminated by the bias makes the front meaningless.
- **Vary gamma across the population.** One gamma pulls every individual to
  the same corner of the plane. `madlib_runner` takes a `gamma_ladder` and
  picks from it by the parent's hash, so the population spans the front and
  siblings share a bias strength. It cannot be inherited: EZGA passes the
  calculator the parent's hash but not the parent's metadata.
- **`selection_method` is `"nondominated_sorted"`.** `pareto_rank` is the
  selector's own internal name for the same thing and is *not* an accepted
  config value -- the `SelectionMethod` enum rejects it before the selector
  sees it.
- **Probe your variation operators before trusting a run.** EZGA's collision
  gate sits between the mutation and the relaxation, so a child that clashes is
  discarded before LAMMPS can repair it, and an operator whose children always
  clash is silently dead rather than an error. See the GA traps in
  [`examples/method_comparison`](../examples/method_comparison/README.md).

## Use

```python
from ezga.factory import build_default_engine, load_config
from madlib_ezga import Potential, madlib_runner, objective_xrd_rel_error
from madlib_ezga.objectives import objective_energy_per_atom

runner = madlib_runner(
    potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
    target="xrd_glassy_carbon_zeng_2017.fq",
    gamma_ladder=(0.0, 0.3, 1.0, 3.0, 10.0),
    links=["/path/to/gap_files", "/path/to/xrd_glassy_carbon_zeng_2017.fq"],
    lmp="/path/to/lmp", ranks=4,
)

config = load_config({
    "evaluator": {"objectives_funcs": [objective_energy_per_atom(),
                                       objective_xrd_rel_error()]},
    "multiobjective": {"selection_method": "nondominated_sorted"},
    "simulator": {"calculator": runner},
    ...
})
build_default_engine(config).run()
```

Afterwards, `front.write_front(db_path, out_dir)` reconstructs the front from
the SAGE database: EZGA keeps every structure but writes no front of its own.

[`examples/method_comparison`](../examples/method_comparison/README.md) has the
install, a complete runnable config with the measured operator settings, and
the traps worth reading before a long run.

## Layout

| file | |
|---|---|
| `deck.py` | the only place a LAMMPS deck is written -- RMC, MAD and the GA share it, which is what makes them comparable |
| `lammps_run.py` | run a deck as a subprocess, parse the log, read the final frame |
| `runner.py` | the EZGA calculator |
| `objectives.py` | the EZGA objectives |
| `front.py` | non-dominated sort over a finished run |
| `constraints.py` | atom-level constraints for the variation operators |
| `tests/test_madlib_ezga.py` | everything that needs no LAMMPS: deck text, log parsing, objectives, front, coordination |
| `tests/test_roundtrip.py` | against a real LAMMPS: round trip through the dump, triclinic cells, and that the reported energy is the potential alone |

```sh
python3 tests/test_madlib_ezga.py

MADLIB_LMP=/path/to/lmp \
MADLIB_GAP_DIR=/path/to/gap_files \
MADLIB_XRD_TARGET=/path/to/target.fq \
  python3 tests/test_roundtrip.py /path/to/structure.data
```
