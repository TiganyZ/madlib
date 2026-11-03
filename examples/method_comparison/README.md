# Four ways to fit a diffraction pattern, at the same cost

| | what it optimises | how it makes a front |
|---|---|---|
| **EZGA** | the potential only | it does not -- one minimum, scored against the data afterwards |
| **EZGA + exp** | the potential *and* the misfit | directly, from one run |
| **MAD** | the potential plus a bias, over a ramp | by scanning gamma, one run per point |
| **HRMC** | the same sum, by Metropolis moves | by scanning gamma, one chain per point |

The first two differ by one line -- whether the misfit is an objective -- so
the pair measures what the measurement buys. The last two differ by how they
move: molecular dynamics down a biased landscape, or Monte Carlo on the same
sum.

## Why there is a budget file

These methods spend compute so differently that comparing them at "the
settings everyone uses" compares the settings. The one currency they share is
the **potential evaluation**: one call returning energy and forces for the
whole cell.

```
MAD     one per MD step, one per minimiser iteration
EZGA    one per MD or relaxation step, per individual
HRMC    one per attempted move -- full_energy re-evaluates the cell
```

`budget.py` takes a single number and derives each method's parameters from
it, so changing the comparison means changing one number:

```sh
./budget.py --budget 60000
```

MAD and HRMC divide their budget across the gamma ladder; EZGA keeps all of
it, because it produces a front from one run. That asymmetry is a property of
the methods and the point is to expose it, not to hide it.

MAD and HRMC spend a fixed number of steps, so their budget is arithmetic. The
GA's is not: how much a generation costs depends on how many children survive
its collision and duplicate gates, which is not known in advance. A generation
count therefore cannot spend a budget exactly -- the first attempt at one
overshot by 41%. So `run_ga.py` enforces the budget **inside the calculator**,
which declines to run once the spend is reached, and `budget.py` only supplies
a generation ceiling. The overshoot is then bounded by one individual.

## Run it

Put `start.data`, `population.xyz`, the target and `gap_files/` in `data/`,
then:

```sh
LMP=/path/to/lmp PY=/path/to/venv/bin/python ./run_all.sh
./analyse.py --runs runs --lmp /path/to/lmp --out results --target target.fq
```

`run_all.sh` takes method names if you want one at a time: `./run_all.sh mad`.
The legs are `seeds mad hrmc ezga ezga_exp`, and **`seeds` must run before
either GA leg** -- it scores the starting population, and the two GA legs run
concurrently so they must not both write that file. Timings land in
`runs/timing.csv` as they finish, so a run that dies still leaves the costs of
what came before.

---

# The GA legs

## Install

EZGA is an unmodified PyPI dependency -- `madlib_ezga` attaches to it through
its config loader, so there is nothing to register.

```sh
python3 -m venv ezga-venv            # Python >= 3.10; 3.12 is what this was run on
./ezga-venv/bin/pip install ezga-lib # pulls in sage_lib, ase, numpy, scipy, pydantic
./ezga-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**torch is not optional even on CPU.** `ezga.factory` imports its `LJ_torch`
calculator unconditionally, so importing the factory without torch fails with
`ModuleNotFoundError: No module named 'torch'`.

Then put `madlib/python` on the path -- `madlib_ezga` is not pip-installed:

```sh
export PYTHONPATH=/path/to/madlib/python${PYTHONPATH:+:$PYTHONPATH}
export LMP=/path/to/lammps/build/lmp
```

Versions this was run against: `ezga-lib 1.2.32`, `sage_lib 0.1.10.4`,
`torch 2.15.0+cpu`, `ase 3.29.0`.

## What the GA needs in `data/`

| | |
|---|---|
| `population.xyz` | the starting population, one frame per individual |
| `population_scored.xyz` | written by the `seeds` leg: the same frames carrying `energy=` and `xrd_rel_error=` |
| `target.fq` | the measured pattern, two columns |
| `gap_files/` | the potential |

`make_population.py` builds the population from one structure by isotropic
strain, with a hard minimum-distance guard:

```sh
./make_population.py --start data/start.data --out data/population.xyz \
    --size 24 --strain 0.03 --rattle 0.006 --min-distance 1.17
```

Build it from a **neutral** starting structure, not from another method's
results. Seeding the GA from a finished MAD ladder made both legs report a
best misfit that was one of the seeds, unchanged.

## Invoking it

```sh
./run_all.sh seeds          # score the population once, ~1 min
./run_all.sh ezga           # potential only
./run_all.sh ezga_exp       # potential + measured misfit
```

Or call the driver directly:

```sh
./run_ga.py --out runs/ezga_energy_exp \
    --population data/population_scored.xyz --data-dir data \
    --lmp "$LMP" --target target.fq --objectives energy+exp \
    --budget 60000 --generations 16 --size 24 \
    --md-steps 100 --relax-steps 150 --gammas 1,3,10,30,100 --ranks 16
```

## A minimal EZGA calculation, without this harness

Everything the GA legs do, in one file. The settings are the measured ones --
see **Probe your operators** below for where they come from.

```python
from ezga.evaluator.features import (feature_density,
                                     feature_minimum_interatomic_distance)
from ezga.factory import build_default_engine, load_config
from ezga.variation.crossover import crossover_two_cut
from ezga.variation.mutation import mutation_random_strain, mutation_rattle

from madlib_ezga import Pdf, Potential, Xrd, madlib_runner, random_subset
from madlib_ezga.features import feature_mean_coordination, feature_sp_fractions
from madlib_ezga.front import write_front
from madlib_ezga.objectives import (objective_energy_per_atom,
                                    objective_xrd_rel_error)

# The calculator: a short biased LAMMPS relaxation per individual. It returns
# a moved structure plus c_pepair, the potential alone.
runner = madlib_runner(
    potential=Potential(style="turbogap", model="gap_files/carbon.gap"),
    target="target.fq",
    pdf=Pdf(rmin=0.1, rmax=8.0, nbins=201, kde=0.1),
    xrd=Xrd(target="target.fq", qmin=1.045, qmax=9.970, nq=201,
            output="qFq", weighting="xray"),
    gamma_ladder=(1.0, 3.0, 10.0, 30.0, 100.0),
    nsteps=100, relax_steps=150,
    budget=60000,                       # the calculator stops at this spend
    lmp="/path/to/lmp", ranks=16,
    links=["data/gap_files", "data/target.fq"],
    workroot="out/work",
)

config = load_config({
    "output_path": "out/engine",
    "max_generations": 16,              # a ceiling; the budget binds first
    "rng": 12345,
    "resume": False,
    "population": {
        "dataset_path": "data/population_scored.xyz",
        "db_path": "out/db",
        "filter_duplicates": True,
        "collision_factor": 0.60,       # a 0.912 A gate; see the traps
        "collision_action": "deny",
        "ef_bounds": (-12.0, 0.0),      # eV/atom; defaults to None
    },
    "multiobjective": {"size": 24, "selection_method": "nondominated_sorted"},
    "thermostat": {"initial_temperature": 0.25, "constant_temperature": True},
    "variation": {"initial_mutation_rate": 1.0, "crossover_probability": 0.30},
    "evaluator": {
        # Features are not objectives: they are the space selection keeps
        # individuals apart in.
        "features_funcs": [feature_density(),
                           feature_minimum_interatomic_distance(),
                           feature_mean_coordination(),
                           feature_sp_fractions()],
        "objectives_funcs": [objective_energy_per_atom(),
                             objective_xrd_rel_error()],
    },
    "mutation_funcs": [
        mutation_rattle(std=0.20, constraints=[random_subset(0.02, 0)], seed=0),
        mutation_rattle(std=0.10, constraints=[random_subset(0.10, 3)], seed=3),
        mutation_random_strain(max_strain=0.04, seed=1),
    ],
    "crossover_funcs": [crossover_two_cut],
    "simulator": {"calculator": runner, "seed": 12345, "mode": "relaxation"},
})

build_default_engine(config).run()

# EZGA keeps every structure but writes no front of its own.
front = write_front("out/db", "out")
print(len(front), "front points,", int(runner.spend["steps"]), "evaluations")
```

Drop `objective_xrd_rel_error()` and set `gamma_ladder=(0.0,)` for the
potential-only leg. Those two changes are the whole difference between the two
legs -- with the misfit as an objective the bias must be on, and without it the
bias must be off, or the legs differ in more than their objective list.

## What comes out

| | |
|---|---|
| `runs/<leg>/timing.json` | wall time, evaluations against budget, individuals, how many were declined once the budget was gone |
| `runs/<leg>/pareto.csv`, `pareto.xyz` | the front, reconstructed by `front.py` |
| `runs/<leg>/engine/logger/gen_*_full.json` | per-generation operator statistics -- the file that tells you whether the GA worked |
| `runs/<leg>/db/` | the SAGE database holding every structure |

`export_db.py` writes every structure out of that database as LAMMPS data
files, which is what `analyse.py` then re-scores. Without it the GA legs
contribute nothing scorable, because the work directories hold only
`start.data`, the structure *before* its relaxation.

Three scripts read the generation logs and are how the run is checked:

```sh
python3 ga_stats.py      runs/ezga_energy_exp   # per-operator attempts/rejections
python3 ga_objectives.py runs/ezga_energy_exp   # objective values of the parents
python3 operator_probe.py data/population.xyz   # acceptance rate of each operator
```

---

## Reading the result

`analyse.py` re-runs **every** structure at `gamma 0` and reads the same two
numbers off each. This is not optional bookkeeping. The methods' own logs are
not comparable:

- HRMC's energy is the biased one, because `fix_modify mad energy yes` is
  what makes the Metropolis test work
- EZGA already reports `c_pepair`, the potential alone
- MAD writes a structure and its own thermo, on its own grid

Scoring them identically afterwards is the only way the four fronts mean the
same thing.

## Traps

**Report `c_pepair`, not `pe`.** `fix mad/*` sets `thermo_energy` on by
default, so `pe` carries the bias. An energy axis with the bias in it is not
the potential's energy and no front built on it is meaningful.

**Check r_max against both the target and the box.** The pair distribution is
truncated at `r_max`, which caps the q-resolution at about `2*pi/r_max`; if
the measured pattern has sharper features than that, the misfit has a floor
set by the grid rather than the structure. And `r_max` must stay below half
the box or the pair distribution counts the cell's own periodic images.
Neither answer transfers between targets or between cell sizes -- scan it for
the case in hand.

### Traps specific to the GA

**Probe your operators. A dead operator is silent.** EZGA's collision gate
sits *between* the mutation and the relaxation, so a child that clashes is
discarded before LAMMPS can repair it, and `mutation_fails_counts` in the
generation log counts those rejections. A run of this example had
**272 rattle attempts and 272 rejections -- a 100% failure rate for the whole
campaign** -- because the rattle was 0.12 A on every atom of a cell whose
closest pair was already 1.178 A, against a 1.140 A gate. The GA never moved
an atom; the only live operator was a +-2% cell strain, and the duplicate rate
climbed to 0.69 while the best energy sat unchanged for 24 generations. None
of that raised an error. `operator_probe.py` measures the acceptance rate of
each candidate operator against the gate before a run:

```
                             accepted @ 1.140 A   @ 0.912 A
rattle 0.20 A on every atom              0.00        0.00
rattle 0.20 A on  2% of atoms            0.08        0.92
rattle 0.10 A on 10% of atoms            0.17        1.00
strain 0.04                              1.00        1.00
crossover_two_cut                        0.92        0.92
crossover_voronoi_bisector_surface       0.00        0.00
```

An all-atom rattle cannot pass a minimum-distance gate in a dense cell: the
gate tests the closest pair out of N, so the chance that no pair falls under it
goes to zero with N however small the displacement. Hence `random_subset`,
which rattles a few atoms hard instead of every atom gently.

**`collision_factor` is a ratio, so work out the distance.** It multiplies the
sum of the two covalent radii: 0.75 for carbon is `0.75 * 2 * 0.76 = 1.140 A`,
which is *inside* the first-neighbour shell of a dense amorphous cell. 0.60
gives 0.912 A, which rejects fused atoms and admits real moves.

**Use `collision_action: "deny"`, not the default `"remove"`.** `"remove"`
strips the overlapping atoms and keeps the structure -- silently changing N,
and so the density and the energy per atom. It strips exactly the
worst-coordinated sites, which flatters both objectives.

**Not every crossover survives a periodic cell.** Of the eight EZGA ships,
`crossover_voronoi_bisector_surface` puts atoms 0.005 A apart and *every* child
is rejected, and `crossover_fractional_mixing` averages Cartesian positions,
which across a periodic boundary is not a structure at all. `crossover_two_cut`
is a real slab splice that survives. With `crossover_funcs` empty -- the
default -- there is no recombination at all, and a GA without recombination is
a parallel hill-climber.

**`ef_bounds` defaults to `None`, so nothing filters a placeholder energy.**
The shipped documentation shows `[-20.0, 100.0]` as though it were the default.
It is not, and without it the children the calculator declines once the budget
is spent enter the archive at their placeholder energy -- a pilot accumulated
1,897 of them and the duplicate rate reached 0.97.

**EZGA only ever simulates offspring, so score the seeds yourself.** A seed
read from a file keeps whatever energy the file gave it; with none,
`objective_energy_per_atom` turns the missing value into 0.0, which is the
worst value a minimisation can see, and the seeds then sit in the parent pool
at a fake objective for the whole run -- 24 of 24 parents at generation 0 and
still 6 of 24 at generation 24. `score_population.py` runs each seed at
`run 0` and writes `energy=` and `xrd_rel_error=` into the extxyz comment
line, which `sage_lib` reads back into `AtomPositionManager.E` and `.metadata`.

**Name objectives carefully.** EZGA string-matches `hull`, `formation`,
`novelty`, `similarity`, `anomality`, `fit` and `regression` against the
function, module and class names to decide whether an objective needs the whole
population. `misfit` contains `fit`, which is why the objective here is
`objective_xrd_rel_error`.

**`gamma` cannot be inherited.** EZGA hands the calculator the parent's hash
but not the parent's metadata, so a child cannot read the gamma its parent was
relaxed at. `runner.py` derives it from that hash instead, so every child of
one parent shares a bias strength rather than a family being split across the
ladder by arrival order. True lineage inheritance would need an `ISimulator`
override.
