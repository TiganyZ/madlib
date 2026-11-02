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

## Run it

Put `start.data`, `population.xyz`, the target and `gap_files/` in `data/`,
then:

```sh
LMP=/path/to/lmp PY=/path/to/venv/bin/python ./run_all.sh
./analyse.py --runs runs --lmp /path/to/lmp --out results --target target.fq
```

`run_all.sh` takes method names if you want one at a time: `./run_all.sh mad`.
Timings land in `runs/timing.csv` as they finish, so a run that dies still
leaves the costs of what came before.

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

## Three traps

**Report `c_pepair`, not `pe`.** `fix mad/*` sets `thermo_energy` on by
default, so `pe` carries the bias. An energy axis with the bias in it is not
the potential's energy and no front built on it is meaningful.

**Set `collision_action: "deny"` in the GA.** EZGA's default is `"remove"`,
which strips overlapping atoms and keeps the structure -- silently changing
N, and so the density and the energy per atom. It strips exactly the
worst-coordinated sites, which flatters both objectives.

**Check r_max against both the target and the box.** The pair distribution is
truncated at `r_max`, which caps the q-resolution at about `2*pi/r_max`; if
the measured pattern has sharper features than that, the misfit has a floor
set by the grid rather than the structure. And `r_max` must stay below half
the box or the pair distribution counts the cell's own periodic images.
Neither answer transfers between targets or between cell sizes -- scan it for
the case in hand.
