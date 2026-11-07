# python/

Two packages, both optional, both importable from this directory.

| | |
|---|---|
| `madlib_models` | any machine-learned per-atom property, into any MADLIB observable |
| `madlib_ezga` | MADlib as the local optimiser of a genetic algorithm |

---

# madlib_models -- a model, an observable, no glue

The quantity between a structure and a spectrum is one per-atom number: a
core-electron binding energy for XPS, a dipole for IR, a chemical shielding --
isotropic or the whole tensor -- for NMR. `compute mad/property/python` has
always accepted such a number from a Python model and asked for one weighted
backward pass in return. This package is the part that should not have to be
written again for every model.

```lammps
compute cebe all mad/property/python madlib_models ncomp 1 cutoff 12.0 &
            config "backend=mace model=cebe.model types=C,O"
fix     xps  all mad/xps c_cebe target measured.dat range 280 300 201 gamma 5.0
```

Nothing in those two lines is specific to MACE. Swap the backend and the same
pair of lines biases an NMR lineshape instead.

## Config keys

Whitespace- or `;`-separated `key=value`, so a config with spaces needs the
quotes LAMMPS strips. An unknown key is an error, not an ignored argument.

| key | |
|---|---|
| `backend` | `mace`, `nequip_nmr` or `callable` (required) |
| `model` | the model file, or a NequIP training directory |
| `types` | element of each LAMMPS atom type, in type order: `Si,O` (required) |
| `output` | which of the model's outputs to read |
| `quantity` | how to reduce it: `scalar` `vector` `tensor6` `tensor9` `iso` `span` `vzz` `etaq` `component:zz` |
| `scale`, `shift` | an affine map, for a unit or sign convention |
| `device`, `dtype` | default: cuda if torch sees one, the model's own dtype |
| `head` | head name, for a multi-head MACE model |
| `rcut` | the graph cutoff in `pbc yes` mode; default the model's `r_max` |
| `fn` | `module:function`, for `backend=callable` |
| `file`, `entry` | read the keys from a JSON file's named entry instead |

## What each observable wants

| | `ncomp` | |
|---|---|---|
| `fix mad/xps`, `fix mad/nmr` | 1 | a per-atom scalar |
| `fix mad/ir` | 3 | a per-atom dipole, summed by the fix |
| `fix mad/nmr/aniso` | 6 | a symmetric tensor, Voigt order `xx yy zz xy xz yz` |

`ncomp` and `quantity` have to agree, and a mismatch is an error rather than a
reshape.

## Backends

**`mace`** reads any MACE model's per-atom outputs: `node_energy` (the
default, and where a model fitted to a per-atom target puts it), `charges`,
`atomic_dipoles`, `node_feats`. A per-graph output such as `energy` has one
row where a property needs one per atom and is refused on that ground.

**A dipole is IR; a polarizability is not.** `fix mad/ir` correlates the
total dipole of the trajectory, so a model with an `atomic_dipoles` output
(`quantity=vector`, `ncomp 3`) drops straight into it — the fix sums the
per-atom vectors. A polarizability tensor is a different observable: its
autocorrelation is the Raman spectrum, and MADLIB has no Raman fix. What this
package will do with one is predict it and reduce it — `tensor6` for the
whole thing, `iso` and `span` for the invariants a Raman lineshape is built
from — which is the per-atom half of that calculation, not the bias.

On a ghost cloud the result is exact for a model that passes one message and
approximate for a deeper one, whatever the cutoff: LAMMPS lists the neighbours
of owned atoms, so a ghost receives no messages of its own. The package says
so once per run, and `pbc yes` has no such error. A cutoff *shorter* than
`r_max` is a different thing and an error.

**`nequip_nmr`** reads the anisotropic NMR parameter models of
[Ben Mahmoud *et al.*](https://github.com/cbenmahm/anistropic-nmr-parameters-data)
-- shielding (`parameter=ms`) and electric field gradient (`parameter=efg`)
tensors -- and needs their [NequIP fork](https://github.com/cbenmahm/nequip-nmr-parameters)
and the `data_mean_std.json` from the data repository, because the network's
output is standardised per species and per irreducible block and nothing in
the weights says so. `pbc yes` only: six layers over a 5.5 Å graph read 33 Å.
See [`examples/nmr_sio2`](../examples/nmr_sio2/README.md).

To put a new quantity or a whole new spectrum behind this, see
[`docs/ADDING-AN-OBSERVABLE.md`](../docs/ADDING-AN-OBSERVABLE.md).

**`callable`** is the escape hatch: `fn=my_module:my_property`, handed the
geometry on the autograd tape, returning per-atom values. The caching, the
reduction and the backward pass stay here.

## One forward pass per geometry

Models are cached by what identifies them -- the backend, the file, the
device, the head -- and each keeps its last forward pass with the autograd
graph intact. Two computes over one model, say the isotropic shielding and the
span of the same tensor, cost one model evaluation a step and two backward
passes. The cache key is the geometry, not the timestep, because a Monte Carlo
fix re-prices a trial move inside one step.

## Testing it

```sh
python -m pytest python/tests/test_madlib_models.py python/tests/test_nequip_nmr_tensor.py
LMP=<build>/lmp tests/run.sh mlprop nmraniso
```

The first two need only torch (and e3nn for the tensor reconstruction, which
is checked against a transcription of the authors' own script). `mlprop`
checks the MACE path against MACE's own data pipeline, which is what catches a
flipped edge convention; it needs `MADLIB_MACE_PYTHON` and
`MADLIB_MACE_SITE` pointing at an environment with `mace-torch`.

---

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
