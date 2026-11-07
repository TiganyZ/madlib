# Amorphous silica against a <sup>29</sup>Si powder lineshape

A classical potential supplies the energy and a graph neural network supplies
the observable. The potential keeps the glass a glass; the NMR lineshape
decides *which* glass.

The observable here is the one a shielding **tensor** buys over its trace. A
spectrum built by broadening isotropic shieldings gives every site a symmetric
peak and throws away the anisotropy of its environment. A static powder
spectrum *is* that anisotropy: a crystallite whose field direction is **n**
resonates at **n**·T·**n**, and what is measured is that over all
orientations — a skewed lineshape with edges at the principal values. Two
sites with the same isotropic shielding and different tensors are two
different patterns.

The tensors come from the models of

> **Graph-neural-network predictions of solid-state NMR parameters from
> spherical tensor decomposition** — Chiheb Ben Mahmoud, Louise A. M. Rosset,
> Jonathan R. Yates, Volker L. Deringer, [arXiv:2412.15063][paper]
> · [data][data] · [code][code]

two NequIP networks for SiO<sub>2</sub>: the magnetic shielding tensor (`ms`,
irreps 0e+1e+2e, ppm) and the electric field gradient (`efg`, irreps 2e,
traceless).

[paper]: https://arxiv.org/abs/2412.15063
[data]: https://github.com/cbenmahm/anistropic-nmr-parameters-data
[code]: https://github.com/cbenmahm/nequip-nmr-parameters

## What you need

A LAMMPS with MADLIB and the embedded interpreter:

```sh
cmake ../cmake -D PKG_MADLIB=on -D MADLIB_ENABLE_PYTHON=on -D PKG_MANYBODY=on
```

`PKG_MANYBODY` is for `pair_style vashishta`, whose `SiO.1990.vashishta`
parameters ship with LAMMPS; set `LAMMPS_POTENTIALS` so it is found. Nothing
in the decks depends on that choice — swap in a GAP, a MACE or
`pair_style zero` and only the energy changes.

Then the environment and the models:

```sh
export PREFIX=/scratch/$USER/nmr_sio2     # not $HOME: torch is a lot of files
./setup/make_venv.sh                      # torch, e3nn, and the authors' NequIP fork
./setup/fetch.sh                          # the two models, the statistics, the structures
./setup/make_inputs.py $PREFIX/models/aSiO2_models.xyz $PREFIX/runs 0
```

**The fork is not optional.** The models' `config.yaml` names a
`MagResTensorModel` builder that upstream NequIP does not have, so without
[cbenmahm/nequip-nmr-parameters][code] the weights cannot be rebuilt into a
model at all. `make_venv.sh` clones it rather than installing it: it is pure
Python, so `PYTHONPATH` is all it needs, and nothing in the package is
executed until a model is loaded.

**Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.** A NequIP 0.6 checkpoint is a
pickled model object rather than a state dict, and torch 2.6 and later refuse
to unpickle one by default. The backend turns that failure into a sentence
saying so, but the switch is the environment's, and it is global — set it for
decks that load models you trust, not in your shell profile.

**One environment per model family.** That fork pins `e3nn < 0.6` and
`mace-torch` pins `0.4.4`; a venv that satisfies both is luck. `compute
mad/property/python` reads `PYTHONPATH`, so a different venv per deck is the
whole of the switch:

```sh
export PYTHONPATH=$PREFIX/madlib/python:$($PREFIX/venv/bin/python -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
```

## Run it

First, check the model against the prediction the structure file already
carries — `ML_ISD_ms`, made by the authors with `nequip-evaluate`, which owes
nothing to this code. It tests the graph, the edge convention, the per-species
de-standardisation, the spherical-to-Cartesian transform and the Voigt
ordering in one number:

```sh
lmp -in .../lammps/in.tensors -var data silica.data
./setup/check_model.py ms_tensors.dump $PREFIX/models/aSiO2_models.xyz 0
```

Then predict, and score against the target without biasing anything:

```sh
cd $PREFIX/runs
export MADLIB_NMR_MODELS=$PREFIX/models
export LAMMPS_POTENTIALS=/path/to/lammps/potentials
lmp -in .../lammps/in.predict -var data silica.data -var target si29_dft.dat
```

The model paths come from the environment, not from `-var`: **LAMMPS does not
substitute `${...}` inside a quoted argument**, and a config string with
spaces in it has to be quoted, so a `-var` path would arrive as the literal
text. `madlib_models` expands `$VAR` in every config value itself, which is
the way round it; an absolute path, or the `file=`/`entry=` JSON route, work
as they stand.

That writes `si29_predicted.dat` (the grid, the prediction, the target) and
`nmr_sites.dump` (per site: isotropic shielding, span, V<sub>zz</sub>,
η<sub>Q</sub>). `f_si29[2]` in the thermo output is the relative error, and
`f_si29[3]` the dissimilarity *D* that sizes gamma: pick gamma so that
γD/2 is a small fraction of the cohesive energy, then

```sh
lmp -in .../lammps/in.mad -var data silica_rattled.data -var target si29_dft.dat \
    -var gamma 200.0 -var every 5 -var steps 2000
```

`silica_rattled.data` is the same structure with every atom displaced by
0.12 Å, which is the thing the bias has to undo: watch `f_si29[2]` fall while
`c_pepair` stays sane. Report `c_pepair`, never `pe` — `fix mad/*` puts the
bias into the potential energy by default.

## Five things that decide whether the number means anything

**The target here is a DFT lineshape, not a measurement.** The structure file
carries CASTEP shielding tensors (`QM_ms`) alongside the models' own
predictions (`ML_ISD_ms`), and `make_inputs.py` builds the target from the
first. A real experiment goes in exactly the same place, as two columns of
shift and intensity. Keeping the DFT one is what makes the example checkable:
you know what the answer should be.

**The numbers to expect**, measured on frame 0 of the 300-atom set (100 Si, a
16.552 Å cube):

| | |
|---|---|
| per-atom tensors vs the file's own `ML_ISD_ms` | max **3e-4 ppm**, rms 1e-4, over shieldings spanning 153–452 ppm |
| DFT lineshape peak | **−106.4 ppm** (the ML one, −106.8) |
| `rel_error` of the ML lineshape against the DFT target | **0.0142** |
| `D` for that, hence gamma for 0.02 eV/atom | 2.0e-4, so gamma ≈ 840 |
| the rattled structure, before biasing | `rel_error` 0.625 |
| after 100 steps at that gamma, `every 5` | `rel_error` 0.285, `c_pepair` −2284 → −2588 eV |

The first row is the one that matters: 3e-4 ppm is float32 round-off, so the
whole chain — graph, edge convention, per-species de-standardisation,
spherical-to-Cartesian transform, Voigt ordering — reproduces the authors' own
`nequip-evaluate` output. The 0.0142 is the model's error against DFT and is
the floor; 100 steps is a demo, not a converged run, and a real one ramps
gamma over the whole trajectory.

As a one-line sanity check with no model at all: the mean stored shielding for
Si, 755.3167/√3 = 436.1 ppm, gives 328.4 − 436.1 = −107.7 ppm.

**The model's own error is a floor.** `si29_ml_in_file.dat` is the lineshape
of the paper's own prediction for this structure, and `si29_dft.dat` the DFT
one. The gap between them is the model, and no amount of gamma closes it —
driving `rel_error` below it means the structure is being distorted to absorb
a model error.

**The referencing constant belongs to the model.** δ = 328.4 − σ is used
here; σ<sub>ref</sub> is specific to the level of theory the model was fitted
at, and borrowing one from elsewhere is the standard way to get a spectrum
that is uniformly wrong. `fix mad/nmr/aniso` will not start until told whether
it has shieldings (`reference`) or shifts (`convention shift`), because
broadening the one against the other mirrors the spectrum and the scale fit
then makes the mirror image look like a fit.

**Orientations are a convergence parameter.** 512 in the biased deck, 1024 in
the prediction, 4096 for the target. Raise them until the lineshape stops
moving at the σ in use; a coarse grid resolves the pattern's edges into a comb
of Gaussians that is smooth enough to look like a spectrum. There is a sharper
test: an exact powder average cannot depend on how the sample is oriented, so
it exerts no torque, and what a finite grid leaves behind is an antisymmetric
part of Σ r⊗f. It falls as 1/N. If a long run's angular momentum drifts,
that is where to look.

**<sup>17</sup>O is quadrupolar, and this fix is a spin-½ lineshape.** The
EFG model is read here for prediction only — V<sub>zz</sub> and η<sub>Q</sub>
per site in `nmr_sites.dump`. An <sup>17</sup>O MAS spectrum is dominated by
the second-order quadrupolar interaction, which is a different lineshape from
**n**·T·**n** and is not what `fix mad/nmr/aniso` computes. Biasing on the
<sup>29</sup>Si pattern and *reporting* the oxygen parameters is the honest
split.

## Cost

Six message-passing layers over a 5.533 Å graph on 300 atoms cost **about 26
seconds per evaluation** — forward and backward — on one CPU core. That is the
whole cost of a biased step: the 100-step run above took 522 s with `every 5`,
so twenty evaluations. Use a GPU (`device=cuda`) for anything longer, and
`every N` regardless: this is a slowly varying restraint, not a force that
needs resolving every femtosecond. The computes in `in.predict` share one
loaded model and one cached forward pass, so the isotropic shielding, the span
and the tensor cost one model evaluation between them.

`pbc yes` is required rather than preferred, and it means **one MPI rank**.
Those six layers read 33 Å around every atom, so the ghost-cloud route would
need a 33 Å ghost shell — hundreds of images of a 16 Å cell — and would still
be approximate, because a ghost atom receives no messages of its own. See the
warning in `doc/src/compute_mad_property.rst`.

## Files

| | |
|---|---|
| `env.sh` | every path and pin |
| `setup/make_venv.sh` | torch, e3nn, the NequIP fork |
| `setup/fetch.sh` | the models, the statistics, the structures (the `.xyz` are git-lfs) |
| `setup/make_inputs.py` | one frame → LAMMPS data files and the target lineshapes |
| `lammps/in.predict` | predict and score, gamma = 0 |
| `lammps/in.mad` | the biased run |
