# MACE-POLAR-1 for a graphite | SEI | electrolyte interface

MACE-POLAR-1 runs every stage of this example:
- The MAD sampling of each system.
- The synthetic experiment.
- The fine-tune.
- The dipole/polarizability model.
- The slab MD.

It carries long-range electrostatics, which matter for the salts, the electrolyte and the SEI. Run it once per
cation.

```sh
CATION=Na ./run_all.sh
CATION=Li ./run_all.sh
REFERENCE=vasp CATION=Na ./run_all.sh     # cluster labels from VASP instead of FHI-aims
```

alt drives the stages. Every stage caches its output under `runs/mace_polar_<cation>/`,
so a rerun resumes. All parameters are in `params.sh`. Each system's species,
melt temperature and MAD temperatures come from `../grace_sei_na/params/`.

| stage | where | what |
|---|---|---|
| **per system** | | |
| 01 structures | alt | the packed mixture and crystals (`grace_sei_na`'s, when it has run) |
| 03 melt_quench | Triton GPU | NPT melt and quench, or NVT for the graphite systems; reports any molecule that broke up |
| 04 reference | Triton GPU, alt | the synthetic experiment: MD with `polar-1-l`, then the mean XRD of its frames |
| 05 mad | Triton GPU array | unbiased MD, whose dissimilarity sets γ, then MAD at each temperature and bias strength |
| 06 select | alt GPU | farthest-point selection in PolarMACE descriptor space, then the train/test split |
| **merged** | | |
| 07 merge | alt | every system's selection in one set |
| 08 label | Roihu | VASP statics at OMat24's level of theory |
| 09 clusters | alt | neutral clusters of whole molecules and ions, cut from the MAD frames |
| 10 label clusters | Roihu | dipole, polarizability, energy and forces per cluster |
| 11 periodic | alt | the labelled frames as extxyz |
| 12 train | Triton GPU | the three fits below |
| 13 validate | Triton GPU | test errors, graphite's pressure and MD stability; picks the model for MD |
| 14 equilibrate | Triton GPU, one per slab | each slab in `$SLABS`: relax, then NPT MD |
| 15 lsc (optional) | alt GPU | LSC-MACE from scratch on the same sets |

## MD with PolarMACE in LAMMPS

PolarMACE computes its electrostatics in k-space over the whole cell. A LAMMPS
pair style only sees local atoms and ghosts, so the model enters through
`fix external` instead:

- `lammps/mace_potential.lmp` sets up the potential: `fix external` calls
  `stages/mace_lammps.py` for PolarMACE's energy, forces and virial on the whole
  cell, and `pair_style dispersion/d3` adds D3(BJ).
- The MAD fixes (`fix mad/xrd`) work as with any other potential.
- `stages/run_deck.py` runs a deck inside the MACE Python environment.
- The runs are serial, one GPU each.

## Graphite

PolarMACE-1 blows up on graphite within a few MD steps, at float32 or float64 and with or
without D3. Its charge flow does not hold up on an extended semimetal, which is
absent from its molecular training data. So the two graphite systems run their melt, synthetic
experiment, MAD and selection with MACE-MH-1 instead (`GRAPHITE_MD_MODEL=mh-1:omat_pbe`, with
the `matpes_r2scan` head as their experiment). MH-1 is a local model with no long-range
electrostatics.

Stage 13 checks whether fine-tuning repairs PolarMACE on graphite. Every PolarMACE model runs
1 ps of MD on bulk graphite and on a test frame of each graphite system, beside MH-1 as the
baseline. If the model chosen for MD does not stay bounded there, the slabs run with MH-1.

## Models

| model | what it is | trained on |
|---|---|---|
| `polar_ef.model` | PolarMACE fine-tuned from `polar-1-m` | periodic E/F/stress and cluster E/F |
| `polar_dipole.model` | `polar_ef`, continued | cluster E/F and dipoles |
| `dielectric.model` | AtomicDielectricMACE, from scratch | cluster dipoles and polarizabilities |

Stage 13 uses `polar_dipole` for MD when its periodic force error is within 10%
of `polar_ef`'s. Otherwise it uses `polar_ef`.

A separate model is needed for the polarizability. PolarMACE responds to a field
only by moving charge between atoms, so it cannot learn the polarizability
within an atom.

## Option: LSC-MACE (local split charges)

`stages/15_train_lsc.sh` trains mace-scf's `LocalSplitCharges` model from scratch on stage 12's
sets, on alt's GPU (`setup/make_lsc_venv.sh` builds its environment; mace-scf needs mace 0.3.14).
Each atom carries a formal charge (`stages/formal_charges.py`: the unit's charge, spread over its
most electronegative element). The model learns charge transfers along local bonds, which conserve
that total, and computes the long-range electrostatics from the resulting charges and dipoles. No
SCF is needed. It also reads out the polarizability directly, so the cluster dipoles and
polarizabilities train the same model as the energies and forces.

It starts without a foundation model, so a few hundred frames are not enough on their own. Train
it iteratively by hand, using the calculator `mace_scf.calculators.localsources.MACELocalSplitCharges`
with `formal_charges_key="formal_charges"`:

1. Train it with stage 15.
2. Run short MD with it on each system and slab.
3. Select frames where it is wrong or uncertain.
4. Label them through `autoplex_soap_turbo/workflows/lifco_mad` (periodic) or `sei_response` (clusters).
5. Add them to the sets and retrain.

mace-scf's self-consistent models (fixed-point SCF, QEq) are left out. They need SCF-aware
losses and training schedules, and their documentation describes them as a sandbox.

## Level of theory

All the DFT is PBE. The labels contain no dispersion, so D3(BJ) is added at run
time.

- **FHI-aims (default).** DFPT on the isolated cluster, with tight species
  defaults. Its basis gives a different energy zero, so cluster energies get
  weight 0; forces, dipoles and polarizabilities are fitted.
- **VASP.** `IDIPOL 4` and `LEPSILON`, with α = V/4π (ε − 1).

## Setup

| | |
|---|---|
| `setup/make_venv.sh` | the MACE environment (torch, `mace-torch[polar]`, torch-dftd), on alt and on Triton |
| `setup/build_lammps_python.sh` | alt: LAMMPS as a shared library with MADLIB and D3, into that environment |
| `setup/triton_lammps_python.sh` | the same on a Triton login node, serial, built in `/tmp` |
| `setup/make_lsc_venv.sh` | alt: the mace-scf environment for stage 15 |

The `stages/*.sbatch` files are the Triton batch scripts.
