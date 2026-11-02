# MACE-POLAR-1 for a graphite | SEI | electrolyte interface

This example samples five SEI systems with MAD and labels the sampled frames with DFT. It then
fine-tunes MACE-POLAR-1, which has long-range electrostatics, together with a
dipole/polarizability model, and equilibrates interface slabs with the result. Run it once per
cation:

```sh
CATION=Na ./run_all.sh
CATION=Li ./run_all.sh
REFERENCE=vasp CATION=Na ./run_all.sh     # cluster labels from VASP instead of FHI-aims
```

Stages cache their output under `runs/mace_polar_<cation>/`, so a rerun resumes. Parameters are
in `params.sh`.

| stage | where | what |
|---|---|---|
| 01 structures | alt | packed mixture and crystals per system |
| 03 melt_quench | Triton GPU | melt and quench |
| 04 reference | Triton GPU | synthetic experiment: MD with a larger model, then its XRD |
| 05 mad | Triton GPU | unbiased and MAD-biased MD |
| 06 select | alt GPU | farthest-point selection in descriptor space; train/test split |
| 07 merge | alt | all systems in one set |
| 08 label | Roihu | VASP statics |
| 09 clusters | alt | neutral clusters cut from the MAD frames |
| 10 label clusters | Roihu | cluster dipoles and polarizabilities (FHI-aims or VASP) |
| 11 periodic | alt | labelled frames as extxyz |
| 12 train | Triton GPU | PolarMACE fine-tune, dipole stage, dielectric model |
| 13 validate | Triton GPU | test errors and graphite stability; picks the MD model |
| 14 equilibrate | Triton GPU | relax and NPT for each slab in `$SLABS` |
| 15 lsc (optional) | alt GPU | LSC-MACE (mace-scf local split charges) from scratch |

Stages 01–06 run once per system; 07 onwards run on the merged set.

## Notes

- **MD in LAMMPS.** PolarMACE enters LAMMPS through `fix external`
  (`lammps/mace_potential.lmp`, `stages/mace_lammps.py`), with D3(BJ) from
  `pair_style dispersion/d3`. The MAD fixes work unchanged.
- **Graphite systems.** These run their MD with MACE-MH-1 (`GRAPHITE_MD_MODEL`), because
  foundation PolarMACE is unstable on graphite. Stage 13 checks whether the fine-tuned model is
  stable there, and falls back to MH-1 if it is not.
- **Polarizability model.** The polarizability needs its own model (`dielectric.model`), because
  PolarMACE's field response only moves charge between atoms.
- **LSC-MACE.** Stage 15 trains a LocalSplitCharges model from scratch. Grow its training set by
  hand: train, run MD, label the frames it gets wrong, retrain.

## Setup

| | |
|---|---|
| `setup/make_venv.sh` | MACE environment (alt and Triton) |
| `setup/build_lammps_python.sh` | alt: LAMMPS shared library with MADLIB and D3 |
| `setup/triton_lammps_python.sh` | the same on Triton |
| `setup/make_lsc_venv.sh` | alt: mace-scf environment for stage 15 |
