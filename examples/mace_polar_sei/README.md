# MACE-POLAR-1 for a graphite | SEI | electrolyte interface

This example fine-tunes MACE-POLAR-1 on the labelled frames from
[`grace_sei_na`](../grace_sei_na). It also trains a dipole/polarizability model
on DFT of neutral clusters. It then relaxes and equilibrates interface slabs with
the result. Run it once per cation, after `grace_sei_na` has labelled its frames.

```sh
CATION=Na ./run_all.sh
CATION=Li ./run_all.sh
REFERENCE=vasp CATION=Na ./run_all.sh     # cluster labels from VASP instead of FHI-aims
```

Every stage caches its output under `runs/mace_polar_<cation>/`, so a rerun
resumes where it stopped. All parameters are in `params.sh`.

| stage | where | what |
|---|---|---|
| 01 clusters | workstation | neutral clusters of whole molecules and ions, cut from the MAD frames |
| 02 label clusters | DFT cluster | dipole, polarizability, energy and forces per cluster (`autoplex_soap_turbo/workflows/sei_response`) |
| 03 periodic | workstation | `grace_sei_na`'s labelled frames as extxyz |
| 04 train | GPU | the three fits below |
| 05 validate | GPU | test errors and the graphite pressure; picks the model for MD |
| 06 equilibrate | GPU, one per slab | relax, then NPT MD of each slab in `$SLABS`, with D3(BJ) dispersion |

## Models

| model | what it is | trained on |
|---|---|---|
| `polar_ef.model` | PolarMACE fine-tuned from `polar-1-m` | periodic E/F/stress and cluster E/F |
| `polar_dipole.model` | `polar_ef`, continued | cluster E/F and dipoles |
| `dielectric.model` | AtomicDielectricMACE, from scratch | cluster dipoles and polarizabilities |

Stage 05 uses `polar_dipole` for MD when its periodic force error is within 10%
of `polar_ef`'s. Otherwise it uses `polar_ef`.

A separate model is needed for the polarizability. PolarMACE responds to a field
only by moving charge between atoms, so it cannot learn the polarizability
within an atom.

## Level of theory

All the DFT is PBE. The labels contain no dispersion, so D3(BJ) is added at run
time (`stages/calc.py`).

- **FHI-aims (default).** DFPT on the isolated cluster, with tight species
  defaults. Its basis gives a different energy zero, so cluster energies get
  weight 0; forces, dipoles and polarizabilities are fitted.
- **VASP.** `IDIPOL 4` and `LEPSILON`, with α = V/4π (ε − 1).

## Setup on the GPU cluster

`setup/triton_venv.sh` builds the environment (torch, `mace-torch[polar]`,
torch-dftd) and caches the foundation models. The `stages/*.sbatch` files are
the batch scripts.
