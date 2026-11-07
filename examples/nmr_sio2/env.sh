# Sourced by every script here. All paths and pins live in this file.
EX=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MADLIB_DIR=$(cd "$EX/../.." && pwd)

# Where the downloads, the venv and the runs go. Not a home directory: the
# venv is a few hundred thousand files and torch alone is gigabytes.
PREFIX=${PREFIX:-/l/$USER/nmr_sio2}

DATA_REPO=cbenmahm/anistropic-nmr-parameters-data
DATA_REF=main
# The magnetic shielding model of the paper: the ISD parameterisation, which
# is the one its Figure 5 uses.
MS_MODEL=models/MS/final/isd/minimal
# The electric field gradient model, for the quadrupolar parameters.
EFG_MODEL=models/EFG/minimal
STRUCTURES=data/aSiO2_models/aSiO2_fast_ml_qm_ms_efg.xyz
STATS=data/data_mean_std.json

NEQUIP_FORK=https://github.com/cbenmahm/nequip-nmr-parameters
PYTHON_VERSION=3.12

# 29Si shielding reference. sigma_ref for the level of theory the model was
# fitted at, NOT a universal constant -- see the README.
SI_REFERENCE=328.4

VENV=$PREFIX/venv
FORK_DIR=$PREFIX/nequip-fork
MODELS=$PREFIX/models
RUNS=$PREFIX/runs
LMP=${LMP:-$PREFIX/lammps/build/lmp}
PY=$VENV/bin/python
