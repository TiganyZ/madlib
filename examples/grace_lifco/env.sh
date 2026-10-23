# Sourced by every script. All pins live here.
EX=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MADLIB_DIR=$(cd "$EX/../.." && pwd)

MACHINE=${MACHINE:-alt}
source "$EX/machines/$MACHINE.sh"

LAMMPS_REPO=https://github.com/lammps/lammps.git
LAMMPS_COMMIT=a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b
GRACE_REPO=https://github.com/yury-lysogorskiy/lammps.git
GRACE_COMMIT=b1f4a6c8cc4e0d0fc93d2dec3a1fcc83558062d4
PYTHON_VERSION=3.12

FOUNDATION=GRACE-FS-OMAT
LABELLER=GRACE-2L-OMAT-large-ft-AM-fp64
SELECTOR=$LABELLER   # needs a UQ signature; GRACE-FS-OMAT has none
SEED=20261022

LAMMPS_DIR=$PREFIX/lammps
GRACE_DIR=$PREFIX/lammps-grace
BUILD_DIR=$PREFIX/build
VENV_DIR=$PREFIX/venv
RUNS=$PREFIX/runs
LMP=$BUILD_DIR/lmp
PY=$VENV_DIR/bin/python

export GRACE_CACHE=$PREFIX/grace_models
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
