#!/bin/bash
#SBATCH --job-name=mace-polar-train
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Usage: triton_train.sbatch WORK FOUNDATION EF_EPOCHS DIPOLE_EPOCHS DIEL_EPOCHS
set -eu
ROOT=/scratch/work/zarrout1/mace-polar
export XDG_CACHE_HOME=$ROOT/cache
export PATH=$ROOT/venv/bin:$PATH
cd "$1"
SETS=$1/sets MODELS=$1/models FOUNDATION=$2 EF_EPOCHS=$3 DIPOLE_EPOCHS=$4 DIEL_EPOCHS=$5 DEVICE=cuda \
    bash "$ROOT/stages/train.sh"
