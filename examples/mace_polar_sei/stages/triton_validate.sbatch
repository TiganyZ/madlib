#!/bin/bash
#SBATCH --job-name=mace-polar-validate
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Usage: triton_validate.sbatch WORK
set -eu
ROOT=/scratch/work/zarrout1/mace-polar
export XDG_CACHE_HOME=$ROOT/cache
cd "$1"
"$ROOT/venv/bin/python" "$ROOT/stages/validate.py" "$1/sets" "$1/models" "$1/validate" cuda
