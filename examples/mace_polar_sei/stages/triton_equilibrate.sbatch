#!/bin/bash
#SBATCH --job-name=mace-polar-equil
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%A_%a.out
# Usage: sbatch --array=0-N triton_equilibrate.sbatch WORK MD_STEPS SLAB...   (one slab per array task)
set -eu
ROOT=/scratch/work/zarrout1/mace-polar
export XDG_CACHE_HOME=$ROOT/cache
work=$1
steps=$2
shift 2
slabs=("$@")
slab=${slabs[$SLURM_ARRAY_TASK_ID]}
cd "$work"
"$ROOT/venv/bin/python" "$ROOT/stages/equilibrate.py" "$work/slabs/$slab/structure.xyz" \
    "$(cat "$work/validate/chosen_model")" "$work/equilibrate/$slab" "$steps" cuda
