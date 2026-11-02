#!/bin/bash
#SBATCH --job-name=mace-polar-md
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/work/zarrout1/mace-polar/logs/%x-%A_%a.out
# Usage: triton_deck.sbatch RUNS_FILE. The array index picks a line: a run directory holding deck.args.
set -eu
ROOT=/scratch/work/zarrout1/mace-polar
export XDG_CACHE_HOME=$ROOT/cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
dir=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$1")
cd "$dir"
mapfile -t args < deck.args
"$ROOT/venv/bin/python" "$ROOT/stages/run_deck.py" "${args[@]}"
date > DONE
