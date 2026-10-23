# Triton: GPU fine-tuning only (no LAMMPS build); TF uses the CUDA wheels in the venv.
PREFIX=${PREFIX:-/scratch/work/zarrout1/madlib-grace}
NPROC=${NPROC:-4}
UV=${UV:-$HOME/.local/bin/uv}
