# Triton: GPU fine-tuning only (no LAMMPS build); TF uses the CUDA wheels in the venv.
PREFIX=${PREFIX:-/scratch/work/zarrout1/madlib-grace}
NPROC=${NPROC:-4}
UV=${UV:-$HOME/.local/bin/uv}
# Home is a 15 GB NFS quota; the TensorFlow wheels alone fill a third of it.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PREFIX/uv_cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PREFIX/pip_cache}
