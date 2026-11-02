# Shared by the stage scripts. Sourced.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source "${PARAMS:-$EX/params.sh}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}
# One thread per MPI rank; TensorFlow otherwise starts one per core in every rank.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}

say() { printf '\n=== %s ===\n' "$1"; }

lmp() { mpirun -np "$NRANKS" "$LMP" -log log.lammps -screen none -var species "$SPECIES" "$@"; }

done_if() {
    if [ -e "$1" ]; then
        echo "[cached] $1"
        exit 0
    fi
}
