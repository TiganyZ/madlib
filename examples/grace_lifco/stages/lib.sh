# Shared by the stage scripts. Sourced.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source "$EX/params.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}

say() { printf '\n=== %s ===\n' "$1"; }

lmp() { mpirun -np "$NRANKS" "$LMP" -log log.lammps -screen none "$@"; }

done_if() {
    if [ -e "$1" ]; then
        echo "[cached] $1"
        exit 0
    fi
}
