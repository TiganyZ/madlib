# Shared by the stage scripts. Sourced.
STAGES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MP_EX=$(cd "$STAGES/.." && pwd)
source "$MP_EX/../grace_lifco/env.sh"
source "$MP_EX/params.sh"

say() { printf '\n=== %s ===\n' "$1"; }

done_if() {
    if [ -e "$1" ]; then
        echo "[cached] $1"
        exit 0
    fi
}

need() {
    if [ ! -e "$1" ]; then
        echo "missing $1: $2" >&2
        exit 1
    fi
}

on_triton() { ssh "$TRITON" "$@"; }

to_triton() {
    on_triton "mkdir -p $2"
    rsync -a "$1" "$TRITON:$2"
}

# Submits $1 (an sbatch file on Triton) once, remembering the job id in $2; waits; fails if the job did.
sbatch_wait() {
    local script=$1
    local idfile=$2
    shift 2
    if [ ! -s "$idfile" ]; then
        on_triton "sbatch --parsable $script $*" > "$idfile"
    fi
    local id state
    id=$(cat "$idfile")
    while true; do
        state=$(on_triton "sacct -n -X -P -j $id -o State" | sort -u | tr '\n' ' ')
        case $state in
            *RUNNING*|*PENDING*|*REQUEUED*|"") sleep "$TRITON_POLL" ;;
            "COMPLETED ") return 0 ;;
            *) echo "Triton job $id: $state" >&2; return 1 ;;
        esac
    done
}
