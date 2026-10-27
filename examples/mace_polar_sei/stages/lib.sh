# Shared by the stage scripts. Sourced. With SYSTEM set, WORK and SPECIES are that system's.
STAGES=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MP_EX=$(cd "$STAGES/.." && pwd)
GRACE_STAGES=$MP_EX/../grace_lifco/stages
source "$MP_EX/../grace_lifco/env.sh"
source "$MP_EX/params.sh"
if [ -n "${SYSTEM:-}" ]; then
    source "$MP_EX/../grace_sei_na/params/$SYSTEM.sh"
fi
case ${SYSTEM:-} in
    graphite_*)
        MD_MODEL=$GRAPHITE_MD_MODEL
        REF_MODEL=$GRAPHITE_REF_MODEL
        SELECT_MODEL=$GRAPHITE_MD_MODEL ;;
esac
MACE_PY=/l/zarrout1/mace-polar/venv/bin/python
export XDG_CACHE_HOME=/l/zarrout1/mace-polar/cache

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

# The Triton path of a path under $RUNS.
remote() { echo "$TRITON_ROOT/runs/${1#"$RUNS"/}"; }

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

# One deck run: DIR/deck.args, one argument of stages/run_deck.py per line: the deck, the given
# name=value pairs, then the PolarMACE ones for MODEL.
write_args() {
    local dir=$1
    local model=$2
    shift 2
    mkdir -p "$dir"
    printf '%s\n' "$@" \
        "potential=$TRITON_ROOT/lammps/mace_potential.lmp" "model=$model" "species=$SPECIES" \
        device=cuda "dtype=$MD_DTYPE" "d3=$MD_D3" > "$dir/deck.args"
}

# The run directories RUN... under STAGE (a stage directory under $RUNS) as one Triton GPU array
# job, remembered as LABEL; then the stage directory comes back.
triton_runs() {
    local stage=$1
    local label=$2
    shift 2
    local list=$stage/runs_$label.txt
    : > "$list"
    for run in "$@"; do
        echo "$(remote "$stage")/$run" >> "$list"
    done
    on_triton "mkdir -p $TRITON_ROOT/logs"
    to_triton "$STAGES/" "$TRITON_ROOT/stages/"
    to_triton "$MP_EX/lammps/" "$TRITON_ROOT/lammps/"
    to_triton "$stage/" "$(remote "$stage")/"
    sbatch_wait "--array=0-$(($# - 1)) $TRITON_ROOT/stages/triton_deck.sbatch $(remote "$list")" \
        "$stage/JOBID_$label"
    rsync -a "$TRITON:$(remote "$stage")/" "$stage/"
}
