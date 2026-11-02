#!/bin/bash
# grace_lifco's stages 01-06 for every system, MAX_PARALLEL at a time, the selections merged, then 07-09 on the merge.
set -eu
NA_EX=$(cd "$(dirname "$0")" && pwd)
STAGES=$NA_EX/../grace_lifco/stages

param() {
    PARAMS=$NA_EX/params/combined.sh bash -c "source '$STAGES/lib.sh'; echo \$$1"
}

per_system() {
    local system=$1
    for stage in "$STAGES"/0[1-6]_*.sh; do
        echo ">>> $system $(basename "$stage")"
        PARAMS=$NA_EX/params/$system.sh bash "$stage"
    done
}

all_systems() {
    local logs=$1
    local max=$2
    local running=0
    mkdir -p "$logs"
    for system in $(param SYSTEMS); do
        per_system "$system" > "$logs/$system.log" 2>&1 &
        running=$((running + 1))
        if [ "$running" -ge "$max" ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
}

combined() {
    PARAMS=$NA_EX/params/combined.sh bash "$NA_EX/stages/06_merge.sh"
    for stage in "$STAGES"/0[7-9]_*.sh; do
        echo ">>> combined $(basename "$stage")"
        PARAMS=$NA_EX/params/combined.sh bash "$stage"
    done
}

all_systems "$(param RUNS)/$(param NAME)/logs" "$(param MAX_PARALLEL)"
combined
