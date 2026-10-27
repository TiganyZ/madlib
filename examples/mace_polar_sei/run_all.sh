#!/bin/bash
# The MACE-POLAR-1 SEI workflow for one cation (CATION=Na|Li), driven from alt.
# Stages 01-06 per system, every system at once (their MD waits on Triton); 07-14 on the merge.
# Every stage caches its output, so a rerun resumes.
set -eu
MP_EX=$(cd "$(dirname "$0")" && pwd)
source "$MP_EX/stages/lib.sh"

per_system() {
    local system=$1
    for stage in "$MP_EX"/stages/0[1-6]_*.sh; do
        echo ">>> $system $(basename "$stage")"
        SYSTEM=$system bash "$stage"
    done
}

all_systems() {
    mkdir -p "$WORK/logs"
    local pids=()
    for system in $SYSTEMS; do
        per_system "$system" > "$WORK/logs/$system.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

combined() {
    for stage in "$MP_EX"/stages/0[7-9]_*.sh "$MP_EX"/stages/1[0-4]_*.sh; do
        echo ">>> combined $(basename "$stage")"
        bash "$stage"
    done
}

all_systems
combined
