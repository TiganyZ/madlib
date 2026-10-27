#!/bin/bash
# The MACE-POLAR-1 workflow for one cation (CATION=Na|Li), on alt, after grace_sei_na for that cation.
# Every stage caches its output, so a rerun resumes. VASP_PILOT=10 ./run_all.sh times the cluster DFT first.
set -eu
MP_EX=$(cd "$(dirname "$0")" && pwd)
for stage in "$MP_EX"/stages/0[1-6]_*.sh; do
    echo ">>> $(basename "$stage")"
    bash "$stage"
done
