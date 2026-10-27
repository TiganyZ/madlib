#!/bin/bash
# grace_sei_na's VASP-labelled train/test frames (OMat24 level) as MACE extxyz.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/03_periodic
done_if "$OUT/test.xyz"
need "$GRACE_RUNS/combined/07_label/test.pkl.gz" "run grace_sei_na (CATION=$CATION) through stage 07 first"
mkdir -p "$OUT"
for split in train test; do
    "$PY" -I "$STAGES/to_extxyz.py" "$GRACE_RUNS/combined/07_label/$split.pkl.gz" "$OUT/$split.xyz"
done
