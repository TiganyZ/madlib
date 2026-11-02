#!/bin/bash
# The VASP-labelled train/test frames of stage 08 (OMat24 level) as MACE extxyz.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$COMBINED/11_periodic
done_if "$OUT/test.xyz"
need "$COMBINED/07_label/test.pkl.gz" "run stage 08 first"
mkdir -p "$OUT"
for split in train test; do
    "$PY" -I "$STAGES/to_extxyz.py" "$COMBINED/07_label/$split.pkl.gz" "$OUT/$split.xyz"
done
