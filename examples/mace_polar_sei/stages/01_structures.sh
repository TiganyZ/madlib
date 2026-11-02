#!/bin/bash
# SYSTEM's packing and crystals: grace_sei_na's own when it has them (same seed; Na keeps its patched
# crystals), else grace_sei_na's structures.py.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/01_structures
done_if "$OUT/mixture.data"
GRACE_OUT=$GRACE_RUNS/$SYSTEM/01_structures
mkdir -p "$OUT"
if [ -f "$GRACE_OUT/mixture.data" ]; then
    cp "$GRACE_OUT/mixture.data" "$OUT/"
    if [ -f "$GRACE_OUT/crystals.xyz" ]; then
        cp "$GRACE_OUT/crystals.xyz" "$OUT/"
    fi
else
    "$PY" -I "$STRUCTURES" "$OUT" "$SEED"
fi
