#!/bin/bash
# Farthest-point selection in PolarMACE descriptor space, N_PER_TEMP frames at each MAD temperature,
# then grace_lifco's seeded train/test split (crystals to train).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/06_select
done_if "$OUT/test_unlabelled.pkl.gz"
need "$WORK/05_mad/DONE" "run stage 05 first"
mkdir -p "$OUT"
cd "$OUT"
parts=()
for temp in $MAD_TEMPS; do
    "$PY" -I "$GRACE_STAGES/pool.py" "$WORK/05_mad" "T${temp}_*" "pool_T$temp.pkl.gz"
    "$MACE_PY" -I "$STAGES/select_fps.py" "pool_T$temp.pkl.gz" "$SELECT_MODEL" "$N_PER_TEMP" "$SEED" \
        "selected_T$temp.pkl.gz"
    parts+=("selected_T$temp.pkl.gz")
done
"$PY" -I "$GRACE_STAGES/merge_selected.py" selected.pkl.gz "${parts[@]}"
"$PY" -I "$GRACE_STAGES/split.py" selected.pkl.gz "$WORK/01_structures/crystals.xyz" \
    "$TEST_FRACTION" "$SEED" "$OUT"
