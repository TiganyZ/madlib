#!/bin/bash
# Farthest-point selection in GRACE feature space, N_PER_TEMP frames at each MAD temperature.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/06_select
done_if "$OUT/test_unlabelled.pkl.gz"
mkdir -p "$OUT"
cd "$OUT"

select_temp() {
    local temp=$1
    "$PY" -I "$EX/stages/pool.py" "$WORK/05_mad" "T${temp}_*" "pool_T$temp.pkl.gz"
    "$VENV_DIR/bin/grace_uq" select --dataset "pool_T$temp.pkl.gz" --model "$GRACE_CACHE/$SELECTOR" \
        -n "$N_PER_TEMP" --strategy fps-all --seed "$SEED" --no-element-stratified \
        --n-workers 1 --output "selected_T$temp.pkl.gz"
}

parts=()
for temp in $MAD_TEMPS; do
    select_temp "$temp"
    parts+=("selected_T$temp.pkl.gz")
done
"$PY" -I "$EX/stages/merge_selected.py" selected.pkl.gz "${parts[@]}"
"$PY" -I "$EX/stages/split.py" selected.pkl.gz "$WORK/01_structures/crystals.xyz" \
    "$TEST_FRACTION" "$SEED" "$OUT"
