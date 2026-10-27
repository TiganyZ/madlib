#!/bin/bash
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/03_melt_quench
done_if "$OUT/quenched.data"
mkdir -p "$OUT"
cd "$OUT"
lmp -in "$MELT_INPUT" \
    -var data "$WORK/01_structures/mixture.data" \
    -var model "$WORK/02_foundation/saved_model.yaml" \
    -var seed "$SEED" -var tmelt "$TMELT" -var out quenched.data
