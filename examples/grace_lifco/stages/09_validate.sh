#!/bin/bash
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/09_validate
done_if "$OUT/test_errors.json"
mkdir -p "$OUT"
cd "$OUT"
FOUND=$WORK/02_foundation/saved_model.yaml
FT=$WORK/08_finetune/seed/$SEED/saved_model.yaml
for split in train test; do
    echo "--- $split"
    "$PY" -I "$EX/stages/validate.py" "$WORK/07_label/$split.pkl.gz" "${split}_errors.json" \
        "foundation=$FOUND" "finetuned=$FT"
done
