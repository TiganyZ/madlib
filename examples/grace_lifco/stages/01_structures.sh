#!/bin/bash
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/01_structures
done_if "$OUT/mixture.data"
"$PY" -I "$EX/stages/structures.py" "$OUT" "$SEED"
