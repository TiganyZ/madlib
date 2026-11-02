#!/bin/bash
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/01_structures
done_if "$OUT/mixture.data"
"$PY" -I "$STRUCTURES" "$OUT" "$SEED"
