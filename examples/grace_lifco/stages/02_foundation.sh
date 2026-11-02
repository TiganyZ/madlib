#!/bin/bash
# GRACE-FS-OMAT in the C++ format grace/fs reads.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/02_foundation
done_if "$OUT/saved_model.yaml"
mkdir -p "$OUT"
cp "$GRACE_CACHE/checkpoints/$FOUNDATION"/{model.yaml,checkpoint.index,checkpoint.data-00000-of-00001} "$OUT/"
cd "$OUT"
"$VENV_DIR/bin/grace_utils" -p model.yaml -c checkpoint.index export -sf
