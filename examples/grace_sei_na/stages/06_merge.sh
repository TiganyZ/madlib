#!/bin/bash
set -eu
source "$(dirname "$0")/../../grace_lifco/stages/lib.sh"
OUT=$WORK/06_select
done_if "$OUT/to_label.xyz"
"$PY" -I "$NA_EX/stages/merge_systems.py" "$RUNS/$NAME" "$OUT" $SYSTEMS
