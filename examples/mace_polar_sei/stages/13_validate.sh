#!/bin/bash
# Test-set errors of every model, graphite's pressure, and the PolarMACE that goes on to MD (validate.py).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$COMBINED/13_validate
done_if "$OUT/validation.json"
need "$COMBINED/12_train/models/dielectric.model" "run stage 12 first"
mkdir -p "$OUT"
to_triton "$STAGES/" "$TRITON_ROOT/stages/"
sbatch_wait "$TRITON_ROOT/stages/triton_validate.sbatch" "$OUT/JOBID" "$TRITON_WORK"
rsync -a "$TRITON:$TRITON_WORK/validate/" "$OUT/"
cat "$OUT/validation.json"
