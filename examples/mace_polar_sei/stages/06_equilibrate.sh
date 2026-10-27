#!/bin/bash
# Each graphite | SEI | electrolyte slab in $SLABS: relax y/z, then MD_STEPS of NPT, one A100 each (array job).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/06_equilibrate
done_if "$OUT/DONE"
need "$WORK/05_validate/chosen_model" "run stage 05 first"
mkdir -p "$OUT"
slabs=$(cd "$SLABS" && ls -d */ | tr -d / | tr '\n' ' ')
for slab in $slabs; do
    to_triton "$SLABS/$slab/structure.xyz" "$TRITON_WORK/slabs/$slab/"
done
to_triton "$STAGES/" "$TRITON_ROOT/stages/"
n=$(echo $slabs | wc -w)
sbatch_wait "--array=0-$((n - 1)) $TRITON_ROOT/stages/triton_equilibrate.sbatch" "$OUT/JOBID" \
    "$TRITON_WORK" "$MD_STEPS" $slabs
rsync -a --exclude md.xyz "$TRITON:$TRITON_WORK/equilibrate/" "$OUT/"
date > "$OUT/DONE"
