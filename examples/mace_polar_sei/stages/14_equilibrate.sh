#!/bin/bash
# Each graphite | SEI | electrolyte slab in $SLABS: relax y/z, then SLAB_MD_STEPS of NPT, one A100 each (array job).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$COMBINED/14_equilibrate
done_if "$OUT/DONE"
need "$COMBINED/13_validate/chosen_model" "run stage 13 first"
mkdir -p "$OUT"
slabs=$(cd "$SLABS" && ls -d */ | tr -d / | tr '\n' ' ')
for slab in $slabs; do
    to_triton "$SLABS/$slab/structure.xyz" "$TRITON_WORK/slabs/$slab/"
done
to_triton "$STAGES/" "$TRITON_ROOT/stages/"
n=$(echo $slabs | wc -w)
sbatch_wait "--array=0-$((n - 1)) $TRITON_ROOT/stages/triton_equilibrate.sbatch" "$OUT/JOBID" \
    "$TRITON_WORK" "$SLAB_MD_STEPS" $slabs
rsync -a --exclude md.xyz "$TRITON:$TRITON_WORK/equilibrate/" "$OUT/"
date > "$OUT/DONE"
