#!/bin/bash
# The training files (make_sets.py), then stages/train.sh as one Triton A100 job; the models come back to 04_train/models.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/04_train
done_if "$OUT/models/dielectric.model"
LABELLED=$WORK/02_label/${NAME}_clusters_$REFERENCE/labelled.xyz
need "$WORK/03_periodic/test.xyz" "run stage 03 first"
need "$LABELLED" "run stage 02 (VASP_PILOT=0) first"
mkdir -p "$OUT"

"$PY" -I "$STAGES/make_sets.py" "$WORK/03_periodic/train.xyz" "$WORK/03_periodic/test.xyz" \
    "$LABELLED" "$OUT/sets" "$TEST_FRACTION" "$CLUSTER_ENERGY_WEIGHT" "$CLUSTER_SEED"
to_triton "$STAGES/" "$TRITON_ROOT/stages/"
to_triton "$OUT/sets/" "$TRITON_WORK/sets/"
sbatch_wait "$TRITON_ROOT/stages/triton_train.sbatch" "$OUT/JOBID" \
    "$TRITON_WORK" "$FOUNDATION" "$EF_EPOCHS" "$DIPOLE_EPOCHS" "$DIEL_EPOCHS"
rsync -a --exclude checkpoints "$TRITON:$TRITON_WORK/models/" "$OUT/models/"
