#!/bin/bash
# The synthetic experiment: MD with REF_MODEL on a Triton GPU, then the mean XRD of its frames.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/04_reference
done_if "$OUT/target_xrd.dat"
need "$WORK/03_melt_quench/quenched.data" "run stage 03 first"
if [ ! -f "$OUT/run/DONE" ]; then
    write_args "$OUT/run" "$REF_MODEL" "$TRITON_ROOT/lammps/in.reference" \
        "start=$(remote "$WORK")/03_melt_quench/run/quenched.data" "temp=$REF_TEMP" "seed=$SEED" \
        "equil=$REF_EQUIL_STEPS" "sample=$REF_SAMPLE_STEPS" "every=$REF_EVERY"
    triton_runs "$OUT" reference run
fi
cd "$OUT"
"$PY" -I "$GRACE_STAGES/xrd_files.py" placeholder placeholder.dat
mpirun -np 4 "$LMP" -log log.lammps -screen none -var species "$SPECIES" -in "$EX/lammps/in.observe" \
    -var start "$WORK/03_melt_quench/quenched.data" -var frames run/frames.dump \
    -var target placeholder.dat -var out frames_xrd.dat
"$PY" -I "$GRACE_STAGES/xrd_files.py" average frames_xrd.dat target_xrd.dat
