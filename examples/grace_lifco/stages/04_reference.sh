#!/bin/bash
# Target XRD from labeller MD: the stand-in for a measured pattern.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/04_reference
done_if "$OUT/target_xrd.dat"
mkdir -p "$OUT"
cd "$OUT"
if [ ! -f frames.dump ]; then
    "$PY" -I "$EX/stages/reference_md.py" "$WORK/03_melt_quench/quenched.data" \
        "$GRACE_CACHE/$LABELLER" "$REF_TEMP" "$REF_EQUIL_STEPS" "$REF_SAMPLE_STEPS" "$REF_EVERY" \
        "$SEED" frames.dump.part
    mv frames.dump.part frames.dump
fi
"$PY" -I "$EX/stages/xrd_files.py" placeholder placeholder.dat
lmp -in "$EX/lammps/in.observe" -var start "$WORK/03_melt_quench/quenched.data" \
    -var frames frames.dump -var target placeholder.dat -var out frames_xrd.dat
"$PY" -I "$EX/stages/xrd_files.py" average frames_xrd.dat target_xrd.dat
