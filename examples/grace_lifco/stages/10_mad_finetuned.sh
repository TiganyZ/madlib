#!/bin/bash
# The finetuned model under the same MAD bias as the foundation, with its extrapolation grade.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/10_mad_finetuned
done_if "$OUT/summary.txt"
mkdir -p "$OUT"

START=$WORK/03_melt_quench/quenched.data
TARGET=$WORK/04_reference/target_xrd.dat
FIT=$WORK/08_finetune/seed/$SEED

mad_run() {
    local tag=$1 gamma=$2 seed=$3
    shift 3
    mkdir -p "$OUT/$tag"
    cd "$OUT/$tag"
    lmp -in "$EX/lammps/in.mad" -var start "$START" -var target "$TARGET" -var gamma "$gamma" \
        -var temp "$REF_TEMP" -var seed "$seed" -var steps "$MAD_STEPS" -var every "$MAD_EVERY" \
        -var prefix mad "$@"
}

report() {
    local tag=$1
    local log=$OUT/$tag/log.lammps
    printf '%-24s rel_error %8.4f  D %10.5g  max_gamma %7.3f\n' "$tag" \
        "$("$PY" -I "$EX/stages/thermo.py" mean "$log" f_xrd[2])" \
        "$("$PY" -I "$EX/stages/thermo.py" mean "$log" f_xrd[3])" \
        "$("$PY" -I "$EX/stages/thermo.py" mean "$log" v_gmax)"
}

GAMMA=$(awk -v t="$REF_TEMP" '$1 == t {g = $3} END {print g}' "$WORK/05_mad/gammas.txt")
for gamma in 0.0 "$GAMMA"; do
    mad_run "foundation_g$gamma" "$gamma" "$SEED" \
        -var pair grace/fs -var coeff "$WORK/02_foundation/saved_model.yaml"
    mad_run "finetuned_g$gamma" "$gamma" "$SEED" \
        -var pair "grace/fs extrapolation" -var coeff "$FIT/saved_model.yaml $FIT/saved_model.asi" -var extrap 1
done
for tag in foundation_g0.0 finetuned_g0.0 "foundation_g$GAMMA" "finetuned_g$GAMMA"; do
    report "$tag"
done | tee "$OUT/summary.txt"
