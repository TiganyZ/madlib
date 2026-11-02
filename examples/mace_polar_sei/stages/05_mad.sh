#!/bin/bash
# MAD-biased MD with PolarMACE + D3, one Triton GPU per run: the unbiased runs first, whose
# dissimilarity at REF_TEMP sets gamma, then every MAD_TEMPS x MAD_EBIAS run.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/05_mad
done_if "$OUT/DONE"
need "$WORK/04_reference/target_xrd.dat" "run stage 04 first"
mkdir -p "$OUT/inputs"
cp "$WORK/04_reference/target_xrd.dat" "$OUT/inputs/"
START=$(remote "$WORK")/03_melt_quench/run/quenched.data
TARGET=$(remote "$OUT")/inputs/target_xrd.dat
NATOMS=$(awk '/ atoms$/ {print $1; exit}' "$WORK/03_melt_quench/quenched.data")

mad_args() {
    local tag=$1 temp=$2 gamma=$3 seed=$4
    write_args "$OUT/$tag" "$MD_MODEL" "$TRITON_ROOT/lammps/in.mad" "start=$START" "target=$TARGET" \
        "gamma=$gamma" "temp=$temp" "seed=$seed" "steps=$MAD_STEPS" "every=$MAD_EVERY" prefix=mad
}

gamma_for() {
    local ebias=$1 d0=$2
    "$PY" -I -c "print(f'{2 * $NATOMS * $ebias / $d0:.6g}')"
}

unbiased=()
i=0
for temp in $MAD_TEMPS; do
    i=$((i + 1))
    mad_args "T${temp}_g0" "$temp" 0.0 $((SEED + i))
    unbiased+=("T${temp}_g0")
done
triton_runs "$OUT" unbiased "${unbiased[@]}"

D0=$("$PY" -I "$GRACE_STAGES/thermo.py" mean "$OUT/T${REF_TEMP}_g0/log.lammps" f_xrd[3])
echo "unbiased dissimilarity at $REF_TEMP K: $D0"
: > "$OUT/gammas.txt"
biased=()
for temp in $MAD_TEMPS; do
    for ebias in $MAD_EBIAS; do
        i=$((i + 1))
        gamma=$(gamma_for "$ebias" "$D0")
        echo "$temp $ebias $gamma" >> "$OUT/gammas.txt"
        mad_args "T${temp}_e${ebias}" "$temp" "$gamma" $((SEED + i))
        biased+=("T${temp}_e${ebias}")
    done
done
triton_runs "$OUT" biased "${biased[@]}"
for run in "${unbiased[@]}" "${biased[@]}"; do
    echo "$run rel_error=$("$PY" -I "$GRACE_STAGES/thermo.py" mean "$OUT/$run/log.lammps" f_xrd[2])"
done
date > "$OUT/DONE"
