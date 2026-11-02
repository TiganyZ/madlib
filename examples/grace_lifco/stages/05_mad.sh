#!/bin/bash
# MAD-biased MD with the foundation model: the configurations to label.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/05_mad
done_if "$OUT/DONE"
mkdir -p "$OUT"

START=$WORK/03_melt_quench/quenched.data
TARGET=$WORK/04_reference/target_xrd.dat
MODEL=$WORK/02_foundation/saved_model.yaml
NATOMS=$(awk '/ atoms$/ {print $1; exit}' "$START")

mad_run() {
    local tag=$1 temp=$2 gamma=$3 seed=$4
    mkdir -p "$OUT/$tag"
    cd "$OUT/$tag"
    lmp -in "$EX/lammps/in.mad" -var start "$START" -var pair grace/fs -var coeff "$MODEL" \
        -var target "$TARGET" -var gamma "$gamma" -var temp "$temp" -var seed "$seed" \
        -var steps "$MAD_STEPS" -var every "$MAD_EVERY" -var prefix mad
    echo "$tag T=$temp gamma=$gamma rel_error=$("$PY" -I "$EX/stages/thermo.py" mean log.lammps f_xrd[2])"
}

gamma_for() {
    local ebias=$1 d0=$2
    "$PY" -I -c "print(f'{2 * $NATOMS * $ebias / $d0:.6g}')"
}

i=0
for temp in $MAD_TEMPS; do
    i=$((i + 1))
    mad_run "T${temp}_g0" "$temp" 0.0 $((SEED + i))
done

D0=$("$PY" -I "$EX/stages/thermo.py" mean "$OUT/T${REF_TEMP}_g0/log.lammps" f_xrd[3])
echo "unbiased dissimilarity at $REF_TEMP K: $D0"
: > "$OUT/gammas.txt"

for temp in $MAD_TEMPS; do
    for ebias in $MAD_EBIAS; do
        i=$((i + 1))
        gamma=$(gamma_for "$ebias" "$D0")
        echo "$temp $ebias $gamma" >> "$OUT/gammas.txt"
        mad_run "T${temp}_e${ebias}" "$temp" "$gamma" $((SEED + i))
    done
done
touch "$OUT/DONE"
