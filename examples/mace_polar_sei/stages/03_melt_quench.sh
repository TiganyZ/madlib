#!/bin/bash
# SYSTEM melted and quenched with PolarMACE + D3 on a Triton GPU: NPT, or NVT for the graphite systems,
# whose cell graphite fixes.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/03_melt_quench
done_if "$OUT/quenched.data"
case $SYSTEM in
    graphite_*) deck=in.anneal_nvt ;;
    *) deck=in.melt_quench ;;
esac
mkdir -p "$OUT/inputs"
cp "$WORK/01_structures/mixture.data" "$OUT/inputs/"
write_args "$OUT/run" "$MD_MODEL" "$TRITON_ROOT/lammps/$deck" \
    "data=$(remote "$OUT")/inputs/mixture.data" "seed=$SEED" "tmelt=$TMELT" out=quenched.data
triton_runs "$OUT" melt run
cp "$OUT/run/quenched.data" "$OUT/quenched.data"
"$MACE_PY" -I "$STAGES/unit_census.py" "$OUT/inputs/mixture.data" "$OUT/quenched.data" "$SPECIES"
