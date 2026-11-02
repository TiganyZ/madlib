#!/bin/bash
# Compare the outputs of two runs: NAME=a ./run_all.sh; NAME=b ./run_all.sh; ./check_reproducible.sh a b
set -eu
source "$(dirname "$0")/env.sh"

A=$RUNS/$1
B=$RUNS/$2

same_file() {
    if cmp -s "$A/$1" "$B/$1"; then echo "identical  $1"; else echo "DIFFERENT  $1"; fi
}

same_numbers() {
    "$PY" -I - "$A/$1" "$B/$1" <<'PY'
import sys
import numpy as np
import pandas as pd
a, b = (pd.read_pickle(p, compression="gzip") for p in sys.argv[1:])
assert list(a["name"]) == list(b["name"]), "different structures"
de = np.max(np.abs(a["energy"].values - b["energy"].values))
df = max(np.abs(x - y).max() for x, y in zip(a["forces"], b["forces"]))
print(f"{'identical ' if de == 0 and df == 0 else 'max diff  '} {sys.argv[1].split('/runs/')[1].split('/', 1)[1]}  dE {de:.2e} eV  dF {df:.2e} eV/A")
PY
}

same_file 01_structures/mixture.data
same_file 03_melt_quench/quenched.data
same_file 04_reference/target_xrd.dat
same_file 05_mad/gammas.txt
same_file 06_select/to_label.xyz
same_numbers 07_label/train.pkl.gz
same_numbers 07_label/test.pkl.gz
same_file 08_finetune/seed/$SEED/saved_model.yaml
same_file 08_finetune/seed/$SEED/saved_model.asi
same_file 09_validate/test_errors.json
same_file 10_mad_finetuned/summary.txt
