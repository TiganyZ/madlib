#!/bin/bash
# Optional, by hand: LSC-MACE (mace-scf's LocalSplitCharges) from scratch on stage 12's sets, on alt's GPU.
# Charge transfer is local and conserves the formal charges, so the electrostatics need no SCF.
# Needs setup/make_lsc_venv.sh. Its models are not used by stages 13-14; see README for the loop.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$COMBINED/15_lsc
done_if "$OUT/lsc.model"
need "$COMBINED/12_train/sets/polar_train.xyz" "run stage 12 first"
mkdir -p "$OUT"
cd "$OUT"
for split in train test; do
    "$PY" -I "$STAGES/lsc_sets.py" "$COMBINED/12_train/sets/polar_$split.xyz" "${split}_e.xyz"
    "$PY" -I "$STAGES/formal_charges.py" "${split}_e.xyz" "$split.xyz"
done
"$LSC_PREFIX/venv/bin/python" "$LSC_PREFIX/mace-scf/scripts/run_train.py" \
    --name lsc --config "$STAGES/lsc_config.yaml" \
    --train_file train.xyz --valid_fraction 0.1 --test_file test.xyz --seed "$CLUSTER_SEED" \
    --model LocalSplitCharges --formal_charges_from_data --compute_polarizability True --compute_stress True \
    --E0s "$("$PY" -I "$STAGES/lsc_e0s.py" train.xyz)" --r_max 6.0 --hidden_irreps "$LSC_HIDDEN_IRREPS" \
    --atomic_multipoles_max_l 1 --atomic_multipoles_smearing_width 1.5 \
    --electrostatic_pbc_method mixed_periodic \
    --batch_size 4 --valid_batch_size 4 --eval_interval 5 --error_table PerAtomRMSE \
    --device cuda --default_dtype float64 --restart_latest --save_cpu
