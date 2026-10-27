#!/bin/bash
# The three fits, in $SETS (make_sets.py output), models to $MODELS. Runs inside triton_train.sbatch.
#   polar_ef      PolarMACE from $FOUNDATION on periodic + cluster E/F/S
#   polar_dipole  polar_ef continued on cluster E/F + dipoles (loss energy_forces_dipole: no stress)
#   dielectric    AtomicDielectricMACE from scratch on cluster dipoles + polarizabilities
set -eu
FOUNDATION=${FOUNDATION:-polar-1-m}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-20261027}
EF_EPOCHS=${EF_EPOCHS:-60}
EF_LR=${EF_LR:-1e-4}
DIPOLE_EPOCHS=${DIPOLE_EPOCHS:-20}
DIPOLE_LR=${DIPOLE_LR:-5e-5}
DIEL_EPOCHS=${DIEL_EPOCHS:-300}
BATCH=${BATCH:-2}
MACE_TRAIN=${MACE_TRAIN:-mace_run_train}
HERE=$(cd "$(dirname "$0")" && pwd)

common() {
    echo --device="$DEVICE" --seed="$SEED" --default_dtype=float64 \
        --model_dir="$MODELS" --checkpoints_dir="$MODELS/checkpoints" \
        --log_dir="$MODELS/logs" --results_dir="$MODELS/results" \
        --total_charge_key=charge --total_spin_key=spin \
        --batch_size="$BATCH" --valid_batch_size="$BATCH" --ema --ema_decay=0.99 \
        --restart_latest
}

foundation64() {
    [ -f "$MODELS/foundation64.model" ] && return
    python "$HERE/foundation64.py" "$FOUNDATION" "$MODELS/foundation64.model"
}

train_polar_ef() {
    [ -f "$MODELS/polar_ef.model" ] && return
    $MACE_TRAIN $(common) --name=polar_ef --model=PolarMACE --foundation_model="$MODELS/foundation64.model" \
        --train_file="$SETS/polar_train.xyz" --test_file="$SETS/polar_test.xyz" --valid_fraction=0.1 \
        --energy_key=REF_energy --forces_key=REF_forces --stress_key=REF_stress \
        --loss=stress --energy_weight=1 --forces_weight=10 --stress_weight=10 --compute_stress=True \
        --E0s=estimated --multiheads_finetuning=False \
        --lr="$EF_LR" --max_num_epochs="$EF_EPOCHS"
}

train_polar_dipole() {
    [ -f "$MODELS/polar_dipole.model" ] && return
    $MACE_TRAIN $(common) --name=polar_dipole --model=PolarMACE --foundation_model="$MODELS/polar_ef.model" \
        --train_file="$SETS/clusters_train.xyz" --test_file="$SETS/clusters_test.xyz" --valid_fraction=0.1 \
        --energy_key=REF_energy --forces_key=REF_forces --dipole_key=REF_dipole \
        --loss=energy_forces_dipole --energy_weight=1 --forces_weight=10 --dipole_weight=10 \
        --error_table=EnergyDipoleRMSE \
        --E0s=foundation --multiheads_finetuning=False --foundation_model_elements=True \
        --lr="$DIPOLE_LR" --max_num_epochs="$DIPOLE_EPOCHS"
}

train_dielectric() {
    [ -f "$MODELS/dielectric.model" ] && return
    $MACE_TRAIN $(common) --name=dielectric --model=AtomicDielectricMACE \
        --train_file="$SETS/clusters_train.xyz" --test_file="$SETS/clusters_test.xyz" --valid_fraction=0.1 \
        --dipole_key=REF_dipole --polarizability_key=REF_polarizability \
        --loss=dipole_polar --error_table=DipolePolarRMSE --compute_polarizability=True \
        --dipole_weight=1 --polarizability_weight=1 --E0s=average \
        --r_max=5.0 --max_L=2 --num_channels=64 --num_interactions=2 \
        --max_num_epochs="$DIEL_EPOCHS"
}

mkdir -p "$MODELS"
foundation64
train_polar_ef
train_polar_dipole
train_dielectric
