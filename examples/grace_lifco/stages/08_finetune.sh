#!/bin/bash
# Full-parameter finetune of GRACE-FS-OMAT, then the C++ export and its active set.
# FT_WHERE=triton runs it as a GPU job instead (stages/finetune_remote.sh).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/08_finetune
FIT=$OUT/seed/$SEED
done_if "$FIT/saved_model.asi"
mkdir -p "$OUT"
cd "$OUT"

write_input() {
    cat > input.yaml <<YAML
seed: $SEED
cutoff: n/a

data:
  filename: $WORK/07_label/train.pkl.gz
  test_filename: $WORK/07_label/test.pkl.gz
  reference_energy: 0
  save_dataset: True

potential:
  finetune_foundation_model: $FOUNDATION
  reduce_elements: True
  shift: auto

fit:
  loss:
    energy: {weight: 16, type: huber, delta: 0.01}
    forces: {weight: 32, type: huber, delta: 0.01}
    stress: {weight: 64, type: huber, delta: 0.01}
  maxiter: $FT_ITERS
  optimizer: BFGS
  opt_params: {gtol: 1.e-8}
  compute_convex_hull: False
  batch_size: 16
  test_batch_size: 32
  jit_compile: True
  eval_init_stats: True
  train_max_n_buckets: 2
  test_max_n_buckets: 1
  checkpoint_freq: 10
  progressbar: False
YAML
}

write_input
if [ "$FT_WHERE" = triton ]; then
    exec bash "$EX/stages/finetune_remote.sh"
fi
rm -rf "$FIT"
TF_DETERMINISTIC_OPS=1 "$VENV_DIR/bin/gracemaker" input.yaml
cd "$FIT"
"$VENV_DIR/bin/grace_utils" -p model.yaml -c checkpoints/checkpoint.best_test_loss.index export -sf
"$VENV_DIR/bin/pace_activeset" -d training_set.pkl.gz saved_model.yaml
