#!/bin/bash
# The fine-tune as a jobflow-remote job on a Triton GPU (autoplex_soap_turbo/workflows/grace_finetune).
# Submits once, waits, unpacks the model into 08_finetune/seed/$SEED.
set -eu
source "$(dirname "$0")/lib.sh"

OUT=$WORK/08_finetune
FIT=$OUT/seed/$SEED
REMOTE=$OUT/remote
mkdir -p "$REMOTE"

autoplex() {
    (source "$AUTOPLEX_ENV" && python "$AUTOPLEX_REPO/workflows/grace_finetune/$1" "${@:2}")
}

write_settings() {
    local repo
    repo=$(source "$AUTOPLEX_ENV" && echo "$AUTOPLEX_REPO")
    sed -e "s|^name: .*|name: grace_ft_$NAME|" \
        -e "s|^train: .*|train: $WORK/07_label/train.pkl.gz|" \
        -e "s|^test: .*|test: $WORK/07_label/test.pkl.gz|" \
        -e "s|^input: .*|input: $OUT/input.yaml|" \
        "$repo/workflows/grace_finetune/training.yaml" > "$REMOTE/settings.yaml"
}

submit() {
    [ -f "$REMOTE/SUBMITTED" ] && return
    autoplex run.py --config "$REMOTE/settings.yaml" --dry-run
    autoplex run.py --config "$REMOTE/settings.yaml" --project "$JF_PROJECT" | tee "$REMOTE/submit.log"
    date > "$REMOTE/SUBMITTED"
}

wait_done() {
    local code
    while true; do
        code=0
        autoplex result.py --project "$JF_PROJECT" --config "$REMOTE/settings.yaml" || code=$?
        [ "$code" -eq 0 ] && return
        [ "$code" -eq 2 ] && { echo "remote fine-tune failed; see jf job info" >&2; exit 1; }
        sleep "$FT_POLL"
    done
}

harvest() {
    rm -rf "$FIT"
    autoplex result.py --project "$JF_PROJECT" --config "$REMOTE/settings.yaml" --out "$FIT"
}

write_settings
submit
wait_done
harvest
