#!/bin/bash
# VASP statics on Roihu for 06_select/to_label.xyz, through autoplex_soap_turbo/workflows/lifco_mad.
# Submits once, waits for every static to finish, writes labelled.xyz.
set -eu
source "$(dirname "$0")/lib.sh"

if [ "$VASP_PILOT" -gt 0 ]; then
    FLOW=lifco_mad_${NAME}_pilot$VASP_PILOT
else
    FLOW=lifco_mad_$NAME
fi
OUT=$WORK/07_label/$FLOW
done_if "$OUT/labelled.xyz"
mkdir -p "$OUT"

autoplex() {
    (source "$AUTOPLEX_ENV" && python "$AUTOPLEX_REPO/workflows/lifco_mad/$1" "${@:2}")
}

write_settings() {
    local repo
    repo=$(source "$AUTOPLEX_ENV" && echo "$AUTOPLEX_REPO")
    sed -e "s|^name: .*|name: $FLOW|" \
        -e "s|^  initial: .*|  initial: $WORK/06_select/to_label.xyz|" \
        -e "s|^  hyperparameters_file: .*|  hyperparameters_file: $repo/workflows/lico/gap_hypers.yaml|" \
        "$repo/workflows/lifco_mad/training.yaml" > "$OUT/training.yaml"
    if [ "$VASP_PILOT" -gt 0 ]; then
        sed -i "s|^  seed: 0|  seed: 0\n  max_initial_frames: $VASP_PILOT|" "$OUT/training.yaml"
    fi
}

submit() {
    [ -f "$OUT/SUBMITTED" ] && return
    autoplex run.py --config "$OUT/training.yaml" --dry-run
    autoplex run.py --config "$OUT/training.yaml" --project "$JF_PROJECT" | tee "$OUT/submit.log"
    date > "$OUT/SUBMITTED"
}

wait_done() {
    until autoplex status.py --project "$JF_PROJECT" --config "$OUT/training.yaml"; do
        sleep "$VASP_POLL"
    done
}

harvest() {
    autoplex harvest.py --project "$JF_PROJECT" --config "$OUT/training.yaml" --out "$OUT/labelled.xyz.part"
    mv "$OUT/labelled.xyz.part" "$OUT/labelled.xyz"
}

write_settings
submit
wait_done
harvest
