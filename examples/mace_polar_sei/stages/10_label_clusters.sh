#!/bin/bash
# Dipole + polarizability per cluster on Roihu, through autoplex_soap_turbo/workflows/sei_response:
# REFERENCE=vasp (IDIPOL 4 + LEPSILON, training.yaml) or aims (DFPT, training_aims.yaml).
# Submits once, waits, harvests labelled.xyz.
set -eu
source "$(dirname "$0")/lib.sh"
if [ "$VASP_PILOT" -gt 0 ]; then
    FLOW=${NAME}_clusters_${REFERENCE}_pilot$VASP_PILOT
else
    FLOW=${NAME}_clusters_$REFERENCE
fi
if [ "$REFERENCE" = aims ]; then
    TEMPLATE=training_aims.yaml
else
    TEMPLATE=training.yaml
fi
OUT=$COMBINED/10_label_clusters/$FLOW
done_if "$OUT/labelled.xyz"
need "$COMBINED/09_clusters/clusters.xyz" "run stage 09 first"
mkdir -p "$OUT"

autoplex() {
    (source "$AUTOPLEX_ENV" && python "$AUTOPLEX_REPO/workflows/sei_response/$1" "${@:2}")
}

write_settings() {
    local repo
    repo=$(source "$AUTOPLEX_ENV" && echo "$AUTOPLEX_REPO")
    sed -e "s|^name: .*|name: $FLOW|" \
        -e "s|^  initial: .*|  initial: $COMBINED/09_clusters/clusters.xyz|" \
        -e "s|^species_list: .*|species_list: [${SPECIES// /, }]|" \
        -e "s|^  hyperparameters_file: .*|  hyperparameters_file: $repo/workflows/sei_response/gap_hypers.yaml|" \
        "$repo/workflows/sei_response/$TEMPLATE" > "$OUT/training.yaml"
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
    autoplex harvest.py --project "$JF_PROJECT" --config "$OUT/training.yaml" --host "$VASP_HOST" \
        --outcars "$OUT/outcars" --out "$OUT/labelled.xyz.part"
    mv "$OUT/labelled.xyz.part" "$OUT/labelled.xyz"
}

write_settings
submit
wait_done
harvest
