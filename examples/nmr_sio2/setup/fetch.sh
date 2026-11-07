#!/bin/bash
# Download the NMR tensor models and the amorphous silica structures of
# Ben Mahmoud et al. (arXiv:2412.15063) into $PREFIX/models.
#
# The .pth weights are ordinary git blobs; the .xyz structures are git-lfs, so
# they come from the media endpoint -- raw.githubusercontent.com returns the
# 130-byte pointer file for those, which looks like a successful download and
# is not one.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

RAW=https://raw.githubusercontent.com/$DATA_REPO/$DATA_REF
LFS=https://media.githubusercontent.com/media/$DATA_REPO/$DATA_REF

get() {
    local url=$1 out=$2
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then
        echo "have $out"
        return
    fi
    echo "get  $out"
    curl -sS -L --fail -o "$out" "$url"
}

fetch_model() {
    local remote=$1 name=$2
    get "$RAW/$remote/best_model.pth" "$MODELS/$name/best_model.pth"
    get "$RAW/$remote/config.yaml"    "$MODELS/$name/config.yaml"
}

fetch_statistics() {
    get "$RAW/$STATS" "$MODELS/data_mean_std.json"
}

fetch_structures() {
    get "$LFS/$STRUCTURES" "$MODELS/aSiO2_models.xyz"
}

check_not_a_pointer() {
    if head -c 40 "$MODELS/aSiO2_models.xyz" | grep -q "git-lfs"; then
        echo "ERROR: $MODELS/aSiO2_models.xyz is a git-lfs pointer, not a structure file" >&2
        exit 1
    fi
}

report() {
    du -sh "$MODELS"
    ls -l "$MODELS"/*/best_model.pth
    head -2 "$MODELS/aSiO2_models.xyz" | cut -c1-120
}

fetch_model "$MS_MODEL" ms
fetch_model "$EFG_MODEL" efg
fetch_statistics
fetch_structures
check_not_a_pointer
report
