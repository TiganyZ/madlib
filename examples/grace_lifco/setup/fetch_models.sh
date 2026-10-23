#!/bin/bash
# Foundation model, its checkpoint, and the labeller; checked against models.sha256.
set -eu
source "$(dirname "$0")/../env.sh"

MANIFEST=$EX/setup/models.sha256

download() {
    "$VENV_DIR/bin/grace_models" download "$FOUNDATION"
    "$VENV_DIR/bin/grace_models" checkpoint "$FOUNDATION"
    "$VENV_DIR/bin/grace_models" download "$LABELLER"
}

digest() {
    cd "$GRACE_CACHE"
    find "$FOUNDATION" "checkpoints/$FOUNDATION" "$LABELLER" -type f ! -name .grace_origin.json \
        | LC_ALL=C sort | xargs sha256sum
}

verify() {
    if [ -f "$MANIFEST" ]; then
        digest | diff - "$MANIFEST" && echo "models match $MANIFEST"
    else
        digest > "$MANIFEST"
        echo "wrote $MANIFEST"
    fi
}

download
verify
