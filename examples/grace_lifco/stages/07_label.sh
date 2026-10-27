#!/bin/bash
# LABEL=grace: the labeller model. LABEL=vasp: VASP statics on Roihu (stages/vasp_label.sh).
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/07_label
done_if "$OUT/test.pkl.gz"
mkdir -p "$OUT"
cd "$OUT"
case $LABEL in
    grace)
        for split in train test; do
            "$PY" -I "$EX/stages/label_grace.py" "$GRACE_CACHE/$LABELLER" \
                "$WORK/06_select/${split}_unlabelled.pkl.gz" "$split.pkl.gz"
        done ;;
    vasp)
        VASP_PILOT=0 bash "$EX/stages/vasp_label.sh"
        for split in train test; do
            "$PY" -I "$EX/stages/label_import.py" "$OUT/${VASP_FLOW}_$NAME/labelled.xyz" \
                "$WORK/06_select/${split}_unlabelled.pkl.gz" "$split.pkl.gz"
        done ;;
esac
