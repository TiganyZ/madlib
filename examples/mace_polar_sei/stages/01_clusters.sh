#!/bin/bash
# N_CLUSTERS neutral clusters of whole units, cut from every system's MAD frames except the test ones.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$WORK/01_clusters
done_if "$OUT/clusters.xyz"
need "$GRACE_RUNS/combined/06_select/test_unlabelled.pkl.gz" "run grace_sei_na (CATION=$CATION) through stage 06 first"
mkdir -p "$OUT"
"$PY" -I "$STAGES/pool_frames.py" "$GRACE_RUNS" "$GRACE_RUNS/combined/06_select/test_unlabelled.pkl.gz" \
    "$OUT/frames.xyz" $SYSTEMS
"$PY" -I "$STAGES/carve_clusters.py" "$OUT/frames.xyz" "$OUT/clusters.xyz.part" "$N_CLUSTERS" "$CLUSTER_SEED"
mv "$OUT/clusters.xyz.part" "$OUT/clusters.xyz"
