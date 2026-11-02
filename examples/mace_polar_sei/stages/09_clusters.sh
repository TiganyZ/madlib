#!/bin/bash
# N_CLUSTERS neutral clusters of whole units, cut from every system's MAD frames except the test ones.
set -eu
source "$(dirname "$0")/lib.sh"
OUT=$COMBINED/09_clusters
done_if "$OUT/clusters.xyz"
need "$COMBINED/06_select/test_unlabelled.pkl.gz" "run stage 07 first"
mkdir -p "$OUT"
"$PY" -I "$STAGES/pool_frames.py" "$WORK" "$COMBINED/06_select/test_unlabelled.pkl.gz" \
    "$OUT/frames.xyz" $SYSTEMS
"$PY" -I "$STAGES/carve_clusters.py" "$OUT/frames.xyz" "$OUT/clusters.xyz.part" "$N_CLUSTERS" "$CLUSTER_SEED"
mv "$OUT/clusters.xyz.part" "$OUT/clusters.xyz"
