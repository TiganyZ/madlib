#!/bin/bash
# Every stage in order; a stage whose output exists is skipped.
set -eu
cd "$(dirname "$0")"
for stage in stages/[0-9][0-9]_*.sh; do
    echo ">>> $stage"
    bash "$stage"
done
