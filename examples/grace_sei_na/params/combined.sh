# The merged selection of every system: labels, fine-tune, validation.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SPECIES="$CATION C O F H P"
WORK=$RUNS/$NAME/combined
