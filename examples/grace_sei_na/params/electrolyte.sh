source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SYSTEM=electrolyte
export SPECIES="$CATION C O F H P"
WORK=$RUNS/$NAME/$SYSTEM
TMELT=400
MAD_TEMPS="300 400 500"
