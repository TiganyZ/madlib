source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SYSTEM=graphite_electrolyte
export SPECIES="$CATION C O F H P"
WORK=$RUNS/$NAME/$SYSTEM
TMELT=500
MAD_TEMPS="300 400 500"
MELT_INPUT=$NA_EX/lammps/in.anneal_nvt
