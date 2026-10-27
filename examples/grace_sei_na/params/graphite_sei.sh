source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SYSTEM=graphite_sei
export SPECIES="$CATION C O F H"
WORK=$RUNS/$NAME/$SYSTEM
TMELT=600
MAD_TEMPS="300 450 600"
MELT_INPUT=$NA_EX/lammps/in.anneal_nvt
