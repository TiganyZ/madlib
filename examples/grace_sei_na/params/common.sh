# Shared by every system; each params/<system>.sh sources this, then sets its own. CATION=Li for the Li run.
NA_EX=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CATION=${CATION:-Na}      # Na or Li
if [ "$CATION" = Li ]; then
    NAME=${NAME:-sei_li}
else
    NAME=${NAME:-sei_na}
fi
NRANKS=4                      # per system
MAX_PARALLEL=2                # systems at once; reference MD holds 5-8 GB of TensorFlow each
SYSTEMS="inorganic organic electrolyte graphite_electrolyte graphite_sei"

STRUCTURES=$NA_EX/stages/structures.py
MELT_INPUT=$EX/lammps/in.melt_quench

REF_TEMP=300
REF_EQUIL_STEPS=1500
REF_SAMPLE_STEPS=1000
REF_EVERY=50

MAD_EBIAS="0.005 0.02 0.05"   # target bias energy, eV/atom; gamma = 2 N e / D0
MAD_STEPS=40000               # 20 ps
MAD_EVERY=2000                # 1 ps

N_PER_TEMP=24                 # per system and temperature
TEST_FRACTION=0.2

FT_ITERS=${FT_ITERS:-200}

LABEL=${LABEL:-vasp}
VASP_PILOT=${VASP_PILOT:-0}
VASP_POLL=600
VASP_FLOW=${NAME}_mad
FT_WHERE=${FT_WHERE:-triton}
FT_POLL=120
