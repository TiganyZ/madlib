# The physics knobs. Sourced after env.sh.
NAME=${NAME:-lifco}
WORK=$RUNS/$NAME
NRANKS=8

TMELT=1300
REF_TEMP=300
REF_EQUIL_STEPS=1500
REF_SAMPLE_STEPS=1000
REF_EVERY=50

MAD_TEMPS="300 600 900"
MAD_EBIAS="0.005 0.02 0.05"   # target bias energy, eV/atom; gamma = 2 N e / D0
MAD_STEPS=60000               # 30 ps
MAD_EVERY=2000                # 1 ps: force correlation between dumps is at baseline

N_PER_TEMP=70                 # selected separately at each MAD temperature
TEST_FRACTION=0.2

FT_ITERS=${FT_ITERS:-50}

LABEL=${LABEL:-grace}          # grace | vasp
VASP_PILOT=${VASP_PILOT:-0}    # >0: label only this many frames, under a separate flow name
VASP_POLL=600                  # s between status checks
FT_WHERE=${FT_WHERE:-local}    # local | triton (GPU job through autoplex_soap_turbo/workflows/grace_finetune)
FT_POLL=120                    # s between status checks
