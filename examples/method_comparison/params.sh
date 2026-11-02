# The system, the target and the budget. Sourced by everything here, so the
# four methods cannot drift apart in what they are fitting or what they spend.

# One number controls the comparison. budget.py turns it into each method's
# parameters; change it here and nothing else.
BUDGET=${BUDGET:-60000}          # potential evaluations per method
GAMMAS=${GAMMAS:-"1 3 10 30 100"}
POPULATION=${POPULATION:-24}
GENERATIONS=${GENERATIONS:-10}
MOVES_PER_CYCLE=${MOVES_PER_CYCLE:-50}

START=${START:-start.data}
SPECIES=${SPECIES:-C}
MASS=${MASS:-12.011}
TARGET=${TARGET:-target.fq}
WEIGHTING=${WEIGHTING:-xray}
OUTPUT=${OUTPUT:-qFq}
QMIN=${QMIN:-1.04525}
QMAX=${QMAX:-9.97006}
NQ=${NQ:-201}

# compute mad/pdf extends its neighbour cutoff to RMAX + 8*KDE, and the ghost
# cutoff has to cover that plus the skin -- and stay under half the box, or
# the pair distribution counts the cell's own periodic images.
RMAX=${RMAX:-8.0}
NBINS=${NBINS:-201}
KDE=${KDE:-0.1}
COMM=${COMM:-9.3}

POTENTIAL=${POTENTIAL:-"pair_style turbogap e0 C -0.16138053"}
POTCOEFF=${POTCOEFF:-"pair_coeff * * gap_files/carbon.gap C"}

TEMP_ANNEAL=${TEMP_ANNEAL:-5000}
TEMP_FINAL=${TEMP_FINAL:-300}
TEMP_HRMC=${TEMP_HRMC:-1000}
MAX_DISP=${MAX_DISP:-0.10}
TIMESTEP=${TIMESTEP:-0.001}
RANKS=${RANKS:-16}
