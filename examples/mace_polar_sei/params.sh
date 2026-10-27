# Every setting of the MACE-POLAR-1 SEI workflow. CATION=Li for the Li run. Sourced by stages/lib.sh.
export CATION=${CATION:-Na}
if [ "$CATION" = Li ]; then
    GRACE_NAME=sei_li
    SLABS=/l/zarrout1/graphite-sei/structures_half
else
    GRACE_NAME=sei_na
    SLABS=/l/zarrout1/graphite-sei/structures_na_half
fi
NAME=mace_polar_${GRACE_NAME#sei_}
SYSTEMS="inorganic organic electrolyte graphite_electrolyte graphite_sei"
SPECIES="$CATION C O F H P"

# alt: where grace_sei_na left its runs (RUNS comes from grace_lifco/env.sh)
GRACE_RUNS=$RUNS/$GRACE_NAME
WORK=$RUNS/$NAME

# 01: clusters carved from the MAD frames
N_CLUSTERS=${N_CLUSTERS:-300}
CLUSTER_SEED=20261027

# 02: the cluster responses on Roihu through autoplex_soap_turbo/workflows/sei_response.
# aims: training_aims.yaml, DFPT on the isolated cluster. vasp: training.yaml, LEPSILON in a box.
# Na pilot, same 30-atom cluster: |mu| 2.18 vs 2.22 eA, alpha 21.2 vs 21.9 A^3, aims ~25x fewer core-hours.
REFERENCE=${REFERENCE:-aims}
VASP_PILOT=${VASP_PILOT:-0}       # >0: only the first N clusters, to time them
VASP_POLL=600
VASP_HOST=roihuc1

# 04-06: Triton (alt reaches it as `tr`)
TRITON=tr
TRITON_ROOT=/scratch/work/zarrout1/mace-polar
TRITON_WORK=$TRITON_ROOT/runs/$NAME
TRITON_POLL=300
FOUNDATION=${FOUNDATION:-polar-1-m}
TEST_FRACTION=0.15                # of the clusters; the periodic split is grace's
# vasp: LREAL False in the cluster statics, Auto in the periodic ones. aims: another basis, another energy zero.
if [ "$REFERENCE" = aims ]; then
    CLUSTER_ENERGY_WEIGHT=0.0
else
    CLUSTER_ENERGY_WEIGHT=0.1
fi
EF_EPOCHS=${EF_EPOCHS:-60}
DIPOLE_EPOCHS=${DIPOLE_EPOCHS:-20}
DIEL_EPOCHS=${DIEL_EPOCHS:-300}
MD_STEPS=${MD_STEPS:-10000}       # 10 ps at 1 fs
