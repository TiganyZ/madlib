# Every setting of the MACE-POLAR-1 SEI workflow. CATION=Li for the Li run. Sourced by stages/lib.sh,
# which with SYSTEM set also sources grace_sei_na/params/$SYSTEM.sh (species, melt and MAD temperatures).
export CATION=${CATION:-Na}
if [ "$CATION" = Li ]; then
    SLABS=/l/zarrout1/graphite-sei/structures_half
else
    SLABS=/l/zarrout1/graphite-sei/structures_na_half
fi
export NAME=mace_polar_$(echo "$CATION" | tr A-Z a-z)
SYSTEMS="inorganic organic electrolyte graphite_electrolyte graphite_sei"
SPECIES="$CATION C O F H P"
WORK=$RUNS/$NAME
COMBINED=$WORK/combined                  # stages 07-14
GRACE_RUNS=$RUNS/sei_$(echo "$CATION" | tr A-Z a-z)   # 01 reuses its packings

# 03-05: MD with PolarMACE through fix external, one Triton GPU per run
MD_MODEL=${MD_MODEL:-polar-1-m}
REF_MODEL=${REF_MODEL:-polar-1-l}         # the synthetic experiment
MD_DTYPE=float32
MD_D3=yes
# Graphite systems: PolarMACE blows up on graphite within a few steps (its charge flow on an
# extended semimetal), so their MD, experiment and selection use MACE-MH-1 instead.
GRAPHITE_MD_MODEL=mh-1:omat_pbe
GRAPHITE_REF_MODEL=mh-1:matpes_r2scan

# 06: farthest-point selection in MACE descriptor space, per MAD temperature (N_PER_TEMP is grace_sei_na's)
SELECT_MODEL=$MD_MODEL

# 09: clusters carved from the MAD frames
N_CLUSTERS=${N_CLUSTERS:-300}
CLUSTER_SEED=20261027

# 10: the cluster responses on Roihu through autoplex_soap_turbo/workflows/sei_response.
# aims: training_aims.yaml, DFPT on the isolated cluster. vasp: training.yaml, LEPSILON in a box.
REFERENCE=${REFERENCE:-aims}
VASP_PILOT=${VASP_PILOT:-0}       # >0: only the first N clusters, to time them
VASP_POLL=600
VASP_HOST=roihuc1

# Triton (alt reaches it as `tr`)
TRITON=tr
TRITON_ROOT=/scratch/work/zarrout1/mace-polar
TRITON_WORK=$TRITON_ROOT/runs/$NAME
TRITON_POLL=300

# 12-14: fits, validation, slabs
FOUNDATION=${FOUNDATION:-polar-1-m}
CLUSTER_TEST_FRACTION=0.15
# vasp: LREAL False in the cluster statics, Auto in the periodic ones. aims: another basis, another energy zero.
if [ "$REFERENCE" = aims ]; then
    CLUSTER_ENERGY_WEIGHT=0.0
else
    CLUSTER_ENERGY_WEIGHT=0.1
fi
EF_EPOCHS=${EF_EPOCHS:-60}
DIPOLE_EPOCHS=${DIPOLE_EPOCHS:-20}
DIEL_EPOCHS=${DIEL_EPOCHS:-300}
SLAB_MD_STEPS=${SLAB_MD_STEPS:-10000}     # 10 ps at 1 fs

# 15 (optional, by hand): LSC-MACE from scratch on alt's GPU (setup/make_lsc_venv.sh)
LSC_PREFIX=/l/zarrout1/mace-scf
LSC_HIDDEN_IRREPS="128x0e + 128x1o"
