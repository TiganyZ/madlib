#!/bin/bash
# Run all four methods at the same budget and record what each one spent.
#
#   LMP=/path/to/lmp PY=/path/to/python ./run_all.sh [method ...]
#
# With no arguments it runs all four. The budget is one number in params.sh;
# budget.py turns it into each method's parameters so they cannot drift apart.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/params.sh"

LMP=${LMP:?set LMP to the lammps binary}
PY=${PY:-python3}
DATA=${DATA:-$HERE/data}
OUT=${OUT:-$HERE/runs}
SEED=${SEED:-12345}
SCORED=${SCORED:-$DATA/population_scored.xyz}

eval "$("$PY" "$HERE/budget.py" --budget "$BUDGET" \
        --gammas "$(echo "$GAMMAS" | wc -w)" \
        --population "$POPULATION" --generations "$GENERATIONS" \
        --moves-per-cycle "$MOVES_PER_CYCLE" --shell)"

# One line per leg, so the comparison carries its own cost.
timing() {
  printf '%s,%s,%s,%s\n' "$1" "$2" "$3" "$4" >> "$OUT/timing.csv"
}

prepare() {
  mkdir -p "$OUT"
  [ -f "$OUT/timing.csv" ] || echo "method,label,wall_seconds,evaluations" > "$OUT/timing.csv"
}

workdir() {
  local d=$OUT/$1
  rm -rf "$d"; mkdir -p "$d"
  ln -sfn "$DATA/gap_files" "$d/gap_files"
  ln -sfn "$DATA/$TARGET" "$d/"
  cp "$DATA/$START" "$d/start.data"
  printf '%s\n%s\n' "$POTENTIAL" "$POTCOEFF" > "$d/potential.lmp"
  echo "$d"
}

run_mad() {
  local melt=$((MAD_STEPS_PER_GAMMA / 5))
  for g in $GAMMAS; do
    local d; d=$(workdir "mad_gamma$g")
    local t0=$SECONDS
    ( cd "$d" && mpirun -np "$RANKS" "$LMP" -in "$HERE/lammps/in.mad" -log log \
        -var target "$TARGET" -var species "$SPECIES" -var mass "$MASS" \
        -var gamma "$g" -var melt "$melt" \
        -var steps "$((MAD_STEPS_PER_GAMMA - melt))" \
        -var minimise "$MAD_MINIMISE_ITERS" \
        -var temp_anneal "$TEMP_ANNEAL" -var temp_final "$TEMP_FINAL" \
        -var rmax "$RMAX" -var nbins "$NBINS" -var kde "$KDE" -var comm "$COMM" \
        -var qmin "$QMIN" -var qmax "$QMAX" -var nq "$NQ" \
        -var output "$OUTPUT" -var weighting "$WEIGHTING" \
        -var timestep "$TIMESTEP" -var seed "$SEED" > stdout 2>&1 )
    timing mad "gamma$g" "$((SECONDS - t0))" \
           "$((MAD_STEPS_PER_GAMMA + MAD_MINIMISE_ITERS))"
  done
}

run_hrmc() {
  for g in $GAMMAS; do
    local d; d=$(workdir "hrmc_gamma$g")
    local t0=$SECONDS
    ( cd "$d" && mpirun -np "$RANKS" "$LMP" -in "$HERE/lammps/in.hrmc" -log log \
        -var target "$TARGET" -var species "$SPECIES" -var mass "$MASS" \
        -var gamma "$g" -var cycles "$HRMC_CYCLES" -var moves "$HRMC_MOVES_PER_CYCLE" \
        -var temp "$TEMP_HRMC" -var maxdisp "$MAX_DISP" \
        -var rmax "$RMAX" -var nbins "$NBINS" -var kde "$KDE" -var comm "$COMM" \
        -var qmin "$QMIN" -var qmax "$QMAX" -var nq "$NQ" \
        -var output "$OUTPUT" -var weighting "$WEIGHTING" \
        -var seed "$SEED" > stdout 2>&1 )
    timing hrmc "gamma$g" "$((SECONDS - t0))" \
           "$((HRMC_CYCLES * HRMC_MOVES_PER_CYCLE))"
  done
}

# EZGA simulates only offspring, so a seed read from a file never gets an
# energy and enters selection at a fake one. Scoring the population is what
# gives the seeds a real objective; the cost is charged to the GA budget
# through seed_spend.json. Its own leg, because the two GA legs run
# concurrently and must not both write the file.
run_seeds() {
  "$PY" "$HERE/score_population.py" --population "$DATA/population.xyz" \
      --out "$SCORED" --data-dir "$DATA" --lmp "$LMP" --target "$TARGET" \
      --rmax "$RMAX" --nbins "$NBINS" --kde "$KDE" \
      --qmin "$QMIN" --qmax "$QMAX" --nq "$NQ" \
      --xrd-output "$OUTPUT" --weighting "$WEIGHTING" --mass "$MASS" \
      --ranks "$RANKS"
}

run_ezga() {
  # Two statements, not one: bash expands every argument to `local` before it
  # assigns any of them, so ${kind} on the same line is expanded in the
  # caller's scope where it does not exist -- and set -u then aborts.
  local kind=$1
  local out=$OUT/ezga_${kind//+/_}
  [ -f "$SCORED" ] || { echo "run the seeds leg first: no $SCORED" >&2; exit 1; }
  local t0=$SECONDS
  "$PY" "$HERE/run_ga.py" --out "$out" --population "$SCORED" \
      --data-dir "$DATA" --lmp "$LMP" --target "$TARGET" --objectives "$kind" \
      --budget "$EZGA_BUDGET" \
      --generations "$EZGA_GENERATIONS" --size "$EZGA_POPULATION" \
      --md-steps "$EZGA_MD_STEPS" --relax-steps "$EZGA_RELAX_STEPS" \
      --gammas "$(echo "$GAMMAS" | tr ' ' ',')" --species "$SPECIES" \
      --rmax "$RMAX" --nbins "$NBINS" --qmin "$QMIN" --qmax "$QMAX" --nq "$NQ" \
      --xrd-output "$OUTPUT" --weighting "$WEIGHTING" --seed "$SEED" \
      --ranks "$RANKS"
  # What run_ga.py measured, not what the plan assumed: the GA simulates only
  # offspring and rejects collided children before evaluating them.
  local spent
  spent=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['evaluations'])" \
          "$out/timing.json" 2>/dev/null || echo 0)
  timing "ezga_${kind//+/_}" all "$((SECONDS - t0))" "$spent"
}

main() {
  prepare
  local want=${*:-"seeds mad hrmc ezga ezga_exp"}
  for m in $want; do
    case $m in
      seeds)    run_seeds ;;
      mad)      run_mad ;;
      hrmc)     run_hrmc ;;
      ezga)     run_ezga energy ;;
      ezga_exp) run_ezga energy+exp ;;
      *) echo "unknown method $m" >&2; exit 1 ;;
    esac
  done
  echo
  column -t -s, "$OUT/timing.csv"
}

main "$@"
