#!/bin/bash
# MADLIB acceptance tests.
#
#   ./run.sh            everything
#   ./run.sh pdf xrd    a subset
#
# LMP        binary under test   (default ../../lammps/build/lmp)
# PYTHON     interpreter         (default python3)
# RANKS      ranks for the mpi leg (default 4)

set -u
cd "$(dirname "$0")"
LMP=${LMP:-../../lammps/build/lmp}
PYTHON=${PYTHON:-python3}
RANKS=${RANKS:-4}
XYZ=/home/tigany/test/turbogap_tests/xrd_mad/atoms.xyz
fail=0

have() { [ -e "$1" ]; }

leg() { printf '\n=== %s ===\n' "$1"; }
ok()  { echo "    PASS: $1"; }
bad() { echo "    FAIL: $1"; fail=1; }

want=${*:-forward gradients invariants mpi guards md mc}

for w in $want; do
case $w in

forward)
  leg "forward g(r) against an independent numpy reference"
  $LMP -in in.pdf -log none >/dev/null 2>&1
  if $PYTHON pdf_reference.py "$XYZ" gr_lammps.dat; then ok "partial g_ab(r)"; else bad "partial g_ab(r)"; fi
  ;;

gradients)
  leg "forces and virial against finite differences of MADLIB's own energy"
  for case in pdf xrd; do
    echo "  -- $case"
    if $PYTHON fd_gradient.py data/gc91.data extra_$case.lmp \
         --lmp "$LMP" --atoms 2 --hs 1e-4,3e-5 --paircut 7.5 >/tmp/fd_$case.log 2>&1; then
      grep -E "ratio|analytic W" /tmp/fd_$case.log | sed 's/^/     /'
      ok "$case forces and virial"
    else
      sed 's/^/     /' /tmp/fd_$case.log | tail -20
      bad "$case forces and virial"
    fi
  done
  ;;

invariants)
  leg "invariants"
  out=$($LMP -in in.invariants -log none 2>&1 | grep -A2 '^   Step' | sed -n 2p)
  $PYTHON - "$out" <<'PY'
import sys
f = sys.argv[1].split()
E, rel, D = float(f[1]), float(f[2]), float(f[3])
fx, fy, fz = float(f[4]), float(f[5]), float(f[6])
bad = 0
half = 0.5 * 100.0 * D
if abs(E - half) > 1e-9 * max(abs(E), 1.0):
    print(f"    FAIL: E != gamma D / 2  ({E} vs {half})"); bad = 1
else:
    print(f"    PASS: E = gamma D / 2   ({E:.10g})")
m = max(abs(fx), abs(fy), abs(fz))
if m > 1e-9:
    print(f"    FAIL: net bias force {m:.3e}"); bad = 1
else:
    print(f"    PASS: sum of bias forces = {m:.3e}")
sys.exit(bad)
PY
  [ $? -eq 0 ] || fail=1
  ;;

mpi)
  leg "1 rank against $RANKS ranks"
  $LMP -in in.invariants -log none 2>&1 | grep -A2 '^   Step' | sed -n 2p > /tmp/mad_1.txt
  mpirun --oversubscribe -np "$RANKS" $LMP -in in.invariants -log none 2>&1 \
      | grep -A2 '^   Step' | sed -n 2p > /tmp/mad_n.txt
  $PYTHON - <<'PY'
import sys
a = open('/tmp/mad_1.txt').read().split()
b = open('/tmp/mad_n.txt').read().split()
names = ["step","E","rel_error","dissim","fx","fy","fz","virial_xx","virial_xy"]
bad = 0
for n, x, y in zip(names, a, b):
    if n == "step": continue
    x, y = float(x), float(y)
    # The force sums are zero by construction, so compare them absolutely;
    # a relative test on two values near 1e-13 says nothing.
    if n in ("fx", "fy", "fz"):
        d, tol, kind = max(abs(x), abs(y)), 1e-9, "abs"
    else:
        d, tol, kind = abs(x - y) / max(abs(x), 1e-12), 1e-10, "rel"
    tag = "PASS" if d < tol else "FAIL"
    if d >= tol: bad = 1
    print(f"    {tag}: {n:10s} {kind} {d:.2e}")
sys.exit(bad)
PY
  [ $? -eq 0 ] || fail=1
  ;;

guards)
  leg "guards refuse rather than degrade"
  if $LMP -in in.guard -log none 2>&1 | grep -q "exceeds ghost atom range"; then
    ok "ghost-range cutoff is an error, not a warning"
  else bad "ghost-range guard did not fire"; fi

  sed 's/kde 0.1/kde 0.0/' in.pdf > /tmp/in.kde0
  cp /tmp/in.kde0 in.kde0
  if $LMP -in in.kde0 -log none 2>&1 | grep -q "histogram has no gradient"; then
    ok "a zero kde width is refused"
  else bad "zero kde width was accepted"; fi
  rm -f in.kde0

  sed 's/9.97006/12.0/' in.xrd0 > in.range
  if $LMP -in in.range -log none 2>&1 | grep -q "is not inside the range"; then
    ok "extrapolating past the target's range is refused"
  else bad "extrapolation was accepted"; fi
  rm -f in.range
  ;;

md)
  leg "biased MD drives the prediction towards the experiment"
  $LMP -in in.mad_md -log none 2>&1 | grep -E '^ +[0-9]+ ' > /tmp/mad_md.txt
  $PYTHON - <<'PY'
import sys
rows = [l.split() for l in open('/tmp/mad_md.txt') if len(l.split()) == 5]
rel = [float(r[4]) for r in rows]
print(f"    rel_error {rel[0]:.4f} -> {rel[-1]:.4f} over {len(rel)} samples")
sys.exit(0 if rel[-1] < 0.25 * rel[0] else 1)
PY
  if [ $? -eq 0 ]; then ok "the bias closes the gap"; else bad "rel_error did not fall"; fi
  ;;

mc)
  leg "the bias reaches the Metropolis test of an unmodified LAMMPS MC fix"
  for b in yes no; do
    $LMP -in in.mad_mc -var bias $b -log none 2>&1 | grep -E '^ +300 ' | tail -1 > /tmp/mad_mc_$b.txt
  done
  $PYTHON - <<'PY'
import sys
y = open('/tmp/mad_mc_yes.txt').read().split()
n = open('/tmp/mad_mc_no.txt').read().split()
ay, an, ry, rn = float(y[4]), float(n[4]), float(y[2]), float(n[2])
print(f"    energy yes: {ay:.0f}/{float(y[3]):.0f} swaps accepted, rel_error {ry:.4f}")
print(f"    energy no : {an:.0f}/{float(n[3]):.0f} swaps accepted, rel_error {rn:.4f}")
sys.exit(0 if (ay < an and ry < rn) else 1)
PY
  if [ $? -eq 0 ]; then ok "acceptance and fit both respond to the bias"
  else bad "the bias did not reach the Metropolis test"; fi
  ;;
esac
done

printf '\n'
[ $fail -eq 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail
