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

want=${*:-forward gradients invariants mpi guards md mc xps gap}

for w in $want; do
case $w in

forward)
  leg "forward g(r) against an independent numpy reference"
  $LMP -in in.pdf -log none >/dev/null 2>&1
  if $PYTHON pdf_reference.py "$XYZ" gr_lammps.dat; then ok "partial g_ab(r)"; else bad "partial g_ab(r)"; fi

  leg "forward coordination and XPS spectrum against the same"
  $LMP -in in.xps0 -log none >/dev/null 2>&1
  if $PYTHON xps_reference.py "$XYZ" cn.dump xps_pred.dat; then
    ok "per-atom property and broadened spectrum"
  else bad "per-atom property and broadened spectrum"; fi
  ;;

gradients)
  leg "forces and virial against finite differences of MADLIB's own energy"
  for case in pdf xrd xps; do
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

gap)
  leg "the soap_turbo backend against TurboGAP itself"
  if ! $LMP -in in.cebe -log none 2>&1 | grep -q "soap_turbo block"; then
    echo "    SKIP: built without the Fortran backend (-D MADLIB_ENABLE_TURBOGAP=on)"
  else
    $LMP -in in.cebe -log none >/dev/null 2>&1
    $PYTHON - <<'PY'
import sys
ref = {int(l.split()[0]): float(l.split()[1]) for l in open('data/cebe_turbogap.dat')}
body = open('cebe.dump').read().split('ITEM: ATOMS')[-1].split('\n')[1:]
got = {int(l.split()[0]): float(l.split()[1]) for l in body if len(l.split()) == 2}
d = [abs(got[i] - ref[i]) for i in ref]
rel = max(x / abs(ref[i]) for x, i in zip(d, ref))
print(f"    {len(ref)} core-electron binding energies, mean {sum(ref.values())/len(ref):.2f} eV")
print(f"    max |LAMMPS - TurboGAP| = {max(d):.2e} eV   ({rel:.1e} relative)")
sys.exit(0 if rel < 1e-7 else 1)
PY
    if [ $? -eq 0 ]; then ok "an existing .gap model reproduces TurboGAP"
    else bad "the .gap model disagrees with TurboGAP"; fi

    leg "soap_turbo derivatives, by finite differences"
    if $PYTHON fd_gradient.py data/gc91.data extra_xps_gap.lmp --lmp "$LMP" \
         --atoms 2 --hs 3e-3,1e-3,3e-4 --paircut 6.0 >/tmp/fd_gap.log 2>&1; then
      grep -E "ratio|analytic W" /tmp/fd_gap.log | sed 's/^/     /'
      ok "dq/dr through the descriptor"
    else
      sed 's/^/     /' /tmp/fd_gap.log | tail -18
      bad "dq/dr through the descriptor"
    fi

    leg "the same, across ranks"
    $LMP -in in.invariants_gap -log none 2>&1 | grep -A2 '^   Step' | sed -n 2p > /tmp/gap_1.txt
    mpirun --oversubscribe -np "$RANKS" $LMP -in in.invariants_gap -log none 2>&1 \
        | grep -A2 '^   Step' | sed -n 2p > /tmp/gap_n.txt
    $PYTHON - <<'PY'
import sys
a, b = open('/tmp/gap_1.txt').read().split(), open('/tmp/gap_n.txt').read().split()
names = ["step","E","rel_error","dissim","fx","fy","fz","virial_xx","virial_xy"]
bad = 0
for n, x, y in zip(names, a, b):
    if n == "step": continue
    x, y = float(x), float(y)
    if n in ("fx","fy","fz"): d, tol, kind = max(abs(x), abs(y)), 1e-9, "abs"
    else: d, tol, kind = abs(x-y)/max(abs(x),1e-12), 1e-10, "rel"
    if d >= tol: bad = 1
    print(f"    {'PASS' if d < tol else 'FAIL'}: {n:10s} {kind} {d:.2e}")
sys.exit(bad)
PY
    if [ $? -eq 0 ]; then ok "the descriptor adjoint is rank-invariant"
    else bad "the descriptor adjoint differs across ranks"; fi
  fi
  ;;

xps)
  leg "XPS: the property layer end to end"
  out=$($LMP -in in.invariants_xps -log none 2>&1 | grep -A2 '^   Step' | sed -n 2p)
  $PYTHON - "$out" <<'PY'
import sys
f = sys.argv[1].split()
E, D = float(f[1]), float(f[3])
m = max(abs(float(f[4])), abs(float(f[5])), abs(float(f[6])))
bad = 0
half = 0.5 * 50.0 * D
if abs(E - half) > 1e-9 * max(abs(E), 1.0): print(f"    FAIL: E != gamma D/2 ({E} vs {half})"); bad = 1
else: print(f"    PASS: E = gamma D / 2   ({E:.10g})")
if m > 1e-9: print(f"    FAIL: net bias force {m:.3e}"); bad = 1
else: print(f"    PASS: sum of bias forces = {m:.3e}")
sys.exit(bad)
PY
  [ $? -eq 0 ] || fail=1

  # the reverse communication of the property adjoint only exists under MPI
  $LMP -in in.invariants_xps -log none 2>&1 | grep -A2 '^   Step' | sed -n 2p > /tmp/xps_1.txt
  mpirun --oversubscribe -np "$RANKS" $LMP -in in.invariants_xps -log none 2>&1       | grep -A2 '^   Step' | sed -n 2p > /tmp/xps_n.txt
  $PYTHON - <<'PY'
import sys
a, b = open('/tmp/xps_1.txt').read().split(), open('/tmp/xps_n.txt').read().split()
names = ["step","E","rel_error","dissim","fx","fy","fz","virial_xx","virial_xy"]
bad = 0
for n, x, y in zip(names, a, b):
    if n == "step": continue
    x, y = float(x), float(y)
    if n in ("fx","fy","fz"): d, tol, kind = max(abs(x), abs(y)), 1e-9, "abs"
    else: d, tol, kind = abs(x-y)/max(abs(x),1e-12), 1e-10, "rel"
    if d >= tol: bad = 1
    print(f"    {'PASS' if d < tol else 'FAIL'}: {n:10s} {kind} {d:.2e}")
sys.exit(bad)
PY
  if [ $? -eq 0 ]; then ok "adjoint reverse communication is rank-invariant"
  else bad "property adjoint differs across ranks"; fi

  leg "XPS reverse Monte Carlo"
  for b in yes no; do
    $LMP -in in.xps_mc -var bias $b -log none 2>&1 | grep -E '^ +300 ' | tail -1 > /tmp/xps_mc_$b.txt
  done
  $PYTHON - <<'PY'
import sys
y, n = open('/tmp/xps_mc_yes.txt').read().split(), open('/tmp/xps_mc_no.txt').read().split()
ry, rn, ay, an = float(y[2]), float(n[2]), float(y[4]), float(n[4])
print(f"    energy yes: {ay:.0f}/{float(y[3]):.0f} moves accepted, rel_error {ry:.4f}")
print(f"    energy no : {an:.0f}/{float(n[3]):.0f} moves accepted, rel_error {rn:.4f}")
sys.exit(0 if (ay < an and ry < rn) else 1)
PY
  if [ $? -eq 0 ]; then ok "the XPS bias drives the walk"
  else bad "the XPS bias did not reach the Metropolis test"; fi
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
