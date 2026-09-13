#!/bin/bash
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
# MADLIB acceptance tests.
#
#   ./run.sh            everything
#   ./run.sh pdf xrd    a subset
#
# LMP        binary under test   (default ../../lammps/build/lmp)
# LMP_GPU    a second lmp built with -D MADLIB_ENABLE_GPU=on, for the pairgpu leg
# PYTHON     interpreter         (default python3)
# RANKS      ranks for the mpi leg (default 4)

set -u
cd "$(dirname "$0")"
# One run at a time. The legs share scratch files in this directory, and two
# suites interleaving produce wrong numbers rather than an obvious clash.
LOCK=.run.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "ERROR: another tests/run.sh is running (remove $LOCK if that is stale)" >&2
  exit 2
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

LMP=${LMP:-../../lammps/build/lmp}
PYTHON=${PYTHON:-python3}
RANKS=${RANKS:-4}
# Open MPI's shared-memory transport uses Cross Memory Attach, which needs
# ptrace permission. On a host with kernel.yama.ptrace_scope = 2 (alt is one)
# every transfer fails with "Read -1, expected N, errno = 1" and the run dies
# after step 0. Turning the single-copy mechanism off costs a little bandwidth
# and makes it work.
MPIRUN_EXTRA=${MPIRUN_EXTRA:---mca btl_vader_single_copy_mechanism none}
# Frames and fitted potentials live in the turbogap_tests data repository, not
# here: they are tens of megabytes and change on a different cadence from this
# code. Legs that need them skip when they are absent.
DATA_ROOT=${MADLIB_DATA_ROOT:-$HOME/test/turbogap_tests}
XYZ=$DATA_ROOT/xrd_mad/atoms.xyz
GAP_DIR=$DATA_ROOT/xrd_mad/gap_files
# -e is false for a dangling symlink, and plain ln then fails against the name
# that is already there -- which leaves the broken link in place and every gap
# leg failing on a missing file. Replace it.
if [ -L data/gap_files ] && [ ! -d data/gap_files ]; then rm -f data/gap_files; fi
if [ -d "$GAP_DIR" ] && [ ! -e data/gap_files ]; then ln -s "$GAP_DIR" data/gap_files; fi
fail=0

have() { [ -e "$1" ]; }

# One host/device comparison. $1 label, $2 host input, $3 kk input, $4 dump
# stem. Both dumps are removed first: a run that dies leaves the previous
# run's dump in place, and comparing those two reports whatever agreement the
# last good pair happened to have.
# Build and run a standalone Fortran driver against libmadlib. $1 is the source
# stem in this directory, $2 what a pass means.
fortran_check() {
  local stem=$1 what=$2
  local lib=../lib/madlib
  if [ ! -f "$lib/libmadlib.a" ] || [ ! -d "$lib/mod" ]; then
    echo "    SKIP: $lib is not built (make -C $lib)"
    return
  fi
  if ! command -v gfortran >/dev/null; then
    echo "    SKIP: no gfortran"
    return
  fi
  if ! gfortran -O2 -I "$lib/mod" "$stem.f90" -o "/tmp/mad_$stem" \
        "$lib/libmadlib.a" -llapack -lblas 2>"/tmp/mad_$stem.log"; then
    sed 's/^/     /' "/tmp/mad_$stem.log" | tail -5
    bad "$stem did not build"
    return
  fi
  if "/tmp/mad_$stem"; then ok "$what"; else bad "$what -- it does not"; fi
}

# Build one source twice: once against the CPU soap_turbo, once against the
# GPU one. They share a module name and an entry point, so they cannot be
# linked into the same binary; two binaries is the only way to compare them.
# $1 source stem, $2 host binary, $3 device binary.
build_cpu_and_gpu() {
  local stem=$1 hostbin=$2 devbin=$3
  local lib=../lib/madlib
  local g=${MADLIB_SOAP_GPU_OBJ:-/tmp/madlib_soap_gpu}
  gfortran -O2 -cpp -ffree-line-length-none -I "$lib/mod" "$stem.f90" -o "$hostbin" \
      "$lib/libmadlib.a" -llapack -lblas 2>/tmp/mad_gpubuild.log || return 1
  ${FC:-mpif90} -O2 -cpp -ffree-line-length-none -fallow-argument-mismatch \
      -DMADLIB_GPU_DESC -I "$g" "$stem.f90" -o "$devbin" "$g"/*.o "$g"/cu/*.o \
      -L"${CUDA_LIB:-/usr/local/cuda-13.4/lib64}" -lcudart -lcublas -lcuda -lstdc++ \
      -llapack -lblas 2>>/tmp/mad_gpubuild.log || return 1
}

compare_kk() {
  local label=$1 hin=$2 kin=$3 stem=$4
  local hlog=/tmp/${stem}_host.log klog=/tmp/${stem}_kk.log
  local hrc=0 krc=0
  rm -f "${stem}.dump" "${stem}_kk.dump"
  TURBOGAP_TERMS=1 $LMP -in "$hin" -log none >"$hlog" 2>&1 || hrc=$?
  TURBOGAP_TERMS=1 TURBOGAP_CHECK_NEIGH=1 $LMP -k on g 1 -sf kk -in "$kin" \
      -log none >"$klog" 2>&1 || krc=$?
  echo "    -- $label"
  if [ $hrc -ne 0 ] || [ $krc -ne 0 ]; then
    echo "    host rc=$hrc  kk rc=$krc"
    grep -ih "error" "$hlog" "$klog" | head -4 | sed 's/^/     /'
    bad "$label: a run did not complete"
    return
  fi
  grep -E "sites, .* pairs|inconsistent" "$klog" | sed 's/^/     /'
  if $PYTHON compare_pair_kk.py "${stem}.dump" "${stem}_kk.dump" "$hlog" "$klog"; then
    ok "$label"
  else
    bad "$label"
  fi
}

leg() { printf '\n=== %s ===\n' "$1"; }
ok()  { echo "    PASS: $1"; }
bad() { echo "    FAIL: $1"; fail=1; }

want=${*:-forward gradients invariants mpi guards md mc xps gap pair kk pairmd pairmpi pairgpu soapbasis soapdesc soapgpu soappredict soapbatch soapswap ir quip torch mliap}

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

mliap)
  leg "an ML-IAP model as a per-atom property"
  if ! $LMP -h 2>/dev/null | grep -q "mad/property/mliap"; then
    echo "    SKIP: built without ML-IAP"
  else
    $LMP -in in.mliap -log none >/dev/null 2>&1
    $PYTHON - <<'PY'
import sys
import numpy as np
rows = sorted([l.split() for l in open("mliap_q.dump").read().split("ITEM: ATOMS")[-1].split("\n")[1:]
               if len(l.split()) == 3], key=lambda r: int(r[0]))
q = np.array([float(r[1]) for r in rows])
pa = np.array([float(r[2]) for r in rows])
d = np.abs(q - pa).max()
print(f"    {len(q)} atoms: max |property - pair mliap per-atom energy| = {d:.2e}")
sys.exit(0 if d < 1e-12 else 1)
PY
    if [ $? -eq 0 ]; then ok "the property is the model's own per-atom value"
    else bad "the property disagrees with pair mliap"; fi

    leg "its adjoint, by finite differences"
    if $PYTHON fd_gradient.py data/ta54.data extra_xps_mliap.lmp --lmp "$LMP" \
         --atoms 2 --hs 1e-3,3e-4,1e-4 --paircut 6.0 >/tmp/fd_mliap.log 2>&1; then
      grep -E "ratio|observed order|analytic W" /tmp/fd_mliap.log | sed 's/^/     /'
      ok "scaled betas give the right force"
    else
      sed 's/^/     /' /tmp/fd_mliap.log | tail -16
      bad "the mliap adjoint is wrong"
    fi
  fi
  ;;

torch)
  leg "a model written in Python, differentiated by autograd"
  if ! $LMP -h 2>/dev/null | grep -q "mad/property/python"; then
    echo "    SKIP: built without the embedded interpreter"
  elif ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "    SKIP: torch not installed"
  else
    export PYTHONPATH=$PWD
    $LMP -in in.torch_ref -log none >/dev/null 2>&1 && mv -f torch_q.dump coord_q.dump
    $LMP -in in.torch -log none >/dev/null 2>&1
    $PYTHON - <<'PY'
import sys
import numpy as np
def rd(f):
    rows = [l.split() for l in open(f).read().split("ITEM: ATOMS")[-1].split("\n")[1:]
            if len(l.split()) == 2]
    return np.array([float(r[1]) for r in sorted(rows, key=lambda r: int(r[0]))])
a, b = rd("coord_q.dump"), rd("torch_q.dump")
d = np.abs(a - b).max()
print(f"    {len(a)} values, mean {a.mean():.6f}: max |torch - C++| = {d:.2e}")
sys.exit(0 if d < 1e-10 else 1)
PY
    if [ $? -eq 0 ]; then ok "the torch model reproduces the C++ one"
    else bad "the torch model disagrees with the C++ one"; fi

    leg "the autograd adjoint, by finite differences"
    if $PYTHON fd_gradient.py data/gc91.data extra_xps_torch.lmp --lmp "$LMP" \
         --atoms 2 --hs 1e-3,3e-4,1e-4 --paircut 7.5 >/tmp/fd_torch.log 2>&1; then
      grep -E "ratio|observed order|analytic W" /tmp/fd_torch.log | sed 's/^/     /'
      ok "one backward pass gives the right force"
    else
      sed 's/^/     /' /tmp/fd_torch.log | tail -16
      bad "the autograd adjoint is wrong"
    fi
  fi
  ;;

quip)
  rm -f /tmp/mad_quip.txt
  leg "the bias composes with a real ML potential (pair_style quip)"
  if ! $LMP -h 2>/dev/null | grep -qw quip; then
    echo "    SKIP: built without ML-QUIP"
  elif [ ! -e data/gap_files/CO.xml ]; then
    echo "    SKIP: no GAP potential (set MADLIB_DATA_ROOT)"
  else
    $LMP -in in.quip -log none >/dev/null 2>&1
    $PYTHON - <<'PY'
import sys
import numpy as np
try:
    from quippy.potential import Potential
    from ase.io import read
except ImportError:
    print("    SKIP: quippy not installed, cannot cross-check"); sys.exit(0)
at = read(os.environ["MADLIB_XYZ"])
at.calc = Potential(param_filename="data/gap_files/CO.xml")
ref_e, ref_f = at.get_potential_energy(), at.get_forces()
rows = [l.split() for l in open("quip_forces.dump").read().split("ITEM: ATOMS")[-1].split("\n")[1:]
        if len(l.split()) == 4]
got = np.array([[float(v) for v in r[1:]] for r in sorted(rows, key=lambda r: int(r[0]))])
df = np.abs(got - ref_f).max()
print(f"    forces through LAMMPS vs quippy: max|dF| = {df:.2e} eV/A  (max|F| = {np.abs(ref_f).max():.3f})")
sys.exit(0 if df < 1e-6 else 1)
PY
    if [ $? -eq 0 ]; then ok "pair_style quip agrees with quippy"
    else bad "pair_style quip disagrees with quippy"; fi

    leg "biased MD on that potential"
    for g in 0.0 50.0; do
      $LMP -in in.quip_mad -var g $g -log none 2>&1 | grep -E '^ +200 ' | tail -1 >> /tmp/mad_quip.txt
    done
    $PYTHON - <<'PY'
import sys
rows = [l.split() for l in open("/tmp/mad_quip.txt")][-2:]
r0, r1 = float(rows[0][4]), float(rows[1][4])
print(f"    gamma 0:  rel_error {r0:.4f}")
print(f"    gamma 50: rel_error {r1:.4f}")
sys.exit(0 if r1 < 0.5 * r0 else 1)
PY
    if [ $? -eq 0 ]; then ok "the bias closes the gap on a real potential"
    else bad "the bias did not improve the fit"; fi
  fi
  ;;

ir)
  rm -f /tmp/mad_mu.txt
  leg "the IR pipeline against TurboGAP's ir_fft, which is verified against GPUMD"
  MADIR=/tmp/mad_ir_driver
  g++ -O2 -std=c++17 -o $MADIR test_ir_spectrum.cpp ../src/MADLIB/mad_ir_spectrum.cpp -lm \
      2>/tmp/mad_ir_build.log
  if [ ! -x $MADIR ]; then
    sed 's/^/     /' /tmp/mad_ir_build.log | head -10
    bad "could not build the IR driver"
  else
    TGSRC=${TURBOGAP_SRC:-../../alt/src}
    if [ -f "$TGSRC/ir_fft.f90" ] && command -v gfortran >/dev/null; then
      gfortran -O2 -o /tmp/irfftverify "$TGSRC/kinds.f90" "$TGSRC/ir_fft.f90" \
          ../../alt/tests/ir_fft/irfftverify.f90 2>/dev/null
    fi
    if [ -x /tmp/irfftverify ]; then
      if $PYTHON ir_compare_turbogap.py $MADIR /tmp/irfftverify data/ir_dipoles.dat; then
        ok "machine-precision agreement with the reference implementation"
      else bad "the IR pipeline disagrees with TurboGAP"; fi
    else
      echo "    SKIP: no TurboGAP source to build the reference against"
    fi

    leg "a vibrational band lands where the force constant puts it"
    $LMP -in in.ir -log none >/dev/null 2>&1
    $PYTHON - <<'PY'
import sys
import numpy as np
d = np.loadtxt('ir_spectrum.dat')
nu, I = d[:, 0], d[:, 1]
K, r0 = 25.0, 1.13                       # bond_coeff, metal units
m = 12.011 * 15.999 / (12.011 + 15.999)  # reduced mass, amu
w = np.sqrt(2 * K / (m * 1.0364e-4))     # 1/ps; harmonic bond gives 2K
exact = w / (2 * np.pi * 0.0299792458)
sel = nu > 300.0
peak = nu[sel][np.argmax(I[sel])]
binw = d[1, 0] - d[0, 0]
print(f"    analytic {exact:.1f} cm^-1, observed {peak:.1f} cm^-1, bin {binw:.1f} cm^-1")
sys.exit(0 if abs(peak - exact) < 2.0 * binw else 1)
PY
    if [ $? -eq 0 ]; then ok "the band is where it should be"
    else bad "the band is in the wrong place"; fi

    leg "a molecular dipole survives being split across ranks"
    for np in 1 $RANKS; do
      rm -f mu.dump
      mpirun --oversubscribe -np $np $LMP -in in.ir_molecular -log none >/dev/null 2>&1
      $PYTHON - $np >> /tmp/mad_mu.txt <<'PY'
import collections, sys
rows = [l.split() for l in open('mu.dump').read().split('ITEM: ATOMS')[-1].split('\n')[1:]
        if len(l.split()) == 6]
m = collections.defaultdict(set)
for r in rows: m[r[1]].add(r[2])
print("%s %.14f %d %d" % (sys.argv[1], sum(float(r[5]) for r in rows),
                          sum(1 for v in m.values() if len(v) > 1), len(m)))
PY
    done
    $PYTHON - <<'PY'
import sys
rows = [l.split() for l in open('/tmp/mad_mu.txt')][-2:]
for r in rows:
    print(f"    {r[0]:>2s} rank(s): total mu_z = {float(r[1]):.12f}"
          f"   ({r[2]} of {r[3]} molecules straddle)")
d = abs(float(rows[0][1]) - float(rows[1][1])) / max(abs(float(rows[0][1])), 1e-12)
print(f"    relative difference {d:.2e}")
sys.exit(0 if d < 1e-10 and int(rows[1][2]) > 0 else 1)
PY
    if [ $? -eq 0 ]; then ok "the anchor is the same atom on every rank"
    else bad "the molecular dipole depends on the rank count"; fi

    leg "sum q_i r_i is refused where it has no meaning"
    if $LMP -in in.ir_guard -log none 2>&1 | grep -q "not defined in a periodic box"; then
      ok "a periodic box without molecules is an error, not a plausible spectrum"
    else bad "a periodic sum q_i r_i was accepted"; fi
  fi
  ;;

kk)
  leg "turbogap/kk reproduces turbogap"
  if ! $LMP -h 2>/dev/null | grep -q "turbogap/kk"; then
    echo "    SKIP: built without the KOKKOS package"
  else
    compare_kk "glassy carbon, 91 atoms" in.pair_turbogap in.pair_turbogap_kk pair_turbogap
    compare_kk "core_pot probe, 3 close pairs" in.core_probe in.core_probe_kk core_probe
  fi
  ;;

soapbasis)
  leg "the exported SOAP basis orthonormalises and is counted right"
  fortran_check soapbasisverify "W, S, skip and multiplicity are what they claim"
  ;;

soapgpu)
  leg "the device SOAP descriptor reproduces the host's"
  LIB=../lib/madlib
  GPUOBJ=${MADLIB_SOAP_GPU_OBJ:-/tmp/madlib_soap_gpu}
  if [ ! -f "$LIB/libmadlib.a" ] || [ ! -d "$LIB/mod" ]; then
    echo "    SKIP: $LIB is not built (make -C $LIB)"
  elif [ ! -d "$GPUOBJ" ]; then
    echo "    SKIP: no device objects -- tools/build_soap_gpu.sh <turbogap-src>"
  else
    # Two binaries, because the CPU and GPU soap_turbo share a module name and
    # an entry point and cannot be linked together. Same source, same
    # configuration; the only difference is which soap_turbo answered.
    rm -f /tmp/mad_desc_cpu.dat /tmp/mad_desc_gpu.dat
    gfortran -O2 -cpp -ffree-line-length-none -I "$LIB/mod" soapdescdump.f90 \
        -o /tmp/mad_desc_cpu "$LIB/libmadlib.a" -llapack -lblas 2>/tmp/mad_desc.log \
      && ${FC:-mpif90} -O2 -cpp -ffree-line-length-none -fallow-argument-mismatch \
        -DMADLIB_GPU_DESC -I "$GPUOBJ" soapdescdump.f90 -o /tmp/mad_desc_gpu \
        "$GPUOBJ"/*.o "$GPUOBJ"/cu/*.o \
        -L"${CUDA_LIB:-/usr/local/cuda-13.4/lib64}" -lcudart -lcublas -lcuda -lstdc++ \
        -llapack -lblas 2>>/tmp/mad_desc.log
    if [ $? -ne 0 ]; then
      sed 's/^/     /' /tmp/mad_desc.log | tail -6
      bad "the descriptor binaries did not build"
    elif ! /tmp/mad_desc_cpu /tmp/mad_desc_cpu.dat >/dev/null 2>&1 \
         || ! /tmp/mad_desc_gpu /tmp/mad_desc_gpu.dat >/dev/null 2>&1; then
      bad "a descriptor binary did not run"
    else
      # The descriptor exercises the forward chain, which is fully converted to
      # the Kokkos dispatch. The derivatives additionally exercise
      # gap_soap_descriptor.cu's raw-CUDA half -- nine launches, including the
      # __shared__ tree reduction at :208 that must not be reassociated. Worth
      # running both: the forward pass agreeing says nothing about that half.
      bad_any=0
      for mode in descriptor derivatives; do
        [ "$mode" = derivatives ] && arg=der || arg=
        /tmp/mad_desc_cpu /tmp/mad_desc_cpu.dat $arg >/dev/null 2>&1
        /tmp/mad_desc_gpu /tmp/mad_desc_gpu.dat $arg >/dev/null 2>&1
        echo "    -- $mode"
        $PYTHON compare_soap_desc.py /tmp/mad_desc_cpu.dat /tmp/mad_desc_gpu.dat || bad_any=1
      done
      if [ $bad_any -eq 0 ]; then ok "device and host agree on descriptor and derivatives"
      else bad "the device SOAP differs from the host's"; fi
    fi
  fi
  ;;

soappredict)
  leg "the device SOAP energies and forces reproduce the host's"
  LIB=../lib/madlib
  GPUOBJ=${MADLIB_SOAP_GPU_OBJ:-/tmp/madlib_soap_gpu}
  if [ ! -f "$LIB/libmadlib.a" ] || [ ! -d "$LIB/mod" ]; then
    echo "    SKIP: $LIB is not built (make -C $LIB)"
  elif [ ! -d "$GPUOBJ" ]; then
    echo "    SKIP: no device objects -- tools/build_soap_gpu.sh <turbogap-src>"
  else
    rm -f /tmp/mad_pred_cpu.dat /tmp/mad_pred_gpu.dat
    if ! build_cpu_and_gpu soappredictdump /tmp/mad_pred_cpu /tmp/mad_pred_gpu; then
      sed 's/^/     /' /tmp/mad_gpubuild.log | tail -6
      bad "the prediction binaries did not build"
    elif ! /tmp/mad_pred_cpu /tmp/mad_pred_cpu.dat >/dev/null 2>&1 \
         || ! /tmp/mad_pred_gpu /tmp/mad_pred_gpu.dat >/dev/null 2>&1; then
      bad "a prediction binary did not run"
    else
      # This is the leg that reaches gap_soap_forces.cu -- get_soap_energy_and_forces
      # calls gpu_final_soap_forces_virial at gap.f90:391 -- and so the three
      # __shared__ tree reductions there. Nothing before it touched them.
      $PYTHON compare_soap_predict.py /tmp/mad_pred_cpu.dat /tmp/mad_pred_gpu.dat
      if [ $? -eq 0 ]; then ok "device and host agree on energies and forces"
      else bad "the device prediction differs from the host's"; fi
    fi
  fi
  ;;

soapswap)
  leg "the same call gives the same answer on either backend"
  LIB=../lib/madlib
  GPUOBJ=${MADLIB_SOAP_GPU_OBJ:-/tmp/madlib_soap_gpu}
  if [ ! -f "$LIB/libmadlib.a" ] || [ ! -d "$LIB/mod" ]; then
    echo "    SKIP: $LIB is not built (make -C $LIB)"
  elif [ ! -d "$GPUOBJ" ]; then
    echo "    SKIP: no device objects -- tools/build_soap_gpu.sh <turbogap-src>"
  else
    # soaplocalpropdump.f90 contains no #ifdef. The host and device
    # implementations export the same C symbol with the same arguments, so the
    # build chooses, not the caller -- which is why pair_turbogap.cpp needs no
    # change to get the device descriptor.
    rm -f /tmp/mad_lp_cpu.dat /tmp/mad_lp_gpu.dat
    sw_bad=0
    gfortran -O2 -cpp -ffree-line-length-none -I "$LIB/mod" soaplocalpropdump.f90 \
        -o /tmp/mad_lp_cpu "$LIB/libmadlib.a" -llapack -lblas 2>/tmp/mad_sw.log \
      && ${FC:-mpif90} -O2 -cpp -ffree-line-length-none -fallow-argument-mismatch \
        -I "$GPUOBJ" soaplocalpropdump.f90 -o /tmp/mad_lp_gpu "$GPUOBJ"/*.o "$GPUOBJ"/cu/*.o \
        -L"${CUDA_LIB:-/usr/local/cuda-13.4/lib64}" -lcudart -lcublas -lcuda -lstdc++ \
        -llapack -lblas 2>>/tmp/mad_sw.log || sw_bad=1
    if [ $sw_bad -ne 0 ]; then
      sed 's/^/     /' /tmp/mad_sw.log | tail -6
      bad "the two backends did not build"
    else
      # Both clamping and not: zero_trunc is where the device predictor is known
      # to differ, which is why this path uses the host one on both sides.
      for zt in 1 0; do
        echo "    -- zero_trunc=$zt"
        MADLIB_ZERO_TRUNC=$zt /tmp/mad_lp_cpu /tmp/mad_lp_cpu.dat >/dev/null 2>&1
        MADLIB_ZERO_TRUNC=$zt /tmp/mad_lp_gpu /tmp/mad_lp_gpu.dat >/dev/null 2>&1
        $PYTHON compare_soap_predict.py /tmp/mad_lp_cpu.dat /tmp/mad_lp_gpu.dat || sw_bad=1
      done
      if [ $sw_bad -eq 0 ]; then ok "madlib_local_property agrees on host and device"
      else bad "the backends disagree"; fi
    fi
  fi
  ;;

soapbatch)
  leg "splitting SOAP into batches changes no answer"
  LIB=../lib/madlib
  GPUOBJ=${MADLIB_SOAP_GPU_OBJ:-/tmp/madlib_soap_gpu}
  if [ ! -f "$LIB/libmadlib.a" ] || [ ! -d "$LIB/mod" ]; then
    echo "    SKIP: $LIB is not built (make -C $LIB)"
  elif [ ! -d "$GPUOBJ" ]; then
    echo "    SKIP: no device objects -- tools/build_soap_gpu.sh <turbogap-src>"
  elif ! build_cpu_and_gpu soappredictdump /tmp/mad_bat_cpu /tmp/mad_bat_gpu; then
    sed 's/^/     /' /tmp/mad_gpubuild.log | tail -6
    bad "the batching binaries did not build"
  else
    /tmp/mad_bat_cpu /tmp/mad_bat_cpu.dat >/dev/null 2>&1
    bat_bad=0
    # Budgets chosen to give one batch and then several. The reported count is
    # checked, because "many batches agree" proves nothing if the budget was
    # silently ignored and one batch ran each time.
    for budget in 0 700000 100000; do
      n=$(MADLIB_SOAP_BATCH_REPORT=1 MADLIB_MAX_BYTES=$budget /tmp/mad_bat_gpu \
            /tmp/mad_bat_gpu.dat 2>&1 >/dev/null | sed -n 's/.*soap batches: \([0-9]*\).*/\1/p')
      echo "    -- budget ${budget:-0}: $n batch(es)"
      [ -n "$n" ] || { bad_any=1; bat_bad=1; continue; }
      $PYTHON compare_soap_predict.py /tmp/mad_bat_cpu.dat /tmp/mad_bat_gpu.dat || bat_bad=1
      echo "$n" >> /tmp/mad_bat_counts
    done
    # The split has to have actually happened at least once.
    if [ "$(sort -u /tmp/mad_bat_counts | wc -l)" -lt 2 ]; then
      echo "    every budget gave the same batch count -- the split was not exercised"
      bat_bad=1
    fi
    rm -f /tmp/mad_bat_counts
    if [ $bat_bad -eq 0 ]; then ok "batched and unbatched agree, and the split happened"
    else bad "batching changes the answer"; fi
  fi
  ;;

soapdesc)
  leg "the exported SOAP descriptor is the one the energy is built from"
  fortran_check soapdescverify "the descriptor reproduces the sparse-GP prediction"
  ;;

pairmpi)
  leg "pair_turbogap on 1 rank against $RANKS"
  rm -f md_mpi.dump
  $LMP -in in.md_mpi -log none >/tmp/mad_pair_1.log 2>&1
  mv md_mpi.dump /tmp/mad_pair_1.dump 2>/dev/null
  rm -f md_mpi.dump
  mpirun --oversubscribe $MPIRUN_EXTRA -np "$RANKS" $LMP -in in.md_mpi -log none \
      >/tmp/mad_pair_n.log 2>&1
  if [ ! -e md_mpi.dump ] || [ ! -e /tmp/mad_pair_1.dump ]; then
    grep -ih "error\|errno" /tmp/mad_pair_1.log /tmp/mad_pair_n.log | head -3 | sed 's/^/     /'
    bad "a run did not produce a trajectory"
  else
    # Ranks partition the sites differently, so a site's neighbours arrive in a
    # different order and the force adjoint reassociates -- the same ~1e-9
    # relative floor every other comparison here sits at. Absolute, because a
    # force component near zero has no meaningful relative error.
    $PYTHON compare_pair_md.py /tmp/mad_pair_1.dump /tmp/mad_pair_1.log \
        "${RANKS}_ranks" md_mpi.dump /tmp/mad_pair_n.log
    if [ $? -eq 0 ]; then ok "the decomposition changes no answer"
    else bad "pair_turbogap disagrees with itself across ranks"; fi
  fi
  ;;

pairmd)
  leg "the pair styles agree after five MD steps, not just at run 0"
  L=/tmp/madlib_md
  rm -f md_ref.dump md_noff.dump md_kk.dump
  $LMP -in in.md_ref  -log none >${L}_ref.log  2>&1
  $LMP -in in.md_noff -log none >${L}_noff.log 2>&1
  args="newton_off md_noff.dump ${L}_noff.log"
  if $LMP -h 2>/dev/null | grep -q "turbogap/kk"; then
    $LMP -k on g 1 -sf kk -in in.md_kk -log none >${L}_kk.log 2>&1
    args="$args turbogap/kk md_kk.dump ${L}_kk.log"
  fi
  if $PYTHON compare_pair_md.py md_ref.dump ${L}_ref.log $args; then
    ok "five steps of NVE agree with newton on"
  else
    bad "the styles diverge under MD"
  fi
  ;;

pairgpu)
  leg "SOAP on the device, through lmp itself"
  # The cmake backend switch, end to end. Two binaries differing only in
  # -D MADLIB_ENABLE_GPU, five steps of the same NVE, compared against each
  # other in this session -- never against a stored dump, because the device
  # does not reproduce itself run to run.
  if [ -z "${LMP_GPU:-}" ]; then
    echo "    SKIP: set LMP_GPU to an lmp built with -D MADLIB_ENABLE_GPU=on"
  else
    L=/tmp/madlib_pairgpu
    rm -f md_ref.dump md_host.dump md_gpu.dump md_gpusplit.dump
    $LMP -in in.md_ref -log none >${L}_host.log 2>&1; hrc=$?
    mv -f md_ref.dump md_host.dump 2>/dev/null
    $LMP_GPU -in in.md_ref -log none >${L}_dev.log 2>&1; drc=$?
    mv -f md_ref.dump md_gpu.dump 2>/dev/null
    # The same run forced into many batches. The descriptor does not fit for a
    # rank of any size, so pair_turbogap always takes this path; a leg that only
    # ran the one-batch case would be testing the rarer half.
    MADLIB_SOAP_MAX_GB=0.02 MADLIB_SOAP_BATCH_REPORT=1 \
      $LMP_GPU -in in.md_ref -log none >${L}_split.log 2>&1; src=$?
    mv -f md_ref.dump md_gpusplit.dump 2>/dev/null
    nb=$(grep -c "soap batches" ${L}_split.log 2>/dev/null)
    if [ $hrc -ne 0 ] || [ $drc -ne 0 ] || [ $src -ne 0 ] \
       || ! have md_host.dump || ! have md_gpu.dump || ! have md_gpusplit.dump; then
      tail -8 ${L}_dev.log | sed 's/^/     /'
      bad "one of the backends did not finish five steps"
    elif [ "${nb:-0}" -eq 0 ]; then
      bad "the forced-batch run reported no batches, so it tested nothing"
    else
      echo "    forced split: $(grep -m1 'soap batches' ${L}_split.log | sed 's/^ *//')"
      if $PYTHON compare_pair_md.py md_host.dump ${L}_host.log \
                                    device md_gpu.dump ${L}_dev.log \
                                    device_batched md_gpusplit.dump ${L}_split.log; then
        ok "the device backend gives the host's trajectory, batched or not"
      else
        bad "the device backend disagrees with the host"
      fi
      # Ranks sharing a card each measure the same free memory, so the budget
      # has to be divided by how many of them there are or they oversubscribe
      # it -- four ranks was an out-of-memory abort. Checked here rather than
      # by running something big enough to OOM, which no leg can afford.
      MADLIB_SOAP_BATCH_REPORT=1 mpirun --oversubscribe $MPIRUN_EXTRA -np "$RANKS" \
        $LMP_GPU -in in.md_ref -log none >${L}_ranks.log 2>&1
      seen=$(grep -m1 -o "ranks on this gpu [0-9]*" ${L}_ranks.log | grep -o "[0-9]*$")
      if [ "${seen:-0}" -eq "$RANKS" ]; then
        ok "the batch budget is shared between the $RANKS ranks on the card"
      else
        bad "$RANKS ranks share the card but the budget says ${seen:-none}"
      fi
    fi
  fi
  ;;

pair)
  leg "a whole TurboGAP potential as pair_style turbogap"
  if ! $LMP -in in.pair_turbogap -log none >/tmp/pair_tg.log 2>&1; then
    if grep -q "Unknown pair style" /tmp/pair_tg.log; then
      echo "    SKIP: built without the Fortran backend (-D MADLIB_ENABLE_TURBOGAP=on)"
    else
      sed 's/^/     /' /tmp/pair_tg.log | tail -12
      bad "pair_style turbogap did not run"
    fi
  else
    $PYTHON compare_pair_turbogap.py
    if [ $? -eq 0 ]; then ok "the four terms reproduce TurboGAP"
    else bad "pair_style turbogap disagrees with TurboGAP"; fi
  fi
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
