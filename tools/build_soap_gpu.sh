#!/bin/bash
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
#
# Build the device SOAP descriptor path against a TurboGAP source tree.
#
# TurboGAP's GPU soap_turbo is a drop-in for the CPU one -- same module name,
# same entry point -- so the two cannot be linked into one binary. This builds
# the GPU side on its own, for the dual-binary comparison in tests/run.sh.
#
# Nothing is copied into this repo: the sources stay in TurboGAP's tree and are
# compiled from there. That is the source-integrated arrangement described in
# docs/TURBOGAP-KK.md, and it is what keeps the vendored manifest from growing
# from twelve files to a directory tree.
#
# usage: build_soap_gpu.sh <turbogap-src-dir> [outdir]

set -u

CUDA_BIN=${CUDA_BIN:-/usr/local/cuda-13.4/bin}
ARCH=${ARCH:-sm_86}
FC=${FC:-mpif90}
FFLAGS="-O2 -fPIC -cpp -ffree-line-length-none -fallow-argument-mismatch -D_GPU -D_MPIF90"

# In dependency order. soap_turbo.f90 last: it uses all of them.
FORTRAN="kinds fortran_cuda_interfaces soap_turbo_functions soap_turbo_radial
         soap_turbo_angular soap_turbo_compress soap_turbo"

# gap_3b.cc and orthonormalization_kernels.cc are deliberately absent. Neither
# is on the descriptor path -- the GPU build computes the orthonormalisation in
# Fortran on the host, and 3b is a different term -- and neither compiles under
# CUDA 13.4 as it stands: gap_3b uses __shfl_down, removed in favour of
# __shfl_down_sync, and orthonormalization_kernels calls hipsolver with its
# #include commented out. Both are worth fixing upstream; neither blocks this.
SKIP_CU='gap_3b|orthonormalization_kernels'

usage() { echo "usage: $0 <turbogap-src-dir> [outdir]" >&2; exit 2; }

require_tree() {
    local src=$1
    for d in gpu hop soap_turbo_gpu/src; do
        if [ ! -d "$src/$d" ]; then
            echo "ERROR: $src/$d is missing -- is that TurboGAP's src/?" >&2
            exit 2
        fi
    done
    if [ ! -f "$src/fortran_cuda_interfaces.f90" ]; then
        echo "ERROR: $src/fortran_cuda_interfaces.f90 is missing" >&2
        exit 2
    fi
}

gather_sources() {
    local src=$1 out=$2
    cp "$src/kinds.f90" "$src/fortran_cuda_interfaces.f90" "$out/"
    cp "$src/soap_turbo_gpu/src/"*.f90 "$out/"
    cp "$MADLIB/lib/madlib/madlib_soap_gpu.f90" "$out/"
}

build_device() {
    local src=$1 out=$2 n=0
    mkdir -p "$out/cu"
    for f in "$src"/gpu/*.cu; do
        local b; b=$(basename "$f" .cu)
        echo "$b" | grep -qE "$SKIP_CU" && continue
        nvcc -O2 -std=c++17 -arch="$ARCH" --expt-extended-lambda --expt-relaxed-constexpr \
             -Xcompiler -fPIC -I"$src/gpu" -I"$src/hop" -I"$src/hop/source/hip" \
             -c "$f" -o "$out/cu/$b.o" > "$out/cu/$b.log" 2>&1 || {
            echo "ERROR: $b.cu did not compile; see $out/cu/$b.log" >&2
            grep -m3 -iE 'error' "$out/cu/$b.log" >&2
            exit 1
        }
        n=$((n + 1))
    done
    echo "  $n device objects"
}

build_fortran() {
    local out=$1
    ( cd "$out" || exit 1
      for f in $FORTRAN madlib_soap_gpu; do
          $FC $FFLAGS -J. -c "$f.f90" -o "$f.o" > "$f.log" 2>&1 || {
              echo "ERROR: $f.f90 did not compile; see $out/$f.log" >&2
              grep -m3 -iE 'error' "$f.log" >&2
              exit 1
          }
      done
    ) || exit 1
    echo "  fortran built"
}

main() {
    [ $# -ge 1 ] || usage
    local src=$1
    local out=${2:-/tmp/madlib_soap_gpu}
    MADLIB=$(cd "$(dirname "$0")/.." && pwd)
    export PATH="$CUDA_BIN:$PATH"

    require_tree "$src"
    rm -rf "$out"; mkdir -p "$out"
    gather_sources "$src" "$out"
    build_device "$src" "$out"
    build_fortran "$out"
    echo "  objects in $out (link with -lcudart -lcublas -lcuda -llapack -lblas)"
}

main "$@"
