#!/bin/bash
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
#
# Does TurboGAP's device layer compile inside a given LAMMPS build?
#
# The SOAP kernels are not like gap_2b.cu, which lifted verbatim because it
# called no vendor API at all. They call ~20 hip* names. hop maps those onto
# CUDA and is header-only -- 45 headers, no library -- so putting it on the
# include path costs nothing at runtime and keeps the ROCm spelling working.
# This checks that claim rather than assuming it, and reports any symbol that
# did not resolve to CUDA.
#
# usage: check_gpu_gate.sh <lammps-build-dir> <turbogap-src-dir>

set -u

CUDA_BIN=${CUDA_BIN:-/usr/local/cuda-13.4/bin}
ARCH=${ARCH:-sm_86}
WORK=${WORK:-/tmp/madlib_gpu_gate}

# The device files a SOAP path needs, in pipeline order.
FILES="gap_soap_radial gap_soap_angular gap_soap_descriptor gap_soap_forces
       gap_soap_radial_operator gpu_blas gpu_scatter gpu_scan gap_predict"

usage() {
    echo "usage: $0 <lammps-build-dir> <turbogap-src-dir>" >&2
    exit 2
}

# Checked separately from the capture below: an exit inside $( ) leaves only
# the subshell, so a missing build directory would print its error and then
# compile anyway.
require_build() {
    local build=$1
    if [ ! -f "$build/CMakeFiles/lammps.dir/flags.make" ]; then
        echo "ERROR: no $build/CMakeFiles/lammps.dir/flags.make" >&2
        echo "       is that a configured LAMMPS build directory?" >&2
        exit 2
    fi
    if [ ! -x "$build/../lib/kokkos/bin/nvcc_wrapper" ]; then
        echo "ERROR: no nvcc_wrapper at $build/../lib/kokkos/bin -- KOKKOS not enabled?" >&2
        exit 2
    fi
}

# LAMMPS's own include list, so this compiles the way LAMMPS will.
lammps_includes() {
    grep -m1 '^CXX_INCLUDES' "$1/CMakeFiles/lammps.dir/flags.make" \
        | sed 's/^CXX_INCLUDES = //'
}

compile_one() {
    local name=$1 src=$2 build=$3 inc=$4
    echo "#include \"$name.cu\"" > "$WORK/probe_$name.cpp"
    "$build/../lib/kokkos/bin/nvcc_wrapper" \
        -std=c++20 -O2 -fPIC -extended-lambda -arch="$ARCH" \
        -Xcudafe --diag_suppress=unrecognized_pragma \
        -D_KOKKOS -DKOKKOS_DEPENDENCE \
        -I"$src/gpu" -I"$src/hop" -I"$src/hop/source/hip" $inc \
        -c "$WORK/probe_$name.cpp" -o "$WORK/probe_$name.o" \
        > "$WORK/$name.log" 2>&1
}

# hop is only worth using if nothing vendor-specific survives it.
check_symbols() {
    local n objs
    objs=$(ls "$WORK"/probe_*.o 2>/dev/null | wc -l)
    echo
    if [ "$objs" -eq 0 ]; then
        echo "  nothing compiled, so there is nothing to check"
        return 1
    fi
    n=$(nm -uC "$WORK"/probe_*.o 2>/dev/null | grep -ciE '\bhip|rocblas')
    if [ "$n" -eq 0 ]; then
        echo "  no hip/rocm symbols left: hop resolved everything to CUDA"
    else
        echo "  $n unresolved hip/rocm symbols -- hop did not cover them:"
        nm -uC "$WORK"/probe_*.o | grep -iE '\bhip|rocblas' | sort -u | head
        return 1
    fi
    echo "  new link dependency:"
    nm -uC "$WORK"/probe_gpu_blas.o 2>/dev/null | grep -oiE 'cublas[A-Za-z_]*' \
        | sort -u | head -6 | sed 's/^/    /'
}

main() {
    [ $# -eq 2 ] || usage
    local build=$1 src=$2 inc bad=0
    require_build "$build"
    rm -rf "$WORK"
    mkdir -p "$WORK"
    export PATH="$CUDA_BIN:$PATH"
    inc=$(lammps_includes "$build")

    for f in $FILES; do
        if compile_one "$f" "$src" "$build" "$inc"; then
            echo "  OK    $f.cu"
        else
            echo "  FAIL  $f.cu"
            grep -m2 -iE 'error' "$WORK/$f.log" | sed 's/^/          /'
            bad=1
        fi
    done

    check_symbols || bad=1
    [ $bad -eq 0 ] || exit 1
    echo
    echo "  gate clear: the device layer compiles unmodified in this LAMMPS build"
}

main "$@"
