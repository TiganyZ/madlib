#!/bin/bash
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
#
# Build LAMMPS with MADLIB and every ML potential backend it can use.
#
#   ./install_turbogap_lammps.sh            everything
#   ./install_turbogap_lammps.sh quip mace  only those stages
#
# Stages are independent and each is skipped when its output already exists, so
# re-running after a failure resumes rather than restarting.

set -u

# ------------------------------------------------------------------ settings
#
# Put PREFIX on a local or scratch disk rather than in a quota'd home. The
# build is tens of gigabytes, but the binding constraint is usually the FILE
# COUNT, not the bytes: a venv with torch and tensorflow in it is a few hundred
# thousand small files, and an inode quota is reported as "disk quota exceeded"
# with plenty of space free, which reads as the wrong problem entirely.
PREFIX=${PREFIX:-$HOME/work/lammps/turbogap-lammps}
LAMMPS_BRANCH=${LAMMPS_BRANCH:-develop}
JOBS=${JOBS:-8}

# GPU. Set GPU_ARCH to a Kokkos arch keyword to build for a device, e.g.
# AMPERE80 for A100, HOPPER90 for H100, VOLTA70 for V100. Empty means CPU only.
# Empty means a CPU-only build; "auto" asks the device. Otherwise give a Kokkos
# arch keyword directly, e.g. AMPERE86 for an RTX A2000, HOPPER90 for an H100.
GPU_ARCH=${GPU_ARCH:-}
KOKKOS_BACKEND=${KOKKOS_BACKEND:-CUDA}

# nvcc refuses a host compiler newer than the CUDA release supports. Leave empty
# to use the default, which is usually right once CUDA_HOME points somewhere
# recent.
HOST_CXX=${HOST_CXX:-}

# SOAP on the device. MADLIB_GPU_SOAP=on swaps the vendored CPU soap_turbo for
# TurboGAP's GPU one, which has to come from a TurboGAP checkout: it is a
# directory tree of Fortran and CUDA rather than the twelve files MADLIB
# vendors, and it is compiled in place. Needs GPU_ARCH, and MPI, because
# gap.f90's device half opens with an unconditional `use mpi`.
#
# The two soap_turbo forks declare the same module and entry names, so a binary
# holds one of them. This is a build-time choice, not a runtime one.
MADLIB_GPU_SOAP=${MADLIB_GPU_SOAP:-off}
TURBOGAP_SRC=${TURBOGAP_SRC:-}

# Which CUDA to build against. "auto" takes the newest /usr/local/cuda-* -- the
# one on PATH is chosen by a system alternatives link and is often not it, and
# Kokkos refuses to build C++20 with nvcc older than 12.2, so the difference
# between the two decides whether a GPU build is possible at all.
CUDA_HOME=${CUDA_HOME:-auto}

# Which stages exist, in dependency order.
ALL_STAGES="deps python lammps quip symmetrix madlib build test"

# ------------------------------------------------------------------ helpers
say()  { printf '\n=== %s ===\n' "$1"; }
step() { printf '  -> %s\n' "$1"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_dir() {
    mkdir -p "$1" || die "cannot create $1"
}

# ------------------------------------------------------------------ stages
stage_deps() {
    say "checking prerequisites"
    local missing=""
    for tool in git make g++ gfortran python3; do
        have $tool || missing="$missing $tool"
    done
    [ -n "$missing" ] && die "missing tools:$missing"

    bootstrap_build_tools
    step "cmake $(cmake --version | head -1 | awk '{print $3}') | meson $(meson --version) (symmetrix needs cmake >= 3.27)"
    step "$(g++ --version | head -1)"
    step "$(gfortran --version | head -1)"
    [ -n "$HOST_CXX" ] && step "host compiler for nvcc: $HOST_CXX"

    if [ "$MADLIB_GPU_SOAP" = on ]; then
        [ -n "$GPU_ARCH" ] || die "MADLIB_GPU_SOAP needs GPU_ARCH"
        [ -n "$TURBOGAP_SRC" ] || die "MADLIB_GPU_SOAP needs TURBOGAP_SRC=<turbogap>/src"
        [ -d "$TURBOGAP_SRC/soap_turbo_gpu/src" ] \
            || die "$TURBOGAP_SRC has no soap_turbo_gpu/src -- is that TurboGAP's src/?"
        [ -e "$TURBOGAP_SRC/hop/hop" ] \
            || die "$TURBOGAP_SRC/hop is empty; run git submodule update --init in TurboGAP"
        step "device SOAP from $TURBOGAP_SRC"
    fi

    if [ -n "$GPU_ARCH" ]; then
        select_cuda
        have nvcc || die "GPU_ARCH is set but nvcc is not on PATH"
        step "cuda: $(nvcc --version | tail -2 | head -1 | sed 's/^ *//') from ${CUDA_HOME}"
        [ "$GPU_ARCH" = auto ] && GPU_ARCH=$(detect_gpu_arch)
        step "kokkos arch: $GPU_ARCH"
    fi
}

bootstrap_build_tools() {
    # cmake, meson and ninja are all hard requirements and a cluster node may
    # have none of them. pip has all three and needs no root.
    [ -d "$PREFIX/venv" ] || python3 -m venv "$PREFIX/venv"
    export PATH="$PREFIX/venv/bin:$PATH"

    local want=""
    if ! have cmake || [ "$(cmake --version | head -1 | awk '{print $3}' | cut -d. -f1-2)" \
                         \< "3.27" ]; then
        want="$want cmake"
    fi
    have meson || want="$want meson"
    have ninja || want="$want ninja"

    if [ -n "$want" ]; then
        step "installing into the venv:$want"
        "$PREFIX/venv/bin/pip" install --quiet --upgrade $want || die "could not install$want"
        hash -r
    fi
    for t in cmake meson ninja; do
        have $t || die "$t still not available"
    done
}

select_cuda() {
    if [ "$CUDA_HOME" = auto ]; then
        # Sort by version, newest last. A bare /usr/local/cuda is a symlink to
        # whichever the system picked and is deliberately not trusted.
        CUDA_HOME=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
        [ -z "$CUDA_HOME" ] && CUDA_HOME=/usr/local/cuda
    fi
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export CUDA_HOME
        hash -r
    fi
}

detect_gpu_arch() {
    # nvidia-smi is not reliable -- a driver/library version skew breaks it
    # while the CUDA runtime still works -- so ask the device directly.
    local d=$PREFIX/.arch
    mkdir -p "$d"
    cat > "$d/arch.cu" <<'CU'
#include <cstdio>
int main(){cudaDeviceProp p;if(cudaGetDeviceProperties(&p,0)!=cudaSuccess)return 1;
printf("%d%d\n",p.major,p.minor);return 0;}
CU
    ( cd "$d" && nvcc -o arch arch.cu >/dev/null 2>&1 && ./arch ) > "$d/cc" 2>/dev/null
    local cc; cc=$(cat "$d/cc" 2>/dev/null)
    case "$cc" in
        70|72) echo VOLTA$cc ;;
        75)    echo TURING75 ;;
        80|86|89) echo AMPERE$cc ;;
        90)    echo HOPPER90 ;;
        *)     die "could not identify the GPU (compute capability '$cc'); set GPU_ARCH by hand" ;;
    esac
}

# AMPERE86 -> 86. Kokkos names an architecture, CUDA numbers it, and the digits
# are the same in both.
kokkos_arch_to_cc() {
    echo "$1" | sed 's/[^0-9]//g'
}

stage_python() {
    say "python environment"
    local venv=$PREFIX/venv
    if [ -d "$venv" ]; then
        step "already at $venv"
    else
        python3 -m venv "$venv" || die "could not create the venv"
    fi
    # shellcheck disable=SC1091
    . "$venv/bin/activate"
    step "installing packages"
    pip install --quiet --upgrade pip
    pip install --quiet numpy scipy ase
    # torch: the CUDA build when a GPU was asked for, otherwise the CPU wheel
    if [ -n "$GPU_ARCH" ]; then
        pip install --quiet torch
    else
        pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    fi
    pip install --quiet mace-torch || step "WARNING: mace-torch did not install"
    pip install --quiet tensorpotential || step "note: GRACE (tensorpotential) not installed"
    step "python: $(python3 -c 'import sys;print(sys.version.split()[0])')"
    step "torch:  $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
}

stage_lammps() {
    say "lammps"
    if [ -d "$PREFIX/lammps/.git" ]; then
        step "already cloned"
    else
        git clone --depth 1 -b "$LAMMPS_BRANCH" \
            https://github.com/lammps/lammps.git "$PREFIX/lammps" || die "lammps clone failed"
    fi
    step "version: $(grep -o '"[^"]*"' "$PREFIX/lammps/src/version.h" | head -1)"
}

stage_quip() {
    say "quip (pair_style quip, for GAP potentials)"
    bootstrap_build_tools
    local quip=$PREFIX/QUIP
    if [ ! -d "$quip/.git" ]; then
        git clone --recursive --depth 1 \
            https://github.com/libAtoms/QUIP.git "$quip" || die "QUIP clone failed"
    fi
    if [ -f "$quip/quiplib/libquip.a" ]; then
        step "libquip.a already built"
        return
    fi
    ( cd "$quip" && meson setup builddir --default-library=static \
        --buildtype=release -Dgap=true >/dev/null 2>&1 || true
      meson compile -C builddir ) || die "QUIP build failed"
    combine_quip_archives "$quip"
}

combine_quip_archives() {
    # QUIP's meson build makes one archive per subdirectory; LAMMPS wants one.
    # libf90wrap_stub is easy to omit and shows up as an undefined f90wrap_abort_.
    local quip=$1
    step "combining archives into libquip.a"
    ensure_dir "$quip/quiplib"
    ( cd "$quip/quiplib" && rm -f libquip.a
      for a in ../builddir/src/Potentials/libPotentials.a \
               ../builddir/src/GAP/libGAP.a \
               ../builddir/src/libAtoms/liblibAtoms.a \
               ../builddir/src/libAtoms/libf90wrap_stub.a \
               ../builddir/src/Utils/libUtils.a \
               ../builddir/src/fox/libfox.a; do
          d=$(basename "$a" .a); mkdir -p "$d" && ( cd "$d" && ar x "../$a" )
      done
      ar rcs libquip.a ./*/*.o && ranlib libquip.a ) || die "could not build libquip.a"
}

stage_symmetrix() {
    say "symmetrix (pair_style symmetrix/mace)"
    local sym=$PREFIX/symmetrix
    if [ ! -d "$sym/.git" ]; then
        git clone --recursive --depth 1 \
            https://github.com/wcwitt/symmetrix.git "$sym" || die "symmetrix clone failed"
    fi
    # --force matters: an interrupted clone leaves a submodule registered but
    # empty, and a plain --init then reports success over an empty directory.
    # Kokkos is only needed for the GPU path, and its submodules are large.
    local subs="libsymmetrix/external/json libsymmetrix/external/sphericart
                libsymmetrix/external/cblas-prototypes"
    [ -n "$GPU_ARCH" ] && subs="$subs libsymmetrix/external/kokkos libsymmetrix/external/kokkos-kernels"
    ( cd "$sym" && git submodule update --init --force --depth 1 $subs ) || die "symmetrix submodules failed"
    # symmetrix registers its Kokkos styles through #define aliases it then
    # undefs, and LAMMPS's style_pair.cpp generator emits the alias name into a
    # file compiled without PAIR_CLASS. Inline the templates the way LAMMPS's
    # own Kokkos styles do.
    if [ -n "$GPU_ARCH" ] && [ -f "$PREFIX/patch_symmetrix_kokkos.py" ]; then
        step "adapting the kokkos style registration"
        python3 "$PREFIX/patch_symmetrix_kokkos.py" \
            "$sym/pair_symmetrix/pair_symmetrix_mace_kokkos.h"
    fi
    step "patching lammps"
    ( cd "$sym/pair_symmetrix" && ./install.sh "$PREFIX/lammps" ) || die "symmetrix install.sh failed"
}

stage_madlib() {
    say "madlib"
    local mad=$PREFIX/madlib
    if [ ! -d "$mad" ]; then
        if [ -n "${MADLIB_SOURCE:-}" ]; then
            cp -r "$MADLIB_SOURCE" "$mad" || die "could not copy MADLIB from $MADLIB_SOURCE"
        else
            die "set MADLIB_SOURCE to a MADLIB checkout, or clone it to $mad"
        fi
    fi
    # A .mod file is tied to the exact gfortran that wrote it, so artefacts
    # carried over from another machine fail with "created by a different
    # version of GNU Fortran" rather than being rebuilt.
    step "clearing any stale fortran build artefacts"
    rm -f "$mad"/lib/madlib/*.mod "$mad"/lib/madlib/*.o \
          "$mad"/lib/madlib/turbogap/*.o "$mad"/lib/madlib/libmadlib.a

    step "linking into lammps"
    ln -sfn "$mad/src/MADLIB" "$PREFIX/lammps/src/MADLIB"
    ln -sfn "$mad/lib/madlib" "$PREFIX/lammps/lib/madlib"
    cp "$mad/cmake-MADLIB.cmake" "$PREFIX/lammps/cmake/Modules/Packages/MADLIB.cmake"
    for f in "$mad"/doc/src/*.rst; do
        ln -sf "$f" "$PREFIX/lammps/doc/src/$(basename "$f")"
    done
    step "registering the package with cmake"
    python3 "$mad/tools/register_madlib.py" "$PREFIX/lammps" || die "registration failed"
}

cmake_flags() {
    # One place for the whole option set, so the GPU and CPU builds differ only
    # by what this adds.
    echo "-D BUILD_MPI=on"
    echo "-D CMAKE_BUILD_TYPE=Release"
    echo "-D CMAKE_CXX_STANDARD=20"
    echo "-D PKG_MANYBODY=on -D PKG_MOLECULE=on -D PKG_EXTRA-FIX=on -D PKG_MC=on"
    echo "-D PKG_ML-IAP=on -D PKG_ML-SNAP=on -D PKG_ML-PACE=on"
    echo "-D PKG_ML-QUIP=on -D QUIP_LIBRARY=$PREFIX/QUIP/quiplib/libquip.a"
    echo "-D PKG_PYTHON=on"
    echo "-D PKG_MADLIB=on -D MADLIB_ENABLE_TURBOGAP=on -D MADLIB_ENABLE_PYTHON=on"
    if [ "$MADLIB_GPU_SOAP" = on ]; then
        echo "-D MADLIB_ENABLE_GPU=on -D MADLIB_TURBOGAP_SRC=$TURBOGAP_SRC"
        echo "-D MADLIB_GPU_ARCH=$(kokkos_arch_to_cc "$GPU_ARCH")"
    fi
    echo "-D Python_EXECUTABLE=$PREFIX/venv/bin/python"
    if [ -n "$GPU_ARCH" ]; then
        echo "-D PKG_KOKKOS=on -D Kokkos_ENABLE_$KOKKOS_BACKEND=on -D Kokkos_ARCH_$GPU_ARCH=on"
        echo "-D Kokkos_ENABLE_OPENMP=on -D SYMMETRIX_KOKKOS=on"
        echo "-D CMAKE_CXX_COMPILER=$PREFIX/lammps/lib/kokkos/bin/nvcc_wrapper"
        # Naming the toolkit ROOT, not just putting its bin on PATH. Without
        # this cmake finds whatever libcudart the system ships -- on a machine
        # with CUDA 12 installed and a newer one under /usr/local, the device
        # code is compiled by the newer nvcc and linked against the older
        # runtime. That builds and links cleanly and then dies as a SIGFPE
        # inside a Kokkos parallel_for launch, naming nothing recognisable.
        echo "-D CUDAToolkit_ROOT=$CUDA_HOME"
    else
        echo "-D SYMMETRIX_KOKKOS=off"
    fi
}

# The runtime the binary resolves must match the nvcc that built its device
# code. Checked rather than assumed: the failure mode is silent at build time.
check_cuda_runtime() {
    [ -n "$GPU_ARCH" ] || return 0
    local lmp=$PREFIX/lammps/build/lmp got want
    [ -x "$lmp" ] || return 0
    got=$(ldd "$lmp" 2>/dev/null | sed -n 's/.*libcudart\.so\.\([0-9]*\).*/\1/p' | head -1)
    want=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    [ -n "$got" ] && [ -n "$want" ] || return 0
    if [ "$got" != "$want" ]; then
        die "linked libcudart.so.$got but nvcc is $want -- set CUDA_HOME to one toolkit"
    fi
    step "cuda runtime $got matches nvcc"
}

stage_build() {
    say "building lammps"
    bootstrap_build_tools
    [ -n "$GPU_ARCH" ] && select_cuda
    # nvcc_wrapper takes its host compiler from the environment. Passing
    # -ccbin as a CXX flag does not work: the wrapper forwards CMAKE_CXX_FLAGS
    # to the host compiler, which then sees -ccbin and rejects it.
    if [ -n "$GPU_ARCH" ] && [ -n "$HOST_CXX" ]; then
        export NVCC_WRAPPER_DEFAULT_COMPILER=$HOST_CXX
        step "nvcc host compiler: $HOST_CXX"
    fi
    local build=$PREFIX/lammps/build
    ensure_dir "$build"
    # shellcheck disable=SC2046
    ( cd "$build" && cmake ../cmake $(cmake_flags) ) || die "cmake failed"
    ( cd "$build" && make -j"$JOBS" ) || die "build failed"
    check_cuda_runtime
    step "binary: $build/lmp"
}

stage_test() {
    say "testing"
    local lmp=$PREFIX/lammps/build/lmp
    [ -x "$lmp" ] || die "no binary at $lmp"
    step "styles present:"
    "$lmp" -h 2>/dev/null | grep -oE "\bmad/[a-z/]+|\bquip\b|symmetrix/mace" \
        | sort -u | tr '\n' ' ' | sed 's/^/     /'
    printf '\n'
    if [ -d "$PREFIX/madlib/tests" ]; then
        step "running the MADLIB suite"
        ( cd "$PREFIX/madlib/tests" && LMP="$lmp" ./run.sh ) || die "tests failed"
    fi
}

# ------------------------------------------------------------------ driver
main() {
    local stages=${*:-$ALL_STAGES}
    ensure_dir "$PREFIX"
    printf 'MADLIB + LAMMPS install\n  prefix: %s\n  stages: %s\n  gpu:    %s\n  soap:   %s\n' \
        "$PREFIX" "$stages" "${GPU_ARCH:-none (cpu build)}" \
        "$([ "$MADLIB_GPU_SOAP" = on ] && echo device || echo host)"
    for s in $stages; do
        case $s in
            deps)      stage_deps ;;
            python)    stage_python ;;
            lammps)    stage_lammps ;;
            quip)      stage_quip ;;
            symmetrix) stage_symmetrix ;;
            madlib)    stage_madlib ;;
            build)     stage_build ;;
            test)      stage_test ;;
            *)         die "unknown stage '$s' (have: $ALL_STAGES)" ;;
        esac
    done
    say "done"
}

main "$@"
