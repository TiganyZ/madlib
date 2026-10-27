#!/bin/bash
# LAMMPS as a shared library with its Python module and MADLIB, installed into the MACE venv.
# PolarMACE's k-space electrostatics needs the whole cell, so it enters through fix external.
#   bash setup/build_lammps_python.sh        (after grace_lifco's build_lammps.sh has checked LAMMPS out)
set -eu
source "$(dirname "$0")/../../grace_lifco/env.sh"
MACE_PREFIX=${MACE_PREFIX:-/l/zarrout1/mace-polar}
MACE_PY=$MACE_PREFIX/venv/bin/python
PYBUILD_DIR=${PYBUILD_DIR:-$PREFIX/build-python}

say() { printf '\n=== %s ===\n' "$1"; }

configure() {
    say "cmake"
    rm -rf "$PYBUILD_DIR"
    mkdir -p "$PYBUILD_DIR"
    cd "$PYBUILD_DIR"
    cmake "$LAMMPS_DIR/cmake" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_STANDARD=20 \
        -D CMAKE_CXX_COMPILER=mpicxx \
        -D CMAKE_Fortran_COMPILER=gfortran \
        -D CMAKE_PREFIX_PATH="$TOOLCHAIN_PREFIX" \
        -D CMAKE_SHARED_LINKER_FLAGS="-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$TOOLCHAIN_PREFIX/lib" \
        -D CMAKE_EXE_LINKER_FLAGS="-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$TOOLCHAIN_PREFIX/lib" \
        -D BUILD_SHARED_LIBS=on \
        -D LAMMPS_EXCEPTIONS=on \
        -D BUILD_MPI=on \
        -D Python_EXECUTABLE="$MACE_PY" \
        -D PKG_PYTHON=on \
        -D PKG_MOLECULE=on \
        -D PKG_EXTRA-FIX=on \
        -D PKG_EXTRA-PAIR=on \
        -D PKG_MC=on \
        -D PKG_MADLIB=on
}

compile() {
    say "make -j$NPROC"
    cd "$PYBUILD_DIR"
    make -j"$NPROC"
}

install_module() {
    say "python module into $MACE_PY"
    cd "$PYBUILD_DIR"
    # make install-python calls python -m pip, which a uv venv lacks.
    VIRTUAL_ENV=$MACE_PREFIX/venv "$UV" pip install -q pip mpi4py
    make install-python
}

check() {
    say "check"
    "$MACE_PY" -c "from lammps import lammps; l = lammps(cmdargs=['-log', 'none', '-screen', 'none']); print(l.version(), l.has_style('fix', 'mad/xrd'), l.has_style('fix', 'external'), l.has_style('pair', 'dispersion/d3'))"
}

configure
compile
install_module
check
