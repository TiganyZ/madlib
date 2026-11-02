#!/bin/bash
# Triton login node: serial LAMMPS as a shared library with MADLIB, installed into the MACE venv.
# Built in /tmp and deleted afterwards: /scratch/work is near its inode quota, and the wheel
# carries liblammps.so into the venv, so nothing else has to stay.
#   bash setup/triton_lammps_python.sh   (after setup/make_venv.sh; madlib synced to $MADLIB_DIR)
set -eu
PREFIX=${PREFIX:-/scratch/work/zarrout1/mace-polar}
MADLIB_DIR=${MADLIB_DIR:-/scratch/work/zarrout1/madlib-grace/madlib}
UV=${UV:-$HOME/.local/bin/uv}
LAMMPS_REPO=https://github.com/lammps/lammps.git
LAMMPS_COMMIT=a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b
TMP_BUILD=${TMP_BUILD:-/tmp/$USER-lammps-python}
NPROC=${NPROC:-8}
MACE_PY=$PREFIX/venv/bin/python

say() { printf '\n=== %s ===\n' "$1"; }

load_modules() {
    module load triton/2025.1-gcc gcc/13.3.0 cmake/3.30.5
}

checkout() {
    say "LAMMPS $LAMMPS_COMMIT"
    rm -rf "$TMP_BUILD"
    git clone -q --filter=blob:none "$LAMMPS_REPO" "$TMP_BUILD/lammps"
    git -C "$TMP_BUILD/lammps" checkout -q "$LAMMPS_COMMIT"
}

register_madlib() {
    say "registering madlib"
    local lmp=$TMP_BUILD/lammps
    ln -sfn "$MADLIB_DIR/src/MADLIB" "$lmp/src/MADLIB"
    ln -sfn "$MADLIB_DIR/lib/madlib" "$lmp/lib/madlib"
    cp "$MADLIB_DIR/cmake-MADLIB.cmake" "$lmp/cmake/Modules/Packages/MADLIB.cmake"
    "$MACE_PY" "$MADLIB_DIR/tools/register_madlib.py" "$lmp"
}

configure() {
    say "cmake"
    mkdir -p "$TMP_BUILD/build"
    cd "$TMP_BUILD/build"
    cmake "$TMP_BUILD/lammps/cmake" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_STANDARD=20 \
        -D BUILD_SHARED_LIBS=on \
        -D LAMMPS_EXCEPTIONS=on \
        -D BUILD_MPI=off \
        -D Python_EXECUTABLE="$MACE_PY" \
        -D PKG_PYTHON=on \
        -D PKG_MOLECULE=on \
        -D PKG_EXTRA-FIX=on \
        -D PKG_EXTRA-PAIR=on \
        -D PKG_MC=on \
        -D PKG_MADLIB=on
}

compile_and_install() {
    say "make -j$NPROC"
    cd "$TMP_BUILD/build"
    make -j"$NPROC"
    # make install-python calls python -m pip, which a uv venv lacks.
    VIRTUAL_ENV=$PREFIX/venv "$UV" pip install -q pip
    make install-python
}

check() {
    say "check"
    "$MACE_PY" -c "from lammps import lammps; l = lammps(cmdargs=['-log', 'none', '-screen', 'none']); print(l.version(), l.has_style('fix', 'mad/xrd'), l.has_style('fix', 'external'), l.has_style('pair', 'dispersion/d3'))"
}

# Clone first: the modules put a libcurl on the path that git's TLS check rejects.
checkout
load_modules
register_madlib
configure
compile_and_install
check
rm -rf "$TMP_BUILD"
