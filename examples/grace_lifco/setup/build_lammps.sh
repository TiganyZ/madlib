#!/bin/bash
# Upstream LAMMPS with MADLIB, and ML-PACE (grace/fs, no TensorFlow) taken from the GRACE fork.
# The fork's own core is too old for MADLIB (NeighRequest::set_cutoff_fixed).
set -eu
source "$(dirname "$0")/../env.sh"

say() { printf '\n=== %s ===\n' "$1"; }

checkout() {
    local repo=$1 commit=$2 dir=$3
    say "$repo $commit"
    [ -d "$dir/.git" ] || git clone -q --filter=blob:none "$repo" "$dir"
    git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin "$commit"
    git -C "$dir" checkout -q --force "$commit"
    git -C "$dir" clean -qfd src cmake
}

overlay_grace() {
    say "ML-PACE from the GRACE fork"
    rm -rf "$LAMMPS_DIR/src/ML-PACE"
    cp -r "$GRACE_DIR/src/ML-PACE" "$LAMMPS_DIR/src/ML-PACE"
    cp "$GRACE_DIR/cmake/Modules/Packages/ML-PACE.cmake" "$LAMMPS_DIR/cmake/Modules/Packages/ML-PACE.cmake"
}

clear_stale_fortran() {
    rm -rf "$MADLIB_DIR/lib/madlib/mod"
    rm -f "$MADLIB_DIR"/lib/madlib/*.mod "$MADLIB_DIR"/lib/madlib/*.o \
          "$MADLIB_DIR"/lib/madlib/turbogap/*.o "$MADLIB_DIR"/lib/madlib/libmadlib.a
}

register_madlib() {
    say "registering madlib"
    ln -sfn "$MADLIB_DIR/src/MADLIB" "$LAMMPS_DIR/src/MADLIB"
    ln -sfn "$MADLIB_DIR/lib/madlib" "$LAMMPS_DIR/lib/madlib"
    cp "$MADLIB_DIR/cmake-MADLIB.cmake" "$LAMMPS_DIR/cmake/Modules/Packages/MADLIB.cmake"
    "$PY" "$MADLIB_DIR/tools/register_madlib.py" "$LAMMPS_DIR"
}

# -L/-rpath: conda's sysroot must not supply libc/libm to a binary started by the system loader.
configure() {
    say "cmake"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$LAMMPS_DIR/cmake" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_STANDARD=20 \
        -D CMAKE_CXX_COMPILER=mpicxx \
        -D CMAKE_Fortran_COMPILER=gfortran \
        -D CMAKE_PREFIX_PATH="$TOOLCHAIN_PREFIX" \
        -D CMAKE_SKIP_BUILD_RPATH=ON \
        -D CMAKE_EXE_LINKER_FLAGS="-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$TOOLCHAIN_PREFIX/lib" \
        -D BUILD_MPI=on \
        -D PKG_ML-PACE=on \
        -D NO_GRACE_TF=ON \
        -D PKG_MANYBODY=on \
        -D PKG_MOLECULE=on \
        -D PKG_EXTRA-FIX=on \
        -D PKG_EXTRA-PAIR=on \
        -D PKG_MC=on \
        -D PKG_MADLIB=on
}

compile() {
    say "make -j$NPROC"
    cd "$BUILD_DIR"
    make -j"$NPROC"
}

record_versions() {
    {
        echo "lammps $LAMMPS_REPO $(git -C "$LAMMPS_DIR" rev-parse HEAD)"
        echo "ml-pace $GRACE_REPO $(git -C "$GRACE_DIR" rev-parse HEAD)"
        echo "madlib $(cat "$MADLIB_DIR/.madlib_commit" 2>/dev/null || git -C "$MADLIB_DIR" rev-parse HEAD)"
        echo "mpicxx $(mpicxx --version | head -1)"
    } > "$BUILD_DIR/VERSIONS"
    cat "$BUILD_DIR/VERSIONS"
}

check_styles() {
    say "styles"
    "$LMP" -h | grep -oE '\bmad/[a-z/]+|\bgrace/fs\b|\bpace\b' | sort -u
}

checkout "$LAMMPS_REPO" "$LAMMPS_COMMIT" "$LAMMPS_DIR"
checkout "$GRACE_REPO" "$GRACE_COMMIT" "$GRACE_DIR"
overlay_grace
clear_stale_fortran
register_madlib
configure
compile
record_versions
check_styles
