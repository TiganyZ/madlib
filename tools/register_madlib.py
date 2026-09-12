#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Register MADLIB with a LAMMPS checkout's cmake and documentation indexes.

Idempotent, so re-running after a LAMMPS update is safe.

Two separate lists matter and only one of them is obvious. STANDARD_PACKAGES
compiles the sources; PKG_WITH_INCL is what causes Packages/MADLIB.cmake to be
read at all, and without it the backend options are accepted, cached and
silently ignored.
"""

import os
import re
import sys

# The Kokkos variants live in src/KOKKOS, not in src/MADLIB. That is LAMMPS's
# own arrangement -- every Kokkos style sits beside the package's Kokkos
# machinery rather than with its host counterpart -- and it is also what makes
# them build only when PKG_KOKKOS is on, without a conditional anywhere.
KOKKOS_FILES = ["pair_turbogap_kokkos.h", "pair_turbogap_kokkos.cpp", "mad_neigh_kokkos.h"]

PAIR_STYLES = [("turbogap", "pair_turbogap"), ("turbogap/kk", "pair_turbogap_kokkos")]

FIX_STYLES = [("mad/ir", "fix_mad_ir"), ("mad/pdf", "fix_mad"), ("mad/sq", "fix_mad"),
              ("mad/xps", "fix_mad"), ("mad/xrd", "fix_mad")]
COMPUTE_STYLES = ["mad/pdf", "mad/property/coord", "mad/property/dipole",
                  "mad/property/mliap", "mad/property/python", "mad/property/table",
                  "mad/property/turbogap"]


def insert_sorted(path, entry, among):
    """Put entry in its alphabetical place among the lines matching `among`."""
    lines = open(path).read().split("\n")
    if entry in lines:
        return False
    idx = [i for i, l in enumerate(lines) if re.match(among, l)]
    if not idx:
        raise SystemExit(f"ERROR: found no list to extend in {path}")
    pos = next((i for i in idx if lines[i] > entry), idx[-1] + 1)
    lines.insert(pos, entry)
    open(path, "w").write("\n".join(lines))
    return True


def add_to_standard_packages(root):
    return insert_sorted(f"{root}/cmake/CMakeLists.txt", "  MADLIB", r"^  [A-Z][A-Z0-9-]*$")


def add_to_pkg_with_incl(root):
    path = f"{root}/cmake/CMakeLists.txt"
    s = open(path).read()
    head = s.split("PKG_WITH_INCL")[1][:400] if "PKG_WITH_INCL" in s else ""
    if "MADLIB" in head:
        return False
    # Insert before MDI, which is in the first list on every version so far.
    new = re.sub(r"(foreach\(PKG_WITH_INCL [^)]*?)\bMDI\b", r"\1MADLIB MDI", s, count=1)
    if new == s:
        raise SystemExit("ERROR: could not find the PKG_WITH_INCL list to extend")
    open(path, "w").write(new)
    return True


def index_docs(root):
    n = 0
    for style, doc in FIX_STYLES:
        n += insert_sorted(f"{root}/doc/src/Commands_fix.rst",
                           f"   * :doc:`{style} <{doc}>`", r"^   \* :doc:`")
    for style in COMPUTE_STYLES:
        doc = "compute_mad_pdf" if style == "mad/pdf" else "compute_mad_property"
        n += insert_sorted(f"{root}/doc/src/Commands_compute.rst",
                           f"   * :doc:`{style} <{doc}>`", r"^   \* :doc:`")
    return n


def add_gitignore(root):
    import os
    names = sorted(os.listdir(f"{root}/src/MADLIB"))
    n = 0
    for name in names:
        if name.endswith((".cpp", ".h")):
            n += insert_sorted(f"{root}/src/.gitignore", "/" + name, r"^/[a-z]")
    return n


def link_kokkos(root, madlib):
    """Put the Kokkos variants where LAMMPS looks for Kokkos styles.

    They go in src/KOKKOS rather than src/MADLIB, and that placement is load
    bearing rather than tidy: LAMMPS builds src/KOKKOS only when PKG_KOKKOS is
    on, so a file there needs no conditional to be left out of a non-Kokkos
    build, and the style scanner that writes style_pair.h only reads package
    directories that are enabled. A Kokkos style left in src/MADLIB would
    either fail to compile without Kokkos or never be registered with it.
    """
    import os
    n = 0
    dest = f"{root}/src/KOKKOS"
    if not os.path.isdir(dest):
        return -1
    for name in KOKKOS_FILES:
        src = os.path.abspath(f"{madlib}/src/KOKKOS/{name}")
        link = f"{dest}/{name}"
        if os.path.islink(link) and os.readlink(link) == src:
            continue
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(src, link)
        n += 1
    return n


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: register_madlib.py path/to/lammps")
    root = sys.argv[1]
    madlib = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("  STANDARD_PACKAGES:", "added" if add_to_standard_packages(root) else "already there")
    print("  PKG_WITH_INCL:    ", "added" if add_to_pkg_with_incl(root) else "already there")
    print("  doc index:        ", index_docs(root), "entries added")
    print("  src/.gitignore:   ", add_gitignore(root), "entries added")
    k = link_kokkos(root, madlib)
    if k < 0:
        print("  KOKKOS variants:   src/KOKKOS is absent; nothing linked")
    else:
        print("  KOKKOS variants:  ", k, "linked into src/KOKKOS")


if __name__ == "__main__":
    main()
