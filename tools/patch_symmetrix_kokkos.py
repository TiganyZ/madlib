#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Make symmetrix's Kokkos pair style build against current LAMMPS.

symmetrix registers its Kokkos styles through #define aliases that it #undefs
at the end of the PAIR_CLASS block. LAMMPS's style_pair.cpp generator copies the
NAME out of the PairStyle macro and emits it into a file compiled without
PAIR_CLASS, so the alias is never defined there and the build fails with
"identifier PairSymmetrixMACEKokkosDeviceDouble is undefined".

LAMMPS's own Kokkos styles avoid this by writing the template arguments inline,
e.g. PairStyle(snap/kk, PairSNAPKokkosDevice<LMPDeviceType>), which is what this
does. Upstream fix belongs in symmetrix; this keeps a GPU build working now.

Idempotent.
"""

import re
import sys

ALIASES = {
    "PairSymmetrixMACEKokkosDeviceDouble": "PairSymmetrixMACEKokkos<LMPDeviceType,double>",
    "PairSymmetrixMACEKokkosHostDouble": "PairSymmetrixMACEKokkos<LMPHostType,double>",
    "PairSymmetrixMACEKokkosDeviceFloat": "PairSymmetrixMACEKokkos<LMPDeviceType,float>",
    "PairSymmetrixMACEKokkosHostFloat": "PairSymmetrixMACEKokkos<LMPHostType,float>",
}


def already_patched(text):
    return "#define PairSymmetrixMACEKokkosDeviceDouble" not in text


def patch(text):
    for alias, real in ALIASES.items():
        text = text.replace(f"#define {alias} {real}\n", "")
        text = text.replace(f"#undef {alias}\n", "")
        text = re.sub(rf"(PairStyle\([^,]+,){alias}\)", rf"\1{real})", text)
    return text


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_symmetrix_kokkos.py path/to/pair_symmetrix_mace_kokkos.h")
    path = sys.argv[1]
    text = open(path).read()
    if already_patched(text):
        print("  already patched")
        return
    open(path, "w").write(patch(text))
    print("  patched", path)


if __name__ == "__main__":
    main()
