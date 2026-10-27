"""A training set as mace-scf reads it.

    python lsc_sets.py IN.xyz OUT.xyz

FHI-aims clusters (config_energy_weight 0, another energy zero) lose REF_energy: mace-scf has no
per-frame energy weight, and a frame without the key adds no energy loss. Every frame with a
dipole gets the per-component dipole weight mace-scf insists on.
"""

import sys

from ase.io import read, write

DIPOLE_WEIGHT = [1.0, 1.0, 1.0]


def main(src, dst):
    frames = read(src, ":")
    no_energy = with_dipole = 0
    for atoms in frames:
        if atoms.info.get("config_energy_weight", 1.0) == 0.0:
            atoms.info.pop("REF_energy", None)
            no_energy += 1
        if "REF_dipole" in atoms.info:
            atoms.info["config_dipole_weight"] = DIPOLE_WEIGHT
            with_dipole += 1
    write(dst, frames, format="extxyz")
    print(f"{len(frames)} frames: energy removed from {no_energy}, dipole weight on {with_dipole} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:])
