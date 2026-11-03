#!/usr/bin/env python3
"""Build the GA's starting population from the same structure the others use.

MAD and HRMC both start from start.data. Seeding the GA from another method's
finished results -- what a production search would do -- makes the comparison
meaningless: the GA's front then contains that method's structures verbatim,
and its best fit can be one of them without the GA improving on anything.

Diversity comes from isotropic strain, not from rattling. A quenched cell sits
close to the collision threshold already (1.24 A against 1.14 A for carbon at
collision_factor 0.75), so a rattle big enough to diversify puts every member
in collision and the GA rejects the whole population before evaluating it.
Strain scales every distance together, and the density spread it produces is
itself one of the features selection keeps individuals apart in.

Every frame is checked against --min-distance; the script fails rather than
writing a population the GA will silently throw away.
"""

import argparse

import numpy as np
from ase.io import read, write


def min_distance(atoms):
    """Closest pair under the minimum image convention, cubic cells only."""
    pos = atoms.get_positions()
    lengths = atoms.cell.lengths()
    delta = pos[:, None, :] - pos[None, :, :]
    delta -= np.round(delta / lengths) * lengths
    r = np.sqrt((delta ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    return r.min()


def population(start, size, rattle, strain, seed):
    rng = np.random.default_rng(seed)
    base = read(start, format="lammps-data", style="atomic")
    # Evenly spaced rather than random, so the density axis is covered rather
    # than clumped at whatever the draws happened to give.
    factors = np.linspace(1.0 - strain, 1.0 + strain, size)
    out = []
    for factor in factors:
        atoms = base.copy()
        atoms.set_cell(atoms.cell * factor, scale_atoms=True)
        atoms.positions += rng.normal(scale=rattle, size=atoms.positions.shape)
        out.append(atoms)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--rattle", type=float, default=0.015, help="Angstrom, per component")
    ap.add_argument("--strain", type=float, default=0.04, help="isotropic, +/- this")
    ap.add_argument("--min-distance", type=float, default=1.17,
                    help="floor; EZGA's collision check is 0.75 * 1.52 = 1.14 for carbon")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = population(args.start, args.size, args.rattle, args.strain, args.seed)
    distances = [min_distance(a) for a in frames]
    worst = min(distances)
    if worst < args.min_distance:
        bad = sum(1 for d in distances if d < args.min_distance)
        raise SystemExit(
            f"{bad}/{len(frames)} frames closer than {args.min_distance} A "
            f"(worst {worst:.3f}); lower --rattle or --strain")

    write(args.out, frames, format="extxyz")
    volumes = np.array([a.get_volume() for a in frames])
    mass = sum(frames[0].get_masses()) / 0.6022140761
    print(f"{len(frames)} structures of {len(frames[0])} atoms -> {args.out}")
    print(f"  min distance {worst:.3f} - {max(distances):.3f} A (floor {args.min_distance})")
    print(f"  density      {mass / volumes.max():.3f} - {mass / volumes.min():.3f} g/cm3")


main()
