#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Per-atom MACE values from MACE's own pipeline, against the ones MADLIB read.

    mace_reference.py <dump> <data file> <model> [tolerance]

The reference runs the same model on the same cell through mace.data.AtomicData
-- MACE's neighbour list, its edge convention, its one-hot encoding -- so what
is compared is MADLIB's reconstruction of the graph. Flip the edge direction
and this is what notices.
"""

import sys

import numpy as np
import torch
from mace import data
from mace.tools import AtomicNumberTable, torch_geometric


def read_data(path):
    """Positions, types and the cell out of a LAMMPS data file."""
    lines = open(path).read().split("\n")
    box, tilt, atoms = {}, [0.0, 0.0, 0.0], []
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if len(w) >= 4 and w[2] in ("xlo", "ylo", "zlo"):
            box[w[2][0]] = (float(w[0]), float(w[1]))
        elif len(w) >= 6 and w[3] == "xy":
            tilt = [float(w[0]), float(w[1]), float(w[2])]
        elif lines[i].startswith("Atoms"):
            i += 2
            while i < len(lines) and lines[i].strip():
                w = lines[i].split()
                atoms.append((int(w[0]), int(w[1]), float(w[2]), float(w[3]), float(w[4])))
                i += 1
            break
        i += 1
    atoms.sort()
    cell = np.array([[box["x"][1] - box["x"][0], 0.0, 0.0],
                     [tilt[0], box["y"][1] - box["y"][0], 0.0],
                     [tilt[1], tilt[2], box["z"][1] - box["z"][0]]])
    return np.array([a[2:] for a in atoms]), np.array([a[1] for a in atoms]), cell


def read_dump(path):
    body = open(path).read().split("ITEM: ATOMS")[-1].split("\n")[1:]
    rows = sorted(([int(r.split()[0]), float(r.split()[1])] for r in body
                   if len(r.split()) == 2), key=lambda r: r[0])
    return np.array([r[1] for r in rows])


def main():
    dump, datafile, modelfile = sys.argv[1:4]
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-10

    # AtomicData builds its tensors at torch's default dtype, and the toy
    # model is float64; MACE's own scripts set this for the same reason
    torch.set_default_dtype(torch.float64)
    model = torch.load(modelfile, map_location="cpu", weights_only=False)
    zs = [int(z) for z in model.atomic_numbers]
    positions, types, cell = read_data(datafile)
    numbers = np.array([zs[t - 1] for t in types])

    # properties/property_weights are required by newer mace and ignored here
    config = data.Configuration(atomic_numbers=numbers, positions=positions, cell=cell,
                                pbc=(True, True, True), properties={}, property_weights={})
    table = AtomicNumberTable(zs)
    batch = next(iter(torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(config, z_table=table, cutoff=float(model.r_max))],
        batch_size=1, shuffle=False)))
    out = model(batch.to_dict(), training=False, compute_force=False)
    reference = out["node_energy"].detach().numpy()

    mine = read_dump(dump)
    err = np.abs(reference - mine).max()
    scale = max(np.abs(reference).max(), 1e-30)
    print(f"    {len(mine)} atoms, spread {reference.min():.6f} to {reference.max():.6f}: "
          f"max |MACE - MADLIB| = {err:.3e}")
    if not np.isfinite(err) or err > tol * scale:
        worst = int(np.argmax(np.abs(reference - mine)))
        print(f"    worst at atom {worst + 1}: {reference[worst]:.12g} vs {mine[worst]:.12g}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
