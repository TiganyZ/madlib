"""Farthest-point selection of MAD frames in PolarMACE descriptor space.

    python select_fps.py POOL.pkl.gz MODEL N SEED OUT.pkl.gz

A frame is the mean invariant descriptor of each element, concatenated, so two frames
are far apart when some element's environments differ.
"""

import sys

import numpy as np
import pandas as pd
from mace.calculators import mace_polar
from mace.modules.utils import extract_invariant
from mace.tools import torch_tools

# After mace, which lets e3nn's constants load under torch's weights_only default.
from e3nn import o3  # noqa: E402


def invariant_descriptors(calc, atoms):
    """Per-atom invariant node features; MACECalculator.get_descriptors refuses PolarMACE."""
    model = calc.models[0]
    batch = calc._atoms_to_batch(atoms)
    with torch_tools.default_dtype(calc.default_dtype):
        feats = model(batch.to_dict(), compute_stress=False)["node_feats"]
    irreps = o3.Irreps(str(model.products[0].linear.irreps_out))
    n_invariant = irreps.dim // (irreps.lmax + 1) ** 2
    d = extract_invariant(feats, num_layers=int(model.num_interactions), num_features=n_invariant,
                          l_max=irreps.lmax)
    return d.detach().cpu().numpy()


def frame_vector(calc, atoms, elements):
    atoms = atoms.copy()
    atoms.info.update(charge=0, spin=1)
    d = invariant_descriptors(calc, atoms)
    blocks = [d[atoms.numbers == z].mean(axis=0) if (atoms.numbers == z).any() else np.zeros(d.shape[1])
              for z in elements]
    return np.concatenate(blocks)


def farthest_points(x, n, rng):
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    chosen = [int(rng.integers(len(x)))]
    dist = np.linalg.norm(x - x[chosen[0]], axis=1)
    while len(chosen) < min(n, len(x)):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(x - x[i], axis=1))
    return sorted(chosen)


def main(pool, model, n, seed, out):
    df = pd.read_pickle(pool, compression="gzip")
    calc = mace_polar(model=model, device="cuda", default_dtype="float64")
    elements = sorted({z for a in df["ase_atoms"] for z in a.numbers})
    x = np.array([frame_vector(calc, a, elements) for a in df["ase_atoms"]])
    keep = farthest_points(x, int(n), np.random.default_rng(int(seed)))
    df.iloc[keep].reset_index(drop=True).to_pickle(out, compression="gzip")
    print(f"{len(keep)} of {len(df)} frames -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
