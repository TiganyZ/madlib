"""Neutral clusters of whole units cut from periodic frames, boxed for a VASP dipole/polarizability.

    python carve_clusters.py FRAMES.xyz OUT.xyz N_CLUSTERS [SEED]

Units are covalent fragments (cations excluded from bonding) with a formal charge
looked up by composition. A cluster is every unit near a random seed atom, then
counter-ions nearest that seed until the charge is zero. The seed's unit type is
drawn uniformly, so salts are not swamped by solvent. Clusters touching a unit of
unknown composition (graphite, a reacted fragment) are skipped.
"""

import sys
from collections import Counter

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CATIONS = {"Li": 1, "Na": 1}
# Hill formula -> formal charge
UNIT_CHARGE = {
    "F": -1, "O": -2, "CO3": -2, "F6P": -1,
    "C4H4O6": -2,    # ethylene dicarbonate
    "C2H3O3": -1,    # methyl carbonate
    "C3H4O3": 0,     # EC
    "C3H6O3": 0,     # DMC
}
BOND_SCALE = 1.15
RADIUS = 4.0          # A; units with any atom this close to the seed atom
MIN_ATOMS = 12
MAX_ATOMS = 60
VACUUM = 7.0          # A each side, so >= 14 A between images
MIN_CONTACT = 0.8     # A; a closer pair means the frame was unphysical


def units_of(atoms):
    """Unit index per atom; a cation is a unit of its own."""
    cation = np.array([s in CATIONS for s in atoms.get_chemical_symbols()])
    i, j = neighbor_list("ij", atoms, [c * BOND_SCALE for c in natural_cutoffs(atoms)])
    keep = ~cation[i] & ~cation[j]
    n = len(atoms)
    graph = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def unit_formulas(atoms, labels):
    symbols = np.array(atoms.get_chemical_symbols())
    return {u: Atoms(list(symbols[labels == u])).get_chemical_formula("hill") for u in np.unique(labels)}


def unit_charges(formulas):
    return {u: CATIONS.get(f, UNIT_CHARGE.get(f)) for u, f in formulas.items()}


def seed_atom(labels, formulas, charges, rng):
    known = sorted({f for u, f in formulas.items() if charges[u] is not None})
    formula = known[rng.integers(len(known))]
    units = [u for u, f in formulas.items() if f == formula]
    unit = units[rng.integers(len(units))]
    members = np.flatnonzero(labels == unit)
    return members[rng.integers(len(members))]


def unwrapped(atoms, center):
    """Positions as minimum-image vectors from `center`."""
    d = atoms.get_positions() - atoms.positions[center]
    frac = np.linalg.solve(atoms.cell.T, d.T).T
    frac -= np.round(frac) * atoms.pbc
    return frac @ atoms.cell


def pick_units(vectors, labels, charges):
    """Units within RADIUS of the seed, then counter-ions by distance until neutral."""
    dist = np.linalg.norm(vectors, axis=1)
    order = sorted(np.unique(labels), key=lambda u: dist[labels == u].min())
    chosen = [u for u in order if dist[labels == u].min() < RADIUS]
    if any(charges[u] is None for u in chosen):
        return None
    total = sum(charges[u] for u in chosen)
    for u in order:
        if total == 0:
            break
        if u in chosen or charges[u] is None or charges[u] == 0 or np.sign(charges[u]) == np.sign(total):
            continue
        if abs(total + charges[u]) < abs(total) or np.sign(total + charges[u]) != np.sign(total):
            chosen.append(u)
            total += charges[u]
    return chosen if total == 0 else None


def boxed(atoms, vectors, mask, name):
    cluster = Atoms(np.array(atoms.get_chemical_symbols())[mask], positions=vectors[mask])
    cluster.center(vacuum=VACUUM)
    cluster.pbc = False
    cluster.info.update(name=name, charge=0, spin=1)
    return cluster


def ok(cluster):
    if not MIN_ATOMS <= len(cluster) <= MAX_ATOMS:
        return False
    d = cluster.get_all_distances()
    return d[np.triu_indices(len(cluster), 1)].min() > MIN_CONTACT


def carve(frames, n_clusters, seed):
    rng = np.random.default_rng(seed)
    parsed = []
    for f in frames:
        labels = units_of(f)
        formulas = unit_formulas(f, labels)
        parsed.append((f, labels, formulas, unit_charges(formulas)))
    out, tries = [], 0
    while len(out) < n_clusters and tries < 50 * n_clusters:
        tries += 1
        k = rng.integers(len(parsed))
        atoms, labels, formulas, charges = parsed[k]
        center = seed_atom(labels, formulas, charges, rng)
        vectors = unwrapped(atoms, center)
        chosen = pick_units(vectors, labels, charges)
        if chosen is None:
            continue
        mask = np.isin(labels, chosen)
        name = f"{atoms.info.get('name', k)}_c{center}"
        cluster = boxed(atoms, vectors, mask, name)
        if ok(cluster):
            out.append(cluster)
    return out


def main(frames_path, out_path, n_clusters, seed="0"):
    frames = read(frames_path, ":")
    clusters = carve(frames, int(n_clusters), int(seed))
    write(out_path, clusters, format="extxyz")
    sizes = [len(c) for c in clusters]
    species = Counter(s for c in clusters for s in c.get_chemical_symbols())
    print(f"{len(clusters)} neutral clusters from {len(frames)} frames, "
          f"{min(sizes)}-{max(sizes)} atoms (mean {np.mean(sizes):.0f}); {dict(species)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
