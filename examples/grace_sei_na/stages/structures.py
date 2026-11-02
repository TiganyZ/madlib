"""Starting cell of one SEI system ($SYSTEM) for cation $CATION (Na or Li), packed from rigid units, plus its crystals.

    SYSTEM=<name> CATION=<Na|Li> python structures.py OUTDIR SEED

Units are placed at random poses, then pushed apart (rigid translations and
rotations) until no two atoms of different units are closer than TOL.
"""

import os
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

UNITS_FILE = Path(__file__).resolve().parents[1] / "data" / "molecules.xyz"
ORGANIC_UNITS = {"Na": {"EDC": "NaEDC", "MC": "NaMC"}, "Li": {"EDC": "LEDC", "MC": "LMC"}}
CRYSTALS = {  # rocksalt MF and fluorite M2O lattice constants, A
    "Na": {"F": 4.63, "2O": 5.55},
    "Li": {"F": 4.03, "2O": 4.61},
}
# Phases are named without the cation (M): "F" is MF, "2CO3" is M2CO3, "EDC" the M ethylene dicarbonate.
# bulk: cubic box at a packing density (NPT compresses); slab: region beside a graphite slab, NVT
SYSTEMS = {
    "inorganic": {"phases": {"F": 32, "2CO3": 12, "2O": 10}, "density": 1.6, "crystals": ["F", "2O"]},
    "organic": {"phases": {"EDC": 8, "MC": 4, "F": 8}, "density": 1.2, "crystals": []},
    "electrolyte": {"phases": {"EC": 7, "DMC": 6, "PF6": 3}, "density": 1.0, "crystals": []},
    "graphite_electrolyte": {"phases": {"EC": 6, "DMC": 6, "PF6": 1}, "width": 12.0,
                             "crystals": ["graphite"]},
    "graphite_sei": {"phases": {"F": 12, "2CO3": 6, "EDC": 4}, "width": 12.0, "crystals": []},
}
TOL = 2.0
GAP = 1.5
MAX_ITER = 20000
STEP = 0.3
MAX_SHIFT = 0.3
MAX_TURN = 0.15

CC = 1.42
CH = 1.09
INTERLAYER = 3.354
SLAB_CELLS_X = 2
SLAB_CELLS_Y = 4
SLAB_LAYERS = 4

STRAINS = (-0.03, 0.0, 0.03)
RATTLES = (0.05, 0.10)


def load_units():
    return {a.info["unit"]: a for a in read(UNITS_FILE, index=":")}


def phase_units(cation):
    ion = f"{cation}+"
    organic = ORGANIC_UNITS[cation]
    return {
        "F": {ion: 1, "F-": 1}, "2O": {ion: 2, "O2-": 1}, "2CO3": {ion: 2, "CO3_2-": 1},
        "PF6": {ion: 1, "PF6-": 1}, "EDC": {organic["EDC"]: 1}, "MC": {organic["MC"]: 1},
        "EC": {"EC": 1}, "DMC": {"DMC": 1},
    }


def unit_names(phases, cation):
    table = phase_units(cation)
    names = []
    for phase, n in phases.items():
        for unit, k in table[phase].items():
            names += [unit] * (n * k)
    return names


def graphite_layers(n_x, n_y, n_layers, terminate):
    a = np.sqrt(3) * CC
    basis = np.array([[0, 0], [CC / 2, a / 2], [3 * CC / 2, a / 2], [2 * CC, 0]])
    cells = np.array([[3 * CC * i, a * j] for i in range(n_x) for j in range(n_y)])
    xy = (basis[None] + cells[:, None]).reshape(-1, 2)
    positions = []
    for k in range(n_layers):
        shifted = xy + [CC if k % 2 else 0.0, 0.0]
        positions.append(np.column_stack([shifted, np.full(len(xy), k * INTERLAYER)]))
    positions = np.vstack(positions)
    length_x = 3 * CC * n_x if not terminate else 3 * CC * n_x + 20.0
    atoms = Atoms(f"C{len(positions)}", positions=positions,
                  cell=[length_x, n_y * a, n_layers * INTERLAYER], pbc=True)
    if terminate:
        atoms = add_edge_hydrogens(atoms)
        atoms.positions[:, 0] -= atoms.positions[:, 0].min()
    atoms.positions[:, 1:] += [0.25 * a, 0.5 * INTERLAYER]
    return atoms


def add_edge_hydrogens(slab):
    i, vec = neighbor_list("iD", slab, 1.6)
    hydrogens = []
    for atom in range(len(slab)):
        bonds = vec[i == atom]
        if len(bonds) == 2:
            missing = -bonds.sum(axis=0)
            hydrogens.append(slab.positions[atom] + CH * missing / np.linalg.norm(missing))
    return slab + Atoms(f"H{len(hydrogens)}", positions=hydrogens)


def random_poses(names, units, x0, x1, cell, rng):
    poses = []
    for name in names:
        radius = np.linalg.norm(units[name].positions, axis=1).max()
        lo, hi = x0 + min(radius, (x1 - x0) / 2), x1 - min(radius, (x1 - x0) / 2)
        com = np.array([rng.uniform(lo, hi), rng.uniform(0, cell[1]), rng.uniform(0, cell[2])])
        poses.append((name, com, Rotation.random(random_state=rng).as_matrix()))
    return poses


def clip_norm(v, cap):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, cap / np.maximum(norm, 1e-12))


def push_apart(fixed, poses, units, x0, x1, cell):
    """Rigid units out of each other's TOL; fixed atoms do not move. Returns symbols, positions."""
    cell = np.asarray(cell, dtype=float)
    names = [p[0] for p in poses]
    com = np.array([p[1] for p in poses])
    rot = np.array([p[2] for p in poses])
    body = np.vstack([units[n].positions for n in names])
    sizes = np.array([len(units[n]) for n in names])
    owner = np.repeat(np.arange(len(names)), sizes)
    n_fixed = len(fixed)
    tag = np.concatenate([np.full(n_fixed, -1), owner])
    for it in range(MAX_ITER):
        pos = com[owner] + np.einsum("nij,nj->ni", rot[owner], body)
        every = np.vstack([fixed.positions, pos])
        pairs = cKDTree(every % cell, boxsize=cell).query_pairs(TOL, output_type="ndarray")
        pairs = pairs[tag[pairs[:, 0]] != tag[pairs[:, 1]]]
        d = every[pairs[:, 1]] - every[pairs[:, 0]]
        d -= cell * np.round(d / cell)
        dist = np.linalg.norm(d, axis=1)
        x = pos[:, 0]
        worst = max((TOL - dist).max(initial=0.0), (x0 - x).max(), (x - x1).max())
        if worst <= 1e-3:
            print(f"packed in {it} steps")
            symbols = [s for n in names for s in units[n].get_chemical_symbols()]
            return symbols, pos
        push = ((TOL - dist) / np.maximum(dist, 1e-6))[:, None] * d
        force = np.zeros((len(every), 3))
        np.add.at(force, pairs[:, 0], -push)
        np.add.at(force, pairs[:, 1], push)
        force = force[n_fixed:]
        force[:, 0] += np.clip(x0 - x, 0, None) - np.clip(x - x1, 0, None)
        arm = pos - com[owner]
        total = np.zeros((len(names), 3))
        torque = np.zeros((len(names), 3))
        inertia = np.zeros(len(names))
        np.add.at(total, owner, force)
        np.add.at(torque, owner, np.cross(arm, force))
        np.add.at(inertia, owner, (arm ** 2).sum(axis=1))
        com += clip_norm(STEP * total / sizes[:, None], MAX_SHIFT)
        rot = Rotation.from_rotvec(clip_norm(STEP * torque / (inertia + 1.0)[:, None], MAX_TURN)).as_matrix() @ rot
    raise RuntimeError(f"not packed after {MAX_ITER} steps, worst overlap {worst:.3f} A")


def bulk_cell(spec, units, cation, rng):
    names = unit_names(spec["phases"], cation)
    mass = sum(units[n].get_masses().sum() for n in names)
    length = (mass * 1.66054 / spec["density"]) ** (1 / 3)
    cell = [length] * 3
    poses = random_poses(names, units, 0.0, length, cell, rng)
    symbols, pos = push_apart(Atoms(), poses, units, -np.inf, np.inf, cell)
    return Atoms(symbols, positions=pos, cell=cell, pbc=True)


def slab_cell(spec, units, cation, rng):
    slab = graphite_layers(SLAB_CELLS_X, SLAB_CELLS_Y, SLAB_LAYERS, terminate=True)
    x0 = slab.positions[:, 0].max() + GAP
    x1 = x0 + spec["width"]
    cell = [x1 + GAP, slab.cell[1, 1], slab.cell[2, 2]]
    slab.cell = cell
    names = unit_names(spec["phases"], cation)
    symbols, pos = push_apart(slab, random_poses(names, units, x0, x1, cell, rng), units, x0, x1, cell)
    return slab + Atoms(symbols, positions=pos, cell=cell, pbc=True)


def crystal(name, cation):
    if name == "F":
        return bulk(f"{cation}F", "rocksalt", a=CRYSTALS[cation]["F"], cubic=True).repeat(2)
    if name == "2O":
        # antifluorite: O on the fcc site; bulk("Na2O", "fluorite") puts Na there (Na-Na 2.40 A, not 2.77)
        return bulk(f"O{cation}2", "fluorite", a=CRYSTALS[cation]["2O"], cubic=True).repeat(2)
    if name == "graphite":
        return graphite_layers(2, 3, 2, terminate=False)
    raise ValueError(name)


def variants(base, name, rng):
    out = []
    for s in STRAINS:
        a = base.copy()
        a.set_cell(base.cell * (1 + s), scale_atoms=True)
        a.info["config_type"] = f"{name}_strain{s:+.2f}"
        out.append(a)
    for r in RATTLES:
        a = base.copy()
        a.positions += rng.normal(scale=r, size=a.positions.shape)
        a.info["config_type"] = f"{name}_rattle{r:.2f}"
        out.append(a)
    return out


def main(outdir, seed, system, cation, species):
    rng = np.random.default_rng(seed)
    spec = SYSTEMS[system]
    units = load_units()
    if "width" in spec:
        atoms = slab_cell(spec, units, cation, rng)
    else:
        atoms = bulk_cell(spec, units, cation, rng)
    atoms.wrap()
    outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / "mixture.data", atoms, format="lammps-data", specorder=species, masses=True)
    crystals = []
    for name in spec["crystals"]:
        label = name if name == "graphite" else f"{cation}{name}"
        crystals += variants(crystal(name, cation), label, rng)
    if crystals:
        write(outdir / "crystals.xyz", crystals)
    print(f"{system}: {atoms.get_chemical_formula()} {len(atoms)} atoms, "
          f"cell {np.round(atoms.cell.lengths(), 2).tolist()}; {len(crystals)} crystal cells")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), os.environ["SYSTEM"], os.environ.get("CATION", "Na"),
         os.environ["SPECIES"].split())
