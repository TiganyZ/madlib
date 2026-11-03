"""Checks that need no LAMMPS: deck text, log parsing, objectives, front."""

import sys
import numpy as np

from madlib_ezga import constraints, deck, features, front, objectives
from madlib_ezga.lammps_run import parse_thermo
from madlib_ezga.runner import _gamma_for

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def test_pdf_cutoffs():
    pdf = deck.Pdf(rmax=8.0, kde=0.1)
    check("pdf rcut", abs(pdf.rcut - 8.8) < 1e-12, pdf.rcut)
    check("pdf comm cutoff", abs(pdf.comm_cutoff - 9.3) < 1e-12, pdf.comm_cutoff)


def test_deck_text():
    text = deck.render("start.data", deck.Potential(), gamma=2.5)
    check("pe pair not pe", "compute pepair all pe pair" in text)
    check("atom_modify map", "atom_modify     map yes" in text)
    check("gamma value", "gamma 2.5" in text)
    check("comm cutoff covers pdf", "comm_modify     cutoff 9.3" in text)
    check("species given", "species C" in text)
    check("run 0 by default", text.rstrip().splitlines()[-1].startswith("write_dump")
          and "run 0" in text)

    ramped = deck.render("s.data", deck.Potential(), gamma=(0.0, 20.0))
    check("gamma ramp", "gamma 0 20" in ramped)

    zero = deck.render("s.data", deck.Potential(style="zero"),
                       moves=deck.rmc(300, 50, 11604.5, 7))
    check("rmc pair zero", "pair_style      zero" in zero)
    check("rmc bias priced", "fix_modify mad energy yes" in zero)
    check("rmc no exchanges", "gcmc 1 0 50 1 7" in zero)


def test_parse_thermo():
    log = """
Setting up Verlet run ...
   Step          Temp          PotEng       c_pepair        f_mad        f_mad[2]
         0   0            -2828.8        -2828.8        0            1.0145
       100   301.2        -3100.5        -3105.5        5.0          0.4210
Loop time of 1.0 on 1 procs
"""
    thermo = parse_thermo(log)
    check("thermo columns", set(thermo) >= {"Step", "c_pepair", "f_mad[2]"})
    check("thermo last pepair", abs(thermo["c_pepair"][-1] + 3105.5) < 1e-9)
    check("thermo last rel_error", abs(thermo["f_mad[2]"][-1] - 0.4210) < 1e-9)


class FakeManager:
    def __init__(self, n, energy, meta):
        self.atomPositions = np.zeros((n, 3))
        self.E = energy
        self.metadata = meta


class FakeStructure:
    def __init__(self, n, energy, meta):
        self.AtomPositionManager = FakeManager(n, energy, meta)


def test_objectives():
    data = [
        FakeStructure(4, -40.0, {"xrd_rel_error": 0.5}),
        FakeStructure(4, -32.0, {"xrd_rel_error": 0.2}),
        FakeStructure(4, -36.0, {}),
    ]
    misfit = objectives.objective_xrd_rel_error()(data)
    check("misfit shape", misfit.shape == (3,), misfit.shape)
    check("misfit values", np.allclose(misfit[:2], [0.5, 0.2]))
    check("missing is not nan", np.isinf(misfit[2]) and not np.isnan(misfit[2]))

    name = objectives.objective_xrd_rel_error.__name__
    globals_kw = ["hull", "formation", "novelty", "similarity", "anomality",
                  "fit", "regression"]
    combined = f"{name} {objectives.__name__}"
    check("name stays local to ezga",
          not any(k in combined for k in globals_kw), combined)


def test_front():
    # energy, misfit; lower is better on both
    points = np.array([
        [-7.5, 0.40],   # dominated by neither -> on the front
        [-7.0, 0.20],   # on the front
        [-6.9, 0.45],   # dominated by both above
        [-7.5, 0.40],   # duplicate of a front point
    ])
    keep = front.non_dominated(points)
    check("front excludes dominated", 2 not in keep, keep)
    check("front keeps the ends", 1 in keep, keep)
    check("duplicates do not erase each other", len(keep) >= 2, keep)


def test_coordination():
    """Against structures whose answer is known by construction."""
    from ase.build import bulk
    from ase.lattice.hexagonal import Graphite

    diamond = bulk("C", "diamond", a=3.567, cubic=True).repeat((2, 2, 2))
    graphite = Graphite(symbol="C", latticeconstant={"a": 2.461, "c": 6.708}).repeat((3, 3, 2))

    cn_d = features._coordination(diamond.get_positions(), np.array(diamond.get_cell()), 1.85)
    cn_g = features._coordination(graphite.get_positions(), np.array(graphite.get_cell()), 1.85)
    check("diamond is 4-coordinate", abs(cn_d - 4.0) < 1e-9, cn_d)
    check("graphite is 3-coordinate", abs(cn_g - 3.0) < 1e-9, cn_g)

    # minimum image: a bond across the boundary must still be counted
    import ase
    pair = ase.Atoms("C2", positions=[[0.1, 0, 0], [9.9, 0, 0]],
                     cell=np.eye(3) * 10.0, pbc=True)
    cn_p = features._coordination(pair.get_positions(), np.array(pair.get_cell()), 1.85)
    check("bond across the boundary counts", abs(cn_p - 1.0) < 1e-9, cn_p)


def test_random_subset():
    """A rattle of every atom cannot pass a minimum-distance gate in a dense
    cell, so the operators move a subset. The subset has to be re-drawn per
    call, or every child rattles the same atoms for the whole run."""
    n = 4000
    pick = constraints.random_subset(0.02, seed=0)
    first = [pick(i, None) for i in range(n)]
    second = [pick(i, None) for i in range(n)]
    share = sum(first) / n
    check("subset is about the asked fraction", abs(share - 0.02) < 0.005, share)
    check("subset is re-drawn per call", first != second)
    check("subset is a bool", all(isinstance(v, bool) for v in first[:50]))

    none_pass = constraints.random_subset(0.0, seed=0)
    check("zero fraction picks nothing", not any(none_pass(i, None) for i in range(100)))


def test_gamma_for():
    """Siblings share a gamma; without that a family is split across the ladder
    by arrival order, which undoes in the child what the parent gained."""
    ladder = (1.0, 3.0, 10.0, 30.0, 100.0)
    fallback = iter([-1.0] * 10)

    a = _gamma_for("hash-a", ladder, fallback)
    check("same parent, same gamma", _gamma_for("hash-a", ladder, fallback) == a)
    check("gamma is on the ladder", a in ladder, a)

    seen = {_gamma_for(f"hash-{i}", ladder, fallback) for i in range(200)}
    check("different parents span the ladder", seen == set(ladder), sorted(seen))

    # No parent means no family to agree with, so the ladder is dealt in turn.
    check("no parent falls back", _gamma_for(None, ladder, iter([7.0])) == 7.0)


for test in [test_pdf_cutoffs, test_deck_text, test_parse_thermo,
             test_objectives, test_front, test_coordination,
             test_random_subset, test_gamma_for]:
    print(test.__name__)
    test()

print()
if FAILURES:
    print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("all passed")
