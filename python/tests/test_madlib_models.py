# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The property bridge without LAMMPS: reductions, and the adjoint by hand.

The adjoint is the one thing here that can be wrong and still look right, so
it is checked against central differences of the weighted sum it is supposed
to be the gradient of -- and by the order of convergence, not a tolerance.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from madlib_models import binding, reduce    # noqa: E402


def random_frame(n=12, box=6.0, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, 3))
    types = np.where(rng.uniform(size=n) > 0.5, 1, 2).astype(np.int32)
    return pos, types, box


def all_pairs(pos, rcut):
    """A full edge list of (centre, neighbour), as the compute would send."""
    ei, ej = [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < rcut:
                ei.append(i)
                ej.append(j)
    return np.array(ei, dtype=np.int32), np.array(ej, dtype=np.int32)


def bind(quantity, ncomp):
    return binding.mad_bind(f"backend=callable fn=mad_tensor_toy:tensor types=C,O "
                            f"output=value quantity={quantity}")


def test_voigt_round_trip():
    t = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]])
    v = reduce.apply("tensor6", t)
    assert np.allclose(v.numpy(), [[1, 4, 6, 2, 3, 5]])
    assert np.allclose(reduce.apply("iso", t).numpy(), [(1 + 4 + 6) / 3])


def test_antisymmetric_part_is_dropped():
    t = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    assert np.allclose(reduce.apply("tensor6", t).numpy(), 0.0)


def test_span_and_quadrupolar():
    t = torch.diag(torch.tensor([1.0, 2.0, 7.0]))[None]
    assert np.allclose(reduce.apply("span", t).numpy(), 6.0)
    # traceless: (-7/3, -4/3, 11/3), ordered by magnitude into
    # Vxx = -4/3, Vyy = -7/3, Vzz = 11/3, so eta = (Vxx - Vyy)/Vzz = 3/11
    assert np.allclose(reduce.apply("vzz", t).numpy(), 11.0 / 3.0)
    assert np.allclose(reduce.apply("etaq", t).numpy(), 3.0 / 11.0)


def test_width_matches_the_compute():
    compute, _ = bind("tensor6", 6)
    pos, types, _ = random_frame()
    ei, ej = all_pairs(pos, 3.0)
    cell = np.zeros((3, 3))
    with pytest.raises(ValueError, match="ncomp"):
        compute(pos.tobytes(), types.tobytes(), ei.tobytes(), ej.tobytes(), len(pos), 1,
                cell.tobytes())


@pytest.mark.parametrize("quantity,ncomp", [("tensor6", 6), ("iso", 1), ("span", 1)])
def test_adjoint_against_central_differences(quantity, ncomp):
    compute, adjoint = bind(quantity, ncomp)
    pos, types, _ = random_frame()
    n = len(pos)
    cell = np.zeros((3, 3))
    rng = np.random.default_rng(7)
    lam = rng.normal(size=(n, ncomp))

    def weighted(p):
        ei, ej = all_pairs(p, 3.0)
        q = compute(p.tobytes(), types.tobytes(), ei.tobytes(), ej.tobytes(), n, ncomp,
                    cell.tobytes())
        return float((q.reshape(n, ncomp) * lam).sum())

    ei, ej = all_pairs(pos, 3.0)
    f = adjoint(pos.tobytes(), types.tobytes(), ei.tobytes(), ej.tobytes(), n, ncomp,
                lam.tobytes(), cell.tobytes()).reshape(n, 3)

    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        worst = 0.0
        for i in (0, n // 2, n - 1):
            for d in range(3):
                plus, minus = pos.copy(), pos.copy()
                plus[i, d] += h
                minus[i, d] -= h
                fd = -(weighted(plus) - weighted(minus)) / (2 * h)
                worst = max(worst, abs(fd - f[i, d]))
        errors.append(worst)

    # second order: a factor of 10 in h is ~100 in the error, and a wrong
    # constant factor would give a flat curve instead
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > 20.0
    assert errors[2] < 1e-6 * max(1.0, abs(f).max())
