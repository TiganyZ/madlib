# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The NMR tensor models' output, rebuilt the way their authors rebuild it.

What the network returns is standardised per species and per irreducible
block, and turning it back into a Cartesian tensor is the one step of that
path which owes nothing to NequIP -- so it is checked here, against a literal
transcription of the authors' 3-transform-to-cartesian.py, with no model
loaded. The columns, the per-species statistics and the two CartesianTensor
transforms all have to agree for this to pass.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
e3nn_io = pytest.importorskip("e3nn.io")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from madlib_models.backends.nequip_nmr import (LAYOUT, Transforms,    # noqa: E402
                                               destandardise)

SI, O = 14, 8


def fake_stats(prefix, seed=11):
    rng = np.random.default_rng(seed)
    stats = {}
    for name in ("si", "o"):
        stats[f"{prefix}_{name}_mean"] = rng.normal(size=6) * 10.0
        stats[f"{prefix}_{name}_std"] = 1.0 + rng.uniform(size=6)
        stats[f"antisymm_{name}_mean"] = rng.normal(size=3)
        stats[f"antisymm_{name}_std"] = 1.0 + rng.uniform(size=3)
    return stats


def authors_version(magres, numbers, stats):
    """3-transform-to-cartesian.py, transcribed: columns, scaling, transforms."""
    ct_symm = e3nn_io.CartesianTensor("ij=ji")
    ct_antisymm = e3nn_io.CartesianTensor("ij=-ji")
    symm_cols = [0, 4, 5, 6, 7, 8]
    antisymm_cols = [1, 2, 3]

    m = magres.copy()
    for z, name in ((SI, "si"), (O, "o")):
        idx = numbers == z
        m[idx, 0] = m[idx, 0] * stats[f"symm_{name}_std"][0] + stats[f"symm_{name}_mean"][0]
        m[idx, 4:] = m[idx, 4:] * stats[f"symm_{name}_std"][1:] + stats[f"symm_{name}_mean"][1:]
        m[idx, 1:4] = (m[idx, 1:4] * stats[f"antisymm_{name}_std"]
                       + stats[f"antisymm_{name}_mean"])
    ms = ct_symm.to_cartesian(torch.from_numpy(m[:, symm_cols]))
    ms = ms + ct_antisymm.to_cartesian(torch.from_numpy(m[:, antisymm_cols]))
    return ms.numpy()


def test_shielding_tensor_matches_the_authors_transform():
    rng = np.random.default_rng(5)
    numbers = np.array([SI, O, O, SI, O])
    magres = rng.normal(size=(len(numbers), 9))
    stats = fake_stats("symm")

    symm, anti = destandardise(torch.from_numpy(magres), numbers, stats, LAYOUT["ms"])
    mine = Transforms().cartesian(symm, anti, LAYOUT["ms"]).numpy()

    # the authors' own transform is built at torch's default dtype, so it is
    # the single-precision one; this is as close as the two can come
    assert np.allclose(mine, authors_version(magres, numbers, stats), atol=1e-6)


def test_the_isotropic_shielding_is_the_stored_mean_when_the_model_says_zero():
    """A standardised output of zero is the training set's mean, not zero ppm."""
    numbers = np.array([SI, O])
    stats = fake_stats("symm")
    symm, anti = destandardise(torch.zeros(2, 9, dtype=torch.float64), numbers, stats,
                               LAYOUT["ms"])
    t = Transforms().cartesian(symm, anti, LAYOUT["ms"]).numpy()
    iso = np.trace(t, axis1=1, axis2=2) / 3.0
    # the l=0 component is the trace, carried by CartesianTensor's 1/sqrt(3)
    expected = [stats["symm_si_mean"][0] / np.sqrt(3.0), stats["symm_o_mean"][0] / np.sqrt(3.0)]
    assert np.allclose(iso, expected, rtol=1e-10)


def test_the_field_gradient_is_traceless_and_symmetric():
    rng = np.random.default_rng(6)
    numbers = np.array([SI, O, O])
    raw = torch.from_numpy(rng.normal(size=(3, 5)))
    stats = fake_stats("efgsymm")
    stats["efgsymm_si_std"][0] = 0.0      # as in the authors' file: no l=0 block
    stats["efgsymm_o_std"][0] = 0.0
    stats["efgsymm_si_mean"][0] = 0.0
    stats["efgsymm_o_mean"][0] = 0.0

    symm, anti = destandardise(raw, numbers, stats, LAYOUT["efg"])
    t = Transforms().cartesian(symm, anti, LAYOUT["efg"]).numpy()
    assert np.allclose(anti.numpy(), 0.0)
    assert np.allclose(np.trace(t, axis1=1, axis2=2), 0.0, atol=1e-12)
    assert np.allclose(t, np.transpose(t, (0, 2, 1)), atol=1e-12)


def test_a_width_the_parameter_does_not_have_is_an_error():
    with pytest.raises(ValueError, match="expects 9 output components"):
        destandardise(torch.zeros(2, 5, dtype=torch.float64), np.array([SI, O]),
                      fake_stats("symm"), LAYOUT["ms"])
