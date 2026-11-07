#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A tiny MACE with random weights, so the bridge to MACE can be tested.

    make_mace_toy.py _mace_toy.model [layers]

Random weights are not a handicap here: what has to be proved is that the
geometry MADLIB hands over reproduces the one MACE's own data pipeline would
build -- the edge direction above all, which is silently wrong rather than
loudly wrong -- and an untrained model separates the two just as well as a
fitted one, in a second rather than a GPU-hour. Two layers over a 2 A graph, so the ghost
shell a test needs is 4 A -- and one layer, which is the control that says
what a difference between the two geometries means.
"""
import sys

import numpy as np
import torch
from e3nn import o3
from mace.modules import MACE, RealAgnosticResidualInteractionBlock

R_MAX = 2.0
LAYERS = 2
NUMBERS = [6, 8]


def build(layers):
    torch.manual_seed(20261107)
    return MACE(
        r_max=R_MAX,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=layers,
        num_elements=len(NUMBERS),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.array([-1.5, -2.5]),
        avg_num_neighbors=4.0,
        atomic_numbers=NUMBERS,
        correlation=2,
        gate=torch.nn.functional.silu,
    ).double()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "_mace_toy.model"
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else LAYERS
    torch.save(build(layers), out)
    print(f"{out}: r_max {R_MAX}, {layers} layer(s), Z {NUMBERS}, float64")


if __name__ == "__main__":
    main()
