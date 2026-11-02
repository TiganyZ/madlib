"""A float64 copy of a MACE-POLAR-1 foundation.

    python foundation64.py polar-1-m OUT.model

mace_run_train (0.3.17) loads a named polar foundation as float32 and then fails
estimating E0s under --default_dtype=float64 ("both inputs should have same dtype");
a path to this copy avoids it.
"""

import sys

import torch
from mace.calculators import mace_polar


def main(name, out):
    model = mace_polar(model=name, device="cpu", return_raw_model=True).double()
    torch.save(model, out)
    print(f"{name} -> {out} (float64)")


if __name__ == "__main__":
    main(*sys.argv[1:])
