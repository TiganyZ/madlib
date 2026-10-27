"""The MACE calculators of this workflow, and D3(BJ) as the GRACE runs use it (15 A, CN 10 A).

A model name is a PolarMACE (polar-1-m, or a fine-tuned .model file), or a mace_mp
foundation with its head after a colon (mh-1:omat_pbe).
"""

from ase.calculators.mixing import SumCalculator
from mace.calculators import mace_mp, mace_polar

D3_CUTOFF = 15.0
D3_CN_CUTOFF = 10.0


def polar(model, device="cuda", dtype="float64"):
    return mace_polar(model=str(model), device=device, default_dtype=dtype)


def foundation(model, device="cuda", dtype="float64"):
    """PolarMACE, or a multi-head mace_mp model named as name:head."""
    model = str(model)
    if ":" in model:
        name, head = model.split(":", 1)
        return mace_mp(model=name, head=head, device=device, default_dtype=dtype)
    return polar(model, device, dtype)


def polar_d3(model, device="cuda", dtype="float64"):
    import torch  # noqa: PLC0415
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator  # noqa: PLC0415

    d3 = TorchDFTD3Calculator(device=device, damping="bj", xc="pbe", cutoff=D3_CUTOFF,
                              cnthr=D3_CN_CUTOFF,
                              dtype=torch.float64 if dtype == "float64" else torch.float32)
    return SumCalculator([foundation(model, device, dtype), d3])
