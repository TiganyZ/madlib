"""The fine-tuned PolarMACE plus D3(BJ), as the GRACE runs use it (15 A, CN 10 A)."""

from ase.calculators.mixing import SumCalculator
from mace.calculators import mace_polar

D3_CUTOFF = 15.0
D3_CN_CUTOFF = 10.0


def polar(model, device="cuda", dtype="float64"):
    return mace_polar(model=str(model), device=device, default_dtype=dtype)


def polar_d3(model, device="cuda", dtype="float64"):
    import torch  # noqa: PLC0415
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator  # noqa: PLC0415

    d3 = TorchDFTD3Calculator(device=device, damping="bj", xc="pbe", cutoff=D3_CUTOFF,
                              cnthr=D3_CN_CUTOFF,
                              dtype=torch.float64 if dtype == "float64" else torch.float32)
    return SumCalculator([polar(model, device, dtype), d3])
