# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The anisotropic NMR parameter models of Ben Mahmoud et al., as a property.

    Graph-neural-network predictions of solid-state NMR parameters from
    spherical tensor decomposition -- C. Ben Mahmoud, L. A. M. Rosset,
    J. R. Yates, V. L. Deringer, arXiv:2412.15063
    data   https://github.com/cbenmahm/anistropic-nmr-parameters-data
    code   https://github.com/cbenmahm/nequip-nmr-parameters

Two models, both NequIP networks that predict a rank-2 tensor per atom through
its irreducible spherical components rather than its nine Cartesian ones:

    parameter=ms    the magnetic shielding tensor, irreps 0e+1e+2e, in ppm
    parameter=efg   the electric field gradient, irreps 2e, traceless

    compute sig all mad/property/python madlib_models ncomp 6 pbc yes &
            config "backend=nequip_nmr parameter=ms quantity=tensor6 &
                    model=models/MS/final/isd/minimal types=Si,O &
                    stats=data/data_mean_std.json"

WHAT COMES OUT OF THE NETWORK IS STANDARDISED, and nothing in the file says
so. The training targets were centred and scaled per species and per
irreducible block, so the raw output is in units of that block's standard
deviation about its mean, and a shielding read straight from the model is a
number near zero rather than near 350 ppm. The inverse is applied here with
the authors' own data_mean_std.json, exactly as their
3-transform-to-cartesian.py does: component 0 and components 4:9 are the
symmetric block in the order (l=0, l=2), components 1:4 the antisymmetric one,
and `stats=` is not optional.

For the EFG the l=0 block is absent from the model and its stored standard
deviation is zero -- an electric field gradient is traceless by construction,
and the tensor is rebuilt that way.

ONE PERIODIC CELL, ONE RANK. Six message-passing layers over a 5.533 A graph
read 33 A around every atom, so the ghost-cloud route would need a 33 A ghost
shell -- hundreds of images of a cell this size. 'pbc yes' is therefore
required, not merely preferred, and the graph is built by NequIP's own
AtomicData so that the edge convention is the one the weights were trained
with rather than one reconstructed here.

SHIELDING, NOT SHIFT. The MS model predicts an absolute shielding. fix mad/nmr
and fix mad/nmr/aniso are what convert it to a shift, and both refuse to run
until told which they have been given.
"""

import json

import numpy as np
import torch
from e3nn import io

from ..elements import SYMBOLS
from .base import Backend, pick_device, pick_dtype

KEYS = {"parameter", "stats", "rcut", "model_name", "model_key"}
DEFAULT_OUTPUT = "tensor"
DEFAULT_QUANTITY = None    # 'iso' for one model is meaningless for the other

# the model's output columns, by parameter: (symmetric l=0, antisymmetric l=1,
# symmetric l=2) and the prefix its statistics are stored under
LAYOUT = {
    "ms": {"width": 9, "l0": [0], "l1": [1, 2, 3], "l2": [4, 5, 6, 7, 8], "prefix": "symm"},
    "efg": {"width": 5, "l0": [], "l1": [], "l2": [0, 1, 2, 3, 4], "prefix": "efgsymm"},
}


def destandardise(raw, numbers, stats, layout, dtype=None):
    """Undo the per-species centring and scaling of each irreducible block.

    Returns the symmetric block as (n, 6) in the order (l=0, l=2) that
    e3nn's CartesianTensor("ij=ji") expects, and the antisymmetric one as
    (n, 3). For a traceless parameter the l=0 entry is left at zero.
    """
    if raw.shape[1] != layout["width"]:
        raise ValueError(f"this parameter expects {layout['width']} output components per atom, "
                         f"but the model gave {raw.shape[1]}")
    dtype = dtype or raw.dtype

    def stat(key):
        if key not in stats:
            raise ValueError(f"the statistics file has no '{key}'; it has {sorted(stats)}")
        return torch.as_tensor(stats[key], dtype=dtype, device=raw.device)

    symm = raw.new_zeros(raw.shape[0], 6)
    anti = raw.new_zeros(raw.shape[0], 3)
    for z in sorted(set(int(v) for v in numbers)):
        name = SYMBOLS[z - 1].lower()
        idx = torch.as_tensor(np.flatnonzero(numbers == z), dtype=torch.long, device=raw.device)
        block = raw.index_select(0, idx)
        mean, std = stat(f"{layout['prefix']}_{name}_mean"), stat(f"{layout['prefix']}_{name}_std")
        if layout["l0"]:
            symm[idx, 0] = block[:, layout["l0"][0]] * std[0] + mean[0]
        symm[idx, 1:] = block[:, layout["l2"]] * std[1:] + mean[1:]
        if layout["l1"]:
            anti[idx] = (block[:, layout["l1"]] * stat(f"antisymm_{name}_std")
                         + stat(f"antisymm_{name}_mean"))
    return symm, anti


class Transforms:
    """e3nn's Cartesian transforms, built once and in double precision.

    Two reasons not to let CartesianTensor.to_cartesian() build its own. The
    change of basis is computed in whatever torch's default dtype happens to
    be and only cast afterwards, so under the float32 default every tensor
    that passes through loses the precision a float64 model was run at -- and
    it is a whole e3nn module, rebuilt on every call, which is an odd thing to
    pay for once per timestep.
    """

    def __init__(self, dtype=torch.float64, device="cpu"):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            self.symm = io.CartesianTensor("ij=ji")
            self.antisymm = io.CartesianTensor("ij=-ji")
            rtp_symm = self.symm.reduced_tensor_products()
            rtp_antisymm = self.antisymm.reduced_tensor_products()
        finally:
            torch.set_default_dtype(previous)
        self.rtp_symm = rtp_symm.to(dtype=dtype, device=device)
        self.rtp_antisymm = rtp_antisymm.to(dtype=dtype, device=device)

    def cartesian(self, symm, anti, layout):
        """The 3x3 tensor from its irreducible blocks."""
        tensor = self.symm.to_cartesian(symm, self.rtp_symm)
        if layout["l1"]:
            tensor = tensor + self.antisymm.to_cartesian(anti, self.rtp_antisymm)
        return tensor



def identity(cfg):
    return (cfg.get("model", ""), cfg.get("model_name", ""), cfg.get("parameter", "ms"),
            cfg.get("stats", ""), cfg.get("device", ""), cfg.get("dtype", ""),
            cfg.get("types", ""))


def build(cfg):
    for key in ("model", "stats"):
        if key not in cfg:
            raise ValueError(f"backend=nequip_nmr needs {key}=<path>")
    parameter = cfg.get("parameter", "ms")
    if parameter not in LAYOUT:
        raise ValueError(f"parameter must be one of {sorted(LAYOUT)}")
    return _NequipNmr(cfg, parameter)


def _load(path, name, device):
    """A NequIP training directory, or a deployed model file."""
    import os
    # Without this a mistyped path falls through to the deployed-model branch
    # and fails somewhere inside NequIP instead, saying nothing about the path.
    if not os.path.exists(path):
        raise ValueError(f"no model at '{path}'; it should be a NequIP training directory "
                         f"holding best_model.pth and config.yaml, or a deployed .pth")
    if os.path.isdir(path):
        from nequip.train import Trainer
        model, config = Trainer.load_model_from_training_session(
            traindir=path, model_name=name or "best_model.pth", device=str(device))
        return model, dict(config)
    from nequip.scripts.deploy import load_deployed_model
    model, metadata = load_deployed_model(path, device=str(device), set_global_options=False)
    return model, metadata


class _NequipNmr(Backend):
    builds_own_graph = True

    def __init__(self, cfg, parameter):
        device = pick_device(cfg)
        self.model, self.config = _load(cfg["model"], cfg.get("model_name"), device)
        self.model.eval()
        self.parameter = parameter
        self.layout = LAYOUT[parameter]
        self.model_key = cfg.get("model_key", "magres")
        self.symbols = list(self.config["chemical_symbols"])
        r_max = float(self.config["r_max"])
        super().__init__(cfg, pick_dtype(cfg, torch.float64), device, rcut=r_max)

        with open(cfg["stats"]) as fh:
            self.stats = {k: np.asarray(v, dtype=np.float64) for k, v in json.load(fh).items()}

        from nequip.data.transforms import TypeMapper
        self.transforms = Transforms(dtype=self.dtype, device=self.device)
        self.mapper = TypeMapper(chemical_symbols=self.symbols)

    def _typemapped(self, graph):
        import ase
        from nequip.data import AtomicData, AtomicDataDict

        atoms = ase.Atoms(numbers=graph.numbers,
                          positions=graph.pos.detach().cpu().numpy(),
                          cell=graph.cell.detach().cpu().numpy(), pbc=True)
        data = AtomicData.to_AtomicDataDict(
            self.mapper(AtomicData.from_ase(atoms, r_max=self.rcut)))
        data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in data.items()}
        # the model's own graph, but OUR positions and cell, so the gradient
        # has somewhere to flow
        data[AtomicDataDict.POSITIONS_KEY] = graph.pos
        data[AtomicDataDict.CELL_KEY] = graph.cell
        return data

    def _destandardise(self, raw, numbers):
        return destandardise(raw, numbers, self.stats, self.layout, self.dtype)

    def forward(self, graph):
        if not graph.periodic:
            raise ValueError(
                "backend=nequip_nmr needs 'pbc yes' on the compute: the model passes "
                f"{self.config.get('num_layers', '?')} messages over a {self.rcut:.3f} A graph, "
                "so the ghost cloud equivalent to one periodic cell is tens of angstroms deep")

        out = self.model(self._typemapped(graph))
        if self.model_key not in out:
            raise ValueError(f"the model has no output '{self.model_key}'; it produces "
                             f"{sorted(out)}")
        raw = out[self.model_key].reshape(graph.pos.shape[0], -1)
        symm, anti = self._destandardise(raw, graph.numbers)
        tensor = self.transforms.cartesian(symm, anti, self.layout)
        return {"tensor": tensor, "magres": raw, "symm": symm, "anti": anti}
