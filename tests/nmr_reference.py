#!/usr/bin/env python3
# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Independent reference for the NMR bias: energy, forces and virial.

The finite-difference harness cannot settle this one. ShiftML is a float32
model, so differencing its energy is dominated by round-off before the step is
small enough for the h^2 term to show -- the error grows as 1/h, which is what
noise looks like and not what a wrong gradient looks like.

So the chain is checked against itself instead: this recomputes the whole of it
-- shieldings, Gaussian broadening, L2 norm, the target's quadrature weights and
the harmonic loss -- from the definitions, in the same arithmetic, and takes the
forces by one autograd call over the lot. Nothing is shared with the C++ but the
model, so an error anywhere in the KDE adjoint, the lambda contraction, the
scatter, the reverse communication or the cell term shows up as a disagreement
far above float32 noise.

    nmr_reference.py <data file> <target> <atom type> <min> <max> <n> <sigma>
                     <gamma> <lammps dump> <lammps energy> [<virial xx..yz>]
"""

import sys

import numpy as np
import torch
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System
from shiftml.ase import ShiftML
from shiftml.utils.tensorial import T_sym_np_inv

ELEMENT = {1: 1, 2: 6, 3: 8}    # LAMMPS type -> Z, as the data file orders them
OUTPUT = {"mtt::cs_iso": ModelOutput(quantity="", unit="", sample_kind="atom")}


def read_data(path):
    """(cell 3x3, types, positions) of a triclinic LAMMPS data file."""
    lines = [l.split("#")[0].rstrip() for l in open(path)]
    box = {}
    tilt = (0.0, 0.0, 0.0)
    for l in lines:
        w = l.split()
        if len(w) == 4 and w[2:] == ["xlo", "xhi"]:
            box["x"] = (float(w[0]), float(w[1]))
        elif len(w) == 4 and w[2:] == ["ylo", "yhi"]:
            box["y"] = (float(w[0]), float(w[1]))
        elif len(w) == 4 and w[2:] == ["zlo", "zhi"]:
            box["z"] = (float(w[0]), float(w[1]))
        elif len(w) == 6 and w[3:] == ["xy", "xz", "yz"]:
            tilt = (float(w[0]), float(w[1]), float(w[2]))
    xy, xz, yz = tilt
    cell = np.array([[box["x"][1] - box["x"][0], 0.0, 0.0],
                     [xy, box["y"][1] - box["y"][0], 0.0],
                     [xz, yz, box["z"][1] - box["z"][0]]])

    i = next(k for k, l in enumerate(lines) if l.strip().startswith("Atoms"))
    rows = [l.split() for l in lines[i + 1:] if l.strip()]
    rows = [r for r in rows if len(r) >= 5]
    rows.sort(key=lambda r: int(r[0]))
    t = np.array([int(r[1]) for r in rows])
    x = np.array([[float(v) for v in r[2:5]] for r in rows])
    return cell, t, x


def shieldings(cell, types, pos):
    """cs_iso from ShiftML, with positions and cell still on the autograd graph."""
    m = ShiftML("ShiftML41", device="cpu")
    w = torch.tensor((T_sym_np_inv[0] + T_sym_np_inv[4] + T_sym_np_inv[8]) / 3.0,
                     dtype=m._dtype)
    x = torch.tensor(pos, dtype=m._dtype, requires_grad=True)
    h = torch.tensor(cell, dtype=m._dtype, requires_grad=True)
    z = torch.as_tensor([ELEMENT[i] for i in types], dtype=torch.int32)
    systems = m._nl_calculators.compute(
        systems=[System(z, x, h, torch.ones(3, dtype=torch.bool))])
    out = m._model(systems=systems,
                   options=ModelEvaluationOptions(length_unit="angstrom", outputs=OUTPUT),
                   check_consistency=False)
    v = torch.cat([b.values.reshape(b.values.shape[0], -1)
                   for b in out["mtt::cs_iso"].blocks()], dim=1)
    return x, h, v @ w


def quadrature_weights(grid):
    """The trapezoidal weights MADTarget forms, normalised to sum to nk."""
    n = len(grid)
    w = np.empty(n)
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return w * (n / w.sum())


def bias(cs, grid, sigma, yexp, weights, gamma):
    """E = 1/2 gamma sum_k w_k (y_k - yexp_k)^2 over the L2-normalised spectrum."""
    g = torch.as_tensor(grid, dtype=cs.dtype)
    u = torch.exp(-((g[:, None] - cs[None, :]) ** 2) / (2 * sigma ** 2)).sum(dim=1)
    y = u / torch.linalg.norm(u)
    w = torch.as_tensor(weights, dtype=cs.dtype)
    e = torch.as_tensor(yexp, dtype=cs.dtype)
    return 0.5 * gamma * (w * (y - e) ** 2).sum()


def read_dump_forces(path):
    txt = open(path).read()
    head = txt.split("ITEM: ATOMS")[1].split("\n")[0].split()
    col = {k: i for i, k in enumerate(head)}
    rows = [l.split() for l in txt.split("ITEM: ATOMS")[1].split("\n")[1:] if l.strip()]
    rows.sort(key=lambda r: int(r[col["id"]]))
    return np.array([[float(r[col[k]]) for k in ("fx", "fy", "fz")] for r in rows])


def main(argv):
    data, tgt, atype, lo, hi, n, sigma, gamma, dump, energy = argv[:10]
    cell, types, pos = read_data(data)
    keep = types == int(atype)

    grid = np.linspace(float(lo), float(hi), int(n))
    yexp = np.loadtxt(tgt)[:, 1]
    weights = quadrature_weights(grid)

    x, h, cs = shieldings(cell, types, pos)
    E = bias(cs[torch.as_tensor(keep)], grid, float(sigma), yexp, weights, float(gamma))
    gx, gh = torch.autograd.grad(E, [x, h])
    f = -gx.numpy().astype(np.float64)
    W = pos.T @ f + cell.T @ (-gh.numpy().astype(np.float64))

    e_ref = float(E.detach())
    e_lmp = float(energy)
    f_lmp = read_dump_forces(dump)
    print("    energy   reference %.10f   LAMMPS %.10f   rel %.3e" %
          (e_ref, e_lmp, abs(e_ref - e_lmp) / max(abs(e_ref), 1e-300)))
    df = np.abs(f - f_lmp).max()
    print("    forces   max|reference - LAMMPS| = %.3e eV/A   on forces up to %.3e" %
          (df, np.abs(f).max()))

    # The model is float32, so a few parts in 1e6 is the floor, not slack: the
    # two sides share only the model and sum the same terms in a different order.
    tol = 1e-4
    ok = abs(e_ref - e_lmp) < tol * max(abs(e_ref), 1.0) and df < tol * max(np.abs(f).max(), 1.0)
    if len(argv) > 10:
        v_lmp = np.array([float(v) for v in argv[10:16]])
        v_ref = np.array([W[0][0], W[1][1], W[2][2],
                          0.5 * (W[0][1] + W[1][0]), 0.5 * (W[0][2] + W[2][0]),
                          0.5 * (W[1][2] + W[2][1])])
        dv = np.abs(v_ref - v_lmp).max()
        print("    virial   max|reference - LAMMPS| = %.3e eV    on %s" %
              (dv, np.array2string(v_ref, precision=4)))
        ok = ok and dv < tol * max(np.abs(v_ref).max(), 1.0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
