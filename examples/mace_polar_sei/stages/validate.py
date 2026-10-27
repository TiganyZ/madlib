"""Test-set errors of every model, bulk graphite's pressure, and which PolarMACE goes on to MD.

    python validate.py SETS_DIR MODELS_DIR OUT_DIR [DEVICE]

polar_dipole is used if its periodic force RMSE is within DIPOLE_TOLERANCE of polar_ef's
(the dipole stage trains on clusters only, without stress).
"""

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.lattice.hexagonal import Graphite
from ase.units import GPa
from mace.calculators import MACECalculator

from calc import polar, polar_d3

POLAR_MODELS = ("foundation64", "polar_ef", "polar_dipole")
DIPOLE_TOLERANCE = 1.10
GRAPHITE_A, GRAPHITE_C = 2.464, 6.708   # experimental, A


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def polar_errors(calc, frames):
    e, f, s, mu = [], [], [], []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        n = len(atoms)
        e.append((atoms.get_potential_energy() / n, atoms.info["REF_energy"] / n))
        f.append((atoms.get_forces(), atoms.arrays["REF_forces"]))
        if "REF_stress" in atoms.info:
            s.append((atoms.get_stress(voigt=False).reshape(9), atoms.info["REF_stress"]))
        if "REF_dipole" in atoms.info:
            mu.append((calc.results["dipole"], atoms.info["REF_dipole"]))
    out: dict = {"n": len(frames)}
    if e:
        pred, ref = map(np.array, zip(*e))
        # a constant offset is a reference choice, not an error
        out["E_meV_per_atom"] = 1000 * rmse(pred - pred.mean(), ref - ref.mean())
        out["F_meV_per_A"] = 1000 * rmse(np.concatenate([p for p, _ in f]), np.concatenate([r for _, r in f]))
    if s:
        out["stress_GPa"] = rmse([p for p, _ in s], [r for _, r in s]) / GPa
    if mu:
        out["dipole_eA"] = rmse([p for p, _ in mu], [r for _, r in mu])
    return out


def dielectric_errors(model, frames, device):
    calc = MACECalculator(model_paths=str(model), model_type="DipolePolarizabilityMACE",
                          device=device, default_dtype="float64")
    mu, alpha = [], []
    for atoms in frames:
        atoms = atoms.copy()
        calc.calculate(atoms)
        mu.append((calc.results["dipole"], atoms.info["REF_dipole"]))
        alpha.append((np.asarray(calc.results["polarizability"]).reshape(9),
                      np.asarray(atoms.info["REF_polarizability"]).reshape(9)))
    iso = [(np.trace(p.reshape(3, 3)) / 3, np.trace(r.reshape(3, 3)) / 3) for p, r in alpha]
    return {"n": len(frames),
            "dipole_eA": rmse([p for p, _ in mu], [r for _, r in mu]),
            "alpha_A3": rmse([p for p, _ in alpha], [r for _, r in alpha]),
            "alpha_iso_A3": rmse([p for p, _ in iso], [r for _, r in iso]),
            "alpha_iso_mean_ref_A3": float(np.mean([r for _, r in iso]))}


def graphite_pressure(model, device):
    """Diagonal stress of AB graphite at the experimental lattice, GPa (positive = compressed)."""
    atoms = Graphite("C", latticeconstant={"a": GRAPHITE_A, "c": GRAPHITE_C}, size=(4, 4, 2))
    atoms.calc = polar_d3(model, device)
    return (-atoms.get_stress(voigt=False).diagonal() / GPa).round(2).tolist()


def main(sets, models, out, device="cuda"):
    sets, models, out = Path(sets), Path(models), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    test = read(sets / "polar_test.xyz", ":")
    periodic = [a for a in test if a.info.get("config_type") != "cluster"]
    clusters = read(sets / "clusters_test.xyz", ":")
    report = {}
    for name in POLAR_MODELS:
        path = models / f"{name}.model"
        if not path.is_file():
            continue
        calc = polar(path, device)
        report[name] = {"periodic": polar_errors(calc, periodic),
                        "clusters": polar_errors(calc, clusters),
                        "graphite_pressure_GPa_with_D3": graphite_pressure(path, device)}
    if (models / "dielectric.model").is_file():
        report["dielectric"] = {"clusters": dielectric_errors(models / "dielectric.model", clusters, device)}
    ef = report["polar_ef"]["periodic"]["F_meV_per_A"]
    dip = report.get("polar_dipole", {}).get("periodic", {}).get("F_meV_per_A", np.inf)
    report["chosen"] = "polar_dipole" if dip <= DIPOLE_TOLERANCE * ef else "polar_ef"
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    (out / "chosen_model").write_text(str(models / f"{report['chosen']}.model") + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
