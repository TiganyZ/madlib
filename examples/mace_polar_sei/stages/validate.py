"""Test-set errors of every model, graphite's pressure and MD stability, and the model that goes on to MD.

    python validate.py SETS_DIR MODELS_DIR OUT_DIR [DEVICE]

polar_dipole is used if its periodic force RMSE is within DIPOLE_TOLERANCE of polar_ef's
(the dipole stage trains on clusters only, without stress). Foundation PolarMACE blows up on
graphite, so every model also runs MD on bulk graphite and on a test frame of each graphite
system, beside MACE-MH-1 as the baseline; if the chosen PolarMACE is not stable there, the
slabs fall back to MH-1.
"""

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase import units
from ase.lattice.hexagonal import Graphite
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import GPa
from mace.calculators import MACECalculator

from calc import polar, polar_d3

POLAR_MODELS = ("foundation64", "polar_ef", "polar_dipole")
DIPOLE_TOLERANCE = 1.10
GRAPHITE_A, GRAPHITE_C = 2.464, 6.708   # experimental, A
BASELINE = "mh-1:omat_pbe"
STABILITY_STEPS = 2000                  # 1 ps at 0.5 fs
STABILITY_TEMP = 300.0
STABLE_MAX_FORCE = 20.0                 # eV/A; thermal forces here stay below ~5


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


def md_stability(atoms, model, device, seed=1):
    """Largest force over STABILITY_STEPS of Langevin MD with D3, and whether it stayed bounded."""
    atoms = atoms.copy()
    atoms.info.update(charge=0, spin=1)
    atoms.calc = polar_d3(model, device, "float32")
    MaxwellBoltzmannDistribution(atoms, temperature_K=STABILITY_TEMP, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=STABILITY_TEMP, friction=0.01)
    worst = 0.0
    for _ in range(STABILITY_STEPS // 10):
        dyn.run(10)
        worst = max(worst, float(np.linalg.norm(atoms.get_forces(), axis=1).max()))
        if not np.isfinite(worst) or worst > STABLE_MAX_FORCE:
            break
    return {"max_force_eV_per_A": round(worst, 2), "stable": bool(np.isfinite(worst) and worst <= STABLE_MAX_FORCE)}


def graphite_checks(model, frames, device):
    bulk = Graphite("C", latticeconstant={"a": GRAPHITE_A, "c": GRAPHITE_C}, size=(4, 4, 2))
    checks = {"pressure_GPa_with_D3": graphite_pressure(model, device),
              "bulk_md": md_stability(bulk, model, device)}
    for name, atoms in frames.items():
        checks[f"{name}_md"] = md_stability(atoms, model, device)
    checks["stable"] = all(v["stable"] for k, v in checks.items() if k.endswith("_md"))
    return checks


def graphite_frames(periodic):
    """One test frame of each graphite system (config_type is the system)."""
    frames = {}
    for atoms in periodic:
        kind = atoms.info.get("config_type", "")
        if kind.startswith("graphite_") and kind not in frames:
            frames[kind] = atoms
    return frames


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
                        "graphite": graphite_checks(path, graphite_frames(periodic), device)}
    report["baseline_" + BASELINE] = {"graphite": graphite_checks(BASELINE, graphite_frames(periodic), device)}
    if (models / "dielectric.model").is_file():
        report["dielectric"] = {"clusters": dielectric_errors(models / "dielectric.model", clusters, device)}
    ef = report["polar_ef"]["periodic"]["F_meV_per_A"]
    dip = report.get("polar_dipole", {}).get("periodic", {}).get("F_meV_per_A", np.inf)
    report["chosen"] = "polar_dipole" if dip <= DIPOLE_TOLERANCE * ef else "polar_ef"
    chosen = str(models / f"{report['chosen']}.model")
    if not report[report["chosen"]]["graphite"]["stable"]:
        print(f"{report['chosen']} is not stable on graphite: the slabs use {BASELINE}")
        report["chosen"] = BASELINE
        chosen = BASELINE
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    (out / "chosen_model").write_text(chosen + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
