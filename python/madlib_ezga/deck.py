"""The one place a LAMMPS deck is written.

RMC, MAD and the GA all run the same physics; they differ only in the move
block and in gamma. Keeping a single renderer is what makes the three stages
comparable rather than three programs that happen to read the same target.
"""

from dataclasses import dataclass, field

# compute mad/pdf extends its neighbour cutoff to rmax + TRUNC * kde_sigma, and
# aborts unless the ghost cutoff covers that plus the skin.
PDF_TRUNC = 8.0
SKIN = 0.5


@dataclass
class Pdf:
    rmin: float = 0.1
    rmax: float = 8.0
    nbins: int = 201
    kde: float = 0.1

    @property
    def rcut(self):
        return self.rmax + PDF_TRUNC * self.kde

    @property
    def comm_cutoff(self):
        return self.rcut + SKIN


@dataclass
class Xrd:
    target: str = "xrd_glassy_carbon_zeng_2017.fq"
    qmin: float = 1.04525
    qmax: float = 9.97006
    nq: int = 201
    species: tuple = ("C",)
    output: str = "qFq"
    weighting: str = "xray"


@dataclass
class Potential:
    """`turbogap` with a .gap file, `mace` with a .json, or `zero` for RMC."""

    style: str = "turbogap"
    model: str = "gap_files/carbon.gap"
    species: tuple = ("C",)
    e0: dict = field(default_factory=lambda: {"C": -0.16138053})
    cutoff: float = 7.5  # pair_style zero only

    def lines(self):
        if self.style == "zero":
            return [f"pair_style      zero {self.cutoff}", "pair_coeff      * *"]
        if self.style == "mace":
            return [
                "pair_style      symmetrix/mace",
                f"pair_coeff      * * {self.model} {' '.join(self.species)}",
            ]
        e0 = " ".join(f"e0 {s} {self.e0[s]}" for s in self.species)
        return [
            f"pair_style      turbogap {e0}",
            f"pair_coeff      * * {self.model} {' '.join(self.species)}",
        ]


def _header(data_file, masses, pdf):
    # atom_modify map yes is required by pair_style symmetrix/mace and harmless
    # for the others, so it is unconditional.
    lines = [
        "units           metal",
        "atom_style      atomic",
        "boundary        p p p",
        "atom_modify     map yes",
        f"read_data       {data_file}",
    ]
    lines += [f"mass            {t} {m}" for t, m in sorted(masses.items())]
    lines += [
        f"comm_modify     cutoff {pdf.comm_cutoff:g}",
        f"neighbor        {SKIN:g} bin",
    ]
    return lines


def _observable(pdf, xrd, gamma, spectrum_file):
    g = f"{gamma[0]:g} {gamma[1]:g}" if isinstance(gamma, (tuple, list)) else f"{gamma:g}"
    fix = (
        f"fix     mad all mad/xrd c_gr target {xrd.target} "
        f"q {xrd.qmin:g} {xrd.qmax:g} {xrd.nq} output {xrd.output} "
        f"species {' '.join(xrd.species)} weighting {xrd.weighting} gamma {g}"
    )
    if spectrum_file:
        fix += f" file {spectrum_file}"
    return [
        # c_pepair is the potential alone. compute pe would include the bias,
        # because fix mad/* sets thermo_energy on by default, and an energy
        # axis contaminated by the bias makes the Pareto front meaningless.
        "compute pepair all pe pair",
        f"compute gr     all mad/pdf r {pdf.rmin:g} {pdf.rmax:g} {pdf.nbins} kde {pdf.kde:g}",
        fix,
    ]


def _thermo(every):
    return [
        "thermo_style custom step temp pe c_pepair f_mad f_mad[2] f_mad[3] f_mad[4]",
        "thermo_modify format float %.10g",
        f"thermo {every}",
    ]


def score():
    return ["run 0"]


def minimise(etol=1e-6, ftol=1e-4, maxiter=2000):
    return [f"min_style fire", f"minimize {etol:g} {ftol:g} {maxiter} {maxiter * 10}"]


def md(steps, temp, seed, damp=0.1):
    return [
        f"velocity all create {temp:g} {seed} mom yes",
        "fix nve all nve",
        f"fix thm all langevin {temp:g} {temp:g} {damp:g} {seed + 1}",
        f"run {steps}",
    ]


def anneal(steps, t_hi, t_lo, seed, damp=0.1):
    return [
        f"velocity all create {t_hi:g} {seed} mom yes",
        "fix nve all nve",
        f"fix thm all langevin {t_hi:g} {t_lo:g} {damp:g} {seed + 1}",
        f"run {steps}",
    ]


def rmc(steps, moves_per_cycle, temp, seed, max_disp=0.1):
    """Displacement-only Metropolis through fix gcmc.

    nexchanges = 0 leaves translation as the only move, which is reverse Monte
    Carlo. fix_modify energy yes is what puts the bias in the acceptance ratio;
    without it the moves are priced on the potential alone.
    """
    return [
        "fix_modify mad energy yes",
        f"fix mc all gcmc 1 0 {moves_per_cycle} 1 {seed} {temp:g} 0.0 {max_disp:g} full_energy",
        f"run {steps}",
    ]


def render(
    data_file,
    potential,
    pdf=None,
    xrd=None,
    gamma=0.0,
    moves=None,
    masses=None,
    thermo_every=100,
    spectrum_file=None,
    dump_file="final.dump",
):
    pdf = pdf or Pdf()
    xrd = xrd or Xrd()
    masses = masses or {1: 12.011}
    moves = moves if moves is not None else score()

    lines = _header(data_file, masses, pdf)
    lines += [""] + potential.lines()
    lines += [""] + _observable(pdf, xrd, gamma, spectrum_file)
    lines += [""] + _thermo(thermo_every)
    lines += [""] + list(moves)
    if dump_file:
        lines += [
            "",
            f"write_dump all custom {dump_file} id type xs ys zs modify sort id",
        ]
    return "\n".join(lines) + "\n"
