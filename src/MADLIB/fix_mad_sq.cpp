/* ----------------------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#include "fix_mad_sq.h"

#include "atom.h"
#include "comm.h"
#include "compute_mad_pdf.h"
#include "domain.h"
#include "error.h"
#include "mad_form_factors.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static const double MY_PI = 3.14159265358979323846;
static const double MY_4PI = 12.566370614359172;

static inline double sinc(double v)
{
  return (fabs(v) < 1.0e-12) ? 1.0 : sin(v) / v;
}

/* ---------------------------------------------------------------------- */

FixMADSq::FixMADSq(LAMMPS *lmp, int narg, char **arg) :
    FixMAD(lmp, narg, arg), pdfid(nullptr), pdf(nullptr), ff(nullptr), xg(nullptr), qg(nullptr),
    mat(nullptr), interf(nullptr), chanf(nullptr), constf(nullptr), wab(nullptr), sab(nullptr),
    dEdg(nullptr)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix mad/sq", error);

  const char *c = arg[3];
  if (strncmp(c, "c_", 2) == 0) c += 2;
  pdfid = utils::strdup(c);

  weighting = (strcmp(style, "mad/xrd") == 0) ? W_XRAY : W_SQ;
  output = (weighting == W_SQ) ? O_RAW : O_QFQ;
  window = 1;
  lorentz = 0;
  q_in_angle = 0;
  nq = 0;
  qmin = qmax = -1.0;
  wavelength = 1.5405981;
  polarization = 1.0;
  sin_theta_min = 1.0e-3;
  window_cut = -1.0;
  std::vector<std::string> labels;

  int iarg = 4;
  while (iarg < narg) {
    if (parse_common(narg, arg, iarg)) continue;
    if (strcmp(arg[iarg], "q") == 0) {
      if (iarg + 3 >= narg) utils::missing_cmd_args(FLERR, "fix mad/sq q", error);
      qmin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      qmax = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      nq = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad/sq units", error);
      if (strcmp(arg[iarg + 1], "q") == 0)
        q_in_angle = 0;
      else if (strcmp(arg[iarg + 1], "twotheta") == 0)
        q_in_angle = 1;
      else
        error->all(FLERR, "fix mad/sq units must be q or twotheta");
      iarg += 2;
    } else if (strcmp(arg[iarg], "weighting") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad/sq weighting", error);
      if (strcmp(arg[iarg + 1], "sq") == 0)
        weighting = W_SQ;
      else if (strcmp(arg[iarg + 1], "xray") == 0)
        weighting = W_XRAY;
      else if (strcmp(arg[iarg + 1], "neutron") == 0)
        weighting = W_NEUTRON;
      else
        error->all(FLERR, "fix mad/sq weighting must be sq, xray or neutron");
      iarg += 2;
    } else if (strcmp(arg[iarg], "output") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad/sq output", error);
      const char *o = arg[iarg + 1];
      if (strcmp(o, "sq") == 0 || strcmp(o, "xrd") == 0 || strcmp(o, "raw") == 0)
        output = O_RAW;
      else if (strcmp(o, "F(q)") == 0 || strcmp(o, "i(q)") == 0)
        output = O_FQ;
      else if (strcmp(o, "q*F(q)") == 0 || strcmp(o, "q*i(q)") == 0 || strcmp(o, "qFq") == 0)
        output = O_QFQ;
      else
        error->all(FLERR, "fix mad/sq unknown output convention {}", o);
      iarg += 2;
    } else if (strcmp(arg[iarg], "species") == 0) {
      const int nt = atom->ntypes;
      if (iarg + nt >= narg) utils::missing_cmd_args(FLERR, "fix mad/sq species", error);
      for (int t = 0; t < nt; t++) labels.emplace_back(arg[iarg + 1 + t]);
      iarg += nt + 1;
    } else if (strcmp(arg[iarg], "wavelength") == 0) {
      wavelength = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "window") == 0) {
      window = (strcmp(arg[iarg + 1], "lorch") == 0 || strcmp(arg[iarg + 1], "yes") == 0);
      iarg += 2;
    } else if (strcmp(arg[iarg], "window_cut") == 0) {
      window_cut = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "lp") == 0) {
      lorentz = (strcmp(arg[iarg + 1], "yes") == 0);
      iarg += 2;
    } else if (strcmp(arg[iarg], "polarization") == 0) {
      polarization = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown fix mad/sq keyword {}", arg[iarg]);
    }
  }

  if (nq < 2) error->all(FLERR, "fix mad/sq requires q <qmin> <qmax> <nq>, nq >= 2");
  if (qmax <= qmin) error->all(FLERR, "fix mad/sq qmax must exceed qmin");

  auto *cmp = modify->get_compute_by_id(pdfid);
  if (!cmp) error->all(FLERR, "fix mad/sq could not find compute {}", pdfid);
  pdf = dynamic_cast<ComputeMADPdf *>(cmp);
  if (!pdf) error->all(FLERR, "fix mad/sq compute {} is not a compute mad/pdf", pdfid);

  if (weighting != W_SQ) {
    if ((int) labels.size() != atom->ntypes)
      error->all(FLERR, "fix mad/sq needs 'species' with one element label per atom type");
    ff = new MADFormFactors(weighting == W_NEUTRON, compat);
    for (auto &l : labels)
      if (!ff->add_species(l))
        error->all(FLERR, "fix mad/sq: {} is not in the {} scattering table", l,
                   weighting == W_NEUTRON ? "neutron" : "x-ray");
  }

  std::vector<double> grid(nq);
  for (int k = 0; k < nq; k++) grid[k] = qmin + (qmax - qmin) * k / (nq - 1);
  build_grid(grid);
  finish_setup(grid);

  const int nb = pdf->n_bins(), nc = pdf->n_channels();
  memory->create(mat, (bigint) nq * nb, "mad/sq:mat");
  memory->create(interf, nq, "mad/sq:interf");
  memory->create(chanf, nq, "mad/sq:chanf");
  memory->create(constf, nq, "mad/sq:constf");
  memory->create(wab, (bigint) nc * nq, "mad/sq:wab");
  memory->create(sab, (bigint) nc * nq, "mad/sq:sab");
  memory->create(dEdg, (bigint) nc * nb, "mad/sq:dEdg");
  build_kernel();
}

/* ---------------------------------------------------------------------- */

FixMADSq::~FixMADSq()
{
  delete[] pdfid;
  delete ff;
  memory->destroy(xg);
  memory->destroy(qg);
  memory->destroy(mat);
  memory->destroy(interf);
  memory->destroy(chanf);
  memory->destroy(constf);
  memory->destroy(wab);
  memory->destroy(sab);
  memory->destroy(dEdg);
}

/* ---------------------------------------------------------------------- */

void FixMADSq::build_grid(const std::vector<double> &user_q)
{
  memory->create(xg, nq, "mad/sq:xg");
  memory->create(qg, nq, "mad/sq:qg");
  for (int k = 0; k < nq; k++) {
    qg[k] = user_q[k];
    // x = 2 sin(theta)/lambda = q/2pi. Every wavevector below is 2 pi x, and
    // form factors are sampled at x/2.
    xg[k] = q_in_angle ? 2.0 * sin(MY_PI * qg[k] / 180.0 / 2.0) / wavelength : qg[k] / (2.0 * MY_PI);
  }
  if (ff) {
    std::vector<double> xv(xg, xg + nq);
    ff->tabulate(xv);
  }
}

/* ---------------------------------------------------------------------- */

void FixMADSq::build_kernel()
{
  const int nb = pdf->n_bins();
  // The Lorch window damps the ringing from truncating the transform, so its
  // cutoff is the extent of the r grid, NOT the neighbour cutoff. They are the
  // same number in TurboGAP and it is easy to carry that coincidence across.
  const double rcut = (window_cut > 0.0) ? window_cut : pdf->r_max();
  const double dr = pdf->bin_r(1) - pdf->bin_r(0);
  for (int k = 0; k < nq; k++) {
    const double q = 2.0 * MY_PI * xg[k];
    for (int l = 0; l < nb; l++) {
      const double r = pdf->bin_r(l);
      const double w = window ? sinc(MY_PI * r / rcut) : 1.0;
      const double m = dr * r * r * sinc(q * r) * w;
      mat[(bigint) k * nb + l] = m;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMADSq::build_weights()
{
  const int nt = atom->ntypes;
  bigint ntot = 0;
  for (int a = 1; a <= nt; a++) ntot += pdf->n_of_type(a);
  const double N = (double) ntot;

  for (int k = 0; k < nq; k++) {
    chanf[k] = 1.0;
    constf[k] = (weighting == W_SQ) ? 1.0 : 0.0;
  }

  double *sth = nullptr, *self = nullptr;
  if (weighting != W_SQ) {
    sth = new double[nq]();
    self = new double[nq]();
    for (int a = 1; a <= nt; a++) {
      const double ca = pdf->concentration(a);
      const double *fa = ff->row(a - 1);
      for (int k = 0; k < nq; k++) {
        sth[k] += ca * fa[k];
        self[k] += ca * fa[k] * fa[k];
      }
    }
  }

  for (int a = 1; a <= nt; a++)
    for (int b = a; b <= nt; b++) {
      const int c = pdf->channel(a, b);
      const double fab = (a == b) ? 1.0 : 2.0;
      double *w = &wab[(bigint) c * nq];
      if (weighting == W_SQ) {
        const double v = fab * sqrt((double) pdf->n_of_type(a) * (double) pdf->n_of_type(b)) / N;
        for (int k = 0; k < nq; k++) w[k] = v;
      } else {
        const double cc = fab * sqrt(pdf->concentration(a) * pdf->concentration(b));
        const double *fa = ff->row(a - 1), *fb = ff->row(b - 1);
        for (int k = 0; k < nq; k++) w[k] = cc * fa[k] * fb[k];
      }
    }

  if (weighting != W_SQ) {
    for (int k = 0; k < nq; k++) {
      const double s2 = sth[k] * sth[k];
      switch (output) {
        case O_RAW:
          chanf[k] = 1.0;
          constf[k] = self[k];
          break;
        case O_FQ:
          chanf[k] = (s2 > 0.0) ? 1.0 / s2 : 0.0;
          constf[k] = 1.0;
          break;
        case O_QFQ:
          chanf[k] = (s2 > 0.0) ? 2.0 * MY_PI * xg[k] / s2 : 0.0;
          constf[k] = 0.0;
          break;
      }
    }
    delete[] sth;
    delete[] self;
  }

  // The powder Lorentz-polarization factor is one multiplicative weight per q,
  // so it scales the gradient channel with the pattern and costs nothing in
  // gradient consistency.
  if (lorentz) {
    for (int k = 0; k < nq; k++) {
      const double s = 0.5 * wavelength * xg[k];
      double lp = 0.0;
      if (s >= sin_theta_min && s < 1.0) {
        const double cth = sqrt(1.0 - s * s);
        const double c2 = 1.0 - 2.0 * s * s;
        lp = (1.0 + polarization * c2 * c2) / (s * s * cth);
      }
      chanf[k] *= lp;
      constf[k] *= lp;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMADSq::init()
{
  pdf = dynamic_cast<ComputeMADPdf *>(modify->get_compute_by_id(pdfid));
  if (!pdf) error->all(FLERR, "fix mad/sq lost its compute {}", pdfid);
}

/* ---------------------------------------------------------------------- */

void FixMADSq::compute_spectrum()
{
  // See fix_mad_pdf.cpp: an MC fix re-evaluates within a timestep, so the
  // once-per-step cache would hide every trial move from the bias.
  pdf->compute_array();
  build_weights();

  const int nb = pdf->n_bins(), nc = pdf->n_channels(), nt = atom->ntypes;
  const double *g = pdf->g();
  const double vol = domain->xprd * domain->yprd * domain->zprd;
  const double rho = (double) atom->natoms / vol;

  for (int a = 1; a <= nt; a++)
    for (int b = a; b <= nt; b++) {
      const int c = pdf->channel(a, b);
      const double pre = MY_4PI * rho * sqrt(pdf->concentration(a) * pdf->concentration(b));
      const double *gc = &g[(bigint) c * nb];
      double *sc = &sab[(bigint) c * nq];
      for (int k = 0; k < nq; k++) {
        const double *m = &mat[(bigint) k * nb];
        double acc = 0.0;
        for (int l = 0; l < nb; l++) acc += m[l] * (gc[l] - 1.0);
        sc[k] = pre * acc;
      }
    }

  for (int k = 0; k < nq; k++) {
    double acc = 0.0;
    for (int c = 0; c < nc; c++) acc += wab[(bigint) c * nq + k] * sab[(bigint) c * nq + k];
    interf[k] = acc;
    y_pred[k] = chanf[k] * acc + constf[k];
  }
}

/* ---------------------------------------------------------------------- */

void FixMADSq::accumulate_bias(const double *dEdy)
{
  const int nb = pdf->n_bins(), nc = pdf->n_channels(), nt = atom->ntypes;
  const double vol = domain->xprd * domain->yprd * domain->zprd;
  const double rho = (double) atom->natoms / vol;

  for (bigint i = 0; i < (bigint) nc * nb; i++) dEdg[i] = 0.0;

  // Two explicit volume dependencies, and they must not be conflated.
  //   compute mad/pdf's own term already carries the one through g_ab, which
  //   is proportional to V.
  //   What is left is the rho = N/V in the prefactor, which multiplies the
  //   whole bracket: V d(S_ab)/dV at fixed g is -(S_ab - delta_ab), so the
  //   virial takes +sum (dE/dS_ab)(S_ab - delta_ab).
  double dev = 0.0;
  for (int a = 1; a <= nt; a++)
    for (int b = a; b <= nt; b++) {
      const int c = pdf->channel(a, b);
      const double pre = MY_4PI * rho * sqrt(pdf->concentration(a) * pdf->concentration(b));
      double *dg = &dEdg[(bigint) c * nb];
      const double *w = &wab[(bigint) c * nq];
      const double *sc = &sab[(bigint) c * nq];
      for (int k = 0; k < nq; k++) {
        const double t = dEdy[k] * chanf[k] * w[k];
        if (t == 0.0) continue;
        dev += t * sc[k];
        const double sfac = t * pre;
        const double *m = &mat[(bigint) k * nb];
        for (int l = 0; l < nb; l++) dg[l] += sfac * m[l];
      }
    }

  pdf->adjoint(dEdg, virial);

  if (comm->me == 0)
    for (int i = 0; i < 3; i++) virial[i] += dev;
}
