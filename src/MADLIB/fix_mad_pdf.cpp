/* ----------------------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#include "fix_mad_pdf.h"

#include "atom.h"
#include "comm.h"
#include "compute_mad_pdf.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static const double MY_4PI = 12.566370614359172;

/* ---------------------------------------------------------------------- */

FixMADPdf::FixMADPdf(LAMMPS *lmp, int narg, char **arg) :
    FixMAD(lmp, narg, arg), pdfid(nullptr), pdf(nullptr), reduced(0), chanw(nullptr), dEdg(nullptr)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix mad/pdf", error);

  // arg[3] names the compute mad/pdf that supplies g_ab(r).
  const char *c = arg[3];
  if (strncmp(c, "c_", 2) == 0) c += 2;
  pdfid = utils::strdup(c);

  int iarg = 4;
  while (iarg < narg) {
    if (parse_common(narg, arg, iarg)) continue;
    if (strcmp(arg[iarg], "output") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad/pdf output", error);
      if (strcmp(arg[iarg + 1], "pdf") == 0 || strcmp(arg[iarg + 1], "g(r)") == 0)
        reduced = 0;
      else if (strcmp(arg[iarg + 1], "D(r)") == 0)
        reduced = 1;
      else
        error->all(FLERR, "fix mad/pdf output must be pdf or D(r)");
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown fix mad/pdf keyword {}", arg[iarg]);
    }
  }

  auto *cmp = modify->get_compute_by_id(pdfid);
  if (!cmp) error->all(FLERR, "fix mad/pdf could not find compute {}", pdfid);
  pdf = dynamic_cast<ComputeMADPdf *>(cmp);
  if (!pdf) error->all(FLERR, "fix mad/pdf compute {} is not a compute mad/pdf", pdfid);

  std::vector<double> grid(pdf->n_bins());
  for (int l = 0; l < pdf->n_bins(); l++) grid[l] = pdf->bin_r(l);
  finish_setup(grid);

  memory->create(chanw, pdf->n_channels(), "mad/pdf:chanw");
  memory->create(dEdg, (bigint) pdf->n_channels() * pdf->n_bins(), "mad/pdf:dEdg");
}

/* ---------------------------------------------------------------------- */

FixMADPdf::~FixMADPdf()
{
  delete[] pdfid;
  memory->destroy(chanw);
  memory->destroy(dEdg);
}

/* ---------------------------------------------------------------------- */

void FixMADPdf::init()
{
  auto *cmp = modify->get_compute_by_id(pdfid);
  pdf = dynamic_cast<ComputeMADPdf *>(cmp);
  if (!pdf) error->all(FLERR, "fix mad/pdf lost its compute {}", pdfid);
}

/* ---------------------------------------------------------------------- */

void FixMADPdf::channel_weights()
{
  const int nt = atom->ntypes;
  bigint ntot = 0;
  for (int a = 1; a <= nt; a++) ntot += pdf->n_of_type(a);
  const double inv = ntot ? 1.0 / ((double) ntot * (double) ntot) : 0.0;

  for (int a = 1; a <= nt; a++)
    for (int b = a; b <= nt; b++) {
      const int c = pdf->channel(a, b);
      const double fab = (a == b) ? 1.0 : 2.0;
      chanw[c] = fab * (double) pdf->n_of_type(a) * (double) pdf->n_of_type(b) * inv;
    }
}

/* ---------------------------------------------------------------------- */

void FixMADPdf::compute_spectrum()
{
  // Not conditioned on invoked_array. An MC fix calls post_force() several
  // times inside one timestep with different configurations, so the usual
  // once-per-step caching would price every trial move against the spectrum
  // of the first one -- silently, and the bias would simply not reach the
  // Metropolis test.
  pdf->compute_array();
  channel_weights();

  const int nb = pdf->n_bins(), nc = pdf->n_channels();
  const double *g = pdf->g();

  for (int l = 0; l < nb; l++) {
    double tot = 0.0;
    for (int c = 0; c < nc; c++) tot += chanw[c] * g[(bigint) c * nb + l];
    y_pred[l] = tot;
  }

  if (reduced) {
    const double rho = (double) atom->natoms / (domain->xprd * domain->yprd * domain->zprd);
    for (int l = 0; l < nb; l++) y_pred[l] = MY_4PI * rho * xgrid[l] * (y_pred[l] - 1.0);
  }
}

/* ---------------------------------------------------------------------- */

void FixMADPdf::accumulate_bias(const double *dEdy)
{
  const int nb = pdf->n_bins(), nc = pdf->n_channels();
  const double vol = domain->xprd * domain->yprd * domain->zprd;
  const double rho = (double) atom->natoms / vol;

  double dchain = 0.0;
  for (int l = 0; l < nb; l++) {
    const double w = reduced ? MY_4PI * rho * xgrid[l] : 1.0;
    for (int c = 0; c < nc; c++) dEdg[(bigint) c * nb + l] = dEdy[l] * w * chanw[c];
    if (reduced) dchain += dEdy[l] * y_pred[l];
  }

  pdf->adjoint(dEdg, virial);

  // D(r) carries its own explicit factor of rho = N/V on top of g's factor of
  // V, and the two do not cancel: V dD/dV = -(D + 4 pi rho r). The g half is
  // already in compute mad/pdf's volume term, so only the extra piece is
  // needed here -- and, like it, on one rank only.
  if (reduced && comm->me == 0) {
    for (int i = 0; i < 3; i++) virial[i] += dchain;
  }
}
