/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
------------------------------------------------------------------------- */

#include "fix_mad_nmr_aniso.h"

#include "atom.h"
#include "comm.h"
#include "compute_mad_property.h"
#include "error.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int NVOIGT = 6;

/* ---------------------------------------------------------------------- */

FixMADNmrAniso::FixMADNmrAniso(LAMMPS *lmp, int narg, char **arg) :
    FixMAD(lmp, narg, arg), propid(nullptr), prop(nullptr), qscale(1.0), qoffset(0.0),
    nmax_lam(0), norient(512), unorm(1.0), u(nullptr), ulocal(nullptr), lambda(nullptr),
    wquad(1.0)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, style, error);

  const char *c = arg[3];
  if (strncmp(c, "c_", 2) == 0) c += 2;
  propid = utils::strdup(c);

  sigma = 1.0;
  emin = emax = 0.0;
  trunc = 8.0;
  normalise = N_L2;
  int nk_req = 0, said_convention = 0, have_ref = 0;
  double slope = 1.0;

  int iarg = 4;
  while (iarg < narg) {
    if (parse_common(narg, arg, iarg)) continue;
    if (strcmp(arg[iarg], "sigma") == 0) {
      sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "range") == 0) {
      if (iarg + 3 >= narg) utils::missing_cmd_args(FLERR, std::string(style) + " range", error);
      emin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      emax = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      nk_req = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;
    } else if (strcmp(arg[iarg], "orientations") == 0) {
      norient = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "norm") == 0) {
      if (strcmp(arg[iarg + 1], "l2") == 0)
        normalise = N_L2;
      else if (strcmp(arg[iarg + 1], "area") == 0)
        normalise = N_AREA;
      else if (strcmp(arg[iarg + 1], "none") == 0)
        normalise = N_NONE;
      else
        error->all(FLERR, "fix {} norm must be l2, area or none", style);
      iarg += 2;
    } else if (strcmp(arg[iarg], "trunc") == 0) {
      trunc = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "reference") == 0) {
      qoffset = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      have_ref = said_convention = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "slope") == 0) {
      slope = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "convention") == 0) {
      if (strcmp(arg[iarg + 1], "shift") != 0)
        error->all(FLERR,
                   "fix {} convention takes only 'shift', meaning the tensor is already on the "
                   "target's axis. To convert a shielding, give 'reference <a>'",
                   style);
      have_ref = 0;
      said_convention = 1;
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown fix {} keyword {}", style, arg[iarg]);
    }
  }

  if (!said_convention)
    error->all(FLERR,
               "fix {} needs the convention stated: 'reference <a> [slope <b>]' to convert a "
               "shielding tensor to shifts as delta = a - b sigma, or 'convention shift' if the "
               "tensor already is one",
               style);
  // After the loop, so 'slope' works on either side of 'reference'.
  if (have_ref) qscale = -slope;
  if (qscale == 0.0) error->all(FLERR, "fix {} slope must not be zero", style);

  if (sigma <= 0.0) error->all(FLERR, "fix {} sigma must be positive", style);
  if (norient < 1) error->all(FLERR, "fix {} orientations must be positive", style);
  if (nk_req < 2) error->all(FLERR, "fix {} requires range <min> <max> <nsamples>", style);
  if (emax <= emin) error->all(FLERR, "fix {} needs max to exceed min", style);

  auto *cmp = modify->get_compute_by_id(propid);
  if (!cmp) error->all(FLERR, "fix {} could not find compute {}", style, propid);
  prop = dynamic_cast<ComputeMADProperty *>(cmp);
  if (!prop) error->all(FLERR, "fix {} compute {} is not a mad/property style", style, propid);
  if (prop->n_components() != NVOIGT)
    error->all(FLERR,
               "fix {} needs a 6-component symmetric tensor property in Voigt order "
               "(xx yy zz xy xz yz), not one with {} components",
               style, prop->n_components());

  build_orientations();

  std::vector<double> grid(nk_req);
  for (int k = 0; k < nk_req; k++) grid[k] = emin + (emax - emin) * k / (nk_req - 1);
  finish_setup(grid);

  memory->create(u, nk, "mad/nmr/aniso:u");
  memory->create(ulocal, nk, "mad/nmr/aniso:ulocal");

  if (comm->me == 0)
    utils::logmesg(lmp, "fix {}: {} orientations, sigma {:.4g}\n", id, norient, sigma);
}

/* ---------------------------------------------------------------------- */

FixMADNmrAniso::~FixMADNmrAniso()
{
  delete[] propid;
  memory->destroy(u);
  memory->destroy(ulocal);
  memory->destroy(lambda);
}

void FixMADNmrAniso::init()
{
  prop = dynamic_cast<ComputeMADProperty *>(modify->get_compute_by_id(propid));
  if (!prop) error->all(FLERR, "fix {} lost its compute {}", style, propid);
}

/* A golden-spiral hemisphere grid -- enough, since omega(n) = omega(-n) --
   stored as v(n) rather than n. */

void FixMADNmrAniso::build_orientations()
{
  const double golden = MY_PI * (3.0 - sqrt(5.0));
  nvec.resize((size_t) norient * NVOIGT);
  wquad = 1.0 / norient;

  for (int m = 0; m < norient; m++) {
    const double z = (m + 0.5) / norient;
    const double r = sqrt(1.0 - z * z);
    const double phi = golden * m;
    const double nx = r * cos(phi), ny = r * sin(phi), nz = z;
    double *v = &nvec[(size_t) m * NVOIGT];
    v[0] = nx * nx;
    v[1] = ny * ny;
    v[2] = nz * nz;
    v[3] = 2.0 * nx * ny;
    v[4] = 2.0 * nx * nz;
    v[5] = 2.0 * ny * nz;
  }
}

void FixMADNmrAniso::grow_lambda()
{
  if (atom->nlocal > nmax_lam) {
    memory->destroy(lambda);
    nmax_lam = atom->nmax;
    memory->create(lambda, nmax_lam, NVOIGT, "mad/nmr/aniso:lambda");
  }
}

/* ---------------------------------------------------------------------- */

void FixMADNmrAniso::compute_spectrum()
{
  // Not cached on the timestep; an MC fix re-evaluates within one.
  prop->compute_property();

  double **t = prop->vector_property();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double dx = xgrid[1] - xgrid[0];
  const double s2inv = 1.0 / (2.0 * sigma * sigma);
  const int ntrunc = (int) (trunc * sigma / dx) + 2;

  for (int k = 0; k < nk; k++) ulocal[k] = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    for (int m = 0; m < norient; m++) {
      const double xi = shifted(t[i], m);
      int lo = (int) ((xi - emin) / dx) - ntrunc;
      int hi = (int) ((xi - emin) / dx) + ntrunc;
      if (lo < 0) lo = 0;
      if (hi >= nk) hi = nk - 1;
      for (int k = lo; k <= hi; k++) {
        const double v = xgrid[k] - xi;
        ulocal[k] += wquad * exp(-v * v * s2inv);
      }
    }
  }

  MPI_Allreduce(ulocal, u, nk, MPI_DOUBLE, MPI_SUM, world);

  if (normalise == N_L2) {
    double s = 0.0;
    for (int k = 0; k < nk; k++) s += u[k] * u[k];
    unorm = sqrt(s);
  } else if (normalise == N_AREA) {
    double s = 0.0;
    for (int k = 0; k < nk; k++) s += u[k];
    unorm = s * dx;
  } else {
    unorm = 1.0;
  }
  if (unorm <= 0.0) unorm = 1.0;

  for (int k = 0; k < nk; k++) y_pred[k] = u[k] / unorm;
}

/* ---------------------------------------------------------------------- */

void FixMADNmrAniso::accumulate_bias(const double *dEdy)
{
  const double dx = xgrid[1] - xgrid[0];

  // the projection term is what a naive 1/||u|| misses; see fix mad/xps
  std::vector<double> dEdu(nk);
  if (normalise == N_L2) {
    double proj = 0.0;
    for (int k = 0; k < nk; k++) proj += dEdy[k] * y_pred[k];
    for (int k = 0; k < nk; k++) dEdu[k] = (dEdy[k] - y_pred[k] * proj) / unorm;
  } else if (normalise == N_AREA) {
    double proj = 0.0;
    for (int k = 0; k < nk; k++) proj += dEdy[k] * y_pred[k];
    for (int k = 0; k < nk; k++) dEdu[k] = (dEdy[k] - proj * dx) / unorm;
  } else {
    for (int k = 0; k < nk; k++) dEdu[k] = dEdy[k];
  }

  grow_lambda();
  double **t = prop->vector_property();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double s2inv = 1.0 / (2.0 * sigma * sigma);
  const double sinv2 = 1.0 / (sigma * sigma);
  const int ntrunc = (int) (trunc * sigma / dx) + 2;

  for (int i = 0; i < nlocal; i++) {
    for (int c = 0; c < NVOIGT; c++) lambda[i][c] = 0.0;
    if (!(mask[i] & groupbit)) continue;
    for (int m = 0; m < norient; m++) {
      const double xi = shifted(t[i], m);
      int lo = (int) ((xi - emin) / dx) - ntrunc;
      int hi = (int) ((xi - emin) / dx) + ntrunc;
      if (lo < 0) lo = 0;
      if (hi >= nk) hi = nk - 1;
      double acc = 0.0;
      for (int k = lo; k <= hi; k++) {
        const double v = xgrid[k] - xi;
        acc += dEdu[k] * v * sinv2 * exp(-v * v * s2inv);
      }
      // d omega_i / d T_(i,c) is v_c; d x_i / d omega_i is qscale
      const double s = acc * wquad * qscale;
      const double *v = &nvec[(size_t) m * NVOIGT];
      for (int c = 0; c < NVOIGT; c++) lambda[i][c] += s * v[c];
    }
  }

  prop->adjoint(lambda, virial);
}
