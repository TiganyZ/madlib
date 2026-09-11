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

#include "fix_mad_xps.h"

#include "atom.h"
#include "comm.h"
#include "compute_mad_property.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixMADXps::FixMADXps(LAMMPS *lmp, int narg, char **arg) :
    FixMAD(lmp, narg, arg), propid(nullptr), prop(nullptr), nmax_lam(0), unorm(1.0), u(nullptr),
    ulocal(nullptr), lambda(nullptr)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix mad/xps", error);

  const char *c = arg[3];
  if (strncmp(c, "c_", 2) == 0) c += 2;
  propid = utils::strdup(c);

  sigma = 0.4;
  emin = emax = 0.0;
  trunc = 8.0;
  normalise = N_L2;
  int nk_req = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (parse_common(narg, arg, iarg)) continue;
    if (strcmp(arg[iarg], "sigma") == 0) {
      sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "energy") == 0 || strcmp(arg[iarg], "range") == 0) {
      if (iarg + 3 >= narg) utils::missing_cmd_args(FLERR, "fix mad/xps energy", error);
      emin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      emax = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      nk_req = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;
    } else if (strcmp(arg[iarg], "norm") == 0) {
      if (strcmp(arg[iarg + 1], "l2") == 0)
        normalise = N_L2;
      else if (strcmp(arg[iarg + 1], "area") == 0)
        normalise = N_AREA;
      else if (strcmp(arg[iarg + 1], "none") == 0)
        normalise = N_NONE;
      else
        error->all(FLERR, "fix mad/xps norm must be l2, area or none");
      iarg += 2;
    } else if (strcmp(arg[iarg], "trunc") == 0) {
      trunc = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown fix mad/xps keyword {}", arg[iarg]);
    }
  }

  if (sigma <= 0.0) error->all(FLERR, "fix mad/xps sigma must be positive");
  if (nk_req < 2) error->all(FLERR, "fix mad/xps requires energy <emin> <emax> <nsamples>");
  if (emax <= emin) error->all(FLERR, "fix mad/xps emax must exceed emin");

  auto *cmp = modify->get_compute_by_id(propid);
  if (!cmp) error->all(FLERR, "fix mad/xps could not find compute {}", propid);
  prop = dynamic_cast<ComputeMADProperty *>(cmp);
  if (!prop) error->all(FLERR, "fix mad/xps compute {} is not a mad/property style", propid);
  if (prop->n_components() != 1)
    error->all(FLERR, "fix mad/xps needs a scalar per-atom property, not one with {} components",
               prop->n_components());

  std::vector<double> grid(nk_req);
  for (int k = 0; k < nk_req; k++) grid[k] = emin + (emax - emin) * k / (nk_req - 1);
  finish_setup(grid);

  memory->create(u, nk, "mad/xps:u");
  memory->create(ulocal, nk, "mad/xps:ulocal");
}

/* ---------------------------------------------------------------------- */

FixMADXps::~FixMADXps()
{
  delete[] propid;
  memory->destroy(u);
  memory->destroy(ulocal);
  memory->destroy(lambda);
}

void FixMADXps::init()
{
  prop = dynamic_cast<ComputeMADProperty *>(modify->get_compute_by_id(propid));
  if (!prop) error->all(FLERR, "fix mad/xps lost its compute {}", propid);
}

void FixMADXps::grow_lambda()
{
  if (atom->nlocal > nmax_lam) {
    memory->destroy(lambda);
    nmax_lam = atom->nmax;
    memory->create(lambda, nmax_lam, 1, "mad/xps:lambda");
  }
}

/* ---------------------------------------------------------------------- */

void FixMADXps::compute_spectrum()
{
  // Not cached on the timestep; an MC fix re-evaluates within one.
  prop->compute_property();

  const double *q = prop->scalar_property();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double dx = xgrid[1] - xgrid[0];
  const double s2inv = 1.0 / (2.0 * sigma * sigma);
  const int ntrunc = (int) (trunc * sigma / dx) + 2;

  for (int k = 0; k < nk; k++) ulocal[k] = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int lo = (int) ((q[i] - emin) / dx) - ntrunc;
    int hi = (int) ((q[i] - emin) / dx) + ntrunc;
    if (lo < 0) lo = 0;
    if (hi >= nk) hi = nk - 1;
    for (int k = lo; k <= hi; k++) {
      const double v = xgrid[k] - q[i];
      ulocal[k] += exp(-v * v * s2inv);
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

void FixMADXps::accumulate_bias(const double *dEdy)
{
  const double dx = xgrid[1] - xgrid[0];

  // Adjoint of the normalisation. For the L2 case
  //   dy_l/du_m = ( delta_lm - y_l y_m ) / ||u||
  // and the projection term is what a naive 1/||u|| misses; it is the whole
  // reason a normalised spectrum's gradient is not simply the unnormalised one
  // scaled.
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
  const double *q = prop->scalar_property();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double s2inv = 1.0 / (2.0 * sigma * sigma);
  const double sinv2 = 1.0 / (sigma * sigma);
  const int ntrunc = (int) (trunc * sigma / dx) + 2;

  for (int i = 0; i < nlocal; i++) {
    lambda[i][0] = 0.0;
    if (!(mask[i] & groupbit)) continue;
    int lo = (int) ((q[i] - emin) / dx) - ntrunc;
    int hi = (int) ((q[i] - emin) / dx) + ntrunc;
    if (lo < 0) lo = 0;
    if (hi >= nk) hi = nk - 1;
    double acc = 0.0;
    for (int k = lo; k <= hi; k++) {
      const double v = xgrid[k] - q[i];
      acc += dEdu[k] * v * sinv2 * exp(-v * v * s2inv);
    }
    lambda[i][0] = acc;
  }

  prop->adjoint(lambda, virial);
}
