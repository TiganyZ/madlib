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

#include "compute_mad_property_coord.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

static const double MY_PI = 3.14159265358979323846;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyCoord::ComputeMADPropertyCoord(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), nmax_q(0), list(nullptr)
{
  r1 = r2 = -1.0;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "cutoff") == 0) {
      if (iarg + 2 >= narg)
        utils::missing_cmd_args(FLERR, "compute mad/property/coord cutoff", error);
      r1 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      r2 = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else {
      error->all(FLERR, "Unknown compute mad/property/coord keyword {}", arg[iarg]);
    }
  }
  if (r2 <= r1 || r1 < 0.0)
    error->all(FLERR, "compute mad/property/coord requires cutoff <r1> <r2> with 0 <= r1 < r2");

  invdr = 1.0 / (r2 - r1);
  ncomp = 1;
  size_peratom_cols = 0;
}

ComputeMADPropertyCoord::~ComputeMADPropertyCoord()
{
  memory->destroy(vector_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyCoord::init()
{
  const double skin = neighbor->skin;
  double cutghost;
  if (force->pair)
    cutghost = MAX(force->pair->cutforce + skin, comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;

  if (r2 + skin > cutghost)
    error->all(FLERR,
               "compute mad/property/coord cutoff plus skin {} exceeds ghost atom range {} - use "
               "comm_modify cutoff command to increase it",
               r2 + skin, cutghost);

  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  req->set_cutoff_fixed(r2 + skin);
}

void ComputeMADPropertyCoord::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyCoord::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyCoord::compute_property()
{
  if (atom->nlocal > nmax_q) {
    memory->destroy(vector_atom);
    nmax_q = atom->nmax;
    memory->create(vector_atom, nmax_q, "mad/property/coord:q");
  }

  neighbor->build_one(list);

  double **x = atom->x;
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double r2sq = r2 * r2;

  for (int i = 0; i < nlocal; i++) vector_atom[i] = 0.0;

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const int *jlist = firstneigh[i];
    double q = 0.0;
    for (int jj = 0; jj < numneigh[i]; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq >= r2sq) continue;
      const double r = sqrt(rsq);
      q += (r <= r1) ? 1.0 : 0.5 * (1.0 + cos(MY_PI * (r - r1) * invdr));
    }
    vector_atom[i] = q;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyCoord::adjoint(double **lambda, double *virial)
{
  zero_adjoint();

  double **x = atom->x;
  const int *mask = atom->mask;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double r2sq = r2 * r2;

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const double li = lambda[i][0];
    if (li == 0.0) continue;
    const int *jlist = firstneigh[i];

    for (int jj = 0; jj < numneigh[i]; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq >= r2sq) continue;
      const double r = sqrt(rsq);
      if (r <= r1 || r == 0.0) continue;

      const double dfc = -0.5 * MY_PI * invdr * sin(MY_PI * (r - r1) * invdr);
      // f = -dE/dr with E carrying lambda_i * fc(r_ij); the pair is equal and
      // opposite, so j takes the negative and the virial is the pair form.
      const double fpair = -li * dfc / r;

      fadj[i][0] += fpair * dx;
      fadj[i][1] += fpair * dy;
      fadj[i][2] += fpair * dz;
      fadj[j][0] -= fpair * dx;
      fadj[j][1] -= fpair * dy;
      fadj[j][2] -= fpair * dz;

      // No halving: the ordered visits (i,j) and (j,i) carry lambda_i and
      // lambda_j, which are different terms rather than the same one twice.
      virial[0] += fpair * dx * dx;
      virial[1] += fpair * dy * dy;
      virial[2] += fpair * dz * dz;
      virial[3] += fpair * dx * dy;
      virial[4] += fpair * dx * dz;
      virial[5] += fpair * dy * dz;
    }
  }

  scatter_adjoint();
}
