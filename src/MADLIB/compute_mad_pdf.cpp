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

#include "compute_mad_pdf.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
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

static const double MY_4PI = 12.566370614359172;
static const double MY_SQRT2PI = 2.5066282746310002;

/* ---------------------------------------------------------------------- */

ComputeMADPdf::ComputeMADPdf(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), invoked_types(-1), chan(nullptr), count(nullptr), conc(nullptr),
    rgrid(nullptr), shellinv(nullptr), cnorm(nullptr), gpart(nullptr), glocal(nullptr),
    list(nullptr)
{
  rmin = 0.0;
  rmax = rcut = -1.0;
  sigma = 0.0;
  nbins = 0;
  trunc = 8.0;
  compat = false;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "r") == 0) {
      if (iarg + 3 >= narg) utils::missing_cmd_args(FLERR, "compute mad/pdf r", error);
      rmin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      rmax = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      nbins = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;
    } else if (strcmp(arg[iarg], "rcut") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/pdf rcut", error);
      rcut = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "kde") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/pdf kde", error);
      sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "trunc") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/pdf trunc", error);
      trunc = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (trunc < 3.0) error->all(FLERR, "compute mad/pdf trunc below 3 sigma is too aggressive");
      iarg += 2;
    } else if (strcmp(arg[iarg], "compat") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/pdf compat", error);
      compat = (strcmp(arg[iarg + 1], "turbogap") == 0);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown compute mad/pdf keyword {}", arg[iarg]);
    }
  }

  if (nbins < 2) error->all(FLERR, "compute mad/pdf requires r <rmin> <rmax> <nbins>, nbins >= 2");
  if (rmax <= rmin) error->all(FLERR, "compute mad/pdf rmax must exceed rmin");

  // A histogram is not differentiable, so a bias built on one is silently
  // inert. TurboGAP defaults this to zero and says nothing; refuse instead.
  if (sigma <= 0.0)
    error->all(FLERR,
               "compute mad/pdf requires a positive kde width: a histogram has no gradient, "
               "so any bias built on it would apply no force");
  if (sigma > 0.25 * (rmax - rmin))
    error->warning(FLERR, "compute mad/pdf kde width is a large fraction of the r range");

  // A pair crossing rcut takes a finite bite out of the pattern, because the
  // Gaussian it contributes to the bins just inside rmax does not vanish
  // there. That discontinuity is invisible in g(r) and lethal to the virial.
  // Placing the cut trunc*sigma beyond rmax makes the largest dropped
  // contribution exp(-trunc^2/2), which at the default is 1e-14.
  const double rsafe = rmax + trunc * sigma;
  if (rcut < 0.0)
    rcut = rsafe;
  else if (rcut < rsafe - 1.0e-12)
    error->warning(FLERR,
                   "compute mad/pdf rcut {} is inside rmax + trunc*sigma = {}; pairs crossing "
                   "it step the pattern, which shows up as a wrong virial",
                   rcut, rsafe);

  ntypes = atom->ntypes;
  nchan = ntypes * (ntypes + 1) / 2;

  array_flag = 1;
  size_array_rows = nbins;
  size_array_cols = nchan + 1;
  extarray = 0;

  memory->create(chan, (ntypes + 1) * (ntypes + 1), "mad/pdf:chan");
  memory->create(count, ntypes + 1, "mad/pdf:count");
  memory->create(conc, ntypes + 1, "mad/pdf:conc");
  memory->create(rgrid, nbins, "mad/pdf:rgrid");
  memory->create(shellinv, nbins, "mad/pdf:shellinv");
  memory->create(cnorm, nchan, "mad/pdf:cnorm");
  memory->create(gpart, (bigint) nbins * nchan, "mad/pdf:gpart");
  memory->create(glocal, (bigint) nbins * nchan, "mad/pdf:glocal");
  memory->create(array, nbins, nchan + 1, "mad/pdf:array");

  // Upper triangle, row-major: (1,1) (1,2) ... (1,n) (2,2) ...
  int c = 0;
  for (int a = 1; a <= ntypes; a++)
    for (int b = a; b <= ntypes; b++) {
      chan[a * (ntypes + 1) + b] = c;
      chan[b * (ntypes + 1) + a] = c;
      c++;
    }

  setup_grid();
}

/* ---------------------------------------------------------------------- */

ComputeMADPdf::~ComputeMADPdf()
{
  memory->destroy(chan);
  memory->destroy(count);
  memory->destroy(conc);
  memory->destroy(rgrid);
  memory->destroy(shellinv);
  memory->destroy(cnorm);
  memory->destroy(gpart);
  memory->destroy(glocal);
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPdf::setup_grid()
{
  delr = (rmax - rmin) / nbins;
  delrinv = 1.0 / delr;
  for (int l = 0; l < nbins; l++) {
    // Bin centre. TurboGAP evaluates the shell volume at the bin's lower edge
    // while placing the sample at its centre; compat reproduces that.
    rgrid[l] = rmin + (l + 0.5) * delr;
    const double rs = compat ? (rmin + l * delr) : rgrid[l];
    shellinv[l] = (rs > 0.0) ? 1.0 / (MY_SQRT2PI * sigma * MY_4PI * rs * rs) : 0.0;
  }
  // Gaussian tails past trunc sigma are dropped. The default 8 puts the
  // truncation error at exp(-32) ~ 1e-14 of the peak, below double round-off
  // on an O(1) g(r), so it is not a source of disagreement with a reference
  // that does not truncate. The cost is linear in it.
  ntrunc = (int) (trunc * sigma * delrinv) + 2;
}

/* ---------------------------------------------------------------------- */

void ComputeMADPdf::init()
{
  const double skin = neighbor->skin;
  double cutghost;
  if (force->pair)
    cutghost = MAX(force->pair->cutforce + skin, comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;

  if (rcut + skin > cutghost)
    error->all(FLERR,
               "compute mad/pdf cutoff plus skin {} exceeds ghost atom range {} - use "
               "comm_modify cutoff command to increase it",
               rcut + skin, cutghost);

  if ((neighbor->style == Neighbor::MULTI) || (neighbor->style == Neighbor::MULTI_OLD))
    error->all(FLERR, "compute mad/pdf requires neighbor style 'bin' or 'nsq'");

  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  req->set_cutoff(rcut + skin);
}

void ComputeMADPdf::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeMADPdf::count_types()
{
  // Deliberately not cached on the timestep. A grand-canonical move changes
  // the counts and a volume move changes V, both within a step, and an
  // Allreduce of ntypes integers is nothing beside the pair loop.
  int *nlocal_count = new int[ntypes + 1]();
  const int *type = atom->type;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) nlocal_count[type[i]]++;
  MPI_Allreduce(nlocal_count, count, ntypes + 1, MPI_INT, MPI_SUM, world);
  delete[] nlocal_count;

  bigint ntot = 0;
  for (int a = 1; a <= ntypes; a++) ntot += count[a];
  for (int a = 1; a <= ntypes; a++) conc[a] = ntot ? (double) count[a] / (double) ntot : 0.0;

  const double vol = domain->xprd * domain->yprd * domain->zprd;
  for (int a = 1; a <= ntypes; a++)
    for (int b = a; b <= ntypes; b++) {
      const int c = chan[a * (ntypes + 1) + b];
      const double fab = (a == b) ? 1.0 : 2.0;
      const double na = (double) count[a], nb = (double) count[b];
      cnorm[c] = (na > 0.0 && nb > 0.0) ? vol / (na * nb * fab) : 0.0;
    }
}

/* ---------------------------------------------------------------------- */

void ComputeMADPdf::compute_array()
{
  invoked_array = update->ntimestep;
  count_types();

  neighbor->build_one(list);

  const bigint n = (bigint) nbins * nchan;
  memset(glocal, 0, n * sizeof(double));

  const double **x = (const double **) atom->x;
  const int *type = atom->type;
  const int *mask = atom->mask;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double rcutsq = rcut * rcut;
  const double s2inv = 1.0 / (2.0 * sigma * sigma);

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= rcutsq || rsq == 0.0) continue;
      const double r = sqrt(rsq);

      const int c = chan[type[i] * (ntypes + 1) + type[j]];
      double *acc = &glocal[(bigint) c * nbins];

      int lo = (int) ((r - rmin) * delrinv) - ntrunc;
      int hi = (int) ((r - rmin) * delrinv) + ntrunc;
      if (lo < 0) lo = 0;
      if (hi >= nbins) hi = nbins - 1;
      for (int l = lo; l <= hi; l++) {
        const double u = rgrid[l] - r;
        acc[l] += exp(-u * u * s2inv);
      }
    }
  }

  MPI_Allreduce(glocal, gpart, n, MPI_DOUBLE, MPI_SUM, world);

  for (int c = 0; c < nchan; c++) {
    double *gc = &gpart[(bigint) c * nbins];
    for (int l = 0; l < nbins; l++) gc[l] *= cnorm[c] * shellinv[l];
  }

  for (int l = 0; l < nbins; l++) {
    array[l][0] = rgrid[l];
    for (int c = 0; c < nchan; c++) array[l][c + 1] = gpart[(bigint) c * nbins + l];
  }
}

/* ----------------------------------------------------------------------
   dEdg is (nchan x nbins), channel-major, matching g().
------------------------------------------------------------------------- */

void ComputeMADPdf::adjoint(const double *dEdg, double *virial)
{
  const double **x = (const double **) atom->x;
  double **f = atom->f;
  const int *type = atom->type;
  const int *mask = atom->mask;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double rcutsq = rcut * rcut;
  const double s2inv = 1.0 / (2.0 * sigma * sigma);
  const double sinv2 = 1.0 / (sigma * sigma);

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;

      const double delx = x[i][0] - x[j][0];
      const double dely = x[i][1] - x[j][1];
      const double delz = x[i][2] - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= rcutsq || rsq == 0.0) continue;
      const double r = sqrt(rsq);

      const int c = chan[type[i] * (ntypes + 1) + type[j]];
      const double *w = &dEdg[(bigint) c * nbins];

      int lo = (int) ((r - rmin) * delrinv) - ntrunc;
      int hi = (int) ((r - rmin) * delrinv) + ntrunc;
      if (lo < 0) lo = 0;
      if (hi >= nbins) hi = nbins - 1;

      double s = 0.0;
      for (int l = lo; l <= hi; l++) {
        const double u = rgrid[l] - r;
        s += w[l] * cnorm[c] * shellinv[l] * u * sinv2 * exp(-u * u * s2inv);
      }
      if (s == 0.0) continue;

      // The unordered pair enters g twice, and this ordered visit carries the
      // whole of dE/dx_i for it. The other visit, from j, carries dE/dx_j, so
      // nothing is written to j and no reverse communication is needed.
      const double fpair = -2.0 * s / r;
      f[i][0] += fpair * delx;
      f[i][1] += fpair * dely;
      f[i][2] += fpair * delz;

      // Halved because each unordered pair is visited from both ends.
      const double h = 0.5 * fpair;
      virial[0] += h * delx * delx;
      virial[1] += h * dely * dely;
      virial[2] += h * delz * delz;
      virial[3] += h * delx * dely;
      virial[4] += h * delx * delz;
      virial[5] += h * dely * delz;
    }
  }

  // g_ab carries an explicit factor of V, so a homogeneous strain moves the
  // prediction through the volume as well as through the distances and
  // sum_a f_a (x) r_a cannot see it. The term belongs to the cell rather than
  // to any pair, so exactly one rank may add it.
  if (comm->me == 0) {
    double dedv = 0.0;
    for (int c = 0; c < nchan; c++)
      for (int l = 0; l < nbins; l++)
        dedv += dEdg[(bigint) c * nbins + l] * gpart[(bigint) c * nbins + l];
    virial[0] -= dedv;
    virial[1] -= dedv;
    virial[2] -= dedv;
  }
}

/* ---------------------------------------------------------------------- */

double ComputeMADPdf::memory_usage()
{
  return (double) (2.0 * nbins * nchan + nbins * (nchan + 1)) * sizeof(double);
}
