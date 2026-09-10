/* ----------------------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#include "compute_mad_property_dipole.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyDipole::ComputeMADPropertyDipole(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), bytype(nullptr), use_atom_q(0), molecular_mode(0),
    nmax_q(0), charge_tol(1.0e-6)
{
  ncomp = 3;
  size_peratom_cols = 3;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "charges") == 0) {
      const int nt = atom->ntypes;
      if (iarg + nt >= narg) utils::missing_cmd_args(FLERR, "compute mad/property/dipole charges", error);
      memory->create(bytype, nt + 1, "mad/property/dipole:bytype");
      for (int t = 1; t <= nt; t++) bytype[t] = utils::numeric(FLERR, arg[iarg + t], false, lmp);
      iarg += nt + 1;
    } else if (strcmp(arg[iarg], "charge") == 0) {
      use_atom_q = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg], "neutral_tol") == 0) {
      charge_tol = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown compute mad/property/dipole keyword {}", arg[iarg]);
    }
  }

  if (!bytype && !use_atom_q)
    error->all(FLERR, "compute mad/property/dipole needs 'charges <per type>' or 'charge'");
  if (use_atom_q && !atom->q_flag)
    error->all(FLERR, "compute mad/property/dipole 'charge' needs an atom style with charges");
}

ComputeMADPropertyDipole::~ComputeMADPropertyDipole()
{
  memory->destroy(bytype);
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

double ComputeMADPropertyDipole::charge(int i) const
{
  return use_atom_q ? atom->q[i] : bytype[atom->type[i]];
}

void ComputeMADPropertyDipole::init()
{
  const int periodic = domain->xperiodic || domain->yperiodic || domain->zperiodic;
  molecular_mode = 0;

  if (periodic) {
    if (atom->molecule_flag) {
      molecular_mode = 1;
    } else {
      error->all(FLERR,
                 "compute mad/property/dipole: sum q_i r_i is not defined in a periodic box. Use "
                 "an atom style with molecule IDs, so that molecular dipoles can be built from "
                 "intramolecular vectors, or run a non-periodic box");
    }
  }
  check_neutrality();
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyDipole::check_neutrality()
{
  const int nlocal = atom->nlocal;
  const int *mask = atom->mask;

  if (!molecular_mode) {
    double qsum = 0.0, all;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) qsum += charge(i);
    MPI_Allreduce(&qsum, &all, 1, MPI_DOUBLE, MPI_SUM, world);
    if (fabs(all) > charge_tol)
      error->all(FLERR,
                 "compute mad/property/dipole: the group carries a net charge of {:.6g}, so its "
                 "dipole depends on where the origin is",
                 all);
    return;
  }

  // Each molecule must be neutral, or its dipole moves with the reference atom
  // the intramolecular vectors are built from.
  const tagint *molecule = atom->molecule;
  std::map<tagint, double> q;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) q[molecule[i]] += charge(i);

  double worst = 0.0;
  tagint worst_id = 0;
  for (auto &kv : q)
    if (fabs(kv.second) > fabs(worst)) { worst = kv.second; worst_id = kv.first; }

  // Molecules can straddle ranks, so the per-rank partials are not the answer;
  // reduce the extreme and let one rank speak.
  struct { double v; int r; } in{fabs(worst), comm->me}, out;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, world);
  double check = 0.0;
  if (comm->me == out.r) check = worst;
  MPI_Bcast(&check, 1, MPI_DOUBLE, out.r, world);

  if (fabs(check) > charge_tol && comm->nprocs == 1)
    error->all(FLERR,
               "compute mad/property/dipole: molecule {} carries a net charge of {:.6g}; a "
               "charged molecule has no origin-independent dipole",
               worst_id, check);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyDipole::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyDipole::compute_property()
{
  if (atom->nlocal > nmax_q) {
    memory->destroy(array_atom);
    nmax_q = atom->nmax;
    memory->create(array_atom, nmax_q, 3, "mad/property/dipole:mu");
  }

  double **x = atom->x;
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) array_atom[i][0] = array_atom[i][1] = array_atom[i][2] = 0.0;

  if (!molecular_mode) {
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      const double q = charge(i);
      for (int d = 0; d < 3; d++) array_atom[i][d] = q * x[i][d];
    }
    return;
  }

  // Each atom's dipole is measured from the lowest-TAGGED atom of its own
  // molecule, reached by the closest image. Summed over a neutral molecule
  // that reference cancels, so the total is the sum of molecular dipoles and
  // does not move with the box origin.
  //
  // The anchor is chosen over local AND ghost atoms. Choosing it among the
  // local ones would make it whichever fragment of the molecule this rank
  // happens to hold, so a molecule split across two ranks would be measured
  // from two different origins and its dipole would come out as zero -- a
  // rank-count-dependent answer with nothing to announce it. Ghosts are what
  // makes the choice the same everywhere, and a bonded molecule is always
  // within the communication cutoff.
  const tagint *molecule = atom->molecule;
  const tagint *tag = atom->tag;
  const int nall = nlocal + atom->nghost;

  std::map<tagint, int> anchor;
  for (int i = 0; i < nall; i++) {
    if (i < nlocal && !(mask[i] & groupbit)) continue;
    auto it = anchor.find(molecule[i]);
    if (it == anchor.end() || tag[i] < tag[it->second]) anchor[molecule[i]] = i;
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const int a = anchor[molecule[i]];
    const int j = domain->closest_image(a, i);
    const double q = charge(i);
    for (int d = 0; d < 3; d++) array_atom[i][d] = q * (x[j][d] - x[a][d]);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyDipole::adjoint(double **lambda, double *virial)
{
  // d mu_(i,a) / d r_(i,b) = q_i delta_ab, with the anchor term cancelling
  // over a neutral molecule. Prediction does not use this; it is here so an
  // IR bias built on a charge model has a gradient rather than a surprise.
  zero_adjoint();

  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  double **x = atom->x;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double q = charge(i);
    for (int d = 0; d < 3; d++) {
      const double f = -lambda[i][d] * q;
      fadj[i][d] += f;
      virial[d] += f * x[i][d];
    }
  }
  scatter_adjoint();
}
