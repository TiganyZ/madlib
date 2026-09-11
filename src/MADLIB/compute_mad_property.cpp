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

#include "compute_mad_property.h"

#include "atom.h"
#include "comm.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADProperty::ComputeMADProperty(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), ncomp(1), nmax_adj(0), fadj(nullptr)
{
  peratom_flag = 1;
  comm_reverse = 3;
}

ComputeMADProperty::~ComputeMADProperty()
{
  memory->destroy(fadj);
}

/* ---------------------------------------------------------------------- */

void ComputeMADProperty::grow_adjoint()
{
  const int n = atom->nlocal + atom->nghost;
  if (n > nmax_adj) {
    memory->destroy(fadj);
    nmax_adj = atom->nmax + atom->nghost;
    memory->create(fadj, nmax_adj, 3, "mad/property:fadj");
  }
}

void ComputeMADProperty::zero_adjoint()
{
  grow_adjoint();
  const int n = atom->nlocal + atom->nghost;
  for (int i = 0; i < n; i++) fadj[i][0] = fadj[i][1] = fadj[i][2] = 0.0;
}

void ComputeMADProperty::scatter_adjoint()
{
  // Unconditionally, not only when nprocs > 1. A single rank with periodic
  // boundaries still has ghosts, and their share of the adjoint has to be
  // folded back onto the atoms that own them. Guarding this on the rank count
  // leaves the serial force wrong by a constant -- which a finite-difference
  // h-scan shows as a flat curve, and a single tolerance would not.
  comm->reverse_comm(this);
  double **f = atom->f;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    f[i][0] += fadj[i][0];
    f[i][1] += fadj[i][1];
    f[i][2] += fadj[i][2];
  }
}

/* ---------------------------------------------------------------------- */

int ComputeMADProperty::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; i++) {
    buf[m++] = fadj[i][0];
    buf[m++] = fadj[i][1];
    buf[m++] = fadj[i][2];
  }
  return m;
}

void ComputeMADProperty::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    fadj[j][0] += buf[m++];
    fadj[j][1] += buf[m++];
    fadj[j][2] += buf[m++];
  }
}

double ComputeMADProperty::memory_usage()
{
  return (double) nmax_adj * 3 * sizeof(double);
}
