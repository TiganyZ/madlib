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

#include "compute_mad_property_mliap.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"
#include "force.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#ifdef MADLIB_MLIAP
#include "mliap_data.h"
#include "mliap_descriptor.h"
#include "mliap_descriptor_snap.h"
#include "mliap_descriptor_so3.h"
#include "mliap_model.h"
#include "mliap_model_linear.h"
#include "mliap_model_nn.h"
#include "mliap_model_quadratic.h"
#endif

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyMliap::ComputeMADPropertyMliap(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), data(nullptr), model(nullptr), descriptor(nullptr),
    map(nullptr), nmax_q(0), built(-1), list(nullptr)
{
#ifndef MADLIB_MLIAP
  error->all(FLERR,
             "compute mad/property/mliap needs the ML-IAP package: build with -D PKG_ML-IAP=on");
#else
  ncomp = 1;
  size_peratom_cols = 0;

  int modelflag = 0, descriptorflag = 0;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 2 >= narg) utils::missing_cmd_args(FLERR, "compute mad/property/mliap model", error);
      if (strcmp(arg[iarg + 1], "linear") == 0)
        model = new MLIAPModelLinear(lmp, arg[iarg + 2]);
      else if (strcmp(arg[iarg + 1], "quadratic") == 0)
        model = new MLIAPModelQuadratic(lmp, arg[iarg + 2]);
      else if (strcmp(arg[iarg + 1], "nn") == 0)
        model = new MLIAPModelNN(lmp, arg[iarg + 2]);
      else
        error->all(FLERR, "Unknown compute mad/property/mliap model {}", arg[iarg + 1]);
      modelflag = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "descriptor") == 0) {
      if (iarg + 2 >= narg)
        utils::missing_cmd_args(FLERR, "compute mad/property/mliap descriptor", error);
      if (strcmp(arg[iarg + 1], "sna") == 0)
        descriptor = new MLIAPDescriptorSNAP(lmp, arg[iarg + 2]);
      else if (strcmp(arg[iarg + 1], "so3") == 0)
        descriptor = new MLIAPDescriptorSO3(lmp, arg[iarg + 2]);
      else
        error->all(FLERR, "Unknown compute mad/property/mliap descriptor {}", arg[iarg + 1]);
      descriptorflag = 1;
      iarg += 3;
    } else {
      error->all(FLERR, "Unknown compute mad/property/mliap keyword {}", arg[iarg]);
    }
  }

  if (!modelflag || !descriptorflag)
    error->all(FLERR, "compute mad/property/mliap needs both a model and a descriptor");

  model->set_ndescriptors(descriptor->ndescriptors);

  memory->create(map, atom->ntypes + 1, "mad/property/mliap:map");
  for (int i = 1; i <= atom->ntypes; i++) map[i] = i - 1;

  // gradgradflag = -1 is ML-IAP's "pair style" mode, the one that fills betas
  // and the per-atom values. The other two build objects for fitting, which is
  // a different question from the one asked here.
  data = new MLIAPData(lmp, -1, map, model, descriptor);

  if (comm->me == 0)
    utils::logmesg(lmp, "compute {}: mliap model with {} descriptors, cutoff {:.4g}\n", id,
                   descriptor->ndescriptors, descriptor->cutmax);
#endif
}

ComputeMADPropertyMliap::~ComputeMADPropertyMliap()
{
#ifdef MADLIB_MLIAP
  delete data;
  delete model;
  delete descriptor;
#endif
  memory->destroy(map);
  memory->destroy(vector_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyMliap::init()
{
#ifdef MADLIB_MLIAP
  if (force->pair == nullptr)
    error->all(FLERR, "compute mad/property/mliap requires a pair style, even pair_style zero");
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  // All three, in this order. Without descriptor->init() the SNAP tables are
  // never built and every descriptor comes back zero -- so the property is the
  // model's constant term for every atom, which looks like a model that simply
  // does not vary rather than like a missing call.
  model->init();
  descriptor->init();
  data->init();
#endif
}

void ComputeMADPropertyMliap::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyMliap::run_model()
{
#ifdef MADLIB_MLIAP
  neighbor->build_one(list);
  // eflag = 1, because the per-atom value only gets written when it is asked
  // for; a model computes betas regardless, but eatoms stays untouched.
  // vflag = 0, since the virial would be tallied through a PairMLIAP that this
  // compute deliberately does not have.
  data->generate_neighdata(list, 1, 0);
  descriptor->compute_descriptors(data);
  // Fills eatoms with the per-atom value and betas with its derivative with
  // respect to the descriptors.
  model->compute_gradients(data);
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyMliap::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyMliap::compute_property()
{
#ifdef MADLIB_MLIAP
  if (atom->nlocal > nmax_q) {
    memory->destroy(vector_atom);
    nmax_q = atom->nmax;
    memory->create(vector_atom, nmax_q, "mad/property/mliap:q");
  }
  for (int i = 0; i < atom->nlocal; i++) vector_atom[i] = 0.0;

  run_model();
  built = 1;

  for (int ii = 0; ii < data->nlistatoms; ii++) {
    const int i = data->iatoms[ii];
    if (i < atom->nlocal) vector_atom[i] = data->eatoms[ii];
  }
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyMliap::adjoint(double **lambda, double *virial)
{
#ifdef MADLIB_MLIAP
  if (built != 1)
    error->all(FLERR, "compute mad/property/mliap: adjoint called before the property");

  zero_adjoint();

  // ML-IAP contracts beta_ik with dB_ik/dr_j to make a force. Scaling the betas
  // by lambda_i first turns that same contraction into sum_i lambda_i dq_i/dr_j,
  // which is the whole of what the adjoint needs -- no second descriptor pass,
  // and no derivative written here by hand.
  for (int ii = 0; ii < data->nlistatoms; ii++) {
    const int i = data->iatoms[ii];
    const double w = (i < atom->nlocal) ? lambda[i][0] : 0.0;
    for (int k = 0; k < data->ndescriptors; k++) data->betas[ii][k] *= w;
  }

  // ML-IAP's descriptors take their force array from atom->f directly rather
  // than from data->f, so redirecting the latter achieves nothing. Point
  // atom->f at this class's own buffer for the duration of the call instead --
  // the same manoeuvre MGPT/pair_mgpt.cpp:1749 makes, and the only way to keep
  // this contribution separable. It has to be separable: the descriptor writes
  // to ghosts, and Verlet reverse-communicates before post_force(), so those
  // contributions would otherwise be lost.
  double **atom_f_save = atom->f;
  atom->f = fadj;
  descriptor->compute_forces(data);
  atom->f = atom_f_save;

  // ML-IAP returns -dq/dr already, so fadj is a force and needs no sign flip.
  double **x = atom->x;
  const int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
    virial[0] += fadj[i][0] * x[i][0];
    virial[1] += fadj[i][1] * x[i][1];
    virial[2] += fadj[i][2] * x[i][2];
    virial[3] += 0.5 * (fadj[i][0] * x[i][1] + fadj[i][1] * x[i][0]);
    virial[4] += 0.5 * (fadj[i][0] * x[i][2] + fadj[i][2] * x[i][0]);
    virial[5] += 0.5 * (fadj[i][1] * x[i][2] + fadj[i][2] * x[i][1]);
  }

  scatter_adjoint();
  built = -1;
#else
  (void) lambda;
  (void) virial;
#endif
}

/* ---------------------------------------------------------------------- */

double ComputeMADPropertyMliap::memory_usage()
{
  double n = ComputeMADProperty::memory_usage();
#ifdef MADLIB_MLIAP
  if (data) n += data->memory_usage();
#endif
  return n;
}
