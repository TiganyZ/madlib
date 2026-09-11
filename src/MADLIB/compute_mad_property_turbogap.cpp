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

#include "compute_mad_property_turbogap.h"

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
#include <stdexcept>

using namespace LAMMPS_NS;

#ifdef MADLIB_TURBOGAP
extern "C" void
madlib_local_property(int n_sites, int n_pairs, int n_species, const int *n_neigh,
                      const int *species_idx, const double *rjs, const double *thetas,
                      const double *phis, const int *alpha_max, int l_max, const double *rcut_hard,
                      const double *rcut_soft, const double *nf, const double *global_scaling,
                      const double *asr, const double *asrs, const double *ast, const double *asts,
                      const double *amp, int radial_enhancement, const double *cw,
                      const char *basis, const char *scaling_mode, int compress, int n_nonzero,
                      const int *p_i, const int *p_j, const double *p_el, int dim, int n_sparse,
                      const double *qs, const double *alphas, double v0, double delta, double zeta,
                      int do_derivatives, double *v, double *v_der);
#endif

/* ---------------------------------------------------------------------- */

ComputeMADPropertyTurbogap::ComputeMADPropertyTurbogap(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), rcut(0.0), rcutsq(0.0), nmax_q(0), built(-1), list(nullptr)
{
#ifndef MADLIB_TURBOGAP
  error->all(FLERR,
             "compute mad/property/turbogap needs the Fortran backend: build lib/madlib and "
             "configure with -D MADLIB_ENABLE_TURBOGAP=on");
#else
  if (narg < 5) utils::missing_cmd_args(FLERR, "compute mad/property/turbogap", error);

  const std::string gapfile = arg[3];
  const std::string label = arg[4];
  std::vector<std::string> labels;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "species") == 0) {
      const int nt = atom->ntypes;
      if (iarg + nt >= narg)
        utils::missing_cmd_args(FLERR, "compute mad/property/turbogap species", error);
      for (int t = 0; t < nt; t++) labels.emplace_back(arg[iarg + 1 + t]);
      iarg += nt + 1;
    } else {
      error->all(FLERR, "Unknown compute mad/property/turbogap keyword {}", arg[iarg]);
    }
  }
  if ((int) labels.size() != atom->ntypes)
    error->all(FLERR, "compute mad/property/turbogap needs 'species' with one label per atom type");

  try {
    blocks = MADGapFile::read(gapfile, label);
  } catch (std::exception &e) {
    error->all(FLERR, e.what());
  }

  // Every block must agree about what the species list means, since one
  // neighbour list feeds all of them.
  type2species.assign(atom->ntypes + 1, 0);
  for (int t = 1; t <= atom->ntypes; t++) {
    for (size_t s = 0; s < blocks[0].species.size(); s++)
      if (blocks[0].species[s] == labels[t - 1]) type2species[t] = (int) s + 1;
    if (type2species[t] == 0)
      error->all(FLERR,
                 "compute mad/property/turbogap: atom type {} is '{}', which the potential "
                 "does not define",
                 t, labels[t - 1]);
  }
  for (auto &b : blocks) {
    if (b.species != blocks[0].species)
      error->all(FLERR,
                 "compute mad/property/turbogap: the soap_turbo blocks disagree about the "
                 "species list");
    rcut = MAX(rcut, b.rcut_max());
  }
  rcutsq = rcut * rcut;

  ncomp = 1;
  size_peratom_cols = 0;
  batch.resize(blocks.size());

  if (comm->me == 0)
    utils::logmesg(lmp, "compute {}: {} soap_turbo block(s) for '{}', cutoff {:.4g} A\n", id,
                   blocks.size(), label, rcut);
#endif
}

ComputeMADPropertyTurbogap::~ComputeMADPropertyTurbogap()
{
  memory->destroy(vector_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTurbogap::init()
{
  const double skin = neighbor->skin;
  double cutghost;
  if (force->pair)
    cutghost = MAX(force->pair->cutforce + skin, comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;

  if (rcut + skin > cutghost)
    error->all(FLERR,
               "compute mad/property/turbogap cutoff plus skin {} exceeds ghost atom range {} - "
               "use comm_modify cutoff command to increase it",
               rcut + skin, cutghost);

  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  req->set_cutoff_fixed(rcut + skin);
}

void ComputeMADPropertyTurbogap::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   soap_turbo wants one flat list per block, in spherical coordinates about
   each central atom, with the atom itself as the first pair of its own run.
   That self pair is where dq_i/dr_i lands, so it is not optional.
------------------------------------------------------------------------- */

void ComputeMADPropertyTurbogap::build_batches(bool derivatives)
{
  neighbor->build_one(list);

  double **x = atom->x;
  const int *type = atom->type;
  const int *mask = atom->mask;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (size_t ib = 0; ib < blocks.size(); ib++) {
    Batch &B = batch[ib];
    const int cs = blocks[ib].central_species;
    const double rc = blocks[ib].rcut_max();
    const double rcsq = rc * rc;

    B.n_neigh.clear();
    B.species_idx.clear();
    B.jatom.clear();
    B.site_atom.clear();
    B.rjs.clear();
    B.thetas.clear();
    B.phis.clear();
    B.xyz.clear();

    for (int ii = 0; ii < list->inum; ii++) {
      const int i = ilist[ii];
      if (!(mask[i] & groupbit)) continue;
      if (type2species[type[i]] != cs) continue;

      B.site_atom.push_back(i);
      int cnt = 1;
      B.species_idx.push_back(cs);
      B.rjs.push_back(0.0);
      B.thetas.push_back(0.0);
      B.phis.push_back(0.0);
      B.jatom.push_back(i);
      B.xyz.insert(B.xyz.end(), {0.0, 0.0, 0.0});

      const int *jlist = firstneigh[i];
      for (int jj = 0; jj < numneigh[i]; jj++) {
        const int j = jlist[jj] & NEIGHMASK;
        if (!(mask[j] & groupbit)) continue;
        const double dx = x[j][0] - x[i][0];
        const double dy = x[j][1] - x[i][1];
        const double dz = x[j][2] - x[i][2];
        const double rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= rcsq || rsq < 1.0e-20) continue;
        const double r = sqrt(rsq);
        B.species_idx.push_back(type2species[type[j]]);
        B.rjs.push_back(r);
        B.thetas.push_back(acos(dz / r));
        B.phis.push_back(atan2(dy, dx));
        B.jatom.push_back(j);
        B.xyz.insert(B.xyz.end(), {dx, dy, dz});
        cnt++;
      }
      B.n_neigh.push_back(cnt);
    }

    const int ns = (int) B.n_neigh.size(), np = (int) B.rjs.size();
    B.q.assign(ns ? ns : 1, 0.0);
    B.qder.assign(derivatives ? 3 * (np ? np : 1) : 3, 0.0);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTurbogap::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyTurbogap::compute_property()
{
#ifdef MADLIB_TURBOGAP
  if (atom->nlocal > nmax_q) {
    memory->destroy(vector_atom);
    nmax_q = atom->nmax;
    memory->create(vector_atom, nmax_q, "mad/property/turbogap:q");
  }
  for (int i = 0; i < atom->nlocal; i++) vector_atom[i] = 0.0;

  // The derivatives cost a second pass through the descriptor, so they are
  // only asked for when something is going to contract them.
  build_batches(true);
  built = 1;

  for (size_t ib = 0; ib < blocks.size(); ib++) {
    const MADGapBlock &b = blocks[ib];
    Batch &B = batch[ib];
    const int ns = (int) B.n_neigh.size(), np = (int) B.rjs.size();
    if (ns == 0) continue;

    madlib_local_property(
        ns, np, b.n_species, B.n_neigh.data(), B.species_idx.data(), B.rjs.data(), B.thetas.data(),
        B.phis.data(), b.alpha_max.data(), b.l_max, b.rcut_hard.data(), b.rcut_soft.data(),
        b.nf.data(), b.global_scaling.data(), b.atom_sigma_r.data(), b.atom_sigma_r_scaling.data(),
        b.atom_sigma_t.data(), b.atom_sigma_t_scaling.data(), b.amplitude_scaling.data(),
        b.radial_enhancement, b.central_weight.data(), b.basis.c_str(), b.scaling_mode.c_str(),
        b.compress ? 1 : 0, b.n_nonzero, b.p_i.data(), b.p_j.data(), b.p_el.data(), b.dim,
        b.n_sparse, b.qs.data(), b.alphas.data(), b.v0, b.delta, b.zeta, 1, B.q.data(),
        B.qder.data());

    for (int s = 0; s < ns; s++) vector_atom[B.site_atom[s]] = B.q[s];
  }
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTurbogap::adjoint(double **lambda, double *virial)
{
  if (built != 1)
    error->all(FLERR, "compute mad/property/turbogap: adjoint called before the property");

  zero_adjoint();

  for (size_t ib = 0; ib < blocks.size(); ib++) {
    Batch &B = batch[ib];
    const int ns = (int) B.n_neigh.size();
    int k = 0;
    for (int s = 0; s < ns; s++) {
      const double li = lambda[B.site_atom[s]][0];
      for (int n = 0; n < B.n_neigh[s]; n++, k++) {
        if (li == 0.0) continue;
        const int j = B.jatom[k];
        // local_property_predict returns a derivative, not a force, so the
        // minus sign is the caller's.
        const double fx = -li * B.qder[3 * k + 0];
        const double fy = -li * B.qder[3 * k + 1];
        const double fz = -li * B.qder[3 * k + 2];
        fadj[j][0] += fx;
        fadj[j][1] += fy;
        fadj[j][2] += fz;

        // sum_j (dq_i/dr_j) r_j = sum_j (dq_i/dr_j) (r_j - r_i), because q is
        // translationally invariant, so the pair separation is the right lever
        // arm and the self pair contributes nothing.
        const double *d = &B.xyz[3 * k];
        virial[0] += fx * d[0];
        virial[1] += fy * d[1];
        virial[2] += fz * d[2];
        virial[3] += 0.5 * (fx * d[1] + fy * d[0]);
        virial[4] += 0.5 * (fx * d[2] + fz * d[0]);
        virial[5] += 0.5 * (fy * d[2] + fz * d[1]);
      }
    }
  }

  scatter_adjoint();
}

/* ---------------------------------------------------------------------- */

double ComputeMADPropertyTurbogap::memory_usage()
{
  double n = ComputeMADProperty::memory_usage();
  for (auto &B : batch)
    n += (double) (B.qder.capacity() + B.xyz.capacity() + B.rjs.capacity() * 3) * sizeof(double);
  return n;
}
