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

#include "compute_mad_property_python.h"

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

#include <cstring>
#include <string>

#ifdef MADLIB_PYTHON
#include <Python.h>
#define PY(x) ((PyObject *) (x))
#endif

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyPython::ComputeMADPropertyPython(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), rcut(-1.0), nmax_q(0), pbc_mode(0), list(nullptr),
    py_module(nullptr), py_compute(nullptr), py_adjoint(nullptr)
{
#ifndef MADLIB_PYTHON
  error->all(FLERR,
             "compute mad/property/python needs the embedded interpreter: configure with "
             "-D MADLIB_ENABLE_PYTHON=on");
#else
  if (narg < 4) utils::missing_cmd_args(FLERR, "compute mad/property/python", error);
  module_name = arg[3];

  ncomp = 1;
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "ncomp") == 0) {
      ncomp = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "cutoff") == 0) {
      rcut = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "config") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/property/python config", error);
      config = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "pbc") == 0) {
      if (strcmp(arg[iarg + 1], "yes") == 0)
        pbc_mode = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        pbc_mode = 0;
      else
        error->all(FLERR, "compute mad/property/python pbc must be yes or no");
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown compute mad/property/python keyword {}", arg[iarg]);
    }
  }
  if (ncomp < 1) error->all(FLERR, "compute mad/property/python ncomp must be positive");
  if (rcut <= 0.0 && !pbc_mode)
    error->all(FLERR, "compute mad/property/python requires a positive cutoff");

  // Handing the model one periodic cell means the cell has to be the whole
  // system, so there is nothing to decompose and nothing outside the group.
  if (pbc_mode) {
    if (comm->nprocs > 1)
      error->all(FLERR,
                 "compute mad/property/python pbc yes passes one periodic cell to the model, so "
                 "it runs on one rank only; drop it and give a cutoff reaching the model's "
                 "interaction range instead");
    if (igroup != 0)
      error->all(FLERR, "compute mad/property/python pbc yes needs the all group");
  }
  for (int i = 0; i < 9; i++) cellbuf[i] = 0.0;

  size_peratom_cols = (ncomp == 1) ? 0 : ncomp;

  // LAMMPS may already be running an interpreter, and finalising one that
  // somebody else started would take their state with it, so this never calls
  // Py_Finalize.
  if (!Py_IsInitialized()) Py_Initialize();

  // The model file normally sits beside the input deck, but an embedded
  // interpreter only gets the working directory on sys.path when PYTHONPATH is
  // unset -- so setting PYTHONPATH to reach numpy loses the model.
  PyRun_SimpleString("import os, sys\n"
                     "if os.getcwd() not in sys.path: sys.path.insert(0, os.getcwd())\n");

  PyObject *mod = PyImport_ImportModule(module_name.c_str());
  if (!mod) fail("could not import the module");
  py_module = mod;

  if (PyObject_HasAttrString(mod, "mad_bind")) {
    PyObject *bind = PyObject_GetAttrString(mod, "mad_bind");
    PyObject *bound = PyObject_CallFunction(bind, "s", config.c_str());
    Py_XDECREF(bind);
    if (!bound) fail("mad_bind raised an exception in");
    if (!PySequence_Check(bound) || PySequence_Size(bound) != 2) {
      Py_DECREF(bound);
      fail("mad_bind must return (compute, adjoint) in");
    }
    py_compute = PySequence_GetItem(bound, 0);
    py_adjoint = PySequence_GetItem(bound, 1);
    Py_DECREF(bound);
  } else {
    // Accepting a config a module cannot read would leave the model on
    // whatever its import-time default is, and it would still return numbers.
    if (!config.empty())
      error->all(FLERR,
                 "compute mad/property/python: module '{}' has no mad_bind, so it cannot be given "
                 "a config string",
                 module_name);
    py_compute = PyObject_GetAttrString(mod, "mad_compute");
    if (!py_compute) fail("the module has no mad_compute");
    py_adjoint = PyObject_GetAttrString(mod, "mad_adjoint");
    if (!py_adjoint) fail("the module has no mad_adjoint");
  }

  if (comm->me == 0)
    utils::logmesg(lmp, "compute {}: python model '{}'{}, {} component(s), {}\n", id, module_name,
                   config.empty() ? std::string("") : fmt::format(" config '{}'", config), ncomp,
                   pbc_mode ? std::string("one periodic cell") :
                              fmt::format("cutoff {:.4g}", rcut));
#endif
}

ComputeMADPropertyPython::~ComputeMADPropertyPython()
{
#ifdef MADLIB_PYTHON
  Py_XDECREF(PY(py_adjoint));
  Py_XDECREF(PY(py_compute));
  Py_XDECREF(PY(py_module));
#endif
  if (ncomp == 1)
    memory->destroy(vector_atom);
  else
    memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyPython::fail(const char *what)
{
#ifdef MADLIB_PYTHON
  // The Python traceback says far more than the return code, so print it
  // before the LAMMPS error swallows the run.
  if (PyErr_Occurred()) PyErr_Print();
  error->all(FLERR, "compute mad/property/python: {} '{}'", what, module_name);
#else
  (void) what;
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyPython::init()
{
  if (pbc_mode) {
    // here rather than in the constructor: change_box can make a box
    // non-periodic after the compute is defined.
    if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
      error->all(FLERR, "compute mad/property/python pbc yes needs a periodic box");
    return;
  }

  const double skin = neighbor->skin;
  double cutghost;
  if (force->pair)
    cutghost = MAX(force->pair->cutforce + skin, comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;

  if (rcut + skin > cutghost)
    error->all(FLERR,
               "compute mad/property/python cutoff plus skin {} exceeds ghost atom range {} - use "
               "comm_modify cutoff command to increase it",
               rcut + skin, cutghost);

  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  req->set_cutoff_fixed(rcut + skin);
}

void ComputeMADPropertyPython::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   A flat edge list over the atoms this rank owns. Ghost neighbours are kept
   as they are: their coordinates are the imaged ones, which is what makes the
   geometry complete without passing a cell.
------------------------------------------------------------------------- */

void ComputeMADPropertyPython::build_edges()
{
  double **x = atom->x;
  const int *type = atom->type;
  const int *mask = atom->mask;

  if (pbc_mode) {
    const int nlocal = atom->nlocal;
    edge_i.clear();
    edge_j.clear();
    pos.resize((size_t) nlocal * 3);
    type_flat.resize(nlocal);
    for (int i = 0; i < nlocal; i++) {
      pos[3 * i + 0] = x[i][0];
      pos[3 * i + 1] = x[i][1];
      pos[3 * i + 2] = x[i][2];
      type_flat[i] = type[i];
    }
    // rows a, b, c of the LAMMPS cell, which is lower triangular
    cellbuf[0] = domain->xprd;
    cellbuf[3] = domain->xy;
    cellbuf[4] = domain->yprd;
    cellbuf[6] = domain->xz;
    cellbuf[7] = domain->yz;
    cellbuf[8] = domain->zprd;
    return;
  }

  neighbor->build_one(list);

  const int nall = atom->nlocal + atom->nghost;
  const double rcutsq = rcut * rcut;

  edge_i.clear();
  edge_j.clear();
  pos.resize((size_t) nall * 3);
  type_flat.resize(nall);
  for (int i = 0; i < nall; i++) {
    pos[3 * i + 0] = x[i][0];
    pos[3 * i + 1] = x[i][1];
    pos[3 * i + 2] = x[i][2];
    type_flat[i] = type[i];
  }

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = list->ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const int *jlist = list->firstneigh[i];
    for (int jj = 0; jj < list->numneigh[i]; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      if (dx * dx + dy * dy + dz * dz >= rcutsq) continue;
      edge_i.push_back(i);
      edge_j.push_back(j);
    }
  }
}

/* ----------------------------------------------------------------------
   Hand the geometry to Python and read one array back. lam is null for the
   forward pass; out is (nlocal x ncomp) then, and (nall x 3) for the adjoint.
------------------------------------------------------------------------- */

void ComputeMADPropertyPython::call_python(void *fn, const double *lam, double *out, int nout)
{
#ifdef MADLIB_PYTHON
  const int nlocal = atom->nlocal;
  // pbc mode sends the owned atoms and the cell; otherwise the ghosts too
  const int nsent = pbc_mode ? nlocal : nlocal + atom->nghost;
  const int nedge = (int) edge_i.size();

  PyObject *mv_pos = PyMemoryView_FromMemory((char *) pos.data(),
                                             (Py_ssize_t) nsent * 3 * sizeof(double), PyBUF_READ);
  PyObject *mv_typ = PyMemoryView_FromMemory((char *) type_flat.data(),
                                             (Py_ssize_t) nsent * sizeof(int), PyBUF_READ);
  PyObject *mv_ei = PyMemoryView_FromMemory((char *) edge_i.data(),
                                            (Py_ssize_t) nedge * sizeof(int), PyBUF_READ);
  PyObject *mv_ej = PyMemoryView_FromMemory((char *) edge_j.data(),
                                            (Py_ssize_t) nedge * sizeof(int), PyBUF_READ);

  PyObject *mv_cell = PyMemoryView_FromMemory((char *) cellbuf, 9 * sizeof(double), PyBUF_READ);

  PyObject *args;
  if (lam) {
    PyObject *mv_lam = PyMemoryView_FromMemory((char *) lam,
                                               (Py_ssize_t) nlocal * ncomp * sizeof(double),
                                               PyBUF_READ);
    args = Py_BuildValue("(OOOOiiOO)", mv_pos, mv_typ, mv_ei, mv_ej, nlocal, ncomp, mv_lam,
                         mv_cell);
    Py_XDECREF(mv_lam);
  } else {
    args = Py_BuildValue("(OOOOiiO)", mv_pos, mv_typ, mv_ei, mv_ej, nlocal, ncomp, mv_cell);
  }
  Py_XDECREF(mv_pos);
  Py_XDECREF(mv_typ);
  Py_XDECREF(mv_ei);
  Py_XDECREF(mv_ej);
  Py_XDECREF(mv_cell);
  if (!args) fail("could not build the argument tuple for");

  PyObject *res = PyObject_CallObject(PY(fn), args);
  Py_DECREF(args);
  if (!res) fail("the model raised an exception in");

  // Read the result through the buffer protocol, so any object exposing a
  // contiguous double buffer works and numpy's C API is never linked.
  Py_buffer buf;
  if (PyObject_GetBuffer(res, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
    Py_DECREF(res);
    fail("the model returned something without a contiguous buffer in");
  }
  if (!buf.format || strcmp(buf.format, "d") != 0) {
    PyBuffer_Release(&buf);
    Py_DECREF(res);
    fail("the model must return float64 in");
  }
  if ((int) (buf.len / (Py_ssize_t) sizeof(double)) != nout) {
    const int got = (int) (buf.len / (Py_ssize_t) sizeof(double));
    PyBuffer_Release(&buf);
    Py_DECREF(res);
    error->all(FLERR,
               "compute mad/property/python: the model returned {} values where {} were expected",
               got, nout);
  }
  memcpy(out, buf.buf, (size_t) nout * sizeof(double));
  PyBuffer_Release(&buf);
  Py_DECREF(res);
#else
  (void) fn;
  (void) lam;
  (void) out;
  (void) nout;
#endif
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyPython::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyPython::compute_property()
{
  if (atom->nlocal > nmax_q) {
    nmax_q = atom->nmax;
    if (ncomp == 1) {
      memory->destroy(vector_atom);
      memory->create(vector_atom, nmax_q, "mad/property/python:q");
    } else {
      memory->destroy(array_atom);
      memory->create(array_atom, nmax_q, ncomp, "mad/property/python:q");
    }
  }

  build_edges();
  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;

  double *out = (ncomp == 1) ? vector_atom : &array_atom[0][0];
  call_python(py_compute, nullptr, out, nlocal * ncomp);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyPython::adjoint(double **lambda, double *virial)
{
  zero_adjoint();

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int nforce = pbc_mode ? nlocal : nall;
  double dedh[9] = {0.0};

  if (nlocal > 0) {
    lam_flat.resize((size_t) nlocal * ncomp);
    for (int i = 0; i < nlocal; i++)
      for (int c = 0; c < ncomp; c++) lam_flat[i * ncomp + c] = lambda[i][c];

    // three rows more in pbc mode: -dE/dh, one per cell vector
    const int nrow = nforce + (pbc_mode ? 3 : 0);
    std::vector<double> f((size_t) nrow * 3, 0.0);
    call_python(py_adjoint, lam_flat.data(), f.data(), nrow * 3);
    for (int i = 0; i < nforce; i++)
      for (int d = 0; d < 3; d++) fadj[i][d] += f[3 * i + d];
    if (pbc_mode)
      for (int k = 0; k < 9; k++) dedh[k] = f[3 * nforce + k];
  }

  // sum_a f_a (x) r_a over owned atoms AND ghosts, whose coordinates are the
  // imaged ones. That is the same quantity virial_fdotr_compute forms, and it
  // has to be taken before the ghost forces are folded back.
  double **x = atom->x;
  for (int i = 0; i < nforce; i++) {
    virial[0] += fadj[i][0] * x[i][0];
    virial[1] += fadj[i][1] * x[i][1];
    virial[2] += fadj[i][2] * x[i][2];
    virial[3] += 0.5 * (fadj[i][0] * x[i][1] + fadj[i][1] * x[i][0]);
    virial[4] += 0.5 * (fadj[i][0] * x[i][2] + fadj[i][2] * x[i][0]);
    virial[5] += 0.5 * (fadj[i][1] * x[i][2] + fadj[i][2] * x[i][1]);
  }

  // In pbc mode the images that carry the rest of r (x) f are inside the model,
  // not in the atom arrays, and what stands in for them is h^T (-dE/dh).
  if (pbc_mode) {
    const double h[3][3] = {{domain->xprd, 0.0, 0.0},
                            {domain->xy, domain->yprd, 0.0},
                            {domain->xz, domain->yz, domain->zprd}};
    double w[3][3] = {{0.0}};
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int c = 0; c < 3; c++) w[a][b] += h[c][a] * dedh[3 * c + b];
    virial[0] += w[0][0];
    virial[1] += w[1][1];
    virial[2] += w[2][2];
    virial[3] += 0.5 * (w[0][1] + w[1][0]);
    virial[4] += 0.5 * (w[0][2] + w[2][0]);
    virial[5] += 0.5 * (w[1][2] + w[2][1]);
  }

  scatter_adjoint();
}

/* ---------------------------------------------------------------------- */

double ComputeMADPropertyPython::memory_usage()
{
  double n = ComputeMADProperty::memory_usage();
  n += (double) (pos.capacity() + lam_flat.capacity()) * sizeof(double);
  n += (double) (edge_i.capacity() + edge_j.capacity() + type_flat.capacity()) * sizeof(int);
  return n;
}
