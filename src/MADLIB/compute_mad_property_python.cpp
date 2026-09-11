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
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cstring>

#ifdef MADLIB_PYTHON
#include <Python.h>
#define PY(x) ((PyObject *) (x))
#endif

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyPython::ComputeMADPropertyPython(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), rcut(-1.0), nmax_q(0), list(nullptr), py_module(nullptr),
    py_compute(nullptr), py_adjoint(nullptr)
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
    } else {
      error->all(FLERR, "Unknown compute mad/property/python keyword {}", arg[iarg]);
    }
  }
  if ((ncomp != 1) && (ncomp != 3))
    error->all(FLERR, "compute mad/property/python supports ncomp 1 or 3, not {}", ncomp);
  if (rcut <= 0.0)
    error->all(FLERR, "compute mad/property/python requires a positive cutoff");

  size_peratom_cols = (ncomp == 1) ? 0 : ncomp;

  // LAMMPS may already be running an interpreter, and finalising one that
  // somebody else started would take their state with it, so this never calls
  // Py_Finalize.
  if (!Py_IsInitialized()) Py_Initialize();

  PyObject *mod = PyImport_ImportModule(module_name.c_str());
  if (!mod) fail("could not import the module");
  py_module = mod;

  py_compute = PyObject_GetAttrString(mod, "mad_compute");
  if (!py_compute) fail("the module has no mad_compute");
  py_adjoint = PyObject_GetAttrString(mod, "mad_adjoint");
  if (!py_adjoint) fail("the module has no mad_adjoint");

  if (comm->me == 0)
    utils::logmesg(lmp, "compute {}: python model '{}', {} component(s), cutoff {:.4g}\n", id,
                   module_name, ncomp, rcut);
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
  req->set_cutoff(rcut + skin);
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
  neighbor->build_one(list);

  double **x = atom->x;
  const int *type = atom->type;
  const int *mask = atom->mask;
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
  const int nall = nlocal + atom->nghost;
  const int nedge = (int) edge_i.size();

  PyObject *mv_pos = PyMemoryView_FromMemory((char *) pos.data(),
                                             (Py_ssize_t) nall * 3 * sizeof(double), PyBUF_READ);
  PyObject *mv_typ = PyMemoryView_FromMemory((char *) type_flat.data(),
                                             (Py_ssize_t) nall * sizeof(int), PyBUF_READ);
  PyObject *mv_ei = PyMemoryView_FromMemory((char *) edge_i.data(),
                                            (Py_ssize_t) nedge * sizeof(int), PyBUF_READ);
  PyObject *mv_ej = PyMemoryView_FromMemory((char *) edge_j.data(),
                                            (Py_ssize_t) nedge * sizeof(int), PyBUF_READ);

  PyObject *args;
  if (lam) {
    PyObject *mv_lam = PyMemoryView_FromMemory((char *) lam,
                                               (Py_ssize_t) nlocal * ncomp * sizeof(double),
                                               PyBUF_READ);
    args = Py_BuildValue("(OOOOiiO)", mv_pos, mv_typ, mv_ei, mv_ej, nlocal, ncomp, mv_lam);
    Py_XDECREF(mv_lam);
  } else {
    args = Py_BuildValue("(OOOOii)", mv_pos, mv_typ, mv_ei, mv_ej, nlocal, ncomp);
  }
  Py_XDECREF(mv_pos);
  Py_XDECREF(mv_typ);
  Py_XDECREF(mv_ei);
  Py_XDECREF(mv_ej);
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
  if (nlocal > 0) {
    lam_flat.resize((size_t) nlocal * ncomp);
    for (int i = 0; i < nlocal; i++)
      for (int c = 0; c < ncomp; c++) lam_flat[i * ncomp + c] = lambda[i][c];

    std::vector<double> f((size_t) nall * 3, 0.0);
    call_python(py_adjoint, lam_flat.data(), f.data(), nall * 3);
    for (int i = 0; i < nall; i++)
      for (int d = 0; d < 3; d++) fadj[i][d] += f[3 * i + d];
  }

  // sum_a f_a (x) r_a over owned atoms AND ghosts, whose coordinates are the
  // imaged ones. That is the same quantity virial_fdotr_compute forms, and it
  // has to be taken before the ghost forces are folded back.
  double **x = atom->x;
  for (int i = 0; i < nall; i++) {
    virial[0] += fadj[i][0] * x[i][0];
    virial[1] += fadj[i][1] * x[i][1];
    virial[2] += fadj[i][2] * x[i][2];
    virial[3] += 0.5 * (fadj[i][0] * x[i][1] + fadj[i][1] * x[i][0]);
    virial[4] += 0.5 * (fadj[i][0] * x[i][2] + fadj[i][2] * x[i][0]);
    virial[5] += 0.5 * (fadj[i][1] * x[i][2] + fadj[i][2] * x[i][1]);
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
