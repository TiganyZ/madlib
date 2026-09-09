/* ----------------------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#include "compute_mad_property_table.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cstring>
#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMADPropertyTable::ComputeMADPropertyTable(LAMMPS *lmp, int narg, char **arg) :
    ComputeMADProperty(lmp, narg, arg), bytype(nullptr), byid(nullptr), maxtag(0), nmax_q(0)
{
  ncomp = 1;
  size_peratom_cols = 0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "values") == 0) {
      const int nt = atom->ntypes;
      if (iarg + nt >= narg) utils::missing_cmd_args(FLERR, "compute mad/property/table values", error);
      memory->create(bytype, nt + 1, "mad/property/table:bytype");
      for (int t = 1; t <= nt; t++)
        bytype[t] = utils::numeric(FLERR, arg[iarg + t], false, lmp);
      iarg += nt + 1;
    } else if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "compute mad/property/table file", error);
      read_file(arg[iarg + 1]);
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown compute mad/property/table keyword {}", arg[iarg]);
    }
  }

  if (!bytype && !byid)
    error->all(FLERR, "compute mad/property/table needs either values or file");
}

ComputeMADPropertyTable::~ComputeMADPropertyTable()
{
  memory->destroy(bytype);
  memory->destroy(byid);
  memory->destroy(vector_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTable::read_file(const char *fname)
{
  std::ifstream in(fname);
  if (!in) error->all(FLERR, "compute mad/property/table cannot open {}", fname);

  std::vector<std::pair<tagint, double>> rows;
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    lineno++;
    const size_t p = line.find_first_not_of(" \t\r\n");
    if (p == std::string::npos || line[p] == '#' || line[p] == '!') continue;
    std::istringstream ss(line);
    tagint id;
    double v;
    if (!(ss >> id >> v))
      error->all(FLERR, "compute mad/property/table: {} line {} is not 'id value'", fname, lineno);
    rows.emplace_back(id, v);
    if (id > maxtag) maxtag = id;
  }
  if (rows.empty()) error->all(FLERR, "compute mad/property/table: {} is empty", fname);

  memory->create(byid, maxtag + 1, "mad/property/table:byid");
  for (tagint i = 0; i <= maxtag; i++) byid[i] = 0.0;
  for (auto &r : rows) byid[r.first] = r.second;
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTable::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  compute_property();
}

void ComputeMADPropertyTable::compute_property()
{
  if (atom->nlocal > nmax_q) {
    memory->destroy(vector_atom);
    nmax_q = atom->nmax;
    memory->create(vector_atom, nmax_q, "mad/property/table:q");
  }

  const int *type = atom->type;
  const tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (byid) {
      if (tag[i] > maxtag)
        error->one(FLERR, "compute mad/property/table has no value for atom {}", tag[i]);
      vector_atom[i] = byid[tag[i]];
    } else {
      vector_atom[i] = bytype[type[i]];
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeMADPropertyTable::adjoint(double ** /*lambda*/, double * /*virial*/)
{
  error->all(FLERR,
             "compute mad/property/table has no gradient, so it cannot drive a bias. Use it for "
             "prediction (gamma 0.0), or supply a model that differentiates.");
}
