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

#include "pair_turbogap.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <stdexcept>

using namespace LAMMPS_NS;

#ifdef MADLIB_TURBOGAP
extern "C" {
void madlib_local_property(int n_sites, int n_pairs, int n_species, const int *n_neigh,
                           const int *species_idx, const double *rjs, const double *thetas,
                           const double *phis, const int *alpha_max, int l_max,
                           const double *rcut_hard, const double *rcut_soft, const double *nf,
                           const double *global_scaling, const double *atom_sigma_r,
                           const double *atom_sigma_r_scaling, const double *atom_sigma_t,
                           const double *atom_sigma_t_scaling, const double *amplitude_scaling,
                           int radial_enhancement, const double *central_weight, const char *basis,
                           const char *scaling_mode, int compress, int n_nonzero, const int *p_i,
                           const int *p_j, const double *p_el, int dim, int n_sparse,
                           const double *qs, const double *alphas, double v0, double delta,
                           double zeta, int zero_trunc, int do_derivatives, double *v,
                           double *v_der);

void madlib_gap_2b(int n_sites, int n_force, int n_pairs, int n_species, int n_sparse, const double *rjs,
                   const double *xyz, const double *alphas, const double *cutoff, double rcut,
                   double buffer, double delta, double sigma, double e0, const double *qs,
                   const int *n_neigh, const int *species, const int *neighbor_species,
                   const char *species1, const char *species2, const char *species_types,
                   int do_forces, double *energies, double *forces, double *virial);

void madlib_gap_3b(int n_sites, int n_force, int n_pairs, int n_species, int n_sparse, int n_dim,
                   const double *rjs, const double *xyz, const double *alphas,
                   const double *cutoff, double rcut, double buffer, double delta,
                   const double *sigma, double e0, const double *qs, const int *n_neigh,
                   const int *neighbors_list, const int *species, const int *neighbor_species,
                   const char *kernel_type, const char *species_center, const char *species1,
                   const char *species2, const char *species_types, int do_forces,
                   double *energies, double *forces, double *virial);

void madlib_gap_core_pot(int n_sites, int n_force, int n_pairs, int n_species, int n_tab, const double *rjs,
                         const double *xyz, const double *x, const double *vtab, double yp1,
                         double ypn, const double *dvdx2, const int *n_neigh, const int *species,
                         const int *neighbor_species, const char *species1, const char *species2,
                         const char *species_types, int do_forces, double *energies,
                         double *forces, double *virial);
}
#endif

/* ---------------------------------------------------------------------- */

PairTurboGAP::PairTurboGAP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;       // no pairwise decomposition to report
  restartinfo = 0;         // the potential is a file, not restart data
  one_coeff = 1;           // pair_coeff * * only
  manybody_flag = 1;       // SOAP and 3b are not pairwise
  centroidstressflag = CENTROID_NOTAVAIL;
  comm_reverse = 3;    // a force vector per ghost; see pack_reverse_comm
  report_terms = (getenv("TURBOGAP_TERMS") != nullptr) ? 1 : 0;
  profile = (getenv("TURBOGAP_PROFILE") != nullptr) ? 1 : 0;
}

/* ---------------------------------------------------------------------- */

PairTurboGAP::~PairTurboGAP()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairTurboGAP::allocate()
{
  allocated = 1;
  const int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
}

/* ----------------------------------------------------------------------
   pair_style turbogap [core_pot_cutoff <r>] [core_pot_buffer <w>]

   Both are TurboGAP input parameters rather than properties of the .gap file:
   they truncate and taper the tabulated core potential AS IT IS READ, so the
   same file gives a different potential under different settings. Defaults are
   TurboGAP's own.
------------------------------------------------------------------------- */

void PairTurboGAP::settings(int narg, char **arg)
{
  int i = 0;
  while (i < narg) {
    if (strcmp(arg[i], "core_pot_cutoff") == 0) {
      if (i + 1 >= narg) error->all(FLERR, "pair_style turbogap: core_pot_cutoff needs a value");
      core_pot_cutoff = utils::numeric(FLERR, arg[i + 1], false, lmp);
      i += 2;
    } else if (strcmp(arg[i], "core_pot_buffer") == 0) {
      if (i + 1 >= narg) error->all(FLERR, "pair_style turbogap: core_pot_buffer needs a value");
      core_pot_buffer = utils::numeric(FLERR, arg[i + 1], false, lmp);
      i += 2;
    } else if (strcmp(arg[i], "e0") == 0) {
      // e0 <species> <value> ... Named rather than positional: pair_style is
      // parsed before pair_coeff says which species exist, and a silent
      // mismatch in a per-species list is a constant energy offset that looks
      // like a bad fit rather than a typo.
      if (i + 2 >= narg)
        error->all(FLERR, "pair_style turbogap: e0 needs a species and a value");
      e0_by_name[arg[i + 1]] = utils::numeric(FLERR, arg[i + 2], false, lmp);
      i += 3;
    } else {
      error->all(FLERR, "pair_style turbogap: unknown keyword {}", arg[i]);
    }
  }
}

/* ----------------------------------------------------------------------
   pair_coeff * * <file.gap> <species for LAMMPS type 1> <type 2> ...

   The species order fixes the numbering TurboGAP's routines index by, so it
   must list one name per LAMMPS type. NULL marks a type the potential does
   not describe.
------------------------------------------------------------------------- */

void PairTurboGAP::coeff(int narg, char **arg)
{
#ifndef MADLIB_TURBOGAP
  error->all(FLERR,
             "pair_style turbogap needs the Fortran backend: build lib/madlib and configure "
             "with -D MADLIB_ENABLE_TURBOGAP=on");
#else
  if (!allocated) allocate();

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR,
               "pair_coeff for turbogap takes * * <file.gap> and one species per atom type "
               "({} expected, {} given)",
               3 + ntypes, narg);
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for turbogap must be * *");

  // The species TurboGAP will number, in LAMMPS type order. A NULL type is
  // kept in the table so the numbering does not shift under it.
  std::vector<std::string> names;
  type2species.assign(ntypes + 1, 0);
  for (int t = 1; t <= ntypes; t++) {
    const std::string s = arg[2 + t];
    if (s == "NULL") continue;
    int idx = 0;
    for (size_t k = 0; k < names.size(); k++)
      if (names[k] == s) idx = (int) k + 1;
    if (idx == 0) {
      names.push_back(s);
      idx = (int) names.size();
    }
    type2species[t] = idx;
  }
  if (names.empty()) error->all(FLERR, "pair_coeff for turbogap names no species");

  try {
    pot = MADGapFile::read_potential(arg[2], names, core_pot_cutoff, core_pot_buffer);
  } catch (std::exception &e) {
    error->all(FLERR, "{}", e.what());
  }
  loaded = true;
  pack_species_table();

  pot.e0.assign(names.size(), 0.0);
  for (const auto &kv : e0_by_name) {
    bool found = false;
    for (size_t k = 0; k < names.size(); k++)
      if (names[k] == kv.first) {
        pot.e0[k] = kv.second;
        found = true;
      }
    if (!found)
      error->all(FLERR, "pair_style turbogap: e0 given for species {}, which pair_coeff does not name",
                 kv.first);
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
#endif
}

/* ----------------------------------------------------------------------
   The character*8 table TurboGAP's routines compare species names against.
   Blank padded, because Fortran compares the whole fixed-length string.
------------------------------------------------------------------------- */

void PairTurboGAP::pack_species_table()
{
  species_buf.assign(pot.species_types.size() * 8, ' ');
  for (size_t i = 0; i < pot.species_types.size(); i++) {
    const std::string &s = pot.species_types[i];
    for (size_t k = 0; k < s.size() && k < 8; k++) species_buf[i * 8 + k] = s[k];
  }
}

/* ---------------------------------------------------------------------- */

void PairTurboGAP::init_style()
{
  if (!loaded) error->all(FLERR, "pair_style turbogap: no potential read; use pair_coeff");

  // A full list: the SOAP descriptor of site i needs every neighbour of i, and
  // a half list would give each pair to only one of its two atoms.
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

double PairTurboGAP::init_one(int /*i*/, int /*j*/)
{
  return pot.rcut_max();
}

/* ----------------------------------------------------------------------
   The neighbour list in TurboGAP's layout: pairs flattened over sites, the
   first pair of each site being the site itself at r = 0.
------------------------------------------------------------------------- */

void PairTurboGAP::build_neighbours()
{
  const double *const *x = atom->x;
  const int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int inum = list->inum;
  const int *ilist = list->ilist;
  const int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double rcut = pot.rcut_max();
  const double rcutsq = rcut * rcut;

  // SITE s IS LOCAL ATOM s. That identity is not tidiness, it is what makes
  // the four terms agree about what an index means.
  //
  // The 2b and core_pot routines write forces(:, i) with i the SITE loop
  // index, while 3b writes forces(:, mod(neighbors_list(k)-1, n_sites0)+1),
  // an ATOM index. The two only coincide if sites are the local atoms in
  // order, so they are built that way -- including atoms the potential does
  // not name, which keep a slot with species 0 and no neighbours so that every
  // later index still lands where it should.
  std::vector<int> pos(nlocal, -1);
  for (int ii = 0; ii < inum; ii++)
    if (ilist[ii] < nlocal) pos[ilist[ii]] = ii;

  nb.n_neigh.assign(nlocal, 1);
  nb.site_atom.resize(nlocal);
  nb.species.assign(nlocal, 0);
  nb.jatom.clear();
  nb.neighbor_species.clear();
  nb.neighbors_list.clear();
  nb.rjs.clear();
  nb.xyz.clear();
  nb.thetas.clear();
  nb.phis.clear();

  for (int i = 0; i < nlocal; i++) {
    nb.site_atom[i] = i;
    const int si = type2species[type[i]];
    nb.species[i] = si;

    // The site itself, first, at r = 0.
    nb.jatom.push_back(i);
    nb.neighbor_species.push_back(si);
    nb.neighbors_list.push_back(i + 1);
    nb.rjs.push_back(0.0);
    nb.xyz.insert(nb.xyz.end(), {0.0, 0.0, 0.0});
    nb.thetas.push_back(0.0);
    nb.phis.push_back(0.0);
    int cnt = 1;

    if (si != 0 && pos[i] >= 0) {
      const int *jlist = firstneigh[i];
      for (int jj = 0; jj < numneigh[i]; jj++) {
        const int j = jlist[jj] & NEIGHMASK;
        const int sj = type2species[type[j]];
        if (sj == 0) continue;
        const double dx = x[j][0] - x[i][0];
        const double dy = x[j][1] - x[i][1];
        const double dz = x[j][2] - x[i][2];
        const double rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= rcutsq || rsq == 0.0) continue;
        const double r = std::sqrt(rsq);

        nb.jatom.push_back(j);
        nb.neighbor_species.push_back(sj);
        // The LAMMPS index, 1-based. forces is sized over local AND ghost
        // atoms below, so 3b's mod() is the identity here and a ghost keeps
        // its own slot rather than being folded onto a local image -- which
        // is what lets LAMMPS reverse communication return it.
        nb.neighbors_list.push_back(j + 1);
        nb.rjs.push_back(r);
        nb.xyz.insert(nb.xyz.end(), {dx, dy, dz});
        nb.thetas.push_back(std::acos(dz / r));
        nb.phis.push_back(std::atan2(dy, dx));
        cnt++;
      }
    }
    nb.n_neigh[i] = cnt;
  }
}

/* ----------------------------------------------------------------------
   The SOAP term.

   One block per central species, so each is evaluated over the sites it
   claims. The GPR is the same one the local-property backend uses -- an
   energy and a local property are both

       V = delta^2 (soap . Qs)^zeta alphas + V0

   -- but with zero_trunc OFF. That flag floors V at zero, which is right for a
   binding energy and catastrophic for a potential energy: it would clamp every
   site with negative energy and drop its force without saying so.

   dV_i/dr_j lands on the NEIGHBOUR, which may be a ghost, so these go straight
   into f and LAMMPS's reverse communication returns them.
------------------------------------------------------------------------- */

void PairTurboGAP::add_soap_terms(int eflag)
{
#ifdef MADLIB_TURBOGAP
  double **f = atom->f;
  const int n_sites_all = (int) nb.n_neigh.size();

  std::vector<int> sub_n_neigh, sub_species, sub_site;
  std::vector<int> sub_jatom;
  std::vector<double> sub_rjs, sub_thetas, sub_phis;
  std::vector<double> v, v_der;

  for (const auto &b : pot.soap) {
    sub_n_neigh.clear();
    sub_species.clear();
    sub_site.clear();
    sub_jatom.clear();
    sub_rjs.clear();
    sub_thetas.clear();
    sub_phis.clear();

    int k = 0;
    for (int s = 0; s < n_sites_all; s++) {
      const int nn = nb.n_neigh[s];
      if (nb.species[s] == b.central_species) {
        sub_n_neigh.push_back(nn);
        sub_site.push_back(nb.site_atom[s]);
        for (int p = 0; p < nn; p++) {
          sub_species.push_back(nb.neighbor_species[k + p]);
          sub_jatom.push_back(nb.jatom[k + p]);
          sub_rjs.push_back(nb.rjs[k + p]);
          sub_thetas.push_back(nb.thetas[k + p]);
          sub_phis.push_back(nb.phis[k + p]);
        }
      }
      k += nn;
    }

    const int ns = (int) sub_n_neigh.size();
    const int np = (int) sub_rjs.size();
    if (ns == 0) continue;

    v.assign(ns, 0.0);
    v_der.assign(3 * np, 0.0);

    madlib_local_property(
        ns, np, b.n_species, sub_n_neigh.data(), sub_species.data(), sub_rjs.data(),
        sub_thetas.data(), sub_phis.data(), b.alpha_max.data(), b.l_max, b.rcut_hard.data(),
        b.rcut_soft.data(), b.nf.data(), b.global_scaling.data(), b.atom_sigma_r.data(),
        b.atom_sigma_r_scaling.data(), b.atom_sigma_t.data(), b.atom_sigma_t_scaling.data(),
        b.amplitude_scaling.data(), b.radial_enhancement, b.central_weight.data(),
        b.basis.c_str(), b.scaling_mode.c_str(), b.compress ? 1 : 0, b.n_nonzero, b.p_i.data(),
        b.p_j.data(), b.p_el.data(), b.dim, b.n_sparse, b.qs.data(), b.alphas.data(), b.v0,
        b.delta, b.zeta, /* zero_trunc */ 0, /* do_derivatives */ 1, v.data(), v_der.data());

    double esum = 0.0;
    for (int s = 0; s < ns; s++) {
      eng_vdwl += v[s];
      esum += v[s];
      if (eflag_atom) eatom[sub_site[s]] += v[s];
    }
    if (report_terms && comm->me == 0)
      fprintf(stderr, "  soap block central=%d ns=%d E=%.17g\n", b.central_species, ns, esum);
    for (int p = 0; p < np; p++) {
      const int a = sub_jatom[p];
      f[a][0] -= v_der[3 * p];
      f[a][1] -= v_der[3 * p + 1];
      f[a][2] -= v_der[3 * p + 2];
    }
  }
  (void) eflag;
#endif
}

/* ----------------------------------------------------------------------
   The 2b, 3b and core_pot terms.

   These write the force only onto the CENTRAL site of each pair, the
   reciprocal being picked up when the neighbour is itself a central site.
   Over a full neighbour list of the rank's own atoms that is complete as it
   stands, so the per-site forces map straight onto local atoms.
------------------------------------------------------------------------- */

// A species name in the fixed-length form TurboGAP's routines compare against.
// Blank padded, because Fortran compares the whole character*8.
static std::string name8(const std::string &s)
{
  std::string buf(8, ' ');
  for (size_t k = 0; k < s.size() && k < 8; k++) buf[k] = s[k];
  return buf;
}

/* ----------------------------------------------------------------------
   The 2b and core_pot terms.

   Split from the 3b term so the Kokkos variant can replace these two and
   inherit that one: TurboGAP has device kernels for both that take raw device
   pointers and touch no vendor API, while its 3b kernel is a template carrying
   a register cap with no Kokkos equivalent.

   Fills site_energy, site_force and virial_term, which add_pair_terms
   accumulates -- so a device implementation copies back into the same buffers
   and the accumulation stays written once.
------------------------------------------------------------------------- */

void PairTurboGAP::add_2b_and_core(int ns, int np, int nall)
{
#ifdef MADLIB_TURBOGAP
  const int nsp = (int) pot.species_types.size();

  for (const auto &b : pot.twob) {
    const std::string s1 = name8(b.species1), s2 = name8(b.species2);
    madlib_gap_2b(ns, nall, np, nsp, b.n_sparse, nb.rjs.data(), nb.xyz.data(), b.alphas.data(),
                  b.cutoff.data(), b.rcut, b.buffer, b.delta, b.sigma, b.e0, b.qs.data(),
                  nb.n_neigh.data(), nb.species.data(), nb.neighbor_species.data(), s1.c_str(),
                  s2.c_str(), species_buf.data(), 1, site_energy.data(), site_force.data(),
                  virial_term.data());
  }
  report_term(ns, "2b", (int) pot.twob.size());

  for (const auto &b : pot.core) {
    const std::string s1 = name8(b.species1), s2 = name8(b.species2);
    madlib_gap_core_pot(ns, nall, np, nsp, (int) b.x.size(), nb.rjs.data(), nb.xyz.data(),
                        b.x.data(), b.V.data(), b.yp1, b.ypn, b.dVdx2.data(), nb.n_neigh.data(),
                        nb.species.data(), nb.neighbor_species.data(), s1.c_str(), s2.c_str(),
                        species_buf.data(), 1, site_energy.data(), site_force.data(),
                        virial_term.data());
  }

  report_term(ns, "core_pot", (int) pot.core.size());
#endif
}

void PairTurboGAP::report_term(int ns, const char *what, int nblocks)
{
  if (!report_terms || comm->me != 0) return;
  double e = 0.0;
  for (int s = 0; s < ns; s++) e += site_energy[s];
  fprintf(stderr, "  after %s: blocks=%d E=%.17g\n", what, nblocks, e);
}

/* ---------------------------------------------------------------------- */

void PairTurboGAP::add_3b(int ns, int np, int nall)
{
#ifdef MADLIB_TURBOGAP
  const int nsp = (int) pot.species_types.size();

  for (const auto &b : pot.threeb) {
    const std::string sc = name8(b.species_center), s1 = name8(b.species1), s2 = name8(b.species2);
    std::string kern = b.kernel_type;
    kern.resize(3, ' ');
    madlib_gap_3b(ns, nall, np, nsp, b.n_sparse, b.dim, nb.rjs.data(), nb.xyz.data(),
                  b.alphas.data(), b.cutoff.data(), b.rcut, b.buffer, b.delta, b.sigma.data(),
                  b.e0, b.qs.data(), nb.n_neigh.data(), nb.neighbors_list.data(),
                  nb.species.data(), nb.neighbor_species.data(), kern.c_str(), sc.c_str(),
                  s1.c_str(), s2.c_str(), species_buf.data(), 1, site_energy.data(),
                  site_force.data(), virial_term.data());
  }

  report_term(ns, "3b", (int) pot.threeb.size());
#endif
}

/* ---------------------------------------------------------------------- */

void PairTurboGAP::add_pair_terms(int eflag)
{
#ifdef MADLIB_TURBOGAP
  double **f = atom->f;
  const int ns = (int) nb.n_neigh.size();
  const int np = (int) nb.rjs.size();
  if (ns == 0) return;

  const int nall = atom->nlocal + atom->nghost;
  site_energy.assign(ns, 0.0);
  // Over local AND ghost atoms: 3b puts a force on every atom of a triplet,
  // and a ghost has to keep its own slot for reverse communication to find it.
  site_force.assign(3 * nall, 0.0);
  virial_term.assign(9, 0.0);

  add_2b_and_core(ns, np, nall);
  add_3b(ns, np, nall);

  for (int s = 0; s < ns; s++) {
    // e0 is a per-atom offset, not a term: TurboGAP folds it into the soap
    // energies, and its own trajectory header reports energy_soap WITHOUT it,
    // so the printed terms do not sum to the printed energy. Added once, here.
    const double e = site_energy[s] + (nb.species[s] > 0 ? pot.e0[nb.species[s] - 1] : 0.0);
    eng_vdwl += e;
    if (eflag_atom) eatom[nb.site_atom[s]] += e;
  }
  for (int a = 0; a < nall; a++) {
    f[a][0] += site_force[3 * a];
    f[a][1] += site_force[3 * a + 1];
    f[a][2] += site_force[3 * a + 2];
  }
  (void) eflag;
#endif
}

/* ---------------------------------------------------------------------- */

void PairTurboGAP::compute(int eflag, int vflag)
{
#ifndef MADLIB_TURBOGAP
  error->all(FLERR, "pair_style turbogap needs the Fortran backend");
#else
  ev_init(eflag, vflag);

  // Verlet::force_clear() zeroes the ghost range of f only when newton is on.
  // With it off this style's own ghost forces from the previous step are still
  // sitting there, and the fold below would add them a second time -- right on
  // the first step, when the array is still fresh, and wrong on every step
  // after. Nothing else writes ghost forces with newton off, because the
  // LAMMPS convention is that a full-list style applies force only to the
  // local atom; this one cannot, since TurboGAP returns de_i/dr_j for every
  // neighbour j of a site.
  if (!force->newton_pair) {
    const int nlocal = atom->nlocal;
    const int nghost = atom->nghost;
    if (nghost) memset(&atom->f[nlocal][0], 0, 3 * nghost * sizeof(double));
  }

  if (!profile) {
    build_neighbours();
    add_soap_terms(eflag);
    add_pair_terms(eflag);
  } else {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t0 = clock::now();
    build_neighbours();
    const auto t1 = clock::now();
    add_soap_terms(eflag);
    const auto t2 = clock::now();
    add_pair_terms(eflag);
    const auto t3 = clock::now();
    if (comm->me == 0)
      fprintf(stderr, "  turbogap profile: neigh %.2f ms, soap %.2f ms, 2b+3b+core %.2f ms\n",
              ms(t0, t1), ms(t1, t2), ms(t2, t3));
  }

  // The virial comes from f.r rather than from TurboGAP's own accumulator.
  // Its routines return a 3x3 built with a sign and a factor of their own, and
  // deriving the LAMMPS convention from it by inspection is exactly the kind
  // of thing that is wrong by a factor of two without looking wrong. f.r needs
  // only the forces, which are checked directly.
  // Before the fold, not after: virial_fdotr_compute sums r.f over local AND
  // ghost atoms, so it wants the ghost forces still sitting where they were
  // computed.
  if (vflag_fdotr) virial_fdotr_compute();

  // With newton pair on, LAMMPS folds the ghosts itself after compute and
  // doing it here too would count them twice.
  if (!force->newton_pair) comm->reverse_comm(this);
#endif
}

/* ----------------------------------------------------------------------
   The ghost-force fold. Only reached with newton pair off; see the header.
------------------------------------------------------------------------- */

int PairTurboGAP::pack_reverse_comm(int n, int first, double *buf)
{
  double **f = atom->f;
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  return m;
}

void PairTurboGAP::unpack_reverse_comm(int n, int *list, double *buf)
{
  double **f = atom->f;
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

double PairTurboGAP::memory_usage()
{
  double bytes = Pair::memory_usage();
  bytes += (double) (nb.rjs.capacity() + nb.thetas.capacity() + nb.phis.capacity()) *
      sizeof(double);
  bytes += (double) nb.xyz.capacity() * sizeof(double);
  bytes += (double) (nb.jatom.capacity() + nb.neighbor_species.capacity() +
                     nb.neighbors_list.capacity()) *
      sizeof(int);
  return bytes;
}
