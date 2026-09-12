/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(turbogap,PairTurboGAP);
// clang-format on
#else

#ifndef LMP_PAIR_TURBOGAP_H
#define LMP_PAIR_TURBOGAP_H

#include "mad_gap_file.h"
#include "pair.h"

#include <map>
#include <string>
#include <vector>

namespace LAMMPS_NS {

/* A TurboGAP potential, evaluated in LAMMPS.

   Reads a .gap file as it stands. That is the point: a fitted potential is
   months of work and there is no reason to refit one to move it.

   FOUR TERMS, NOT ONE.

       E = E_soap_turbo + E_distance_2b + E_angle_3b + E_core_pot

   Every fitted TurboGAP potential uses all four -- of the ones in the test
   data only water_dipole.gap is soap_turbo alone, and that is a dipole model
   with no energy at all. A pair style evaluating only the SOAP term would
   report a plausible number missing three contributions, so all four are
   evaluated and a file that declares a term this cannot handle is an error
   rather than a silent omission.

   The arithmetic is TurboGAP's own, not a transcription of it: the 2b, 3b and
   core_pot routines in lib/madlib/turbogap/gap.f90 are compiled and called,
   and the SOAP term goes through the same GPR the local-property backend uses
   -- with zero_trunc off, because an energy is routinely negative and the
   property layer floors what it evaluates.

   Forces divide into two kinds and they are NOT handled the same way.

   The 2b, 3b and core_pot routines write the force only onto the CENTRAL site
   of each pair, the reciprocal being picked up when the neighbour is itself a
   central site. Over a full neighbour list of local atoms that is already
   complete per rank, and needs no communication.

   The SOAP term is many-body: dE_i/dr_j lands on the NEIGHBOUR, which may be a
   ghost, so those contributions are written to ghost indices and LAMMPS's own
   reverse communication returns them. */

class PairTurboGAP : public Pair {
 public:
  PairTurboGAP(class LAMMPS *);
  ~PairTurboGAP() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

  // Fold forces from ghosts back onto the atoms that own them.
  //
  // LAMMPS does this for us when newton pair is on, and the KOKKOS package
  // forbids newton on with a full neighbour list -- which a SOAP descriptor
  // has to have, since site i needs every neighbour of i. So with newton off
  // the fold is ours to do, and without it every contribution a site makes to
  // a neighbour it does not own is silently dropped.
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  MADGapPotential pot;
  bool loaded = false;

  // TurboGAP input parameters that change the potential on read, so they
  // belong to pair_style rather than to the .gap file.
  double core_pot_cutoff = 1.0e10;
  double core_pot_buffer = 1.0;

  // Per-species energy offsets, by species name because pair_style is parsed
  // before pair_coeff names the species. TurboGAP takes these from its input
  // file rather than from the .gap file, so they belong here too.
  std::map<std::string, double> e0_by_name;

  // LAMMPS type -> 1-based index into pot.species_types, or 0 for a type the
  // potential does not name.
  std::vector<int> type2species;

  // One TurboGAP-layout neighbour list over the local atoms, shared by every
  // term. Pairs are flattened over sites and the FIRST pair of each site is
  // the site itself at r = 0, which is the convention TurboGAP's loops assume
  // when they start at j = 2.
  struct NeighbourData {
    std::vector<int> n_neigh;             // per site
    std::vector<int> site_atom;           // site -> LAMMPS index
    std::vector<int> jatom;               // pair -> LAMMPS index
    std::vector<int> species;             // per site, 1-based
    std::vector<int> neighbor_species;    // per pair, 1-based
    std::vector<int> neighbors_list;      // per pair, 1-based site index (3b needs it)
    std::vector<double> rjs, xyz;         // per pair; xyz is 3 per pair
    std::vector<double> thetas, phis;     // per pair, for the SOAP descriptor
  };
  NeighbourData nb;

  std::vector<double> site_energy, site_force, virial_term;
  std::vector<char> species_buf;    // n_species * 8, the character*8 table

  // Print each term's energy separately, when TURBOGAP_TERMS is set in the
  // environment. Read once rather than per step. This exists because the terms
  // are the only useful unit of debugging here: a wrong total says nothing,
  // whereas "2b and soap agree and 3b does not" names the bug -- which is
  // exactly how the 3b sparse-set layout was found.
  int report_terms = 0;

  void allocate();
  // Virtual because the Kokkos variant replaces exactly this and nothing
  // else: it builds the same arrays on the device and copies them back, so
  // the terms below cannot tell which built them. That is what makes
  // turbogap/kk checkable against turbogap -- same compute, one input.
  virtual void build_neighbours();
  void pack_species_table();
  void add_soap_terms(int eflag);
  void add_pair_terms(int eflag);
};

}    // namespace LAMMPS_NS

#endif
#endif
