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

#ifndef LMP_MAD_NEIGH_KOKKOS_H
#define LMP_MAD_NEIGH_KOKKOS_H

#include "kokkos_type.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

/* LAMMPS's neighbour list, rewritten on the device into the layout TurboGAP's
   routines read.

   The two disagree about almost everything except the physics. LAMMPS gives
   per-atom lists indexed by atom, with ghosts; TurboGAP wants ONE flat array
   of pairs, sites contiguous and in order, the site itself first in its own
   list at r = 0, and spherical coordinates rather than Cartesian. Converting
   that on the host and copying it down each step would cost more than the
   compute it feeds, so it is built where the data already is.

   SITE s IS LOCAL ATOM s, the same invariant the host pair style keeps and for
   the same reason: the 2b and core_pot routines index forces by SITE while 3b
   indexes them by ATOM, and the two coincide only under that identity. Atoms
   the potential does not name keep a slot with species 0 and no neighbours, so
   every later index still lands where it should.

   Three passes, which is one more than it looks like it needs. Counting first
   is what lets the pair arrays be exactly the right size: a single pass would
   have to guess a maximum neighbour count, and either waste the memory or
   discover it was wrong on the one configuration that mattered.

     count   neighbours per site, in range and named by the potential
     scan    exclusive prefix sum, giving each site its slice
     fill    write the pairs, self first

   Nothing here is a reduction, so no ordering question arises: every entry is
   written by exactly one thread to an index computed from the scan. */

template <class DeviceType> struct MADNeighKokkos {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // TurboGAP's layout. Per site:
  Kokkos::View<int *, DeviceType> d_n_neigh;     // neighbours, self included
  Kokkos::View<int *, DeviceType> d_species;     // 1-based, 0 = not in the potential
  Kokkos::View<int *, DeviceType> d_offset;      // first pair of each site
  // Per pair:
  Kokkos::View<int *, DeviceType> d_jatom;              // LAMMPS index
  Kokkos::View<int *, DeviceType> d_neighbor_species;   // 1-based
  Kokkos::View<int *, DeviceType> d_neighbors_list;     // LAMMPS index + 1
  Kokkos::View<double *, DeviceType> d_rjs;
  Kokkos::View<double *, DeviceType> d_xyz;      // 3 per pair
  Kokkos::View<double *, DeviceType> d_thetas;
  Kokkos::View<double *, DeviceType> d_phis;

  int n_sites = 0;
  int n_pairs = 0;

  // The view typedefs are LAMMPS's current spelling, not the older
  // t_x_array_randomread: develop renamed them around the mixed-precision
  // work, and KK_FLOAT is now whatever precision the build was configured for.
  void build(typename AT::t_kkfloat_1d_3_lr_randomread x, typename AT::t_int_1d_randomread type,
             Kokkos::View<int *, DeviceType> d_type2species, NeighListKokkos<DeviceType> *k_list,
             int nlocal, double rcut);
};

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MADNeighKokkos<DeviceType>::build(typename AT::t_kkfloat_1d_3_lr_randomread x,
                                       typename AT::t_int_1d_randomread type,
                                       Kokkos::View<int *, DeviceType> d_type2species,
                                       NeighListKokkos<DeviceType> *k_list, int nlocal,
                                       double rcut)
{
  n_sites = nlocal;
  const double rcutsq = rcut * rcut;

  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;

  if ((int) d_n_neigh.extent(0) < nlocal) {
    d_n_neigh = Kokkos::View<int *, DeviceType>("mad:n_neigh", nlocal);
    d_species = Kokkos::View<int *, DeviceType>("mad:species", nlocal);
    d_offset = Kokkos::View<int *, DeviceType>("mad:offset", nlocal + 1);
  }

  auto n_neigh = d_n_neigh;
  auto species = d_species;

  // --- count. The site itself is pair one, so every site starts at 1 even
  //     when the potential does not name it: that empty slot is what keeps
  //     site s equal to atom s.
  Kokkos::parallel_for(
      "mad/neigh:count", Kokkos::RangePolicy<DeviceType>(0, nlocal), KOKKOS_LAMBDA(const int i) {
        const int si = d_type2species(type(i));
        species(i) = si;
        int cnt = 1;
        if (si != 0) {
          const int jnum = d_numneigh(i);
          for (int jj = 0; jj < jnum; jj++) {
            const int j = d_neighbors(i, jj) & NEIGHMASK;
            if (d_type2species(type(j)) == 0) continue;
            const double dx = x(j, 0) - x(i, 0);
            const double dy = x(j, 1) - x(i, 1);
            const double dz = x(j, 2) - x(i, 2);
            const double rsq = dx * dx + dy * dy + dz * dz;
            if (rsq >= rcutsq || rsq == 0.0) continue;
            cnt++;
          }
        }
        n_neigh(i) = cnt;
      });

  // --- scan, giving each site the index its first pair lands on
  auto offset = d_offset;
  Kokkos::parallel_scan(
      "mad/neigh:scan", Kokkos::RangePolicy<DeviceType>(0, nlocal),
      KOKKOS_LAMBDA(const int i, int &running, const bool final) {
        if (final) offset(i) = running;
        running += n_neigh(i);
      });

  {
    // The total is the last offset plus the last count, read back once. One
    // copy of one integer per step, which is what it costs to size the pair
    // arrays exactly rather than guessing a maximum neighbour count.
    auto h_offset = Kokkos::create_mirror_view(d_offset);
    auto h_n_neigh = Kokkos::create_mirror_view(d_n_neigh);
    Kokkos::deep_copy(h_offset, d_offset);
    Kokkos::deep_copy(h_n_neigh, d_n_neigh);
    n_pairs = (nlocal > 0) ? h_offset(nlocal - 1) + h_n_neigh(nlocal - 1) : 0;
  }

  if ((int) d_rjs.extent(0) < n_pairs) {
    d_jatom = Kokkos::View<int *, DeviceType>("mad:jatom", n_pairs);
    d_neighbor_species = Kokkos::View<int *, DeviceType>("mad:nspecies", n_pairs);
    d_neighbors_list = Kokkos::View<int *, DeviceType>("mad:nlist", n_pairs);
    d_rjs = Kokkos::View<double *, DeviceType>("mad:rjs", n_pairs);
    d_xyz = Kokkos::View<double *, DeviceType>("mad:xyz", 3 * n_pairs);
    d_thetas = Kokkos::View<double *, DeviceType>("mad:thetas", n_pairs);
    d_phis = Kokkos::View<double *, DeviceType>("mad:phis", n_pairs);
  }

  auto jatom = d_jatom;
  auto nspec = d_neighbor_species;
  auto nlist = d_neighbors_list;
  auto rjs = d_rjs;
  auto xyz = d_xyz;
  auto thetas = d_thetas;
  auto phis = d_phis;

  // --- fill. Each site writes its own slice, so the order within a slice is
  //     the order LAMMPS's list has, which is what the host builder produces
  //     too -- the two are comparable element for element.
  Kokkos::parallel_for(
      "mad/neigh:fill", Kokkos::RangePolicy<DeviceType>(0, nlocal), KOKKOS_LAMBDA(const int i) {
        int k = offset(i);
        const int si = species(i);

        jatom(k) = i;
        nspec(k) = si;
        nlist(k) = i + 1;
        rjs(k) = 0.0;
        xyz(3 * k) = 0.0;
        xyz(3 * k + 1) = 0.0;
        xyz(3 * k + 2) = 0.0;
        thetas(k) = 0.0;
        phis(k) = 0.0;
        k++;

        if (si == 0) return;
        const int jnum = d_numneigh(i);
        for (int jj = 0; jj < jnum; jj++) {
          const int j = d_neighbors(i, jj) & NEIGHMASK;
          const int sj = d_type2species(type(j));
          if (sj == 0) continue;
          const double dx = x(j, 0) - x(i, 0);
          const double dy = x(j, 1) - x(i, 1);
          const double dz = x(j, 2) - x(i, 2);
          const double rsq = dx * dx + dy * dy + dz * dz;
          if (rsq >= rcutsq || rsq == 0.0) continue;
          const double r = sqrt(rsq);

          jatom(k) = j;
          nspec(k) = sj;
          nlist(k) = j + 1;
          rjs(k) = r;
          xyz(3 * k) = dx;
          xyz(3 * k + 1) = dy;
          xyz(3 * k + 2) = dz;
          thetas(k) = acos(dz / r);
          phis(k) = atan2(dy, dx);
          k++;
        }
      });
}

}    // namespace LAMMPS_NS

#endif
