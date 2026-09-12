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
// Written out rather than through #define'd aliases. symmetrix uses aliases
// because it templates on two parameters and the comma inside PairStyle(...)
// would be read as a macro argument separator; with one parameter there is no
// comma to hide, and the alias form is actively wrong here -- the #undef that
// closes it leaves the name undefined in the generated style_pair.cpp, which
// then fails to compile on a symbol that looks like it should exist.
PairStyle(turbogap/kk,PairTurboGAPKokkos<LMPDeviceType>);
PairStyle(turbogap/kk/device,PairTurboGAPKokkos<LMPDeviceType>);
PairStyle(turbogap/kk/host,PairTurboGAPKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_TURBOGAP_KOKKOS_H
#define LMP_PAIR_TURBOGAP_KOKKOS_H

#include "kokkos_base.h"
#include "mad_neigh_kokkos.h"
#include "pair_kokkos.h"
#include "pair_turbogap.h"

namespace LAMMPS_NS {

/* pair_style turbogap, with the neighbour list built on the device.

   STAGE ONE of docs/TURBOGAP-KK.md, and deliberately only that. It replaces
   build_neighbours() and nothing else: the four energy terms still run through
   the host routines the plain style uses and which are checked against
   TurboGAP to 1.1e-09.

   Why stop there when the point is to compute on the GPU. Because it makes the
   step checkable. The device build either produces the same neighbour list as
   the host build or it does not, and with the compute held fixed, any
   difference in the answer is the list. Moving both at once would leave two
   candidates for every discrepancy, and the neighbour list is the part with
   the index conventions -- sites equal to atoms, self first at r = 0, ghosts
   addressed rather than folded -- that are easy to get subtly wrong.

   Stage two swaps the compute for the device entry points, on the same list.

   The list is built where the data already is and copied back once per step.
   That copy is the cost of stage one and it is the thing stage two removes:
   the device arrays it produces are what the device compute will read
   directly, so nothing built here is thrown away. */

template <class DeviceType> class PairTurboGAPKokkos : public PairTurboGAP {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairTurboGAPKokkos(class LAMMPS *);
  ~PairTurboGAPKokkos() override;

  void compute(int, int) override;
  void init_style() override;

 protected:
  void build_neighbours() override;

  MADNeighKokkos<DeviceType> nbk;
  Kokkos::View<int *, DeviceType> d_type2species;

  // Set once per run, when the device list first differs in size from what a
  // host build would produce. See the note in build_neighbours.
  int checked_against_host = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
