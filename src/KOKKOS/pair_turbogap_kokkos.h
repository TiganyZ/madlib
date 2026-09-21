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

#include <vector>

namespace LAMMPS_NS {

/* pair_style turbogap, with the neighbour list and the 2b and core_pot terms
   on the device.

   NOT REPRODUCIBLE RUN TO RUN. The device kernels atomic_add into the site
   energies and forces, and the hardware does not order those: four runs of the
   same 91-atom cell gave four different force dumps, differing at ~1e-11
   relative. Two GPU trajectories from identical input will separate. The host
   style is deterministic; this one is not, and no amount of holding the input
   fixed changes that.

   STAGES ONE AND TWO of docs/TURBOGAP-KK.md. It replaces build_neighbours(),
   and the 2b and core_pot terms with TurboGAP's own device kernels, lifted
   verbatim. soap_turbo and 3b still run through the host routines the plain
   style uses and which are checked against TurboGAP to 1.1e-09.

   Why stop there when the point is to compute on the GPU. Because it makes the
   step checkable. The device build either produces the same neighbour list as
   the host build or it does not, and with the compute held fixed, any
   difference in the answer is the list. Moving both at once would leave two
   candidates for every discrepancy, and the neighbour list is the part with
   the index conventions -- sites equal to atoms, self first at r = 0, ghosts
   addressed rather than folded -- that are easy to get subtly wrong.

   The list is built where the data already is, and the device terms read those
   arrays directly -- .data() on them is the raw pointer the kernels want, with
   no copy and no conversion. That is what stage one was separated out for.

   The copy back to the host remains, because soap_turbo and 3b are still host
   routines. Stage four removes it. */

template <class DeviceType> class PairTurboGAPKokkos : public PairTurboGAP {
 public:
  // Public only because nvcc refuses an extended __host__ __device__ lambda
  // inside a private or protected member function.
  void report_device_term(int ns, const char *what, int nblocks);

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairTurboGAPKokkos(class LAMMPS *);
  ~PairTurboGAPKokkos() override;

  void compute(int, int) override;
  void init_style() override;

 protected:
  void build_neighbours() override;

  // STAGE TWO: the 2b and core_pot terms, on the device.
  //
  // These call TurboGAP's own gap_2b.cu kernels, lifted verbatim and compiled
  // against LAMMPS's Kokkos (see lib/madlib/turbogap_gpu). They take raw
  // device pointers, which is what makes this cheap: the neighbour arrays
  // stage one already builds on the device are handed over as they stand, with
  // no copy and no conversion.
  //
  // The 3b term is inherited and still runs on the host. Its device kernel is
  // a template carrying __maxnreg__, which Kokkos has no equivalent for; see
  // docs/TURBOGAP-KK.md.
  void add_2b_and_core(int ns, int np, int nall) override;

  MADNeighKokkos<DeviceType> nbk;

  // The sparse sets, uploaded once. A potential does not change during a run,
  // so re-uploading per step would be pure cost.
  struct DeviceTwoB {
    Kokkos::View<double *, DeviceType> alphas, cutoff, qs;
    int sp1 = 0, sp2 = 0;
  };
  struct DeviceCore {
    Kokkos::View<double *, DeviceType> x, V, dVdx2;
    int sp1 = 0, sp2 = 0;
  };
  std::vector<DeviceTwoB> d_twob;
  std::vector<DeviceCore> d_core;
  Kokkos::View<double *, DeviceType> d_energies, d_forces, d_virial;
  int uploaded = 0;

  void upload_potential();
  Kokkos::View<int *, DeviceType> d_type2species;

  // Set once per run, when the device list first differs in size from what a
  // host build would produce. See the note in build_neighbours.
  int checked_against_host = 0;
};

}    // namespace LAMMPS_NS

#endif
#endif
