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

#include "pair_turbogap_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cstdlib>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairTurboGAPKokkos<DeviceType>::PairTurboGAPKokkos(LAMMPS *lmp) : PairTurboGAP(lmp)
{
  respa_enable = 0;
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;

  // HOST, in stage one, and that is not a placeholder: the four energy terms
  // still run on the host, so the forces are written there and LAMMPS's
  // bookkeeping has to agree. Declaring the device space instead makes LAMMPS
  // mark atom:f modified on the device while this writes it on the host, and
  // Kokkos rejects the pair outright -- "Concurrent modification of host and
  // device views in DualView". Stage two moves this to the device along with
  // the compute, which is the same commit.
  execution_space = Host;
  datamask_read = X_MASK | TYPE_MASK | F_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairTurboGAPKokkos<DeviceType>::~PairTurboGAPKokkos()
{
  if (copymode) return;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void PairTurboGAPKokkos<DeviceType>::init_style()
{
  PairTurboGAP::init_style();

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

/* ----------------------------------------------------------------------
   The neighbour list, built on the device and copied back.

   The copy is stage one's cost and stage two's starting point: the device
   arrays are what the device compute will read directly, so nothing built
   here is wasted, only read twice for now.
------------------------------------------------------------------------- */

template <class DeviceType> void PairTurboGAPKokkos<DeviceType>::build_neighbours()
{
  const int nlocal = atom->nlocal;
  const int ntypes = atom->ntypes;

  if ((int) d_type2species.extent(0) != ntypes + 1) {
    d_type2species = Kokkos::View<int *, DeviceType>("mad:type2species", ntypes + 1);
    auto h = Kokkos::create_mirror_view(d_type2species);
    for (int t = 0; t <= ntypes; t++) h(t) = (t < (int) type2species.size()) ? type2species[t] : 0;
    Kokkos::deep_copy(d_type2species, h);
  }

  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto x = atomKK->k_x.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();

  nbk.build(x, type, d_type2species, k_list, nlocal, pot.rcut_max());

  // --- back to the host, in the layout the terms expect
  const int ns = nbk.n_sites;
  const int np = nbk.n_pairs;

  nb.n_neigh.resize(ns);
  nb.site_atom.resize(ns);
  nb.species.resize(ns);
  nb.jatom.resize(np);
  nb.neighbor_species.resize(np);
  nb.neighbors_list.resize(np);
  nb.rjs.resize(np);
  nb.xyz.resize(3 * np);
  nb.thetas.resize(np);
  nb.phis.resize(np);

  // Generic over the view type and the element type, because Kokkos 5 dropped
  // View::HostMirror and create_mirror_view is the spelling that survives a
  // Kokkos version change.
  auto copy_back = [](auto d, auto &h, int n) {
    if (n == 0) return;
    auto m = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(m, d);
    for (int i = 0; i < n; i++) h[i] = m(i);
  };
  auto copy_int = copy_back;
  auto copy_dbl = copy_back;

  copy_int(nbk.d_n_neigh, nb.n_neigh, ns);
  copy_int(nbk.d_species, nb.species, ns);
  copy_int(nbk.d_jatom, nb.jatom, np);
  copy_int(nbk.d_neighbor_species, nb.neighbor_species, np);
  copy_int(nbk.d_neighbors_list, nb.neighbors_list, np);
  copy_dbl(nbk.d_rjs, nb.rjs, np);
  copy_dbl(nbk.d_xyz, nb.xyz, 3 * np);
  copy_dbl(nbk.d_thetas, nb.thetas, np);
  copy_dbl(nbk.d_phis, nb.phis, np);
  for (int s = 0; s < ns; s++) nb.site_atom[s] = s;

  // --- cheap invariants, on request
  //
  // NOT a comparison against a host-built list, which was the first thing
  // tried and cannot work: under KOKKOS the host numneigh/firstneigh arrays
  // are empty because the list lives on the device, so the host builder
  // returns one pair per site -- the atom itself -- and "device 4929, host 91"
  // reads like a catastrophic bug in the device build rather than an empty
  // input to the thing it was being compared with.
  //
  // What can be checked here is that the list is internally consistent. What
  // actually establishes stage one is end to end: turbogap/kk against
  // turbogap, same configuration, in tests/run.sh.
  if (!checked_against_host && getenv("TURBOGAP_CHECK_NEIGH")) {
    checked_against_host = 1;
    long total = 0;
    int bad_self = 0;
    for (int i = 0; i < ns; i++) total += nb.n_neigh[i];
    for (int i = 0, k = 0; i < ns; i++) {
      if (nb.jatom[k] != i || nb.rjs[k] != 0.0) bad_self++;
      k += nb.n_neigh[i];
    }
    if (total != np || bad_self)
      error->warning(FLERR,
                     "turbogap/kk: neighbour list is inconsistent -- sum(n_neigh) {} against {} "
                     "pairs, {} sites whose first pair is not themselves at r = 0",
                     total, np, bad_self);
    else
      utils::logmesg(lmp, "turbogap/kk: {} sites, {} pairs, each site first in its own list\n", ns,
                     np);
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void PairTurboGAPKokkos<DeviceType>::compute(int eflag, int vflag)
{
  // Positions and types are needed in both spaces in stage one: on the device
  // for the neighbour build, on the host for the terms that read the list back.
  // Stage two drops the second sync along with the copy-back.
  atomKK->sync(Device, X_MASK | TYPE_MASK);
  atomKK->sync(Host, X_MASK | TYPE_MASK | F_MASK);

  PairTurboGAP::compute(eflag, vflag);

  atomKK->modified(Host, F_MASK);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairTurboGAPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairTurboGAPKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
