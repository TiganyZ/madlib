/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
------------------------------------------------------------------------- */

/* TurboGAP's two-body and core-potential device kernels, compiled into LAMMPS.

   The kernels themselves are NOT here. lib/madlib/turbogap_gpu/gap_2b.cu is a
   verbatim copy of TurboGAP's src/gpu/gap_2b.cu, and it is included rather
   than transcribed so that it stays diffable against the original -- these are
   two pieces of fitted-potential arithmetic and a spline evaluation that
   nobody should be re-deriving, and an edited copy could not be compared with
   what it came from.

   What makes the verbatim include possible is that gap_2b.cu contains NO
   vendor API call at all: every kernel in it goes through tg_parallel_for, so
   the three headers it includes can be supplied by shims that sit on LAMMPS's
   own Kokkos instead of on hop and hipBLAS. See lib/madlib/turbogap_gpu.

   It is compiled as a .cpp because that is what LAMMPS's build understands,
   and it needs no CUDA language support of its own: under Kokkos the kernels
   are lambdas, and nvcc_wrapper is already the compiler for this package. */

#include "gap_2b.cu"
