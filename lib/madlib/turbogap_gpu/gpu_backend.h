/* ---------------------------------------------------------------------------
   TurboGAP's kernel-dispatch layer, on LAMMPS's Kokkos.

   TurboGAP's own src/gpu/gpu_backend.h dispatches either to a raw <<<>>>
   launch or, under -D_KOKKOS, to Kokkos on a stream its Fortran context owns.
   Inside LAMMPS neither of those is wanted: there is exactly one Kokkos, it
   belongs to LAMMPS, and the raw-launch path would mean a second device
   runtime beside it.

   So this supplies the same three names against LAMMPS's Kokkos and nothing
   else, which is what lets the kernel files be lifted from TurboGAP VERBATIM.
   That is the point of the exercise: an edited copy could not be diffed
   against the original, and these kernels are fitted-potential arithmetic that
   nobody should be re-deriving.

   The stream argument is accepted and ignored. TurboGAP passes the stream its
   gpu_context owns; here the work runs in Kokkos's default execution space,
   which is LAMMPS's, and callers pass nullptr. Keeping the parameter is what
   keeps the signatures identical.
--------------------------------------------------------------------------- */

#ifndef MADLIB_GPU_BACKEND_H
#define MADLIB_GPU_BACKEND_H

#include "gpu_common.h"

#include <Kokkos_Core.hpp>

#define TG_LAMBDA KOKKOS_LAMBDA
#define TG_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION

// Kokkos picks its own team size, so the block hint the kernels pass is
// accepted and dropped, exactly as the stream is.
template <class Functor>
inline void tg_parallel_for(const char *name, int n, hipStream_t * /*stream*/, const Functor &f,
                            int /*block*/ = 256)
{
  if (n <= 0) return;
  Kokkos::parallel_for(name, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), f);
}

template <class Functor>
inline void tg_parallel_for_long(const char *name, long long n, hipStream_t * /*stream*/,
                                 const Functor &f, int /*block*/ = 256)
{
  if (n <= 0) return;
  Kokkos::parallel_for(
      name,
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<long long>>(0, n), f);
}

// Kokkos::atomic_add returns void; only atomic_fetch_add returns the old value.
#define TG_ATOMIC_ADD(addr, val) Kokkos::atomic_add((addr), (val))
#define TG_ATOMIC_FETCH_ADD(addr, val) Kokkos::atomic_fetch_add((addr), (val))

#endif
