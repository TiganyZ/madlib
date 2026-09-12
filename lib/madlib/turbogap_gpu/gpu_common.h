/* ---------------------------------------------------------------------------
   A stand-in for TurboGAP's src/gpu/gpu_common.h, for building its device
   kernels inside LAMMPS.

   TurboGAP's own gpu_common.h pulls in hop -- its HIP-on-CUDA shim -- and
   hipBLAS, because the files that include it collectively need both. The two
   kernels lifted here need NEITHER: gap_2b.cu contains no vendor API call at
   all, only kernel bodies dispatched through tg_parallel_for.

   Bringing hop and hipBLAS into a LAMMPS build to satisfy an include that is
   not used would mean two device runtimes and two allocators beside LAMMPS's
   own Kokkos, for nothing. So the names those files expect are supplied here
   and the dependency is not taken. See docs/TURBOGAP-KK.md section 4, where
   this is "option 3" applied at one point of use.

   hipStream_t exists only because it appears in the signatures. Nothing here
   dereferences it: under Kokkos the work runs in LAMMPS's execution space
   instance, and a caller passes nullptr.
--------------------------------------------------------------------------- */

#ifndef MADLIB_GPU_COMMON_H
#define MADLIB_GPU_COMMON_H

#include <cmath>
#include <cstdio>

typedef void *hipStream_t;

// TurboGAP's launch-geometry constants. Kokkos chooses its own team sizes, so
// these only survive as the block-size hints the kernels pass along.
#define WARP_SIZE 32
#define BLOCK_SIZE 512

#endif
