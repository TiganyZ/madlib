/* ---------------------------------------------------------------------------
   The declarations of the TurboGAP device entry points MADLIB calls.

   A subset of TurboGAP's src/gpu/gap_gpu.h, carrying only what the lifted
   kernels define, and kept character-for-character identical to it so that a
   drift in either signature is a compile error here rather than a silent
   mismatch across a C boundary.
--------------------------------------------------------------------------- */

#ifndef MADLIB_GAP_GPU_H
#define MADLIB_GAP_GPU_H

#include "gpu_common.h"

extern "C" {

// ---- gap_2b.cu
void gpu_get_2b_forces_energies(int i_beg, int i_end, int n_sparse, double* energies_d, double e0, int* n_neigh_d, bool do_forces,
                                double* forces_d, double* virial_d, double* rjs_d, double rcut, int* species_d,
                                int* neighbor_species_d, int sp1, int sp2, double buffer, double delta, double* cutoff_d,
                                double* Qs_d, double sigma, double* alphas_d, double* xyz_d, hipStream_t* stream);
void gpu_get_core_pot_energy_and_forces(int i_beg, int i_end, bool do_forces, int* species_d, int sp1, int sp2, int* n_neigh_d,
                                        int* neighbor_species_d, double* rjs_d, int n_sparse, double* x_d, double* V_d,
                                        double* dVdx2_d, double yp1, double ypn, double* xyz_d, double* forces_d, double* virial_d,
                                        double* energies_d, hipStream_t* stream);

}    // extern "C"

#endif
