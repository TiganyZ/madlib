! ---------------------------------------------------------------------------
!  SOAP on the device: the descriptor, and the energy and forces from it.
!
!  TurboGAP's GPU soap_turbo is a drop-in for the CPU one at the module level --
!  same module name, same entry point -- and it takes cublas_handle and
!  gpu_stream as arguments rather than reading them from gpu_context. So it is
!  callable, and this is a shim rather than a reimplementation: the ~1500 lines
!  of orchestration inside get_soap, and the sparse-GP prediction in
!  get_soap_energy_and_forces, are TurboGAP's own.
!
!  gap.f90's get_soap_energy_and_forces is the one place that DOES read module
!  state: cublas_handle and gpu_stream from gpu_context, and alphas_d/Qs_d which
!  are private to gap.f90 and set by soap_backend_begin. Both are reachable --
!  gpu_context is `public`, so the handle and stream can be assigned directly
!  without gpu_context_init, whose paired finalize calls gpu_device_reset and
!  would destroy the host application's context. soap_backend_begin needs only
!  five fields of a soap_turbo, so a minimal one is built here.
!
!  Built only in a GPU configuration: it needs F_B_C and the device objects.
! ---------------------------------------------------------------------------

module madlib_soap_gpu

   use iso_c_binding
   use F_B_C
   use soap_turbo_desc, only: get_soap
   use gap, only: get_soap_energy_and_forces, soap_backend_begin, soap_backend_end
   use types, only: soap_turbo
   use gpu_context, only: ctx_cublas_handle => cublas_handle, ctx_gpu_stream => gpu_stream

   implicit none

   integer, parameter :: dp = kind(0.d0)

!  Created once and kept. create_cublas_handle makes the stream too, so this
!  owns both. A later revision should adopt LAMMPS's Kokkos stream instead of
!  making one, so the two do not serialise against each other; for now the
!  question being answered is whether the numbers are right.
   type(c_ptr), save :: cublas_handle
   type(c_ptr), save :: gpu_stream
   logical, save :: have_handle = .false.

!  The basis state get_soap computes on its first call and then reuses. It
!  depends on n_max, l_max and rcut_hard only -- not on the number of sites --
!  so it survives a change of problem size, which is what makes keeping it here
!  safe across MD steps.
   type(c_ptr), save :: W_d
   type(c_ptr), save :: S_d
   type(c_ptr), save :: multiplicity_array_d
   integer(c_size_t), save :: st_W_d = 0
   integer(c_size_t), save :: st_S_d = 0
   integer(c_size_t), save :: st_multiplicity_array_d = 0
   logical, save :: recompute_basis = .true.

!  What one descriptor call allocates. Bundled so the two public entries can
!  share the setup without an argument list of thirteen device pointers, and so
!  that releasing them is one call that cannot half-happen.
   type :: soap_buffers
      type(c_ptr) :: soap_d
      type(c_ptr) :: der_d
      type(c_ptr) :: hyper_d(11)
      integer(c_size_t) :: st_soap = 0
      integer(c_size_t) :: st_der = 0
   end type soap_buffers

contains

   subroutine madlib_soap_gpu_init() bind(c, name="madlib_soap_gpu_init")
      if (have_handle) return
      call create_cublas_handle(cublas_handle, gpu_stream)
!     gap.f90 reads these from the module rather than taking them as arguments.
!     Assigned rather than obtained through gpu_context_init, because that
!     selects a device and its paired finalize calls gpu_device_reset -- which
!     inside a host application would tear down a context this code does not own.
      ctx_cublas_handle = cublas_handle
      ctx_gpu_stream = gpu_stream
      have_handle = .true.
   end subroutine madlib_soap_gpu_init

!  Drop the cached basis, so the next call recomputes it. Needed when the
!  hyperparameters change under a running process -- a second pair_style, or a
!  potential reloaded -- because get_soap will otherwise reuse matrices built
!  for the old ones and say nothing.
   subroutine madlib_soap_gpu_reset() bind(c, name="madlib_soap_gpu_reset")
      if (st_W_d > 0) then
         call gpu_free_async(W_d, gpu_stream)
         call gpu_free_async(S_d, gpu_stream)
         call gpu_free_async(multiplicity_array_d, gpu_stream)
         st_W_d = 0
         st_S_d = 0
         st_multiplicity_array_d = 0
      end if
      recompute_basis = .true.
   end subroutine madlib_soap_gpu_reset

!  The descriptor, left on the device. Shared by both public entries.
   subroutine run_get_soap(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, phis, &
                           alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                           atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                           atom_sigma_t_scaling, amplitude_scaling, radial_enhancement, &
                           central_weight, basis, scaling_mode, compress, n_nonzero, p_j, dim, &
                           do_der, buf, soap, soap_cart_der)
      integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim
      integer(c_int), intent(in), target :: n_neigh(*), species_idx(*), alpha_max(*), p_j(*)
      real(c_double), intent(in), target :: rjs(*), thetas(*), phis(*)
      real(c_double), intent(in), target :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in), target :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      logical, intent(in) :: do_der
      type(soap_buffers), intent(out) :: buf
      real(dp), allocatable, intent(out) :: soap(:, :)
      real(dp), allocatable, intent(out) :: soap_cart_der(:, :, :)

      type(c_ptr) :: n_neigh_d
      integer(c_size_t) :: st_dbl, st_int
      logical, allocatable :: mask(:, :)
      integer, allocatable, target :: species(:, :), species_multiplicity(:)
      integer, allocatable, target :: comp_idx(:)
      integer :: ncomp
      real(dp) :: time_get_soap
      character(len=32) :: bas, smode

      call madlib_soap_gpu_init()
      bas = c_to_f_string_gpu(basis)
      smode = c_to_f_string_gpu(scaling_mode)
      time_get_soap = 0.d0

      call build_mask_gpu(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                          mask, species, species_multiplicity)

!     get_soap indexes compress_soap_indices unconditionally, so an
!     uncompressed run still needs a length-1 array to point at.
      if (compress /= 0) then
         ncomp = n_nonzero
         allocate (comp_idx(1:ncomp))
         comp_idx(1:ncomp) = p_j(1:ncomp)
      else
         ncomp = 1
         allocate (comp_idx(1:1))
         comp_idx(1) = 0
      end if

      st_dbl = int(n_species, c_size_t)*8_c_size_t
      st_int = int(n_species, c_size_t)*4_c_size_t
      call upload(c_loc(nf), buf%hyper_d(1), st_dbl)
      call upload(c_loc(rcut_hard), buf%hyper_d(2), st_dbl)
      call upload(c_loc(rcut_soft), buf%hyper_d(3), st_dbl)
      call upload(c_loc(global_scaling), buf%hyper_d(4), st_dbl)
      call upload(c_loc(atom_sigma_r), buf%hyper_d(5), st_dbl)
      call upload(c_loc(atom_sigma_r_scaling), buf%hyper_d(6), st_dbl)
      call upload(c_loc(atom_sigma_t), buf%hyper_d(7), st_dbl)
      call upload(c_loc(atom_sigma_t_scaling), buf%hyper_d(8), st_dbl)
      call upload(c_loc(amplitude_scaling), buf%hyper_d(9), st_dbl)
      call upload(c_loc(central_weight), buf%hyper_d(10), st_dbl)
      call upload(c_loc(alpha_max), buf%hyper_d(11), st_int)

      buf%st_soap = int(dim, c_size_t)*int(n_sites, c_size_t)*8_c_size_t
      if (do_der) then
         buf%st_der = 3_c_size_t*int(dim, c_size_t)*int(n_pairs, c_size_t)*8_c_size_t
      else
         buf%st_der = 8_c_size_t
      end if
      call gpu_malloc_async(buf%soap_d, buf%st_soap, gpu_stream)
      call gpu_malloc_async(buf%der_d, buf%st_der, gpu_stream)

!     n_neigh_d is NOT allocated here. It sits in get_soap's argument list but
!     is declared there without an intent and is both allocated (:698) and
!     freed (:1276) inside, so it is scratch wearing an argument's clothes.
!     Allocating it here leaks; freeing it here is a double free, which is what
!     it was -- hipFreeAsync returning "invalid argument" from gpu_memory.cu.

      allocate (soap(1:dim, 1:n_sites))
      if (do_der) then
         allocate (soap_cart_der(1:3, 1:dim, 1:n_pairs))
      else
         allocate (soap_cart_der(1:3, 1:dim, 1:1))
      end if
      soap = 0.d0
      soap_cart_der = 0.d0

      call get_soap(n_sites, n_neigh(1:n_sites), n_species, species, species_multiplicity, &
                    n_pairs, mask, rjs(1:n_pairs), thetas(1:n_pairs), phis(1:n_pairs), &
                    buf%hyper_d(11), alpha_max(1:n_species), l_max, buf%hyper_d(2), &
                    rcut_hard(1:n_species), buf%hyper_d(3), buf%hyper_d(1), buf%hyper_d(4), &
                    buf%hyper_d(5), atom_sigma_r(1:n_species), buf%hyper_d(6), &
                    buf%hyper_d(7), buf%hyper_d(8), buf%hyper_d(9), &
                    radial_enhancement, buf%hyper_d(10), central_weight(1:n_species), &
                    trim(bas), trim(smode), .false., do_der, compress /= 0, &
                    comp_idx(1:ncomp), soap, soap_cart_der, time_get_soap, &
                    buf%soap_d, buf%der_d, n_neigh_d, W_d, S_d, multiplicity_array_d, &
                    st_W_d, st_S_d, st_multiplicity_array_d, recompute_basis, &
                    cublas_handle, gpu_stream)

      deallocate (mask, species, species_multiplicity, comp_idx)
   end subroutine run_get_soap

   subroutine release(buf)
      type(soap_buffers), intent(inout) :: buf
      integer :: i
      call gpu_free_async(buf%soap_d, gpu_stream)
      call gpu_free_async(buf%der_d, gpu_stream)
      do i = 1, 11
         call gpu_free_async(buf%hyper_d(i), gpu_stream)
      end do
   end subroutine release

   subroutine madlib_soap_descriptor_gpu(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                                         rjs, thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, &
                                         nf, global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                         atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, &
                                         radial_enhancement, central_weight, basis, scaling_mode, &
                                         compress, n_nonzero, p_j, dim, do_derivatives, &
                                         soap_out, der_out) &
      bind(c, name="madlib_soap_descriptor_gpu")

      integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, do_derivatives
      integer(c_int), intent(in), target :: n_neigh(*), species_idx(*), alpha_max(*), p_j(*)
      real(c_double), intent(in), target :: rjs(*), thetas(*), phis(*)
      real(c_double), intent(in), target :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in), target :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      real(c_double), intent(out) :: soap_out(*), der_out(*)

      type(soap_buffers) :: buf
      real(dp), allocatable, target :: soap(:, :), soap_cart_der(:, :, :)
      logical :: do_der
      integer :: i, j, k, n

      do_der = (do_derivatives /= 0)
      call run_get_soap(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, phis, &
                        alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                        atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                        amplitude_scaling, radial_enhancement, central_weight, basis, &
                        scaling_mode, compress, n_nonzero, p_j, dim, do_der, buf, soap, &
                        soap_cart_der)

!     get_soap leaves the descriptor on the device; nothing copies it back.
      call cpy_dtoh_blocking(buf%soap_d, c_loc(soap), buf%st_soap)
      if (do_der) call cpy_dtoh_blocking(buf%der_d, c_loc(soap_cart_der), buf%st_der)

      n = 0
      do j = 1, n_sites
         do i = 1, dim
            n = n + 1
            soap_out(n) = soap(i, j)
         end do
      end do
      if (do_der) then
         n = 0
         do k = 1, n_pairs
            do j = 1, dim
               do i = 1, 3
                  n = n + 1
                  der_out(n) = soap_cart_der(i, j, k)
               end do
            end do
         end do
      end if

      call release(buf)
      deallocate (soap, soap_cart_der)
   end subroutine madlib_soap_descriptor_gpu

!  Energy, forces and virial for one SOAP descriptor, all on the device.
!
!  forces is (3, n_force) and n_force may exceed n_sites: TurboGAP writes
!  de_i/dr_j for every neighbour j, so under LAMMPS it spans local atoms and
!  ghosts. energies is over sites.
   subroutine madlib_soap_energy_forces_gpu(n_sites, n_pairs, n_force, n_species, n_neigh, &
                                            neighbors_list, species_idx, rjs, thetas, phis, xyz, &
                                            alpha_max, l_max, rcut_hard, rcut_soft, nf, &
                                            global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                            atom_sigma_t, atom_sigma_t_scaling, &
                                            amplitude_scaling, radial_enhancement, &
                                            central_weight, basis, scaling_mode, compress, &
                                            n_nonzero, p_j, dim, n_sparse, qs, alphas, &
                                            delta, zeta, e0, do_forces, &
                                            energies, forces, virial) &
      bind(c, name="madlib_soap_energy_forces_gpu")

      integer(c_int), value :: n_sites, n_pairs, n_force, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, n_sparse, do_forces
      integer(c_int), intent(in), target :: n_neigh(*), species_idx(*), alpha_max(*), p_j(*)
      integer(c_int), intent(in) :: neighbors_list(*)
      real(c_double), intent(in), target :: rjs(*), thetas(*), phis(*), xyz(*)
      real(c_double), intent(in), target :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in), target :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      real(c_double), intent(in), target :: qs(*), alphas(*)
      real(c_double), value :: delta, zeta, e0
      real(c_double), intent(out) :: energies(*), forces(*), virial(*)

      type(soap_buffers) :: buf
      type(soap_turbo), target :: hypers
      type(c_ptr) :: n_neigh_d
      real(dp), allocatable, target :: soap(:, :), soap_cart_der(:, :, :)
      real(dp), allocatable :: e_loc(:), f_loc(:, :), v_loc(:, :)
      real(dp) :: solo_time
      integer(c_int) :: np
      integer :: i, j, n
      logical :: do_f

      do_f = (do_forces /= 0)
      call run_get_soap(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, phis, &
                        alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                        atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                        amplitude_scaling, radial_enhancement, central_weight, basis, &
                        scaling_mode, compress, n_nonzero, p_j, dim, do_f, buf, soap, &
                        soap_cart_der)

!     soap_backend_begin reads only n_sparse, dim, alphas, Qs and the kind of
!     nf(1), so a minimal hypers carries what it needs rather than a whole
!     parsed descriptor.
      hypers%n_sparse = n_sparse
      hypers%dim = dim
      allocate (hypers%nf(1:1))
      hypers%nf(1) = 0.d0
      allocate (hypers%alphas(1:n_sparse))
      hypers%alphas(1:n_sparse) = alphas(1:n_sparse)
      allocate (hypers%Qs(1:dim, 1:n_sparse))
      n = 0
      do j = 1, n_sparse
         do i = 1, dim
            n = n + 1
            hypers%Qs(i, j) = qs(n)
         end do
      end do
      call soap_backend_begin(hypers)

      allocate (e_loc(1:n_sites), f_loc(1:3, 1:n_force), v_loc(1:3, 1:3))
      e_loc = 0.d0
      f_loc = 0.d0
      v_loc = 0.d0
      solo_time = 0.d0
      np = n_pairs

!     soap and soap_cart_der are passed for their SHAPE: the routine reads
!     n_soap and n_sites from them and takes the values from the device.
!     n_neigh_d is the same scratch-wearing-an-argument as in run_get_soap, and
!     is not referenced in the body.
      call get_soap_energy_and_forces(n_sparse, soap, soap_cart_der, delta, zeta, e0, &
                                      n_neigh(1:n_sites), neighbors_list(1:n_pairs), &
                                      reshape(xyz(1:3*n_pairs), [3, n_pairs]), do_f, .false., &
                                      e_loc, f_loc, v_loc, solo_time, buf%soap_d, buf%der_d, &
                                      n_neigh_d, np)

      call soap_backend_end()

      energies(1:n_sites) = e_loc(1:n_sites)
      if (do_f) then
         n = 0
         do j = 1, n_force
            do i = 1, 3
               n = n + 1
               forces(n) = f_loc(i, j)
            end do
         end do
         n = 0
         do j = 1, 3
            do i = 1, 3
               n = n + 1
               virial(n) = v_loc(i, j)
            end do
         end do
      end if

      call release(buf)
      deallocate (soap, soap_cart_der, e_loc, f_loc, v_loc)
      deallocate (hypers%nf, hypers%alphas, hypers%Qs)
   end subroutine madlib_soap_energy_forces_gpu

   subroutine upload(src, dst, nbytes)
      type(c_ptr), intent(in) :: src
      type(c_ptr), intent(out) :: dst
      integer(c_size_t), intent(in) :: nbytes
      call gpu_malloc_async(dst, nbytes, gpu_stream)
      call cpy_htod(src, dst, nbytes, gpu_stream)
   end subroutine upload

   subroutine build_mask_gpu(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                             mask, species, species_multiplicity)
      integer, intent(in) :: n_sites, n_pairs, n_species
      integer(c_int), intent(in) :: n_neigh(*), species_idx(*)
      logical, allocatable, intent(out) :: mask(:, :)
      integer, allocatable, intent(out) :: species(:, :)
      integer, allocatable, intent(out) :: species_multiplicity(:)

      integer :: k, s

      allocate (mask(1:n_pairs, 1:n_species))
      mask = .false.
      do k = 1, n_pairs
         s = species_idx(k)
         if (s >= 1 .and. s <= n_species) mask(k, s) = .true.
      end do

      allocate (species(1:1, 1:n_sites))
      allocate (species_multiplicity(1:n_sites))
      species_multiplicity = 1
      k = 1
      do s = 1, n_sites
         species(1, s) = species_idx(k)
         k = k + n_neigh(s)
      end do
   end subroutine build_mask_gpu

   function c_to_f_string_gpu(s) result(f)
      character(kind=c_char), intent(in) :: s(*)
      character(len=32) :: f
      integer :: i
      f = " "
      do i = 1, 32
         if (s(i) == c_null_char) exit
         f(i:i) = s(i)
      end do
   end function c_to_f_string_gpu

end module madlib_soap_gpu
