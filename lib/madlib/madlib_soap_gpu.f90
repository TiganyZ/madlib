! ---------------------------------------------------------------------------
!  The SOAP descriptor on the device.
!
!  TurboGAP's GPU soap_turbo is a drop-in for the CPU one at the module level:
!  same module name, same entry point, same semantics. What differs is that the
!  hyperparameters arrive as device pointers, the descriptor is left on the
!  device, and the orthonormalisation matrices are caller-owned state computed
!  on the first call.
!
!  So this is a shim, not a reimplementation. The 1500 lines of orchestration
!  inside get_soap -- radial expansion, angular expansion, cnk, power spectrum,
!  normalisation, and the batching of all of it -- are TurboGAP's own and are
!  called rather than restated. What is here is the uploading, the persistent
!  state, and the copy back.
!
!  Built only in a GPU configuration: it needs F_B_C and the device objects.
! ---------------------------------------------------------------------------

module madlib_soap_gpu

   use iso_c_binding
   use F_B_C
   use soap_turbo_desc, only: get_soap

   implicit none

   integer, parameter :: dp = kind(0.d0)

!  Created once and kept. create_cublas_handle makes the stream too, so this
!  owns both. A later revision should accept LAMMPS's Kokkos stream instead of
!  making one, so the two do not serialise against each other; for now the
!  question being answered is whether the descriptor is right, not whether it
!  overlaps.
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

contains

   subroutine madlib_soap_gpu_init() bind(c, name="madlib_soap_gpu_init")
      if (have_handle) return
      call create_cublas_handle(cublas_handle, gpu_stream)
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

      type(c_ptr) :: alpha_max_d, rcut_hard_d, rcut_soft_d, nf_d, global_scaling_d
      type(c_ptr) :: atom_sigma_r_d, atom_sigma_r_scaling_d
      type(c_ptr) :: atom_sigma_t_d, atom_sigma_t_scaling_d
      type(c_ptr) :: amplitude_scaling_d, central_weight_d
      type(c_ptr) :: soap_d, soap_cart_der_d, n_neigh_d
      integer(c_size_t) :: st_dbl, st_int, st_soap, st_der
      real(dp), allocatable, target :: soap(:, :), soap_cart_der(:, :, :)
      logical, allocatable :: mask(:, :)
      integer, allocatable, target :: species(:, :), species_multiplicity(:)
      integer, allocatable, target :: comp_idx(:)
      integer :: i, j, k, n, ncomp
      real(dp) :: time_get_soap
      logical :: do_der
      character(len=32) :: bas, smode

      call madlib_soap_gpu_init()

      do_der = (do_derivatives /= 0)
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

      call upload(c_loc(nf), nf_d, st_dbl)
      call upload(c_loc(rcut_hard), rcut_hard_d, st_dbl)
      call upload(c_loc(rcut_soft), rcut_soft_d, st_dbl)
      call upload(c_loc(global_scaling), global_scaling_d, st_dbl)
      call upload(c_loc(atom_sigma_r), atom_sigma_r_d, st_dbl)
      call upload(c_loc(atom_sigma_r_scaling), atom_sigma_r_scaling_d, st_dbl)
      call upload(c_loc(atom_sigma_t), atom_sigma_t_d, st_dbl)
      call upload(c_loc(atom_sigma_t_scaling), atom_sigma_t_scaling_d, st_dbl)
      call upload(c_loc(amplitude_scaling), amplitude_scaling_d, st_dbl)
      call upload(c_loc(central_weight), central_weight_d, st_dbl)
      call upload(c_loc(alpha_max), alpha_max_d, st_int)

      st_soap = int(dim, c_size_t)*int(n_sites, c_size_t)*8_c_size_t
      if (do_der) then
         st_der = 3_c_size_t*int(dim, c_size_t)*int(n_pairs, c_size_t)*8_c_size_t
      else
         st_der = 8_c_size_t
      end if
      call gpu_malloc_async(soap_d, st_soap, gpu_stream)
      call gpu_malloc_async(soap_cart_der_d, st_der, gpu_stream)

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
                    alpha_max_d, alpha_max(1:n_species), l_max, rcut_hard_d, &
                    rcut_hard(1:n_species), rcut_soft_d, nf_d, global_scaling_d, &
                    atom_sigma_r_d, atom_sigma_r(1:n_species), atom_sigma_r_scaling_d, &
                    atom_sigma_t_d, atom_sigma_t_scaling_d, amplitude_scaling_d, &
                    radial_enhancement, central_weight_d, central_weight(1:n_species), &
                    trim(bas), trim(smode), .false., do_der, compress /= 0, &
                    comp_idx(1:ncomp), soap, soap_cart_der, time_get_soap, &
                    soap_d, soap_cart_der_d, n_neigh_d, W_d, S_d, multiplicity_array_d, &
                    st_W_d, st_S_d, st_multiplicity_array_d, recompute_basis, &
                    cublas_handle, gpu_stream)

!     get_soap leaves the descriptor on the device; nothing copies it back.
      call cpy_dtoh_blocking(soap_d, c_loc(soap), st_soap)
      if (do_der) call cpy_dtoh_blocking(soap_cart_der_d, c_loc(soap_cart_der), st_der)

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

      call gpu_free_async(soap_d, gpu_stream)
      call gpu_free_async(soap_cart_der_d, gpu_stream)
      call gpu_free_async(nf_d, gpu_stream)
      call gpu_free_async(rcut_hard_d, gpu_stream)
      call gpu_free_async(rcut_soft_d, gpu_stream)
      call gpu_free_async(global_scaling_d, gpu_stream)
      call gpu_free_async(atom_sigma_r_d, gpu_stream)
      call gpu_free_async(atom_sigma_r_scaling_d, gpu_stream)
      call gpu_free_async(atom_sigma_t_d, gpu_stream)
      call gpu_free_async(atom_sigma_t_scaling_d, gpu_stream)
      call gpu_free_async(amplitude_scaling_d, gpu_stream)
      call gpu_free_async(central_weight_d, gpu_stream)
      call gpu_free_async(alpha_max_d, gpu_stream)

      deallocate (soap, soap_cart_der, mask, species, species_multiplicity, comp_idx)
   end subroutine madlib_soap_descriptor_gpu

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
