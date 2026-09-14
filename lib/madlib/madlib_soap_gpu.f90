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
   use mpi
   use F_B_C
   use madlib_soap_dims, only: c_to_f_string
   use soap_turbo_desc, only: get_soap
   use gap, only: get_soap_energy_and_forces, soap_backend_begin, soap_backend_end
   use local_prop, only: local_property_predict, gpu_local_property_predict
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

!  Ranks sharing this rank's GPU; 0 until device_share() works it out.
   integer, save :: ranks_per_device = 0

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

!  What one SOAP batch costs, per pair and per site.
!
!  Restated from neighbors.f90's soap_batch_memory_model (the _GPU branch, which
!  differs from the CPU one -- the device holds more per pair). Restated rather
!  than called because that routine lives in a module this does not otherwise
!  need, and the formula is six terms. If it drifts upstream, soapbatchverify
!  is what notices: it checks the split against the real allocation.
   subroutine batch_memory_model(l_max, n_max, n_soap, n_species, bytes_per_pair, bytes_per_site)
      integer, intent(in) :: l_max
      integer, intent(in) :: n_max
      integer, intent(in) :: n_soap
      integer, intent(in) :: n_species
      real(dp), intent(out) :: bytes_per_pair
      real(dp), intent(out) :: bytes_per_site

      integer :: k_max

      k_max = 1 + l_max*(l_max + 1)/2 + l_max

      bytes_per_pair = 48.d0*dfloat(k_max)*dfloat(n_max) &
                       + 64.d0*dfloat(k_max) &
                       + 72.d0*dfloat(n_soap) &
                       + 40.d0*dfloat(n_max) &
                       + 8.d0*dfloat(n_species) &
                       + 88.d0

      bytes_per_site = 16.d0*dfloat(k_max)*dfloat(n_max) &
                       + 16.d0*dfloat(n_soap) &
                       + 24.d0
   end subroutine batch_memory_model

!  How many batches this configuration needs to fit in max_bytes, and where the
!  site boundaries fall.
!
!  Boundaries are chosen so each batch holds roughly the same number of PAIRS,
!  not the same number of sites: the cost is dominated by pairs and a site with
!  many neighbours costs many times one with few. Sites stay contiguous and in
!  order, which is what lets a batch's energies slot straight back into the
!  caller's array.
!
!  This is the loop at turbogap.f90:1792, which is in the driver rather than in
!  anything callable -- so an external caller reimplements it. TurboGAP takes
!  the batch count from a keyword on the GPU path; here it is derived from a
!  byte budget, because a LAMMPS user has no keyword to set and the number of
!  atoms on a rank is not theirs to choose.
   subroutine plan_batches(n_sites, n_pairs, n_neigh, l_max, n_max, n_soap, n_species, &
                           max_bytes, n_chunks, i_beg, i_end, j_beg, j_end)
      integer, intent(in) :: n_sites, n_pairs, l_max, n_max, n_soap, n_species
      integer(c_int), intent(in) :: n_neigh(*)
      integer(c_size_t), intent(in) :: max_bytes
      integer, intent(out) :: n_chunks
      integer, allocatable, intent(out) :: i_beg(:), i_end(:), j_beg(:), j_end(:)

      real(dp), parameter :: SAFETY = 1.10d0
      real(dp) :: bytes_per_pair, bytes_per_site, need, pairs_per_chunk
      integer :: i, k, k2, chunk

      call batch_memory_model(l_max, n_max, n_soap, n_species, bytes_per_pair, bytes_per_site)
      need = SAFETY*(dfloat(n_pairs)*bytes_per_pair + dfloat(n_sites)*bytes_per_site)

      if (max_bytes <= 0_c_size_t) then
         n_chunks = 1
      else
         n_chunks = max(1, ceiling(need/dfloat(max_bytes)))
      end if
      if (n_chunks > n_sites) n_chunks = n_sites

      allocate (i_beg(1:n_chunks), i_end(1:n_chunks))
      allocate (j_beg(1:n_chunks), j_end(1:n_chunks))

      i_beg(1) = 1
      j_beg(1) = 1
      i_end(n_chunks) = n_sites
      j_end(n_chunks) = n_pairs
      if (n_chunks == 1) return

      pairs_per_chunk = dfloat(n_pairs)/dfloat(n_chunks)
      k = 0
      k2 = 0
      chunk = 1
      do i = 1, n_sites
         do k2 = 1, n_neigh(i)
            k = k + 1
         end do
         if (dfloat(k) >= dfloat(chunk)*pairs_per_chunk) then
            i_end(chunk) = i
            j_end(chunk) = k
            chunk = chunk + 1
            if (chunk > n_chunks) exit
            i_beg(chunk) = i + 1
            j_beg(chunk) = k + 1
            if (chunk == n_chunks) exit
         end if
      end do
   end subroutine plan_batches

!  The descriptor, left on the device. Shared by both public entries.
   subroutine run_get_soap(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, phis, &
                           alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                           atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                           atom_sigma_t_scaling, amplitude_scaling, radial_enhancement, &
                           central_weight, basis, scaling_mode, compress, n_nonzero, p_j, dim, &
                           do_der, buf, soap, soap_cart_der, zero_host)
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
!     Whether the caller is going to read the host arrays. The device get_soap
!     never writes them -- its copy back to the host is commented out and the
!     results are read from soap_d and der_d -- so for a caller that stays on
!     the device they are write-only scratch that exists to satisfy the
!     argument list, and touching them is hundreds of MB of page faults per
!     batch for nothing.
      logical, intent(in), optional :: zero_host
      type(soap_buffers), intent(out) :: buf
      real(dp), allocatable, intent(out) :: soap(:, :)
      real(dp), allocatable, intent(out) :: soap_cart_der(:, :, :)
      logical :: zero_them

      type(c_ptr) :: n_neigh_d
      integer(c_size_t) :: st_dbl, st_int
      logical, allocatable :: mask(:, :)
      integer, allocatable, target :: species(:, :), species_multiplicity(:)
      integer, allocatable, target :: comp_idx(:)
      integer :: ncomp
      real(dp) :: time_get_soap
      character(len=32) :: bas, smode

      call madlib_soap_gpu_init()
      bas = c_to_f_string(basis)
      smode = c_to_f_string(scaling_mode)
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
      zero_them = .true.
      if (present(zero_host)) zero_them = zero_host
      if (zero_them) then
         soap = 0.d0
         soap_cart_der = 0.d0
      end if

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
                                            delta, zeta, e0, do_forces, max_bytes, &
                                            energies, forces, virial) &
      bind(c, name="madlib_soap_energy_forces_gpu")

      integer(c_int), value :: n_sites, n_pairs, n_force, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, n_sparse, do_forces
      integer(c_size_t), value :: max_bytes
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

      type(soap_turbo), target :: hypers
      real(dp), allocatable :: e_b(:), f_b(:, :), v_b(:, :)
      integer, allocatable :: i_beg(:), i_end(:), j_beg(:), j_end(:)
      integer :: n_chunks, c, ns_b, np_b, ib, ie, jb, je, i, j, n
      logical :: do_f

      do_f = (do_forces /= 0)
      call madlib_soap_gpu_init()

      energies(1:n_sites) = 0.d0
      if (do_f) then
         forces(1:3*n_force) = 0.d0
         virial(1:9) = 0.d0
      end if

!     The sparse set is uploaded once, not once per batch: it does not depend on
!     which sites are in flight, and it is the largest single upload here.
      call build_hypers(n_sparse, dim, qs, alphas, hypers)
      call soap_backend_begin(hypers)

      call plan_batches(n_sites, n_pairs, n_neigh, l_max, sum(alpha_max(1:n_species)), dim, &
                        n_species, max_bytes, n_chunks, i_beg, i_end, j_beg, j_end)

      call report_batches(n_chunks, n_sites, n_pairs)

      allocate (e_b(1:n_sites), f_b(1:3, 1:n_force), v_b(1:3, 1:3))

      do c = 1, n_chunks
         ib = i_beg(c)
         ie = i_end(c)
         jb = j_beg(c)
         je = j_end(c)
         ns_b = ie - ib + 1
         np_b = je - jb + 1
         if (ns_b <= 0 .or. np_b <= 0) cycle

!        Each batch overwrites rather than accumulates -- get_soap_energy_and_forces
!        declares energies, forces and virial intent(out) and zeroes them -- so the
!        accumulation happens here.
         call one_batch(ns_b, np_b, n_force, n_species, n_neigh(ib), neighbors_list(jb), &
                        species_idx(jb), rjs(jb), thetas(jb), phis(jb), xyz(3*jb - 2), &
                        alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                        atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                        atom_sigma_t_scaling, amplitude_scaling, radial_enhancement, &
                        central_weight, basis, scaling_mode, compress, n_nonzero, p_j, &
                        dim, n_sparse, delta, zeta, e0, do_f, e_b, f_b, v_b)

         energies(ib:ie) = e_b(1:ns_b)
         if (do_f) then
            n = 0
            do j = 1, n_force
               do i = 1, 3
                  n = n + 1
                  forces(n) = forces(n) + f_b(i, j)
               end do
            end do
            n = 0
            do j = 1, 3
               do i = 1, 3
                  n = n + 1
                  virial(n) = virial(n) + v_b(i, j)
               end do
            end do
         end if
      end do

      call soap_backend_end()
      deallocate (e_b, f_b, v_b, i_beg, i_end, j_beg, j_end)
      deallocate (hypers%nf, hypers%alphas, hypers%Qs)
   end subroutine madlib_soap_energy_forces_gpu

!  Say how the work was split, when asked.
!
!  A test that forces many batches and finds the answer unchanged proves nothing
!  unless the split actually happened -- if the budget were silently ignored it
!  would agree for that reason. MADLIB_SOAP_BATCH_REPORT is how the test checks.
   subroutine report_batches(n_chunks, n_sites, n_pairs)
      integer, intent(in) :: n_chunks, n_sites, n_pairs
      character(len=32) :: env

      call get_environment_variable("MADLIB_SOAP_BATCH_REPORT", env)
      if (len_trim(env) == 0) return
      write (0, '(a,i0,a,i0,a,i0,a)') "    soap batches: ", n_chunks, " for ", n_sites, &
         " sites, ", n_pairs, " pairs"
   end subroutine report_batches

!  soap_backend_begin reads only n_sparse, dim, alphas, Qs and the kind of
!  nf(1), so a minimal hypers carries what it needs rather than a whole parsed
!  descriptor.
   subroutine build_hypers(n_sparse, dim, qs, alphas, hypers)
      integer(c_int), intent(in) :: n_sparse, dim
      real(c_double), intent(in) :: qs(*), alphas(*)
      type(soap_turbo), intent(out) :: hypers

      integer :: i, j, n

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
   end subroutine build_hypers

!  One batch: descriptor, then prediction, on a contiguous run of sites.
   subroutine one_batch(n_sites, n_pairs, n_force, n_species, n_neigh, neighbors_list, &
                        species_idx, rjs, thetas, phis, xyz, alpha_max, l_max, rcut_hard, &
                        rcut_soft, nf, global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                        atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, &
                        radial_enhancement, central_weight, basis, scaling_mode, compress, &
                        n_nonzero, p_j, dim, n_sparse, delta, zeta, e0, do_f, e_out, f_out, v_out)
      integer, intent(in) :: n_sites, n_pairs, n_force
      integer(c_int), value :: n_species, l_max, radial_enhancement, compress, n_nonzero
      integer(c_int), value :: dim, n_sparse
      integer(c_int), intent(in), target :: n_neigh(*), species_idx(*), alpha_max(*), p_j(*)
      integer(c_int), intent(in) :: neighbors_list(*)
      real(c_double), intent(in), target :: rjs(*), thetas(*), phis(*), xyz(*)
      real(c_double), intent(in), target :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in), target :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      real(c_double), value :: delta, zeta, e0
      logical, intent(in) :: do_f
      real(dp), intent(out) :: e_out(:), f_out(:, :), v_out(:, :)

      type(soap_buffers) :: buf
      type(c_ptr) :: n_neigh_d
      real(dp), allocatable, target :: soap(:, :), soap_cart_der(:, :, :)
      real(dp) :: solo_time
      integer(c_int) :: np

      call run_get_soap(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, phis, &
                        alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                        atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling, &
                        amplitude_scaling, radial_enhancement, central_weight, basis, &
                        scaling_mode, compress, n_nonzero, p_j, dim, do_f, buf, soap, &
                        soap_cart_der)

      e_out = 0.d0
      f_out = 0.d0
      v_out = 0.d0
      solo_time = 0.d0
      np = n_pairs

!     soap and soap_cart_der are passed for their SHAPE: the routine reads
!     n_soap and n_sites from them and takes the values from the device.
!     n_neigh_d is scratch wearing an argument's clothes, as in run_get_soap.
      call get_soap_energy_and_forces(n_sparse, soap, soap_cart_der, delta, zeta, e0, &
                                      n_neigh(1:n_sites), neighbors_list(1:n_pairs), &
                                      reshape(xyz(1:3*n_pairs), [3, n_pairs]), do_f, .false., &
                                      e_out, f_out, v_out, solo_time, buf%soap_d, buf%der_d, &
                                      n_neigh_d, np)

      call release(buf)
      deallocate (soap, soap_cart_der)
   end subroutine one_batch

!  A per-atom property with the descriptor computed on the device.
!
!  Same C symbol and same argument list as the host madlib_local_property, so a
!  caller does not choose between them -- the build does. Exactly one of the two
!  is compiled, which is forced anyway: the CPU and GPU soap_turbo share a
!  module name and cannot both be linked.
!
!  The PREDICTION deliberately runs on the host, through the same
!  local_property_predict the CPU path uses, even though gpu_local_property_predict
!  exists. That routine takes no zero_trunc argument and does not clamp, while
!  the host one floors negative values and zeroes their gradients. Core electron
!  binding energies and Hirshfeld volumes both set zero_trunc, so the device
!  predictor would quietly return different numbers for exactly the quantities
!  this is for. The descriptor is the expensive part and it is on the device;
!  the contraction that follows is not worth a silent change in physics.
   subroutine madlib_local_property_gpu(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                                        rjs, thetas, phis, alpha_max, l_max, rcut_hard, &
                                        rcut_soft, nf, global_scaling, atom_sigma_r, &
                                        atom_sigma_r_scaling, atom_sigma_t, &
                                        atom_sigma_t_scaling, amplitude_scaling, &
                                        radial_enhancement, central_weight, basis, &
                                        scaling_mode, compress, n_nonzero, p_i, p_j, p_el, &
                                        dim, n_sparse, qs, alphas, v0, delta, zeta, &
                                        zero_trunc, do_derivatives, v, v_der) &
      bind(c, name="madlib_local_property")

      integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, n_sparse, zero_trunc, do_derivatives
      integer(c_int), intent(in), target :: n_neigh(*), species_idx(*), alpha_max(*)
      integer(c_int), intent(in), target :: p_i(*), p_j(*)
      real(c_double), intent(in), target :: rjs(*), thetas(*), phis(*), p_el(*)
      real(c_double), intent(in), target :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in), target :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in), target :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      real(c_double), intent(in), target :: qs(*), alphas(*)
      real(c_double), value :: v0, delta, zeta
      real(c_double), intent(out), target :: v(*), v_der(*)

      type(soap_buffers) :: buf
      real(dp), allocatable, target :: soap(:, :), soap_cart_der(:, :, :)
      real(dp), allocatable :: qs_m(:, :)
      real(dp), pointer :: der_p(:, :)
      integer, allocatable :: i_beg(:), i_end(:), j_beg(:), j_end(:)
      integer :: n_chunks, c, ib, ie, jb, je, ns_b, np_b
      integer(c_size_t) :: max_bytes
      logical :: do_der
      logical :: on_device

      do_der = (do_derivatives /= 0)
      call madlib_soap_gpu_init()

      ! dV/dr is one vector per PAIR, not per component: (3, n_pairs). The
      ! descriptor derivative it comes from is (3, dim, n_pairs); the
      ! contraction over dim happens inside local_property_predict.
      if (do_der) then
         call c_f_pointer(c_loc(v_der), der_p, [3, n_pairs])
      else
         call c_f_pointer(c_loc(v_der), der_p, [3, 1])
      end if

!     Batched, because a rank's whole neighbour list does not fit: 2457 atoms
!     of glassy carbon is already an out-of-memory abort on a 6 GB card. Unlike
!     madlib_soap_energy_forces_gpu this takes no budget argument -- the C entry
!     point is TurboGAP's local-property one and pair_turbogap passes what that
!     needs -- so the budget comes from the device.
!
!     The split is exact rather than approximate: local_property_predict is per
!     site and per pair with no coupling between them, so a batch's V and dV/dr
!     go straight into their slice of the caller's arrays.
      max_bytes = soap_budget_bytes()
      call plan_batches(n_sites, n_pairs, n_neigh, l_max, sum(alpha_max(1:n_species)), dim, &
                        n_species, max_bytes, n_chunks, i_beg, i_end, j_beg, j_end)
      call report_batches(n_chunks, n_sites, n_pairs)

      on_device = .not. host_predict_requested()

      do c = 1, n_chunks
         ib = i_beg(c)
         ie = i_end(c)
         jb = j_beg(c)
         je = j_end(c)
         ns_b = ie - ib + 1
         np_b = je - jb + 1
         if (ns_b <= 0 .or. np_b <= 0) cycle

         call run_get_soap(ns_b, np_b, n_species, n_neigh(ib), species_idx(jb), rjs(jb), &
                           thetas(jb), phis(jb), alpha_max, l_max, rcut_hard, rcut_soft, nf, &
                           global_scaling, atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                           atom_sigma_t_scaling, amplitude_scaling, radial_enhancement, &
                           central_weight, basis, scaling_mode, compress, n_nonzero, p_j, dim, &
                           do_der, buf, soap, soap_cart_der, zero_host=.not. on_device)

         if (on_device) then
!           der_p is shaped (3, 1) when there are no derivatives, so the batch's
!           pair slice is only meaningful in the branch that has one.
            if (do_der) then
               call predict_on_device(ns_b, np_b, dim, n_sparse, qs, alphas, v0, delta, &
                                      zeta, (zero_trunc /= 0), do_der, buf, n_neigh(ib), &
                                      soap, v(ib), der_p(:, jb:je))
            else
               call predict_on_device(ns_b, np_b, dim, n_sparse, qs, alphas, v0, delta, &
                                      zeta, (zero_trunc /= 0), do_der, buf, n_neigh(ib), &
                                      soap, v(ib), der_p)
            end if
         else
!           The host route, kept reachable so the device one can be checked
!           against it in one process rather than across two builds.
            call cpy_dtoh_blocking(buf%soap_d, c_loc(soap), buf%st_soap)
            if (do_der) call cpy_dtoh_blocking(buf%der_d, c_loc(soap_cart_der), buf%st_der)
            if (.not. allocated(qs_m)) qs_m = reshape(qs(1:dim*n_sparse), [dim, n_sparse])
            if (do_der) then
               call local_property_predict(soap, qs_m, alphas(1:n_sparse), v0, delta, zeta, &
                                           v(ib:ie), do_der, soap_cart_der, n_neigh(ib:ie), &
                                           der_p(:, jb:je), zero_trunc=(zero_trunc /= 0))
            else
               call local_property_predict(soap, qs_m, alphas(1:n_sparse), v0, delta, zeta, &
                                           v(ib:ie), do_der, soap_cart_der, n_neigh(ib:ie), &
                                           der_p, zero_trunc=(zero_trunc /= 0))
            end if
         end if

         call release(buf)
         deallocate (soap, soap_cart_der)
      end do
   end subroutine madlib_local_property_gpu

!  The sparse-GP contraction, on the device.
!
!  What it replaces: copying the descriptor and its (3, dim, n_pairs) derivative
!  tensor back and calling local_property_predict on the host. That tensor is
!  the largest array in the calculation -- hundreds of MB for a real cell -- and
!  taking it off the card was most of the reason the device path was only 1.3x.
!
!  Correct only since TurboGAP's gpu_local_property_predict learned zero_trunc.
!  Core electron binding energies and Hirshfeld volumes both set it, so before
!  that this would have quietly changed their values and their forces.
!
!  soap is passed unfilled: gpu_local_property_predict reads it for its shape
!  alone, the values it uses being the ones already on the card in buf%soap_d.
   subroutine predict_on_device(ns, np, dim, n_sparse, qs, alphas, v0, delta, zeta, &
                                zero_trunc, do_der, buf, n_neigh, soap, v, v_der)
      integer, intent(in) :: ns
      integer, intent(in) :: np
      integer(c_int), intent(in) :: dim
      integer(c_int), intent(in) :: n_sparse
      real(c_double), intent(in), target :: qs(*)
      real(c_double), intent(in), target :: alphas(*)
      real(c_double), intent(in) :: v0
      real(c_double), intent(in) :: delta
      real(c_double), intent(in) :: zeta
      logical, intent(in) :: zero_trunc
      logical, intent(in) :: do_der
      type(soap_buffers), intent(inout) :: buf
      integer(c_int), intent(in) :: n_neigh(*)
      real(dp), intent(in), target :: soap(:, :)
      real(c_double), intent(out) :: v(*)
      real(dp), intent(out) :: v_der(:, :)

      integer(c_int), allocatable, target :: l_index(:)
      real(c_double), allocatable, target :: lp(:)
      real(c_double), allocatable, target :: lp_der(:, :)
      type(c_ptr) :: qs_d
      type(c_ptr) :: alphas_d
      type(c_ptr) :: lp_d
      type(c_ptr) :: lp_der_d
      type(c_ptr) :: l_index_d
      integer(c_size_t) :: st_lp
      integer(c_size_t) :: st_lp_der
      integer :: i
      integer :: j
      integer :: l

      allocate (l_index(1:max(np, 1)))
      allocate (lp(1:ns))
      allocate (lp_der(1:3, 1:max(np, 1)))

!     Pair to site, one-based: what gap.f90 builds and what the derivative
!     kernel indexes Qss with.
      l = 0
      do i = 1, ns
         do j = 1, n_neigh(i)
            l = l + 1
            l_index(l) = i
         end do
      end do

      st_lp = int(ns, c_size_t)*sizeof(lp(1))
      st_lp_der = 3_c_size_t*int(max(np, 1), c_size_t)*sizeof(lp(1))

!     Per call rather than cached: two computes in one run carry different
!     models, and dim*n_sparse doubles is nothing beside the tensor this
!     avoids moving.
      call upload(c_loc(qs), qs_d, int(dim, c_size_t)*int(n_sparse, c_size_t)*sizeof(lp(1)))
      call upload(c_loc(alphas), alphas_d, int(n_sparse, c_size_t)*sizeof(lp(1)))
      call upload(c_loc(l_index), l_index_d, int(max(np, 1), c_size_t)*sizeof(l_index(1)))
      call gpu_malloc_async(lp_d, st_lp, gpu_stream)
      call gpu_malloc_async(lp_der_d, st_lp_der, gpu_stream)

      call gpu_local_property_predict(n_sparse, soap, buf%soap_d, qs_d, alphas_d, v0, delta, &
                                      zeta, lp, lp_d, do_der, buf%der_d, lp_der, lp_der_d, &
                                      int(np, c_int), l_index_d, cublas_handle, gpu_stream, &
                                      zero_trunc)

      v(1:ns) = lp(1:ns)
      if (do_der) v_der(1:3, 1:np) = lp_der(1:3, 1:np)

      call gpu_free_async(qs_d, gpu_stream)
      call gpu_free_async(alphas_d, gpu_stream)
      call gpu_free_async(l_index_d, gpu_stream)
      call gpu_free_async(lp_d, gpu_stream)
      call gpu_free_async(lp_der_d, gpu_stream)
      deallocate (l_index, lp, lp_der)
   end subroutine predict_on_device

!  Fall back to the host contraction. A diagnostic, not a tuning knob: it is
!  how the device predictor gets compared against the host one inside a single
!  process, which is the only comparison that means anything when the device
!  does not reproduce itself run to run.
   function host_predict_requested() result(yes)
      logical :: yes

      character(len=32) :: env

      call get_environment_variable("MADLIB_SOAP_HOST_PREDICT", env)
      yes = (len_trim(env) > 0 .and. trim(env) /= "0")
   end function host_predict_requested

!  Device bytes one SOAP batch may use.
!
!  Half of what is free, not all of it: the memory model in plan_batches is an
!  estimate over six terms, cuBLAS takes a workspace this does not see, and in
!  a LAMMPS run Kokkos owns device memory too. MADLIB_SOAP_MAX_GB overrides it
!  for a run that wants a fixed decomposition -- the batch boundaries move the
!  last digits of a sum, so a bit-exact comparison needs the budget pinned.
!
!  Divided by the ranks sharing the card, because every one of them asks the
!  same question and gets the same answer. Four ranks each budgeting half of
!  free memory is an out-of-memory abort, which is what it was.
   function soap_budget_bytes() result(b)
      integer(c_size_t) :: b

      real(dp), parameter :: FRACTION = 0.5d0
      integer(c_size_t) :: dev_free, dev_total, live, peak
      character(len=32) :: env
      real(dp) :: gb
      integer :: st, share

      call get_environment_variable("MADLIB_SOAP_MAX_GB", env)
      if (len_trim(env) > 0) then
         read (env, *, iostat=st) gb
         if (st == 0 .and. gb > 0.d0) then
            b = int(gb*1024.d0**3, c_size_t)
            return
         end if
      end if

      call gpu_mem_stats(dev_free, dev_total, live, peak)
      share = device_share()
      b = int(dfloat(dev_free)*FRACTION/dfloat(share), c_size_t)

      call get_environment_variable("MADLIB_SOAP_BATCH_REPORT", env)
      if (len_trim(env) > 0) then
         write (0, '(a,i0,a,i0)') "    soap budget: ", b, " bytes, ranks on this gpu ", share
      end if
   end function soap_budget_bytes

!  How many ranks share this rank's GPU.
!
!  Node-local rank count over visible devices, because cuda_set_device assigns
!  them round-robin -- that is what fortran_cuda_interfaces.f90:796 says it
!  does. Cached: the topology cannot change during a run, and
!  MPI_Comm_split_type allocates a communicator on every call.
   function device_share() result(n)
      integer :: n

      integer :: ierr, shared, sz, ndev
      logical :: inited

      if (ranks_per_device > 0) then
         n = ranks_per_device
         return
      end if

      n = 1
      call mpi_initialized(inited, ierr)
      if (inited) then
         call mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                                  MPI_INFO_NULL, shared, ierr)
         if (ierr == MPI_SUCCESS) then
            call mpi_comm_size(shared, sz, ierr)
            call mpi_comm_free(shared, ierr)
            ndev = max(1, gpu_device_count())
            n = max(1, (sz + ndev - 1)/ndev)
         end if
      end if
      ranks_per_device = n
   end function device_share

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

end module madlib_soap_gpu
