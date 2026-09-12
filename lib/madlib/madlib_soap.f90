! ---------------------------------------------------------------------------
!  MADLIB - a C binding over soap_turbo and TurboGAP's local-property GPR.
!
!  Exposes exactly what a per-atom property model needs and nothing else:
!  given a neighbour list in spherical coordinates, return q_i for every site
!  and dq_i/dr_j for every pair.
!
!  Array layout matches soap_turbo's own. Pairs are flattened over sites in
!  order, and the FIRST pair of each site is the site itself at r = 0, which is
!  where dq_i/dr_i lands. Callers must honour that.
! ---------------------------------------------------------------------------

module madlib_soap

   use iso_c_binding
   use soap_turbo_compress_module, only: get_compress_indices
   use soap_turbo_desc, only: get_soap
   use local_prop, only: local_property_predict

   implicit none

   integer, parameter :: dp = kind(0.d0)

contains

!  SOAP vector length for a set of hyperparameters, so the caller can size its
!  own arrays before committing to anything.
   subroutine madlib_soap_dim(n_species, alpha_max, l_max, compress, mode, dim, nonzero) &
      bind(c, name="madlib_soap_dim")
      integer(c_int), value :: n_species, l_max, compress
      integer(c_int), intent(in) :: alpha_max(*)
      character(kind=c_char), intent(in) :: mode(*)
      integer(c_int), intent(out) :: dim, nonzero

      integer :: am(1:n_species), n_max, i, pi(1), pj(1)
      real(dp) :: pel(1)
      character(len=32) :: m

      am = alpha_max(1:n_species)
      n_max = sum(am)

      if (compress == 0) then
         dim = n_max*(n_max + 1)/2*(l_max + 1)
         nonzero = dim
      else
         m = c_to_f_string(mode)
         call get_compress_indices(trim(m), am, l_max, dim, nonzero, pi, pj, pel, "get_dim")
      end if
   end subroutine madlib_soap_dim

   subroutine madlib_compress_indices(n_species, alpha_max, l_max, mode, nonzero, pi, pj, pel) &
      bind(c, name="madlib_compress_indices")
      integer(c_int), value :: n_species, l_max, nonzero
      integer(c_int), intent(in) :: alpha_max(*)
      character(kind=c_char), intent(in) :: mode(*)
      integer(c_int), intent(out) :: pi(*), pj(*)
      real(c_double), intent(out) :: pel(*)

      integer :: am(1:n_species), dim, nz
      character(len=32) :: m

      am = alpha_max(1:n_species)
      m = c_to_f_string(mode)
      dim = 0
      nz = nonzero
      call get_compress_indices(trim(m), am, l_max, dim, nz, pi(1:nonzero), pj(1:nonzero), &
                                pel(1:nonzero), "set_indices")
   end subroutine madlib_compress_indices

!  q_i and dq_i/dr_j for one descriptor and one local-property model.
!
!  species_idx(k)  1-based species of the neighbour in pair k
!  n_neigh(i)      neighbours of site i INCLUDING itself; pair 1 of each site
!                  is the site, at rjs = 0
   subroutine madlib_local_property(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                                    rjs, thetas, phis, &
                                    alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                                    atom_sigma_r, atom_sigma_r_scaling, &
                                    atom_sigma_t, atom_sigma_t_scaling, &
                                    amplitude_scaling, radial_enhancement, central_weight, &
                                    basis, scaling_mode, &
                                    compress, n_nonzero, p_i, p_j, p_el, dim, &
                                    n_sparse, qs, alphas, v0, delta, zeta, &
                                    zero_trunc, do_derivatives, v, v_der) &
      bind(c, name="madlib_local_property")

      integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, n_sparse, do_derivatives
!  Floor V at zero, and zero the derivative where it fires.
!
!  Passed explicitly and never defaulted, because local_property_predict
!  defaults it to TRUE. That is right for a binding energy or a Hirshfeld
!  volume, which cannot be negative; it is catastrophic for anything that can
!  be, such as a potential energy, and it fails silently -- the clamped sites
!  simply stop contributing a force. A caller must say which it has.
      integer(c_int), value :: zero_trunc
      integer(c_int), intent(in) :: n_neigh(*), species_idx(*), alpha_max(*), p_i(*), p_j(*)
      real(c_double), intent(in) :: rjs(*), thetas(*), phis(*), p_el(*)
      real(c_double), intent(in) :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
      real(c_double), intent(in) :: atom_sigma_r(*), atom_sigma_r_scaling(*)
      real(c_double), intent(in) :: atom_sigma_t(*), atom_sigma_t_scaling(*)
      real(c_double), intent(in) :: amplitude_scaling(*), central_weight(*)
      character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
      real(c_double), intent(in) :: qs(*), alphas(*)
      real(c_double), value :: v0, delta, zeta
      real(c_double), intent(out) :: v(*), v_der(*)

      real(dp), allocatable :: soap(:, :), soap_cart_der(:, :, :)
      logical, allocatable :: mask(:, :)
      integer, allocatable :: species(:, :), species_multiplicity(:)
      integer :: k, s
      logical :: do_der
      character(len=32) :: bas, smode

      do_der = (do_derivatives /= 0)
      bas = c_to_f_string(basis)
      smode = c_to_f_string(scaling_mode)

!     soap_turbo selects neighbours by species through a logical mask rather
!     than by an index, and carries the central atom's species as a (possibly
!     fractional) multiplicity, for alloy sites. Neither is needed here, so the
!     mask is one-hot and the multiplicity is one.
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
                    alpha_max(1:n_species), l_max, rcut_hard(1:n_species), rcut_soft(1:n_species), &
                    nf(1:n_species), global_scaling(1:n_species), &
                    atom_sigma_r(1:n_species), atom_sigma_r_scaling(1:n_species), &
                    atom_sigma_t(1:n_species), atom_sigma_t_scaling(1:n_species), &
                    amplitude_scaling(1:n_species), radial_enhancement, &
                    central_weight(1:n_species), trim(bas), trim(smode), .false., &
                    do_der, compress /= 0, n_nonzero, p_i(1:n_nonzero), p_j(1:n_nonzero), &
                    p_el(1:n_nonzero), soap, soap_cart_der)

      call local_property_predict(soap, reshape(qs(1:dim*n_sparse), [dim, n_sparse]), &
                                  alphas(1:n_sparse), v0, delta, zeta, v(1:n_sites), &
                                  do_der, soap_cart_der, n_neigh(1:n_sites), &
                                  v_der_view(v_der, do_der, n_pairs), &
                                  zero_trunc=(zero_trunc /= 0))

      deallocate (soap, soap_cart_der, mask, species, species_multiplicity)
   end subroutine madlib_local_property

!  A (3, n_pairs) view of the caller's flat buffer, so nothing is copied.
   function v_der_view(buf, do_der, n_pairs) result(p)
      real(c_double), intent(inout), target :: buf(*)
      logical, intent(in) :: do_der
      integer, intent(in) :: n_pairs
      real(dp), pointer :: p(:, :)
      if (do_der) then
         call c_f_pointer(c_loc(buf(1)), p, [3, n_pairs])
      else
         call c_f_pointer(c_loc(buf(1)), p, [3, 1])
      end if
   end function v_der_view

!  The derived basis state get_soap builds once and caches in module `save`
!  variables: the orthonormalisation matrices W and S, which components a
!  compressed descriptor keeps, and the multiplicity of each kept component.
!
!  Exported because a device path has to own them -- the kernels take them as
!  device pointers, and the Fortran routine that would otherwise compute them
!  is the one being replaced. Computed here by calling soap_turbo's own
!  orthonormalisation rather than re-deriving it; only the bookkeeping around
!  it is restated.
!
!  W and S are n_max x n_max column-major, block diagonal over species, with
!  n_max = sum(alpha_max). `multiplicity` must be at least
!  n_max*(n_max+1)/2 * (l_max+1)*(l_max+2)/2 long, which is the uncompressed
!  bound; n_mult says how much was written. `skip` is n_soap_uncompressed long.
   subroutine madlib_soap_basis(n_species, alpha_max, l_max, atom_sigma_r, rcut_hard, basis, &
                                compress, n_nonzero, p_j, W, S, skip, multiplicity, n_mult) &
      bind(c, name="madlib_soap_basis")
      use soap_turbo_radial, only: get_orthonormalization_matrix_poly3, &
                                   get_orthonormalization_matrix_poly3gauss, &
                                   get_orthonormalization_matrix_poly3_tabulated

      integer(c_int), value :: n_species
      integer(c_int), value :: l_max
      integer(c_int), value :: compress
      integer(c_int), value :: n_nonzero
      integer(c_int), intent(in) :: alpha_max(*)
      integer(c_int), intent(in) :: p_j(*)
      real(c_double), intent(in) :: atom_sigma_r(*)
      real(c_double), intent(in) :: rcut_hard(*)
      character(kind=c_char), intent(in) :: basis(*)
      real(c_double), intent(out) :: W(*)
      real(c_double), intent(out) :: S(*)
      integer(c_int), intent(out) :: skip(*)
      real(c_double), intent(out) :: multiplicity(*)
      integer(c_int), intent(out) :: n_mult

      real(dp), allocatable :: S_temp(:, :)
      real(dp), allocatable :: W_temp(:, :)
      integer :: am(1:n_species)
      integer :: i, j, k, n, np, l, m, n_max, i_beg, i_end, counter, counter2
      integer :: n_soap_uncompressed
      real(dp) :: mult
      character(len=32) :: b

      am = alpha_max(1:n_species)
      n_max = sum(am)
      b = c_to_f_string(basis)

      do i = 1, n_max*n_max
         W(i) = 0.d0
         S(i) = 0.d0
      end do

      i_end = 0
      do i = 1, n_species
         i_beg = i_end + 1
         i_end = i_end + am(i)
         allocate (S_temp(1:am(i), 1:am(i)))
         allocate (W_temp(1:am(i), 1:am(i)))
         S_temp = 0.d0
         W_temp = 0.d0
         if (trim(b) == "poly3gauss") then
            call get_orthonormalization_matrix_poly3gauss(am(i), atom_sigma_r(i), rcut_hard(i), S_temp, W_temp)
         else if (trim(b) == "poly3") then
            call get_orthonormalization_matrix_poly3(am(i), S_temp, W_temp)
         else
            ! poly3tab and poly3operator share the tabulated basis.
            call get_orthonormalization_matrix_poly3_tabulated(am(i), S_temp, W_temp)
         end if
         do j = 1, am(i)
            do k = 1, am(i)
               W((i_beg + j - 2)*n_max + i_beg + k - 1) = W_temp(k, j)
               S((i_beg + j - 2)*n_max + i_beg + k - 1) = S_temp(k, j)
            end do
         end do
         deallocate (S_temp, W_temp)
      end do

      n_soap_uncompressed = n_max*(n_max + 1)/2*(l_max + 1)
      if (compress == 0) then
         skip(1:n_soap_uncompressed) = 0
      else
         skip(1:n_soap_uncompressed) = 1
         do i = 1, n_nonzero
            skip(p_j(i)) = 0
         end do
      end if

      counter = 0
      counter2 = 0
      do n = 1, n_max
         do np = n, n_max
            do l = 0, l_max
               counter = counter + 1
               if (skip(counter) /= 0) cycle
               do m = 0, l
                  counter2 = counter2 + 1
                  mult = 1.d0
                  if (n /= np) mult = mult*dsqrt(2.d0)
                  if (m > 0) mult = mult*2.d0
                  multiplicity(counter2) = mult
               end do
            end do
         end do
      end do
      n_mult = counter2
   end subroutine madlib_soap_basis

   function c_to_f_string(s) result(f)
      character(kind=c_char), intent(in) :: s(*)
      character(len=32) :: f
      integer :: i
      f = " "
      do i = 1, 32
         if (s(i) == c_null_char) exit
         f(i:i) = s(i)
      end do
   end function c_to_f_string

end module madlib_soap
