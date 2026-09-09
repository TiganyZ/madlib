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
                                    do_derivatives, v, v_der) &
      bind(c, name="madlib_local_property")

      integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
      integer(c_int), value :: compress, n_nonzero, dim, n_sparse, do_derivatives
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
                                  v_der_view(v_der, do_der, n_pairs))

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
