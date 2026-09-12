! Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
!
! madlib_soap_descriptor returns the descriptor the energy is actually built
! from, in the layout it claims.
!
! The descriptor is the reference a device implementation gets compared against,
! so it has to be trustworthy on its own. get_soap is soap_turbo's and already
! verified; what is new is the copy-out, and a transposed or mis-strided copy is
! exactly the kind of error that would make a later device comparison agree with
! itself and still be wrong.
!
! So: take the exported soap, run the sparse-GP prediction over it by hand --
! V = delta^2 * (soap . Qs)^zeta . alphas + V0, from local_properties.f90 -- and
! check it against what madlib_local_property returns for the same input. The
! two share get_soap but nothing else, so a copy-out error cannot cancel.
program soapdescverify

   use iso_c_binding

   implicit none

   integer, parameter :: dp = kind(0.d0)
   integer, parameter :: n_sites = 4
   integer, parameter :: n_per = 6              ! neighbours per site, self included
   integer, parameter :: n_pairs = n_sites*n_per
   integer, parameter :: n_species = 2
   integer, parameter :: l_max = 3
   integer, parameter :: n_sparse = 5

   interface
      subroutine madlib_soap_dim(n_species, alpha_max, l_max, compress, mode, dim, nonzero) &
         bind(c, name="madlib_soap_dim")
         use iso_c_binding
         integer(c_int), value :: n_species, l_max, compress
         integer(c_int), intent(in) :: alpha_max(*)
         character(kind=c_char), intent(in) :: mode(*)
         integer(c_int), intent(out) :: dim, nonzero
      end subroutine madlib_soap_dim

      subroutine madlib_soap_descriptor(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                                        rjs, thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, &
                                        nf, global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                        atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, &
                                        radial_enhancement, central_weight, basis, scaling_mode, &
                                        compress, n_nonzero, p_i, p_j, p_el, dim, do_derivatives, &
                                        soap_out, der_out) bind(c, name="madlib_soap_descriptor")
         use iso_c_binding
         integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
         integer(c_int), value :: compress, n_nonzero, dim, do_derivatives
         integer(c_int), intent(in) :: n_neigh(*), species_idx(*), alpha_max(*), p_i(*), p_j(*)
         real(c_double), intent(in) :: rjs(*), thetas(*), phis(*), p_el(*)
         real(c_double), intent(in) :: rcut_hard(*), rcut_soft(*), nf(*), global_scaling(*)
         real(c_double), intent(in) :: atom_sigma_r(*), atom_sigma_r_scaling(*)
         real(c_double), intent(in) :: atom_sigma_t(*), atom_sigma_t_scaling(*)
         real(c_double), intent(in) :: amplitude_scaling(*), central_weight(*)
         character(kind=c_char), intent(in) :: basis(*), scaling_mode(*)
         real(c_double), intent(out) :: soap_out(*), der_out(*)
      end subroutine madlib_soap_descriptor

      subroutine madlib_local_property(n_sites, n_pairs, n_species, n_neigh, species_idx, &
                                       rjs, thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, &
                                       nf, global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                       atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, &
                                       radial_enhancement, central_weight, basis, scaling_mode, &
                                       compress, n_nonzero, p_i, p_j, p_el, dim, n_sparse, &
                                       qs, alphas, v0, delta, zeta, zero_trunc, do_derivatives, &
                                       v, v_der) bind(c, name="madlib_local_property")
         use iso_c_binding
         integer(c_int), value :: n_sites, n_pairs, n_species, l_max, radial_enhancement
         integer(c_int), value :: compress, n_nonzero, dim, n_sparse, zero_trunc, do_derivatives
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
      end subroutine madlib_local_property
   end interface

   integer(c_int) :: n_neigh(n_sites), species_idx(n_pairs), alpha_max(n_species)
   integer(c_int) :: p_i(1), p_j(1), dim, nonzero
   real(c_double) :: rjs(n_pairs), thetas(n_pairs), phis(n_pairs), p_el(1)
   real(c_double) :: rcut_hard(n_species), rcut_soft(n_species), nf(n_species)
   real(c_double) :: global_scaling(n_species), atom_sigma_r(n_species)
   real(c_double) :: atom_sigma_r_scaling(n_species), atom_sigma_t(n_species)
   real(c_double) :: atom_sigma_t_scaling(n_species), amplitude_scaling(n_species)
   real(c_double) :: central_weight(n_species)
   real(c_double) :: v0, delta, zeta
   real(c_double), allocatable :: soap(:), qs_sp(:), alphas(:), v_ref(:), v_mine(:), der(:)
   real(dp) :: k, acc, worst, nrm, worst_nrm
   integer :: i, j, s, p
   integer(8) :: seed

   alpha_max = [6, 6]
   rcut_hard = 3.5d0
   rcut_soft = 3.0d0
   nf = 4.d0
   global_scaling = 1.d0
   atom_sigma_r = 0.5d0
   atom_sigma_r_scaling = 0.d0
   atom_sigma_t = 0.5d0
   atom_sigma_t_scaling = 0.d0
   amplitude_scaling = 1.d0
   central_weight = 1.d0
   p_i = 0; p_j = 0; p_el = 0.d0
   v0 = 0.3d0; delta = 1.7d0; zeta = 2.d0

   call madlib_soap_dim(n_species, alpha_max, l_max, 0, "none"//c_null_char, dim, nonzero)
   allocate (soap(dim*n_sites), qs_sp(dim*n_sparse), alphas(n_sparse))
   allocate (v_ref(n_sites), v_mine(n_sites), der(1))

   ! A reproducible pseudo-random configuration. Self first at r = 0, which is
   ! the layout soap_turbo requires and the pair style honours.
   n_neigh = n_per
   seed = 7_8
   p = 0
   do i = 1, n_sites
      do j = 1, n_per
         p = p + 1
         if (j == 1) then
            rjs(p) = 0.d0
            thetas(p) = 0.d0
            phis(p) = 0.d0
         else
            rjs(p) = 0.9d0 + 2.0d0*frand(seed)
            thetas(p) = 3.14159265358979d0*frand(seed)
            phis(p) = 6.28318530717959d0*frand(seed)
         end if
         species_idx(p) = 1 + mod(i + j, n_species)
      end do
   end do
   do i = 1, dim*n_sparse
      qs_sp(i) = 2.d0*frand(seed) - 1.d0
   end do
   do i = 1, n_sparse
      alphas(i) = 2.d0*frand(seed) - 1.d0
   end do

   call madlib_soap_descriptor(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, &
                               phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                               atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                               atom_sigma_t_scaling, amplitude_scaling, 0, central_weight, &
                               "poly3gauss"//c_null_char, "polynomial"//c_null_char, 0, 1, &
                               p_i, p_j, p_el, dim, 0, soap, der)

   call madlib_local_property(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, &
                              phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                              atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                              atom_sigma_t_scaling, amplitude_scaling, 0, central_weight, &
                              "poly3gauss"//c_null_char, "polynomial"//c_null_char, 0, 1, &
                              p_i, p_j, p_el, dim, n_sparse, qs_sp, alphas, v0, delta, zeta, &
                              0, 0, v_ref, der)

   ! V = delta^2 * (soap . Qs)^zeta . alphas + V0, by hand.
   do i = 1, n_sites
      acc = 0.d0
      do s = 1, n_sparse
         k = 0.d0
         do j = 1, dim
            k = k + soap((i - 1)*dim + j)*qs_sp((s - 1)*dim + j)
         end do
         acc = acc + (k**zeta)*alphas(s)
      end do
      v_mine(i) = delta**2*acc + v0
   end do

   ! soap_turbo returns a normalised descriptor, so this is a second, cheaper
   ! statement about the copy-out: a mis-strided read would not be unit length.
   worst_nrm = 0.d0
   do i = 1, n_sites
      nrm = dsqrt(sum(soap((i - 1)*dim + 1:i*dim)**2))
      worst_nrm = max(worst_nrm, dabs(nrm - 1.d0))
   end do

   worst = 0.d0
   do i = 1, n_sites
      worst = max(worst, dabs(v_mine(i) - v_ref(i)))
   end do

   write (*, '(a,i0,a,i0,a)') "    ", n_sites, " sites, descriptor length ", dim, ""
   write (*, '(a,es10.3)') "    max | |soap_i| - 1 |              = ", worst_nrm
   write (*, '(a,es10.3)') "    max |V(by hand) - V(local_prop)| = ", worst

   if (worst < 1.d-12 .and. worst_nrm < 1.d-12) then
      write (*, '(a)') "    PASS: the exported descriptor is the one the energy is built from"
   else
      write (*, '(a)') "    FAIL: the exported descriptor does not reproduce the prediction"
      stop 1
   end if

contains

   ! A small LCG, so the configuration is the same on every machine. 64-bit
   ! because the multiplier overflows a default integer.
   function frand(seed) result(r)
      integer(8), intent(inout) :: seed
      real(dp) :: r
      seed = mod(1103515245_8*seed + 12345_8, 2147483648_8)
      r = dble(seed)/2147483648.d0
   end function frand

end program soapdescverify
