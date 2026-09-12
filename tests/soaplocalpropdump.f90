! Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
!
! madlib_local_property, dumped. Note there is no #ifdef anywhere in this file.
!
! That is the point of it. The host and device implementations export the SAME
! C symbol and take the SAME arguments, so a caller does not choose between
! them -- the build does, by which soap_turbo was linked. This source is
! compiled twice, unchanged, and the two dumps must agree.
!
! Which is also what makes pair_turbogap.cpp need no change to get the device
! descriptor: it already calls madlib_local_property.
program soaplocalpropdump

   use iso_c_binding

   implicit none

   integer, parameter :: dp = kind(0.d0)
   integer, parameter :: n_species = 2
   integer, parameter :: l_max = 3
   integer, parameter :: n_sparse = 6

   interface
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

   integer :: n_sites, n_per, n_pairs
   integer(c_int), allocatable :: n_neigh(:), species_idx(:)
   integer(c_int) :: alpha_max(n_species), p_i(1), p_j(1), dim
   real(c_double), allocatable :: rjs(:), thetas(:), phis(:)
   real(c_double) :: p_el(1)
   real(c_double) :: rcut_hard(n_species), rcut_soft(n_species), nf(n_species)
   real(c_double) :: global_scaling(n_species), atom_sigma_r(n_species)
   real(c_double) :: atom_sigma_r_scaling(n_species), atom_sigma_t(n_species)
   real(c_double) :: atom_sigma_t_scaling(n_species), amplitude_scaling(n_species)
   real(c_double) :: central_weight(n_species)
   real(c_double) :: v0, delta, zeta
   real(c_double), allocatable :: qs_sp(:), alphas(:), v(:), v_der(:)
   integer(8) :: seed
   integer :: i, j, p, ztrunc
   character(len=256) :: outfile, env

   call get_command_argument(1, outfile)
   if (len_trim(outfile) == 0) outfile = "localprop.dat"

   n_sites = 8
   n_per = 10
   ! zero_trunc is the one place the host and device predictors are known to
   ! differ -- gpu_local_property_predict does not clamp. Exercised on purpose,
   ! because both sides must go through the clamping predictor.
   ztrunc = 1
   call get_environment_variable("MADLIB_ZERO_TRUNC", env)
   if (len_trim(env) > 0) read (env, *) ztrunc
   n_pairs = n_sites*n_per
   allocate (n_neigh(n_sites), species_idx(n_pairs))
   allocate (rjs(n_pairs), thetas(n_pairs), phis(n_pairs))

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
   ! v0 is chosen so several sites come out NEGATIVE. Without that, zero_trunc
   ! never fires and comparing zero_trunc=1 against 0 compares nothing -- which
   ! is what the first version of this did, giving identical numbers for both
   ! and looking like a pass.
   v0 = -2.5d0; delta = 1.3d0; zeta = 2.d0

   dim = sum(alpha_max)*(sum(alpha_max) + 1)/2*(l_max + 1)
   allocate (qs_sp(dim*n_sparse), alphas(n_sparse), v(n_sites), v_der(3*n_pairs))
   v = 0.d0
   v_der = 0.d0

   n_neigh = n_per
   seed = 23_8
   p = 0
   do i = 1, n_sites
      do j = 1, n_per
         p = p + 1
         if (j == 1) then
            rjs(p) = 0.d0; thetas(p) = 0.d0; phis(p) = 0.d0
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

   call madlib_local_property(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, &
                              phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                              atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                              atom_sigma_t_scaling, amplitude_scaling, 0, central_weight, &
                              "poly3gauss"//c_null_char, "polynomial"//c_null_char, 0, 1, &
                              p_i, p_j, p_el, dim, n_sparse, qs_sp, alphas, v0, delta, zeta, &
                              ztrunc, 1, v, v_der)

   open (unit=11, file=trim(outfile), status="replace")
   write (11, '(i0,1x,i0)') n_sites, 3*n_pairs
   do i = 1, n_sites
      write (11, '(es24.17)') v(i)
   end do
   do i = 1, 3*n_pairs
      write (11, '(es24.17)') v_der(i)
   end do
   close (11)
   j = 0
   do i = 1, n_sites
      if (v(i) <= 0.d0) j = j + 1
   end do
   write (*, '(a,i0,a,i0,a,i0,a,i0,a)') "    wrote ", n_sites, " values and ", 3*n_pairs, &
      " derivatives, zero_trunc=", ztrunc, " (", j, " at or below zero)"

contains

   function frand(seed) result(r)
      integer(8), intent(inout) :: seed
      real(dp) :: r
      seed = mod(1103515245_8*seed + 12345_8, 2147483648_8)
      r = dble(seed)/2147483648.d0
   end function frand

end program soaplocalpropdump
