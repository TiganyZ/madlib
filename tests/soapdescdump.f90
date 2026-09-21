! Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
!
! Dump the SOAP descriptor for a fixed pseudo-random configuration.
!
! Built twice from one source -- once against the CPU soap_turbo and once
! against the GPU one -- because the two share a module name and entry point and
! cannot be linked into the same binary. The two dumps are then compared. That
! is the only honest way to compare them: they are the same call, and the only
! difference is which soap_turbo answered it.
program soapdescdump

   use iso_c_binding
#ifdef MADLIB_GPU_DESC
   use madlib_soap_gpu, only: madlib_soap_descriptor_gpu
#else
   use madlib_soap, only: madlib_soap_descriptor
#endif

   implicit none

   integer, parameter :: dp = kind(0.d0)
   integer, parameter :: n_sites = 8
   integer, parameter :: n_per = 10
   integer, parameter :: n_pairs = n_sites*n_per
   integer, parameter :: n_species = 2
   integer, parameter :: l_max = 3

   integer(c_int) :: n_neigh(n_sites), species_idx(n_pairs), alpha_max(n_species)
   integer(c_int) :: p_i(1), p_j(1), dim
   real(c_double) :: rjs(n_pairs), thetas(n_pairs), phis(n_pairs), p_el(1)
   real(c_double) :: rcut_hard(n_species), rcut_soft(n_species), nf(n_species)
   real(c_double) :: global_scaling(n_species), atom_sigma_r(n_species)
   real(c_double) :: atom_sigma_r_scaling(n_species), atom_sigma_t(n_species)
   real(c_double) :: atom_sigma_t_scaling(n_species), amplitude_scaling(n_species)
   real(c_double) :: central_weight(n_species)
   real(c_double), allocatable :: soap(:), der(:)
   integer :: do_der
   integer(8) :: seed
   integer :: i, j, p
   character(len=256) :: outfile, arg2

   call get_command_argument(1, outfile)
   if (len_trim(outfile) == 0) outfile = "soapdesc.dat"
   call get_command_argument(2, arg2)
   do_der = 0
   if (trim(arg2) == "der") do_der = 1

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

   ! Uncompressed: n_max(n_max+1)/2 * (l_max+1), with n_max = sum(alpha_max).
   ! Computed rather than asked for, so this driver states the size it expects
   ! instead of agreeing with the library about it.
   dim = sum(alpha_max)*(sum(alpha_max) + 1)/2*(l_max + 1)
   if (do_der /= 0) then
      allocate (der(3*dim*n_pairs))
   else
      allocate (der(1))
   end if
   allocate (soap(dim*n_sites))
   soap = 0.d0
   der = 0.d0

   n_neigh = n_per
   seed = 7_8
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

#ifdef MADLIB_GPU_DESC
   call madlib_soap_descriptor_gpu(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, &
                                   thetas, phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, &
                                   global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                   atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, 0, &
                                   central_weight, "poly3gauss"//c_null_char, &
                                   "polynomial"//c_null_char, 0, 1, p_j, dim, do_der, soap, der)
#else
   call madlib_soap_descriptor(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, &
                               phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                               atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                               atom_sigma_t_scaling, amplitude_scaling, 0, central_weight, &
                               "poly3gauss"//c_null_char, "polynomial"//c_null_char, 0, 1, &
                               p_i, p_j, p_el, dim, do_der, soap, der)
#endif

   open (unit=11, file=trim(outfile), status="replace")
   if (do_der /= 0) then
      write (11, '(i0,1x,i0)') n_pairs, 3*dim
      do i = 1, 3*dim*n_pairs
         write (11, '(es24.17)') der(i)
      end do
   else
      write (11, '(i0,1x,i0)') n_sites, dim
      do i = 1, dim*n_sites
         write (11, '(es24.17)') soap(i)
      end do
   end if
   close (11)
   if (do_der /= 0) then
      write (*, '(a,i0,a,i0,a,a)') "    wrote ", n_pairs, " x ", 3*dim, " derivatives to ", trim(outfile)
   else
      write (*, '(a,i0,a,i0,a,a)') "    wrote ", n_sites, " x ", dim, " descriptor to ", trim(outfile)
   end if

contains

   function frand(seed) result(r)
      integer(8), intent(inout) :: seed
      real(dp) :: r
      seed = mod(1103515245_8*seed + 12345_8, 2147483648_8)
      r = dble(seed)/2147483648.d0
   end function frand

end program soapdescdump
