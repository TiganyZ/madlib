! Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
!
! SOAP energies and forces, from the host path and from the device one.
!
! Built twice from this source, as soapdescdump is, because the two soap_turbo
! implementations share a module name and cannot be linked together.
!
! The two sides are NOT the same call. The host runs madlib_local_property,
! which returns a per-site value and a per-pair dV/dr, and the scatter into
! per-atom forces is done here -- the same scatter pair_turbogap.cpp does. The
! device runs madlib_soap_energy_forces_gpu, which goes through TurboGAP's own
! get_soap_energy_and_forces and scatters on the device. So this compares the
! prediction AND the scatter, which is the point: gap_soap_forces.cu is only
! reached by the second, and it carries three __shared__ tree reductions whose
! summation order is load bearing.
program soappredictdump

   use iso_c_binding
#ifdef MADLIB_GPU_DESC
   use madlib_soap_gpu, only: madlib_soap_energy_forces_gpu
#else
   use madlib_soap, only: madlib_local_property
#endif

   implicit none

   integer, parameter :: dp = kind(0.d0)
   integer, parameter :: n_sites = 8
   integer, parameter :: n_per = 10
   integer, parameter :: n_pairs = n_sites*n_per
   integer, parameter :: n_species = 2
   integer, parameter :: l_max = 3
   integer, parameter :: n_sparse = 6

   integer(c_int) :: n_neigh(n_sites), species_idx(n_pairs), alpha_max(n_species)
   integer(c_int) :: neighbors_list(n_pairs), p_i(1), p_j(1), dim
   real(c_double) :: rjs(n_pairs), thetas(n_pairs), phis(n_pairs), xyz(3*n_pairs), p_el(1)
   real(c_double) :: rcut_hard(n_species), rcut_soft(n_species), nf(n_species)
   real(c_double) :: global_scaling(n_species), atom_sigma_r(n_species)
   real(c_double) :: atom_sigma_r_scaling(n_species), atom_sigma_t(n_species)
   real(c_double) :: atom_sigma_t_scaling(n_species), amplitude_scaling(n_species)
   real(c_double) :: central_weight(n_species)
   real(c_double) :: v0, delta, zeta
   real(c_double), allocatable :: qs_sp(:), alphas(:), v(:), v_der(:)
   real(c_double), allocatable :: energies(:), forces(:), virial(:)
   integer(8) :: seed
   integer :: i, j, k, p, a
   character(len=256) :: outfile

   call get_command_argument(1, outfile)
   if (len_trim(outfile) == 0) outfile = "soappredict.dat"

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
   v0 = 0.0d0; delta = 1.7d0; zeta = 2.d0

   dim = sum(alpha_max)*(sum(alpha_max) + 1)/2*(l_max + 1)
   allocate (qs_sp(dim*n_sparse), alphas(n_sparse))
   allocate (v(n_sites), v_der(3*n_pairs))
   allocate (energies(n_sites), forces(3*n_sites), virial(9))
   v = 0.d0; v_der = 0.d0; energies = 0.d0; forces = 0.d0; virial = 0.d0

   ! Self first at r = 0, then neighbours drawn from the other sites. xyz is
   ! derived from the spherical coordinates rather than chosen independently,
   ! so the two descriptions cannot disagree.
   n_neigh = n_per
   seed = 11_8
   p = 0
   do i = 1, n_sites
      do j = 1, n_per
         p = p + 1
         if (j == 1) then
            rjs(p) = 0.d0; thetas(p) = 0.d0; phis(p) = 0.d0
            neighbors_list(p) = i
         else
            rjs(p) = 0.9d0 + 2.0d0*frand(seed)
            thetas(p) = 3.14159265358979d0*frand(seed)
            phis(p) = 6.28318530717959d0*frand(seed)
            neighbors_list(p) = 1 + mod(i + j - 2, n_sites)
         end if
         xyz(3*p - 2) = rjs(p)*dsin(thetas(p))*dcos(phis(p))
         xyz(3*p - 1) = rjs(p)*dsin(thetas(p))*dsin(phis(p))
         xyz(3*p) = rjs(p)*dcos(thetas(p))
         species_idx(p) = 1 + mod(i + j, n_species)
      end do
   end do
   do i = 1, dim*n_sparse
      qs_sp(i) = 2.d0*frand(seed) - 1.d0
   end do
   do i = 1, n_sparse
      alphas(i) = 2.d0*frand(seed) - 1.d0
   end do

#ifdef MADLIB_GPU_DESC
   call madlib_soap_energy_forces_gpu(n_sites, n_pairs, n_sites, n_species, n_neigh, &
                                      neighbors_list, species_idx, rjs, thetas, phis, xyz, &
                                      alpha_max, l_max, rcut_hard, rcut_soft, nf, &
                                      global_scaling, atom_sigma_r, atom_sigma_r_scaling, &
                                      atom_sigma_t, atom_sigma_t_scaling, amplitude_scaling, 0, &
                                      central_weight, "poly3gauss"//c_null_char, &
                                      "polynomial"//c_null_char, 0, 1, p_j, dim, n_sparse, &
                                      qs_sp, alphas, delta, zeta, v0, 1, energies, forces, virial)
#else
   call madlib_local_property(n_sites, n_pairs, n_species, n_neigh, species_idx, rjs, thetas, &
                              phis, alpha_max, l_max, rcut_hard, rcut_soft, nf, global_scaling, &
                              atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, &
                              atom_sigma_t_scaling, amplitude_scaling, 0, central_weight, &
                              "poly3gauss"//c_null_char, "polynomial"//c_null_char, 0, 1, &
                              p_i, p_j, p_el, dim, n_sparse, qs_sp, alphas, v0, delta, zeta, &
                              0, 1, v, v_der)
   energies(1:n_sites) = v(1:n_sites)
   ! The scatter pair_turbogap.cpp does: force on the neighbour atom of each
   ! pair is minus that pair's dV/dr.
   do k = 1, n_pairs
      a = neighbors_list(k)
      forces(3*a - 2) = forces(3*a - 2) - v_der(3*k - 2)
      forces(3*a - 1) = forces(3*a - 1) - v_der(3*k - 1)
      forces(3*a) = forces(3*a) - v_der(3*k)
   end do
#endif

   open (unit=11, file=trim(outfile), status="replace")
   write (11, '(i0,1x,i0)') n_sites, 3*n_sites
   do i = 1, n_sites
      write (11, '(es24.17)') energies(i)
   end do
   do i = 1, 3*n_sites
      write (11, '(es24.17)') forces(i)
   end do
   close (11)
   write (*, '(a,i0,a,i0,a)') "    wrote ", n_sites, " energies and ", 3*n_sites, " force components"

contains

   function frand(seed) result(r)
      integer(8), intent(inout) :: seed
      real(dp) :: r
      seed = mod(1103515245_8*seed + 12345_8, 2147483648_8)
      r = dble(seed)/2147483648.d0
   end function frand

end program soappredictdump
