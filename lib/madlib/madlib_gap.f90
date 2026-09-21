! ---------------------------------------------------------------------------
!  MADLIB - a C binding over TurboGAP's energy and force terms.
!
!  A fitted TurboGAP potential is a SUM OF FOUR CONTRIBUTIONS, not one:
!
!      E = E_soap_turbo + E_distance_2b + E_angle_3b + E_core_pot
!
!  and every fitted potential worth running uses all four -- carbon.gap,
!  CO.gap, phosphorus.gap and the rest all carry 2b, 3b and core_pot blocks
!  beside their soap_turbo one. A pair style that evaluated only the SOAP term
!  would reproduce none of them, and would do it silently, so each term gets an
!  entry point here and the caller sums them.
!
!  The three non-SOAP routines are TurboGAP's own, called unmodified: their CPU
!  paths take every argument plainly, use no MPI and touch no derived type, so
!  compiling the vendored gap.f90 is cheaper and safer than transcribing four
!  pieces of fitted-potential arithmetic.
!
!  Array layout is soap_turbo's, as everywhere else in MADLIB. Pairs are
!  flattened over sites in order and the FIRST pair of each site is the site
!  itself at r = 0, which is the convention TurboGAP's own loops assume when
!  they start at j = 2.
!
!  n_sites and n_force are DIFFERENT lengths and both are needed. Energies are
!  per site; forces are per atom, and the 3b term puts a force on every atom of
!  a triplet, which under LAMMPS's domain decomposition includes ghosts. So the
!  force array spans local and ghost atoms while the energy array spans sites,
!  and TurboGAP reads the force extent back as its n_sites0 -- which is exactly
!  the index its neighbours list is wrapped against.
!
!  Each entry ADDS into energies, forces and virial rather than overwriting, so
!  a caller accumulates the four terms by calling them in turn. TurboGAP's
!  routines declare those intent(out) and overwrite their own buffers; the
!  addition happens here.
!
!  THE SOAP TERM IS NOT HERE, and needs no code of its own. It is
!  madlib_local_property in madlib_soap.f90 called with zero_trunc = 0: a
!  soap_turbo energy and a soap_turbo local property are the same GPR,
!
!      V = delta^2 * (soap . Qs)^zeta * alphas + V0
!
!  with V0 the e0 offset, and that path is already checked against TurboGAP to
!  1.2e-09 relative. The zero_trunc argument is the whole difference and it is
!  not cosmetic: a local property is floored at zero, and flooring an energy
!  would clamp sites and drop their forces without saying so.
! ---------------------------------------------------------------------------

module madlib_gap

   use iso_c_binding
   use gap, only: get_2b_energy_and_forces, get_3b_energy_and_forces, &
                  get_core_pot_energy_and_forces

   implicit none

   integer, parameter :: dp = kind(0.d0)

contains

!  A character*8 array from the flat n*8 buffer C can pass.
   function species_table(buf, n) result(t)
      character(kind=c_char), intent(in) :: buf(*)
      integer, intent(in) :: n
      character*8 :: t(1:n)
      integer :: i, k

      do i = 1, n
         do k = 1, 8
            t(i) (k:k) = buf((i - 1)*8 + k)
         end do
      end do
   end function species_table

   function one_species(buf) result(s)
      character(kind=c_char), intent(in) :: buf(*)
      character*8 :: s
      integer :: k

      do k = 1, 8
         s(k:k) = buf(k)
      end do
   end function one_species

!  E_distance_2b and its forces, for one species pair.
   subroutine madlib_gap_2b(n_sites, n_force, n_pairs, n_species, n_sparse, rjs, xyz, &
                            alphas, cutoff, rcut, buffer, delta, sigma, e0, qs, &
                            n_neigh, species, neighbor_species, &
                            species1, species2, species_types, &
                            do_forces, energies, forces, virial) &
      bind(c, name="madlib_gap_2b")
      integer(c_int), value :: n_sites, n_force, n_pairs, n_species, n_sparse, do_forces
      real(c_double), intent(in) :: rjs(*), xyz(*), alphas(*), cutoff(*), qs(*)
      real(c_double), value :: rcut, buffer, delta, sigma, e0
      integer(c_int), intent(in) :: n_neigh(*), species(*), neighbor_species(*)
      character(kind=c_char), intent(in) :: species1(*), species2(*), species_types(*)
      real(c_double), intent(inout) :: energies(*), forces(*), virial(*)

      real(dp) :: e(1:n_sites), f(1:3, 1:n_force), v(1:3, 1:3)

      e = 0.0_dp
      f = 0.0_dp
      v = 0.0_dp

      call get_2b_energy_and_forces(rjs(1:n_pairs), reshape(xyz(1:3*n_pairs), [3, n_pairs]), &
                                    alphas(1:n_sparse), cutoff(1:n_sparse), rcut, buffer, &
                                    delta, sigma, e0, qs(1:n_sparse), n_neigh(1:n_sites), &
                                    do_forces /= 0, .false., species(1:n_sites), &
                                    neighbor_species(1:n_pairs), &
                                    one_species(species1), one_species(species2), &
                                    species_table(species_types, n_species), e, f, v)

      call accumulate(n_sites, n_force, e, f, v, do_forces /= 0, energies, forces, virial)
   end subroutine madlib_gap_2b

!  E_angle_3b and its forces, for one species triplet.
   subroutine madlib_gap_3b(n_sites, n_force, n_pairs, n_species, n_sparse, n_dim, rjs, xyz, &
                            alphas, cutoff, rcut, buffer, delta, sigma, e0, qs, &
                            n_neigh, neighbors_list, species, neighbor_species, &
                            kernel_type, species_center, species1, species2, species_types, &
                            do_forces, energies, forces, virial) &
      bind(c, name="madlib_gap_3b")
      integer(c_int), value :: n_sites, n_force, n_pairs, n_species, n_sparse, n_dim, do_forces
      real(c_double), intent(in) :: rjs(*), xyz(*), alphas(*), cutoff(*), qs(*), sigma(*)
      real(c_double), value :: rcut, buffer, delta, e0
      integer(c_int), intent(in) :: n_neigh(*), neighbors_list(*), species(*), neighbor_species(*)
      character(kind=c_char), intent(in) :: kernel_type(*), species_center(*)
      character(kind=c_char), intent(in) :: species1(*), species2(*), species_types(*)
      real(c_double), intent(inout) :: energies(*), forces(*), virial(*)

      real(dp) :: e(1:n_sites), f(1:3, 1:n_force), v(1:3, 1:3)
      character*3 :: kern
      integer :: k

      do k = 1, 3
         kern(k:k) = kernel_type(k)
      end do

      e = 0.0_dp
      f = 0.0_dp
      v = 0.0_dp

      call get_3b_energy_and_forces(rjs(1:n_pairs), reshape(xyz(1:3*n_pairs), [3, n_pairs]), &
                                    alphas(1:n_sparse), cutoff(1:n_sparse), rcut, buffer, &
                                    delta, sigma(1:3), e0, &
!                                   Qs is (n_sparse, 3) here, NOT (3, n_sparse):
!                                   TurboGAP stores the 3b sparse set with the
!                                   indices the opposite way round from SOAP,
!                                   and says so in read_files.f90 -- "the array
!                                   indices are the opposite of SOAP
!                                   convention, this makes execution faster".
!                                   The file still holds three consecutive
!                                   values per sparse point, so order=[2,1]
!                                   fills it the way the file is written.
!                                   Passing the SOAP shape instead costs no
!                                   error and no warning: it silently scrambles
!                                   the sparse set and the 3b energy comes back
!                                   plausible and wrong.
                                    reshape(qs(1:n_dim*n_sparse), [n_sparse, n_dim], order=[2, 1]), &
                                    n_neigh(1:n_sites), neighbors_list(1:n_pairs), &
                                    do_forces /= 0, .false., kern, &
                                    species(1:n_sites), neighbor_species(1:n_pairs), &
                                    one_species(species_center), one_species(species1), &
                                    one_species(species2), &
                                    species_table(species_types, n_species), e, f, v)

      call accumulate(n_sites, n_force, e, f, v, do_forces /= 0, energies, forces, virial)
   end subroutine madlib_gap_3b

!  E_core_pot: a tabulated repulsive spline, for one species pair.
   subroutine madlib_gap_core_pot(n_sites, n_force, n_pairs, n_species, n_tab, rjs, xyz, &
                                  x, vtab, yp1, ypn, dvdx2, &
                                  n_neigh, species, neighbor_species, &
                                  species1, species2, species_types, &
                                  do_forces, energies, forces, virial) &
      bind(c, name="madlib_gap_core_pot")
      integer(c_int), value :: n_sites, n_force, n_pairs, n_species, n_tab, do_forces
      real(c_double), intent(in) :: rjs(*), xyz(*), x(*), vtab(*), dvdx2(*)
      real(c_double), value :: yp1, ypn
      integer(c_int), intent(in) :: n_neigh(*), species(*), neighbor_species(*)
      character(kind=c_char), intent(in) :: species1(*), species2(*), species_types(*)
      real(c_double), intent(inout) :: energies(*), forces(*), virial(*)

      real(dp) :: e(1:n_sites), f(1:3, 1:n_force), v(1:3, 1:3)

      e = 0.0_dp
      f = 0.0_dp
      v = 0.0_dp

      call get_core_pot_energy_and_forces(rjs(1:n_pairs), &
                                          reshape(xyz(1:3*n_pairs), [3, n_pairs]), &
                                          x(1:n_tab), vtab(1:n_tab), yp1, ypn, dvdx2(1:n_tab), &
                                          n_neigh(1:n_sites), do_forces /= 0, .false., &
                                          species(1:n_sites), neighbor_species(1:n_pairs), &
                                          one_species(species1), one_species(species2), &
                                          species_table(species_types, n_species), e, f, v)

      call accumulate(n_sites, n_force, e, f, v, do_forces /= 0, energies, forces, virial)
   end subroutine madlib_gap_core_pot

!  Add one term's contribution to the caller's running totals. TurboGAP's
!  routines declare their outputs intent(out) and overwrite, so summing the
!  four terms has to happen somewhere; here is the one place it does.
   subroutine accumulate(n_sites, n_force, e, f, v, do_forces, energies, forces, virial)
      integer, intent(in) :: n_sites
      integer, intent(in) :: n_force
      real(dp), intent(in) :: e(:)
      real(dp), intent(in) :: f(:, :)
      real(dp), intent(in) :: v(:, :)
      logical, intent(in) :: do_forces
      real(c_double), intent(inout) :: energies(*), forces(*), virial(*)

      integer :: i, k1, k2

      do i = 1, n_sites
         energies(i) = energies(i) + e(i)
      end do

      if (do_forces) then
         do i = 1, n_force
            do k1 = 1, 3
               forces(3*(i - 1) + k1) = forces(3*(i - 1) + k1) + f(k1, i)
            end do
         end do
         do k2 = 1, 3
            do k1 = 1, 3
               virial(3*(k2 - 1) + k1) = virial(3*(k2 - 1) + k1) + v(k1, k2)
            end do
         end do
      end if
   end subroutine accumulate

end module madlib_gap
