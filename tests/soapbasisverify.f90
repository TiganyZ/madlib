! Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
!
! madlib_soap_basis against the properties that define it.
!
! The routine exports the derived basis state get_soap caches in module `save`
! variables, because a device path has to own it. The orthonormalisation itself
! is soap_turbo's -- what is restated here is the bookkeeping around it, so that
! is what this checks:
!
!   * W and S are block diagonal over species. Species share no basis functions,
!     so every off-block entry must be exactly zero.
!   * W orthonormalises the basis whose overlap is S: W^T S W = I.
!   * The multiplicities are sqrt(2)^[n/=np] * 2^[m>0], so each is one of four
!     values, and there are as many as the kept (n,np,l,m) quadruples.
!
! On W^T S W = I the tolerance is asserted only for a small radial basis. The
! poly3 overlap is a Hilbert-like matrix and its conditioning degrades fast:
! measured max |W^T S W - I| at alpha_max 4 / 7 / 10 is 6e-13 / 9e-09 / 2e-03
! for poly3gauss and 2e-11 / 8e-07 / 3e-02 for poly3. That is the basis, not
! this code -- soap_turbo carries the same error, which is why MADLIB still
! reproduces TurboGAP to 1e-09. It is asserted where double precision can
! represent the identity and reported above that, since a silent 3% error in
! the orthonormalisation at alpha_max = 10 is worth seeing.
program soapbasisverify

   use iso_c_binding

   implicit none

   integer, parameter :: dp = kind(0.d0)
   integer, parameter :: n_species = 2
   integer, parameter :: l_max = 4
   integer, parameter :: amx_tight = 4        ! where the identity must hold
   real(dp), parameter :: tol_tight = 1.d-10

   interface
      subroutine madlib_soap_basis(n_species, alpha_max, l_max, atom_sigma_r, rcut_hard, basis, &
                                   compress, n_nonzero, p_j, W, S, skip, multiplicity, n_mult) &
         bind(c, name="madlib_soap_basis")
         use iso_c_binding
         integer(c_int), value :: n_species, l_max, compress, n_nonzero
         integer(c_int), intent(in) :: alpha_max(*), p_j(*)
         real(c_double), intent(in) :: atom_sigma_r(*), rcut_hard(*)
         character(kind=c_char), intent(in) :: basis(*)
         real(c_double), intent(out) :: W(*), S(*), multiplicity(*)
         integer(c_int), intent(out) :: skip(*)
         integer(c_int), intent(out) :: n_mult
      end subroutine madlib_soap_basis
   end interface

   integer :: nfail, amx, ib
   character(len=32) :: basis_name

   nfail = 0
   do amx = 4, 10, 3
      basis_name = "poly3gauss"
      call check_one(amx, basis_name, nfail)
      basis_name = "poly3"
      call check_one(amx, basis_name, nfail)
   end do

   call check_compression(nfail)

   if (nfail == 0) then
      write (*, '(a)') "    PASS: the exported basis is block diagonal, orthonormalising and correctly counted"
   else
      write (*, '(a,i0,a)') "    FAIL: ", nfail, " check(s)"
      stop 1
   end if

contains

   subroutine check_one(amx, basis_name, nfail)
      integer, intent(in) :: amx
      character(len=*), intent(in) :: basis_name
      integer, intent(inout) :: nfail

      integer(c_int) :: alpha_max(n_species), p_j(1), n_mult
      real(c_double) :: atom_sigma_r(n_species), rcut_hard(n_species)
      real(dp), allocatable :: W(:), S(:), mult(:)
      integer(c_int), allocatable :: skip(:)
      integer :: n_max, i, j, l, ib, ie, isp, n_unc, bound
      real(dp) :: acc, worst, off

      alpha_max = amx
      atom_sigma_r = 0.5d0
      rcut_hard = 3.5d0
      p_j = 0
      n_max = sum(alpha_max)
      n_unc = n_max*(n_max + 1)/2*(l_max + 1)
      bound = n_unc*(l_max + 2)/2
      allocate (W(n_max*n_max), S(n_max*n_max), skip(n_unc), mult(bound))

      call madlib_soap_basis(n_species, alpha_max, l_max, atom_sigma_r, rcut_hard, &
                             trim(basis_name)//c_null_char, 0, 0, p_j, W, S, skip, mult, n_mult)

      ! W^T S W = I, per species block. (W^T S W)(i,j) = sum_l [sum_k W(k,i) S(k,l)] W(l,j),
      ! with W(k,i) stored at W((i-1)*n_max + k).
      worst = 0.d0
      ie = 0
      do isp = 1, n_species
         ib = ie + 1
         ie = ie + alpha_max(isp)
         do i = ib, ie
            do j = ib, ie
               acc = 0.d0
               do l = ib, ie
                  acc = acc + dot_product(W((i - 1)*n_max + ib:(i - 1)*n_max + ie), &
                                          S((l - 1)*n_max + ib:(l - 1)*n_max + ie)) &
                        *W((j - 1)*n_max + l)
               end do
               if (i == j) acc = acc - 1.d0
               worst = max(worst, dabs(acc))
            end do
         end do
      end do
      write (*, '(a,i2,1x,a,a,es10.3)') "    alpha_max=", amx, trim(basis_name), &
         ": max |W^T S W - I| = ", worst
      if (amx <= amx_tight .and. worst > tol_tight) then
         write (*, '(a)') "      ^ above tolerance where conditioning cannot excuse it"
         nfail = nfail + 1
      end if

      ! Off-block entries: exactly zero, not nearly.
      off = 0.d0
      ie = 0
      do isp = 1, n_species
         ib = ie + 1
         ie = ie + alpha_max(isp)
         do i = 1, n_max
            do j = 1, n_max
               if ((i >= ib .and. i <= ie) .eqv. (j >= ib .and. j <= ie)) cycle
               off = max(off, dabs(W((j - 1)*n_max + i)), dabs(S((j - 1)*n_max + i)))
            end do
         end do
      end do
      if (off /= 0.d0) then
         write (*, '(a,es10.3)') "      off-block leak: ", off
         nfail = nfail + 1
      end if

      if (any(skip(1:n_unc) /= 0)) then
         write (*, '(a)') "      an uncompressed run skipped a component"
         nfail = nfail + 1
      end if
      if (n_mult /= bound) then
         write (*, '(a,i0,a,i0)') "      n_mult = ", n_mult, " expected ", bound
         nfail = nfail + 1
      end if
      do i = 1, n_mult
         if (dabs(mult(i) - 1.d0) > 1.d-12 .and. dabs(mult(i) - dsqrt(2.d0)) > 1.d-12 .and. &
             dabs(mult(i) - 2.d0) > 1.d-12 .and. dabs(mult(i) - 2.d0*dsqrt(2.d0)) > 1.d-12) then
            write (*, '(a,i0,a,es14.6)') "      multiplicity(", i, ") is none of the four: ", mult(i)
            nfail = nfail + 1
            exit
         end if
      end do

      deallocate (W, S, skip, mult)
   end subroutine check_one

   ! Compression keeps exactly the components p_j names, and the multiplicity
   ! count must drop to match. Without this the compressed branch is unchecked,
   ! and it is the branch a real potential with a compress file takes.
   subroutine check_compression(nfail)
      integer, intent(inout) :: nfail

      integer, parameter :: amx = 4
      integer(c_int) :: alpha_max(n_species), n_mult
      integer(c_int), allocatable :: p_j(:), skip(:)
      real(c_double) :: atom_sigma_r(n_species), rcut_hard(n_species)
      real(dp), allocatable :: W(:), S(:), mult(:)
      integer :: n_max, n_unc, bound, nz, i, n, np, l, counter, expect

      alpha_max = amx
      atom_sigma_r = 0.5d0
      rcut_hard = 3.5d0
      n_max = sum(alpha_max)
      n_unc = n_max*(n_max + 1)/2*(l_max + 1)
      bound = n_unc*(l_max + 2)/2
      nz = n_unc/3
      allocate (W(n_max*n_max), S(n_max*n_max), skip(n_unc), mult(bound), p_j(nz))
      do i = 1, nz
         p_j(i) = 3*i - 2          ! keep every third component
      end do

      call madlib_soap_basis(n_species, alpha_max, l_max, atom_sigma_r, rcut_hard, &
                             "poly3gauss"//c_null_char, 1, nz, p_j, W, S, skip, mult, n_mult)

      if (count(skip(1:n_unc) == 0) /= nz) then
         write (*, '(a,i0,a,i0)') "    compressed: kept ", count(skip(1:n_unc) == 0), " expected ", nz
         nfail = nfail + 1
      end if
      do i = 1, nz
         if (skip(p_j(i)) /= 0) then
            write (*, '(a,i0,a)') "    compressed: component ", p_j(i), " named by p_j was skipped"
            nfail = nfail + 1
            exit
         end if
      end do

      ! The kept quadruples, counted independently of the routine.
      expect = 0
      counter = 0
      do n = 1, n_max
         do np = n, n_max
            do l = 0, l_max
               counter = counter + 1
               if (skip(counter) == 0) expect = expect + l + 1
            end do
         end do
      end do
      if (n_mult /= expect) then
         write (*, '(a,i0,a,i0)') "    compressed: n_mult = ", n_mult, " expected ", expect
         nfail = nfail + 1
      end if
      write (*, '(a,i0,a,i0,a)') "    compressed: ", nz, " of ", n_unc, " components kept"

      deallocate (W, S, skip, mult, p_j)
   end subroutine check_compression

end program soapbasisverify
