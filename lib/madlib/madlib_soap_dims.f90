! ---------------------------------------------------------------------------
!  The part of the MADLIB C binding that does not depend on which soap_turbo
!  is linked.
!
!  Split out so a GPU build can have it. TurboGAP ships two soap_turbo forks
!  which declare the SAME module and entry names, so a binary gets one of them;
!  madlib_soap.f90 and madlib_soap_gpu.f90 are the two halves that choose, and
!  neither may define what the other already does. These entries belong to
!  neither: they size the descriptor and expand a compression mode, and the
!  compression module they use (soap_turbo_compress_module, from the CPU
!  soap_turbo) has a different module name from the GPU fork's
!  soap_turbo_compress, so it coexists with either.
!
!  That naming difference is not cosmetic. The CPU fork returns the sparse P
!  matrix (P_i, P_j, P_el) and so supports P_transformation compression; the
!  GPU fork returns a plain index list and does not. Sizing and index expansion
!  therefore always come from the CPU one, and the GPU descriptor path uses the
!  P_j column of the result as its index list.
! ---------------------------------------------------------------------------

module madlib_soap_dims

   use iso_c_binding
   use soap_turbo_compress_module, only: get_compress_indices

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

      integer :: am(1:n_species), n_max, pi(1), pj(1)
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

end module madlib_soap_dims
