# The MADLIB soap_turbo backend, as a static library the package links against.

project(madlib Fortran)

add_library(madlib STATIC
  turbogap/kinds.f90
  turbogap/soap_turbo_functions.f90
  turbogap/soap_turbo_radial.f90
  turbogap/soap_turbo_angular.f90
  turbogap/soap_turbo_compress.f90
  turbogap/soap_turbo.f90
  turbogap/local_properties.f90
  madlib_soap.f90)

# -cpp is load-bearing. turbogap/local_properties.f90 is a verbatim copy of
# TurboGAP's and carries #ifdef _GPU around its device path; without the
# preprocessor those lines reach the compiler and it fails. _GPU is never
# defined here, so the blocks compile out. See tools/vendor_turbogap.sh.
target_compile_options(madlib PRIVATE -cpp)

set_target_properties(madlib PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(madlib PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
