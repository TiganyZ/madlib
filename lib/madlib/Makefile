# Standalone build of the MADLIB soap_turbo backend.
#
#   make                     libmadlib.a
#   make FC=ifx FFLAGS=-O2   a different compiler
#
# BLAS is the only external dependency.

FC      = gfortran
# -cpp is load-bearing, not decoration. turbogap/local_properties.f90 is a
# VERBATIM copy of TurboGAP's, and carries #ifdef _GPU blocks around its device
# path. Without the preprocessor gfortran does not remove them and the compile
# fails; with it, and _GPU never defined, they vanish. Vendoring the file
# unedited is what makes it diffable against upstream, which is the whole point
# of tools/vendor_turbogap.sh.
FFLAGS  = -O2 -fPIC -cpp
BLAS    = -llapack -lblas

SRC = turbogap/kinds.f90 \
      turbogap/elements.f90 \
      turbogap/topology.f90 \
      turbogap/splines.f90 \
      turbogap/soap_turbo_functions.f90 \
      turbogap/soap_turbo_radial.f90 \
      turbogap/soap_turbo_angular.f90 \
      turbogap/soap_turbo_compress.f90 \
      turbogap/soap_turbo.f90 \
      turbogap/types.f90 \
      turbogap/local_properties.f90 \
      turbogap/gap.f90 \
      madlib_soap.f90 \
      madlib_gap.f90

OBJ = $(SRC:.f90=.o)

# Fortran modules must be compiled in dependency order, and SRC is already in
# it. Nothing states that to make, so -j reorders them and the build fails on a
# missing .mod -- which reads like a broken source tree rather than a race.
# .NOTPARALLEL is the whole fix; the build is ~20 s serial.
.NOTPARALLEL:

libmadlib.a: $(OBJ)
	ar rcs $@ $(OBJ)

%.o: %.f90
	$(FC) $(FFLAGS) -J. -c $< -o $@

clean:
	rm -f $(OBJ) *.mod turbogap/*.mod libmadlib.a

.PHONY: clean
