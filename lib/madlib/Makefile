# Standalone build of the MADLIB soap_turbo backend.
#
#   make                     libmadlib.a
#   make FC=ifx FFLAGS=-O2   a different compiler
#
# BLAS is the only external dependency.

FC      = gfortran
FFLAGS  = -O2 -fPIC
BLAS    = -llapack -lblas

SRC = turbogap/kinds.f90 \
      turbogap/soap_turbo_functions.f90 \
      turbogap/soap_turbo_radial.f90 \
      turbogap/soap_turbo_angular.f90 \
      turbogap/soap_turbo_compress.f90 \
      turbogap/soap_turbo.f90 \
      turbogap/local_properties.f90 \
      madlib_soap.f90

OBJ = $(SRC:.f90=.o)

libmadlib.a: $(OBJ)
	ar rcs $@ $(OBJ)

%.o: %.f90
	$(FC) $(FFLAGS) -J. -c $< -o $@

clean:
	rm -f $(OBJ) *.mod libmadlib.a

.PHONY: clean
