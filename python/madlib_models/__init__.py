# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Any machine-learned per-atom property, into any MADLIB observable.

MADLIB biases a structure towards a measured spectrum, and what stands between
the two is one per-atom quantity: a core-electron binding energy for XPS, a
dipole for IR, a chemical shielding -- isotropic or the whole tensor -- for
NMR. `compute mad/property/python` already accepts such a quantity from a
Python model and asks for one weighted backward pass in return. This package
is the part that should not have to be written again for every model: load the
model, run it on the geometry LAMMPS hands over, reduce its output to the
quantity the observable wants, and differentiate.

    compute cebe all mad/property/python madlib_models ncomp 1 cutoff 10.0 &
            config "backend=mace model=cebe.model types=C,O"
    fix     xps  all mad/xps c_cebe target measured.dat range 280 300 201 gamma 5.0

The deck never names a quantity twice, and nothing in it is specific to MACE:
swap `backend=mace` for `backend=nequip_nmr` and the same two lines bias an
NMR lineshape instead.

CONFIG, not environment. The module is imported once per run, so configuring
it by environment variable allows exactly one model and one quantity per deck
-- and a deck wanting a model's binding energies AND its dipoles wants two
computes. `mad_bind(config)` is therefore given the `config` string of the
compute that is binding, and returns that compute's own pair of callables.

    backend   mace | nequip_nmr | callable                        (required)
    model     the model file, or a NequIP training directory
    types     element of each LAMMPS atom type, in type order: "Si,O" (required)
    output    which of the model's outputs to read
                mace:       node_energy (default), atomic_dipoles, charges, ...
                nequip_nmr: magres (default)
    quantity  how to reduce that output to the ncomp values asked for
                scalar vector tensor6 tensor9 iso span vzz etaq component:zz
    scale     y -> scale*y + shift, for a unit or sign convention   (default 1)
    shift                                                          (default 0)
    device    cpu | cuda                      (default cuda if torch sees one)
    dtype     float32 | float64                        (default the model's)
    head      head name, for a multi-head MACE model
    rcut      neighbour cutoff the model's own graph is built with, in pbc mode
              (default the model's r_max)
    fn        module:function, for backend=callable
    file      a JSON file of named configs, and
    entry     which of its entries to use; keys given here override it

Keys are separated by whitespace or ';', so a config with spaces needs the
quotes LAMMPS strips: config "a=1 b=2".

ONE FORWARD PASS PER GEOMETRY, however many computes read it. Models are
cached by what identifies them -- the backend, the file, the device, the head
-- and each caches its own last forward pass with the autograd graph intact.
Two computes over one model, say an isotropic shielding and the span of the
same tensor, cost one model evaluation a step and two backward passes. MADLIB
deliberately does not cache a property on the timestep, since an MC fix
re-prices within one, so the geometry itself is the cache key.

WHAT EACH OBSERVABLE WANTS BACK:

    fix mad/xps, fix mad/nmr    ncomp 1    a per-atom scalar
    fix mad/ir                  ncomp 3    a per-atom dipole, summed by the fix
    fix mad/nmr/aniso           ncomp 6    a symmetric tensor, Voigt order

`ncomp` on the compute and `quantity` here have to agree, and a mismatch is an
error rather than a reshape: a 6-vector read as six scalars is still numbers.
"""

from .binding import mad_bind

__all__ = ["mad_bind"]
