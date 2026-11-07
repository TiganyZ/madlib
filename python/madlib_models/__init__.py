# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Any machine-learned per-atom property, into any MADLIB observable.

What stands between a structure and a spectrum is one per-atom quantity: a
binding energy for XPS, a dipole for IR, a shielding -- isotropic or the whole
tensor -- for NMR. `compute mad/property/python` takes such a quantity from a
Python model and asks for one weighted backward pass back. This package is the
part that should not have to be written again per model.

    compute cebe all mad/property/python madlib_models ncomp 1 cutoff 10.0 &
            config "backend=mace model=cebe.model types=C,O"
    fix     xps  all mad/xps c_cebe target measured.dat range 280 300 201 gamma 5

Config keys, separators whitespace or ';', unknown keys an error:

    backend   mace | nequip_nmr | callable                        (required)
    model     the model file, or a NequIP training directory
    types     element of each LAMMPS atom type, in order: "Si,O"   (required)
    output    which model output to read (mace: node_energy, atomic_dipoles...)
    quantity  scalar vector tensor6 tensor9 iso span vzz etaq component:zz
    scale     y -> scale*y + shift, for a unit or sign convention
    shift
    device    cpu | cuda          dtype  float32 | float64
    head      a multi-head MACE model's head
    rcut      the graph cutoff in pbc mode; default the model's r_max
    fn        module:function, for backend=callable
    file      a JSON file of named configs, with entry= choosing one

What each observable wants: ncomp 1 for mad/xps and mad/nmr, 3 for mad/ir (a
dipole, summed by the fix), 6 for mad/nmr/aniso (Voigt order). ncomp and
quantity must agree; a mismatch is an error, not a reshape.

ONE FORWARD PASS PER GEOMETRY, however many computes read it: models are
cached by what identifies them and each keeps its last pass with the graph
intact, so an isotropic shielding and the span of the same tensor cost one
model evaluation and two backward passes. The key is the geometry, not the
timestep, because an MC fix re-prices inside one.

See python/README.md for the backends, and docs/ADDING-AN-OBSERVABLE.md for
putting a new quantity or a new spectrum behind it.
"""

from .binding import mad_bind

__all__ = ["mad_bind"]
