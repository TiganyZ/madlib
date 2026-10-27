"""PolarMACE as the potential of a LAMMPS run, through fix external.

PolarMACE's electrostatics is a k-space sum over the whole cell, so it cannot be
a LAMMPS pair style working on local atoms and ghosts. `lammps/mace_potential.lmp`
defines the fix and calls attach() with the string variables model, species,
device and dtype. Serial only: the model needs every atom of the cell on one rank.
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from lammps import LMP_VAR_STRING, lammps

from calc import polar

FIX_ID = "ext"
STATE = {}


def variable(lmp, name):
    return lmp.extract_variable(name, None, LMP_VAR_STRING)


def cell_of(lmp):
    boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
    hx, hy, hz = np.subtract(boxhi, boxlo)
    return np.array([[hx, 0.0, 0.0], [xy, hy, 0.0], [xz, yz, hz]])


def virial_of(stress, volume):
    """ASE stress (xx yy zz yz xz xy, eV/A^3) to a LAMMPS virial (xx yy zz xy xz yz, eV)."""
    s = -np.asarray(stress) * volume
    return [s[0], s[1], s[2], s[5], s[4], s[3]]


def callback(lmp, ntimestep, nlocal, tag, x, f):
    if nlocal != lmp.get_natoms():
        raise RuntimeError("mace_lammps runs on one MPI rank only")
    types = lmp.numpy.extract_atom("type")[:nlocal]
    atoms = Atoms(numbers=STATE["numbers"][types - 1], positions=np.array(x[:nlocal]),
                  cell=cell_of(lmp), pbc=True)
    atoms.info.update(charge=0, spin=1)
    atoms.calc = STATE["calc"]
    f[:nlocal] = atoms.get_forces()
    lmp.fix_external_set_energy_global(FIX_ID, atoms.get_potential_energy())
    lmp.fix_external_set_virial_global(FIX_ID, virial_of(atoms.get_stress(), atoms.get_volume()))


def attach(ptr):
    lmp = lammps(ptr=ptr)
    species = variable(lmp, "species").split()
    STATE["numbers"] = np.array([atomic_numbers[s] for s in species])
    STATE["calc"] = polar(variable(lmp, "model"), variable(lmp, "device"), variable(lmp, "dtype"))
    lmp.set_fix_external_callback(FIX_ID, callback, lmp)
