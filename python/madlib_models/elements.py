# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Element symbol to atomic number, without pulling in ASE for one lookup."""

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
           "Mc Lv Ts Og").split()

NUMBER = {s: i + 1 for i, s in enumerate(SYMBOLS)}


def numbers_of_types(species):
    """Atomic number per LAMMPS atom type, indexable by the 1-based type."""
    import numpy as np
    for s in species:
        if s not in NUMBER:
            raise ValueError(f"'{s}' is not an element symbol")
    return np.array([0] + [NUMBER[s] for s in species], dtype=np.int64)
