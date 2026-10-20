"""MADlib as an EZGA plugin: experimental data as an objective, MAD as the
local optimiser that reaches it.

    from madlib_ezga import madlib_runner, objective_xrd_rel_error

Nothing here is registered with EZGA. Its config loader resolves
"madlib_ezga.runner:madlib_runner" or a bare path to a file, so the plugin is
an ordinary import.
"""

from .deck import Pdf, Potential, Xrd, render
from .front import non_dominated, write_front
from .objectives import objective_energy_per_atom, objective_xrd_rel_error
from .runner import madlib_runner

__all__ = [
    "Pdf",
    "Potential",
    "Xrd",
    "render",
    "madlib_runner",
    "objective_xrd_rel_error",
    "objective_energy_per_atom",
    "non_dominated",
    "write_front",
]
