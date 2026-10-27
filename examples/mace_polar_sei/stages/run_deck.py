"""Run a LAMMPS deck in this Python process, so mace_lammps.attach can hand it the model.

    python run_deck.py DECK [name=value ...]

Each name=value becomes a LAMMPS -var; a value with spaces stays one string.
"""

import sys
from pathlib import Path

from lammps import lammps

STAGES = Path(__file__).resolve().parent


def lammps_args(pairs):
    args = ["-log", "log.lammps", "-var", "stages", str(STAGES)]
    for pair in pairs:
        name, value = pair.split("=", 1)
        args += ["-var", name, value]
    return args


def main(deck, *pairs):
    sys.path.insert(0, str(STAGES))
    lmp = lammps(cmdargs=lammps_args(pairs))
    lmp.file(deck)
    lmp.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
