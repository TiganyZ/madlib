"""Run a deck and read back what it produced.

A subprocess rather than the in-process `lammps` module: a MAD relaxation
takes seconds, so startup is noise, and this buys MPI ranks inside one
evaluation plus isolation from a LAMMPS abort.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read

_NUMBER = re.compile(r"^\s*-?\d")


class LammpsError(RuntimeError):
    pass


def parse_thermo(log_text):
    """Every thermo row of the last run, as a dict of column -> array."""
    blocks = []
    columns = None
    rows = []
    for line in log_text.splitlines():
        if line.lstrip().startswith("Step"):
            if columns and rows:
                blocks.append((columns, rows))
            columns, rows = line.split(), []
            continue
        if columns is None:
            continue
        if _NUMBER.match(line):
            parts = line.split()
            if len(parts) == len(columns):
                rows.append([float(p) for p in parts])
        elif rows:
            blocks.append((columns, rows))
            columns, rows = None, []
    if columns and rows:
        blocks.append((columns, rows))
    if not blocks:
        raise LammpsError("no thermo output in log")
    columns, rows = blocks[-1]
    table = np.asarray(rows, dtype=float)
    return {name: table[:, i] for i, name in enumerate(columns)}


def read_dump(path):
    """Final frame of a LAMMPS text dump, as an ASE Atoms."""
    return read(str(path), format="lammps-dump-text", index=-1)


def run(workdir, deck, lmp, ranks=1, links=(), timeout=3600, mpirun="mpirun", env=None):
    """Write the deck into workdir, run it, return (thermo, atoms_or_None)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for src in links:
        src = Path(src)
        dst = workdir / src.name
        if not dst.exists():
            os.symlink(src.resolve(), dst)

    deck_path = workdir / "in.deck"
    deck_path.write_text(deck)
    log_path = workdir / "log.lammps"

    cmd = [lmp, "-in", deck_path.name, "-log", log_path.name]
    if ranks > 1:
        cmd = [mpirun, "-np", str(ranks)] + cmd

    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    log_text = log_path.read_text() if log_path.exists() else ""
    if proc.returncode != 0:
        tail = (log_text or proc.stderr)[-2000:]
        raise LammpsError(f"lmp exited {proc.returncode} in {workdir}:\n{tail}")

    thermo = parse_thermo(log_text)
    dump = workdir / "final.dump"
    atoms = read_dump(dump) if dump.exists() else None
    return thermo, atoms


def find_lmp(explicit=None):
    for candidate in (explicit, os.environ.get("MADLIB_LMP"), shutil.which("lmp")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise LammpsError("no lmp binary; pass lmp= or set MADLIB_LMP")
