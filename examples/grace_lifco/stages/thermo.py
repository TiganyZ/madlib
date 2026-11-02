"""Thermo columns of a LAMMPS log.

    python thermo.py mean LOG COLUMN     mean over the second half of the last run
    python thermo.py last LOG COLUMN
"""

import sys

import numpy as np


def last_block(path):
    header, rows, blocks = None, [], []
    for line in open(path):
        words = line.split()
        if words[:1] == ["Step"]:
            header, rows = words, []
        elif line.startswith("Loop time") and header:
            blocks.append((header, rows))
            header = None
        elif header:
            try:
                rows.append([float(w) for w in words])
            except ValueError:
                pass
    if header:
        blocks.append((header, rows))
    header, rows = blocks[-1]
    return header, np.array([r for r in rows if len(r) == len(header)])


def column(path, name):
    header, rows = last_block(path)
    return rows[:, header.index(name)]


if __name__ == "__main__":
    mode, path, name = sys.argv[1:]
    x = column(path, name)
    print(f"{(x[len(x) // 2:].mean() if mode == 'mean' else x[-1]):.10g}")
