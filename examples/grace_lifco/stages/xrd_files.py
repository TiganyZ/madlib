"""XRD file helpers.

    python xrd_files.py placeholder OUT          a flat target on the q grid, for prediction only
    python xrd_files.py average PRED OUT         mean of every block madlib wrote to PRED
"""

import sys

import numpy as np

Q_MIN, Q_MAX, NQ = 0.5, 10.5, 201


def placeholder(out):
    q = np.linspace(Q_MIN, Q_MAX, NQ)
    np.savetxt(out, np.column_stack([q, np.ones_like(q)]), fmt="%.10f")


def blocks(path):
    out, cur = [], []
    for line in open(path):
        if line.startswith("# step"):
            if cur:
                out.append(np.array(cur))
            cur = []
        elif not line.startswith("#"):
            cur.append([float(x) for x in line.split()[:2]])
    if cur:
        out.append(np.array(cur))
    return out


def average(pred, out):
    b = blocks(pred)
    mean = np.mean([x[:, 1] for x in b], axis=0)
    np.savetxt(out, np.column_stack([b[0][:, 0], mean]), fmt="%.12e")
    print(f"averaged {len(b)} frames into {out}")


if __name__ == "__main__":
    {"placeholder": placeholder, "average": average}[sys.argv[1]](*sys.argv[2:])
