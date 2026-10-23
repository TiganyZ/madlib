"""Concatenate per-temperature selections into one dataframe.

    python merge_selected.py OUT IN...
"""

import sys

import pandas as pd

if __name__ == "__main__":
    parts = [pd.read_pickle(p, compression="gzip") for p in sys.argv[2:]]
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(sys.argv[1], compression="gzip")
    print(f"{len(df)} selected: " + ", ".join(str(len(p)) for p in parts))
