# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The config string of one compute, parsed strictly.

    "backend=mace model=cebe.model types=C,O quantity=scalar"

Whitespace or ';' between keys. An unknown key is an error: a mistyped
'quantitiy=span' that fell back to a default would produce a spectrum rather
than a complaint.

'file=models.json entry=cebe' reads the keys from a JSON file instead, for a
config too long for the deck; keys in the string override the file's.

LAMMPS DOES NOT SUBSTITUTE ${var} INSIDE A QUOTED ARGUMENT, and a config with
spaces must be quoted, so a model path cannot come from a LAMMPS variable --
the literal text ${var} arrives. Every value is therefore passed through
expandvars and expanduser:

    config "backend=mace model=$MACE_MODELS/cebe.model types=C,O"
"""

import json
import os

COMMON = {"backend", "model", "types", "output", "quantity", "scale", "shift", "device",
          "dtype", "file", "entry"}


def parse(text):
    """The config string as a dict of strings."""
    cfg = {}
    for token in text.replace(";", " ").split():
        if "=" not in token:
            raise ValueError(f"config token '{token}' is not key=value")
        key, value = token.split("=", 1)
        cfg[key.strip()] = os.path.expanduser(os.path.expandvars(value.strip()))

    if "file" in cfg:
        entry = cfg.get("entry")
        if not entry:
            raise ValueError("config 'file' needs 'entry' naming which of its configs to use")
        with open(cfg["file"]) as fh:
            entries = json.load(fh)
        if entry not in entries:
            raise ValueError(f"config file {cfg['file']} has no entry '{entry}'; "
                             f"it has {sorted(entries)}")
        merged = {str(k): str(v) for k, v in entries[entry].items()}
        merged.update(cfg)
        cfg = merged
    return cfg


def validate(cfg, extra):
    """Refuse a key no backend reads."""
    allowed = COMMON | set(extra)
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"unknown config key(s) {unknown}; this backend reads "
                         f"{sorted(allowed)}")


def number(cfg, key, default):
    return float(cfg[key]) if key in cfg else default


def species(cfg):
    """The element symbol of each LAMMPS atom type, in type order."""
    if "types" not in cfg:
        raise ValueError("config needs types=<element per LAMMPS atom type>, e.g. types=Si,O")
    return [s for s in cfg["types"].replace(",", " ").split()]
