# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The model families this bridge knows how to run.

A backend is a module with four names:

    KEYS              config keys it reads beyond the common ones
    DEFAULT_OUTPUT    which of the model's outputs a config may leave unsaid
    DEFAULT_QUANTITY  and how it is reduced by default
    identity(cfg)     what makes two configs the same loaded model
    build(cfg)        -> an object with .graph(...) and .forward(graph) -> dict
"""

import importlib

KNOWN = ("mace", "nequip_nmr", "callable")


def load(name):
    if name not in KNOWN:
        raise ValueError(f"unknown backend '{name}'; one of {list(KNOWN)}")
    return importlib.import_module(f".{name}", __package__)
