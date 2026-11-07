# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""What `compute mad/property/python ... config "..."` calls, once, at setup."""

from . import backends, config, engine


def mad_bind(text):
    """The (compute, adjoint) pair for one compute's config string."""
    cfg = config.parse(text or "")
    if "backend" not in cfg:
        raise ValueError("config needs backend=<" + "|".join(backends.KNOWN) + ">; see "
                         "the madlib_models docstring for the rest of the keys")
    backend = backends.load(cfg["backend"])
    config.validate(cfg, backend.KEYS)

    quantity = cfg.get("quantity", backend.DEFAULT_QUANTITY)
    if quantity is None:
        raise ValueError(f"backend={cfg['backend']} has no default quantity; give "
                         f"quantity=<scalar|vector|tensor6|iso|span|vzz|etaq|component:zz>")
    prop = engine.Property(engine.model_for(cfg, backend),
                           cfg.get("output", backend.DEFAULT_OUTPUT), quantity,
                           config.number(cfg, "scale", 1.0), config.number(cfg, "shift", 0.0))
    return prop.compute, prop.adjoint
