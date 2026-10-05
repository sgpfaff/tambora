"""Finding the backend for an object, without importing packages the user never used.

Backends are listed by import path, so a backend module (and its package) is only
imported once an object that could belong to it has been seen: an object can only
come from a package that is already in ``sys.modules``. tambora's own models (e.g.
``tambora.ic.Plummer``) belong to no package, so any installed backend may take them.
"""

import importlib
import importlib.util
import sys
from typing import NamedTuple

from ._backend import PotentialBackend, SamplerBackend


class _Entry(NamedTuple):
    name: str           # the backend class's `name`
    package: str        # top-level module of the external package
    module: str         # tambora module defining the backend class
    cls: str            # backend class name


#: Potential backends in priority order: the first whose ``accepts`` is True wins.
_POTENTIAL_BACKENDS = (
    _Entry('galpy', 'galpy', 'tambora.interop._galpy.potential', 'GalpyPotential'),
)

#: Sampler backends, in the same form and order of priority.
_SAMPLER_BACKENDS = (
    _Entry('galpy', 'galpy', 'tambora.interop._galpy.sampling', 'GalpySampler'),
)


def _load(entry: _Entry) -> type:
    return getattr(importlib.import_module(entry.module), entry.cls)


def _installed(entries) -> tuple:
    return tuple(e.name for e in entries if importlib.util.find_spec(e.package) is not None)


def available_potential_backends() -> tuple:
    """Names of the potential backends whose package is installed."""
    return _installed(_POTENTIAL_BACKENDS)


def _is_tamboras(obj) -> bool:
    """Whether ``obj`` is one of tambora's own models, which mark themselves (``_tambora_model``)."""
    return getattr(type(obj), '_tambora_model', False) is True


def _backend_for(entries, cant, hint, obj, **kwargs):
    """Dispatch shared by the potential and sampler registries.

    ``cant`` completes "can't ..." for this object, and ``hint`` ends the error for an
    object no backend accepts. The backend is chosen by ``obj`` alone: the first, in
    priority order, that accepts it. It is built with ``obj`` and ``kwargs``.
    """
    for entry in entries:
        if entry.package not in sys.modules:    # obj can't come from a package never imported,
            if not _is_tamboras(obj) or importlib.util.find_spec(entry.package) is None:
                continue                        # unless it's tambora's own
        cls = _load(entry)
        if cls.accepts(obj):
            return cls(obj, **kwargs)

    installed = _installed(entries)
    supported = ', '.join(e.name for e in entries)
    raise TypeError(f"Can't {cant}. Supported packages: {supported} "
                    f"(installed: {', '.join(installed) or 'none'}). {hint}")


def potential_backend_for(obj) -> PotentialBackend:
    """Wrap ``obj`` in the backend for its package.

    Parameters
    ----------
    obj : object
        A potential from a supported package (e.g. a galpy ``Potential``, or a
        list of them).

    Raises
    ------
    TypeError
        If no backend accepts ``obj``.
    """
    return _backend_for(
        _POTENTIAL_BACKENDS, f"use a {type(obj).__name__} as an external potential",
        "For a custom force, subclass tambora.dynamics.forces.ExternalConservativeForce.",
        obj)


def sampler_backend_for(obj, *, potential=None) -> SamplerBackend:
    """Wrap the model ``obj`` in the sampler backend for its package.

    Parameters
    ----------
    obj : object
        A model from a supported package, e.g. a galpy potential or distribution function.
    potential : object, optional
        A potential to draw ``obj``'s density in, rather than its own; passed to the backend.

    Raises
    ------
    TypeError
        If no backend accepts ``obj``.
    """
    return _backend_for(
        _SAMPLER_BACKENDS, f"sample a {type(obj).__name__}",
        "Particles made another way can go straight into Sim.add_particles.",
        obj, potential=potential)
