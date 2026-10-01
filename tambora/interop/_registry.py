"""Finding the backend for an object, without importing packages the user never used.

Backends are listed by import path, so a backend module (and its package) is only
imported once an object that could belong to it has been seen: an object can only
come from a package that is already in ``sys.modules``.
"""

import importlib
import importlib.util
import sys
from typing import NamedTuple, Optional

from ._backend import PotentialBackend, SamplerBackend


class _Entry(NamedTuple):
    name: str           # value for backend=
    package: str        # top-level module of the external package
    module: str         # tambora module defining the backend class
    cls: str            # backend class name
    install: str        # where to point users who don't have the package


#: Potential backends in priority order: the first whose ``accepts`` is True wins.
_POTENTIAL_BACKENDS = (
    _Entry('galpy', 'galpy', 'tambora.interop._galpy.potential', 'GalpyPotential',
           'https://docs.galpy.org/en/stable/installation.html'),
)

#: Sampler backends, in the same form and order of priority.
_SAMPLER_BACKENDS = (
    _Entry('galpy', 'galpy', 'tambora.interop._galpy.sampling', 'GalpySampler',
           'https://docs.galpy.org/en/stable/installation.html'),
)


def _load(entry: _Entry) -> type:
    return getattr(importlib.import_module(entry.module), entry.cls)


def _installed(entries) -> tuple:
    return tuple(e.name for e in entries if importlib.util.find_spec(e.package) is not None)


def available_potential_backends() -> tuple:
    """Names of the potential backends whose package is installed."""
    return _installed(_POTENTIAL_BACKENDS)


def _backend_for(entries, kind, cant, hint, obj, backend):
    """Dispatch shared by the potential and sampler registries.

    ``kind`` names the backends in errors (``'potential'``), ``cant`` completes "can't ..."
    for this object, and ``hint`` ends the error for an object no backend accepts.
    """
    if backend is not None:
        entry = next((e for e in entries if e.name == backend), None)
        if entry is None:
            known = ', '.join(repr(e.name) for e in entries)
            raise ValueError(f"Unknown {kind} backend {backend!r}. Known backends: {known}.")
        if importlib.util.find_spec(entry.package) is None:
            raise ImportError(f"The {entry.name!r} backend needs {entry.package}, which isn't "
                              f"installed. See {entry.install}")
        cls = _load(entry)
        if not cls.accepts(obj):
            raise TypeError(f"The {entry.name!r} backend can't {cant}.")
        return cls(obj)

    for entry in entries:
        if entry.package not in sys.modules:    # obj can't come from a package never imported
            continue
        cls = _load(entry)
        if cls.accepts(obj):
            return cls(obj)

    installed = _installed(entries)
    supported = ', '.join(e.name for e in entries)
    raise TypeError(f"Can't {cant}. Supported packages: {supported} "
                    f"(installed: {', '.join(installed) or 'none'}). {hint}")


def potential_backend_for(obj, backend: Optional[str] = None) -> PotentialBackend:
    """Wrap ``obj`` in the backend for its package.

    Parameters
    ----------
    obj : object
        A potential from a supported package (e.g. a galpy ``Potential``, or a
        list of them).
    backend : str, optional
        Name of the backend to use, e.g. ``'galpy'``. By default it is chosen
        from ``obj``: the first backend, in priority order, that accepts it.

    Raises
    ------
    ValueError
        If ``backend`` names no known backend.
    ImportError
        If ``backend`` is given but its package isn't installed.
    TypeError
        If no backend accepts ``obj`` (or the named one doesn't).
    """
    return _backend_for(
        _POTENTIAL_BACKENDS, 'potential', f"use a {type(obj).__name__} as an external potential",
        "For a custom force, subclass tambora.dynamics.forces.ExternalConservativeForce.",
        obj, backend)


def sampler_backend_for(obj, backend: Optional[str] = None) -> SamplerBackend:
    """Wrap the model ``obj`` in the sampler backend for its package.

    Parameters
    ----------
    obj : object
        A model from a supported package, e.g. a galpy potential or distribution function.
    backend : str, optional
        Name of the backend to use, e.g. ``'galpy'``. By default it is chosen
        from ``obj``: the first backend, in priority order, that accepts it.

    Raises
    ------
    ValueError, ImportError, TypeError
        As for :func:`potential_backend_for`.
    """
    return _backend_for(
        _SAMPLER_BACKENDS, 'sampler', f"sample a {type(obj).__name__}",
        "Particles made some other way can go straight into Sim.add_particles.",
        obj, backend)
