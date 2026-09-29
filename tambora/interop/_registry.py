"""Finding the backend for an object, without importing packages the user never used.

Backends are listed by import path, so a backend module (and its package) is only
imported once an object that could belong to it has been seen: an object can only
come from a package that is already in ``sys.modules``.
"""

import importlib
import importlib.util
import sys
from typing import NamedTuple, Optional

from ._backend import PotentialBackend


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


def _load(entry: _Entry) -> type:
    return getattr(importlib.import_module(entry.module), entry.cls)


def available_potential_backends() -> tuple:
    """Names of the potential backends whose package is installed."""
    return tuple(e.name for e in _POTENTIAL_BACKENDS
                 if importlib.util.find_spec(e.package) is not None)


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
    if backend is not None:
        entry = next((e for e in _POTENTIAL_BACKENDS if e.name == backend), None)
        if entry is None:
            known = ', '.join(repr(e.name) for e in _POTENTIAL_BACKENDS)
            raise ValueError(f"Unknown potential backend {backend!r}. Known backends: {known}.")
        if importlib.util.find_spec(entry.package) is None:
            raise ImportError(f"The {entry.name!r} backend needs {entry.package}, which isn't "
                              f"installed. See {entry.install}")
        cls = _load(entry)
        if not cls.accepts(obj):
            raise TypeError(f"The {entry.name!r} backend can't use a {type(obj).__name__} "
                            f"as an external potential.")
        return cls(obj)

    for entry in _POTENTIAL_BACKENDS:
        if entry.package not in sys.modules:    # obj can't come from a package never imported
            continue
        cls = _load(entry)
        if cls.accepts(obj):
            return cls(obj)

    installed = available_potential_backends()
    supported = ', '.join(e.name for e in _POTENTIAL_BACKENDS)
    raise TypeError(
        f"Can't use a {type(obj).__name__} as an external potential. Supported packages: "
        f"{supported} (installed: {', '.join(installed) or 'none'}). For a custom force, "
        f"subclass tambora.dynamics.forces.ExternalConservativeForce.")
