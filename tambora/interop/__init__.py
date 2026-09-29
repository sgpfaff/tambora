"""Bridges between tambora and external galactic-dynamics packages.

One backend per package adapts that package's objects to tambora internal units.
Importing this module never imports an optional package; a backend is only loaded
once an object from its package has been seen.
"""

from ._backend import PotentialBackend
from ._registry import available_potential_backends, potential_backend_for

__all__ = ['PotentialBackend', 'available_potential_backends', 'potential_backend_for']
