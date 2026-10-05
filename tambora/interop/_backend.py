"""The interfaces a package backend implements."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Hashable, Optional

import numpy as np


class PotentialBackend(ABC):
    """Adapter from one package's potential/force objects to tambora internal units.

    A backend wraps a single object (which may itself be a composite of several
    potentials) and converts everything to internal units once, when wrapped.
    Backends are found through :func:`tambora.interop.potential_backend_for`;
    :class:`~tambora.dynamics.forces.ExternalPotential` is the force built on them.
    """

    #: Short name, e.g. ``'galpy'``, shown in reprs and part of the duplicate check. Must
    #: match the backend's registry entry, which also records the package it wraps.
    name: ClassVar[str]

    #: The (possibly normalised) object this backend wraps.
    obj: Any

    @classmethod
    @abstractmethod
    def accepts(cls, obj) -> bool:
        """Whether ``obj`` belongs to this backend's package.

        Only called once the backend's package has been imported, so it may import it.
        A ``True`` here doesn't promise success: ``__init__`` still validates and
        raises a specific error for objects of the right package but the wrong kind.
        """

    @abstractmethod
    def __init__(self, obj):
        """Validate ``obj``, record its units, and precompute conversion factors."""

    @abstractmethod
    def acc(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Acceleration (N, 3) [kpc/Gyr^2] at Cartesian ``pos`` (N, 3) [kpc], time ``t`` [Gyr]."""

    @abstractmethod
    def potential(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Potential (N,) [(kpc/Gyr)^2] at Cartesian ``pos`` (N, 3) [kpc], time ``t`` [Gyr]."""

    def check_still_valid(self) -> None:
        """Raise if the wrapped object can no longer be trusted.

        Called at the start of every ``Sim.run()``. The default does nothing; a
        backend overrides it when outside state can change what the object
        returns (e.g. Agama's global unit system).
        """

    def dedup_key(self) -> Optional[Hashable]:
        """Key identifying the wrapped model, or ``None`` to opt out of dedup.

        It must not change when the object is evaluated: some objects cache results in
        themselves, and the same object added again after a run must still match.
        """
        return None

    def describe(self) -> str:
        """Short label for ``Sim.__repr__``."""
        return type(self.obj).__name__


class SamplerBackend(ABC):
    """Adapter that draws particles from one package's models.

    A backend wraps a single model and prepares it for sampling once, when wrapped, so
    drawing again is cheap. Backends are found through
    :func:`tambora.interop.sampler_backend_for`; :func:`tambora.ic.sample` is built on them.
    Seeds, masses and the returned :class:`~tambora.ic.ParticleSet` are tambora's job.
    """

    #: Short name, e.g. ``'galpy'``, recorded in a sample's ``meta['backend']``. Must
    #: match the backend's registry entry, which also records the package it wraps.
    name: ClassVar[str]

    #: The model this backend wraps.
    obj: Any

    @classmethod
    @abstractmethod
    def accepts(cls, obj) -> bool:
        """Whether ``obj`` is a model this backend can sample.

        Only called once the backend's package has been imported, so it may import it.
        """

    @abstractmethod
    def __init__(self, obj, potential=None):
        """Validate ``obj``, record its units, and prepare it for sampling.

        Given a ``potential``, prepare to draw ``obj``'s density in equilibrium in it, rather
        than in ``obj``'s own potential. A backend that can't do that raises TypeError.
        """

    @property
    @abstractmethod
    def total_mass(self) -> float:
        """Mass of the model being sampled [Msun]. Each of n particles gets ``total_mass / n``."""

    @abstractmethod
    def draw(self, n: int, seed: int) -> tuple:
        """Positions (n, 3) [kpc] and velocities (n, 3) [km/s] of n particles.

        The same ``seed`` must give the same particles, and drawing must leave any random
        state outside the backend as it found it.
        """

    def describe(self) -> str:
        """Short label for ``ParticleSet.meta``."""
        return type(self.obj).__name__
