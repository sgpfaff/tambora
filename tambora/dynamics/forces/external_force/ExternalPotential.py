import numpy as np

from .ExternalConservativeForce import ExternalConservativeForce
from ....interop import potential_backend_for
from ....interop._registry import potential_backend_named


class ExternalPotential(ExternalConservativeForce):
    """A potential from a supported package, as a conservative external force.

    The package is detected from the object, so the same class wraps any backend::

        sim.add_force(ExternalPotential(MWPotential2014))    # galpy

    Combine with other forces via ``+``.

    Parameters
    ----------
    potential : object
        A potential from a supported package, e.g. a galpy ``Potential``, a galpy
        ``CompositePotential``, or a list of galpy potentials.
    """

    def __init__(self, potential):
        self._attach(potential_backend_for(potential))

    @classmethod
    def _from_backend(cls, backend):
        """The force of an already-made backend, e.g. one loaded from a run file."""
        force = cls.__new__(cls)
        force._attach(backend)
        return force

    def _attach(self, backend):
        self._backend = backend
        key = self._backend.dedup_key()
        self._key = None if key is None else (type(self), self._backend.name, key)

    @property
    def backend(self):
        """The :class:`~tambora.interop.PotentialBackend` wrapping the potential."""
        return self._backend

    def acc(self, pos: np.ndarray, t) -> np.ndarray:
        return self._backend.acc(pos, t)

    def potential(self, pos: np.ndarray, t) -> np.ndarray:
        return self._backend.potential(pos, t)

    def check_still_valid(self) -> None:
        """Raise if outside state has invalidated the wrapped potential (see the backend)."""
        self._backend.check_still_valid()

    def save_problem(self):
        return self._backend.save_problem()

    def save(self):
        """The backend's record with the backend's name, or None if the backend doesn't save."""
        saved = self._backend.save()
        return None if saved is None else {'backend': self._backend.name, **saved}

    @classmethod
    def load(cls, record):
        return cls._from_backend(potential_backend_named(record['backend']).load(record))

    def _dedup_key(self):
        """Key for ``Sim``'s duplicate check: the backend's key, tagged with this class and
        the backend's name so it can't collide with another force's."""
        return self._key

    def __repr__(self):
        return f"ExternalPotential({self._backend.name}: {self._backend.describe()})"
