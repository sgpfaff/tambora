import numpy as np

from .ExternalConservativeForce import ExternalConservativeForce
from ....interop import potential_backend_for


class ExternalPotential(ExternalConservativeForce):
    """A potential from a supported package, as a conservative external force.

    The package is detected from the object, so the same class wraps any backend::

        sim.add_external_force(ExternalPotential(MWPotential2014))    # galpy

    Combine with other forces via ``+``.

    Parameters
    ----------
    potential : object
        A potential from a supported package, e.g. a galpy ``Potential``, a galpy
        ``CompositePotential``, or a list of galpy potentials.
    """

    def __init__(self, potential):
        self._backend = potential_backend_for(potential)
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

    def _dedup_key(self):
        """Key for ``Sim``'s duplicate check: the backend's key, tagged with this class and
        the backend's name so it can't collide with another force's."""
        return self._key

    def __repr__(self):
        return f"ExternalPotential({self._backend.name}: {self._backend.describe()})"
