from abc import ABC, abstractmethod
import numpy as np


class Force(ABC):
    """Root of all forces acting on N-body particles.

    The canonical interface used at the integrator boundary is
    :meth:`_eval_acc`, which always takes the full ``(pos, vel, mass, t)``
    state. Concrete subclasses implement the author-facing :meth:`acc`, whose
    signature may be narrower (e.g. ``acc(pos, t)`` for an external potential);
    the matching ``_eval_*`` adapter maps the uniform call onto it.

    Forces split along two orthogonal axes:

    * **Coupling** — :class:`ExternalForce` (single-particle, evaluated per
      particle against a fixed external field) vs :class:`InteractionForce`
      (collective N-body, requires a solver). This determines how the force is
      routed and whether it needs the whole-ensemble arrays.
    * **Conservative** — the :class:`~.Conservative` mixin adds a scalar
      potential (and hence velocity-independence) for energy diagnostics.

    Forces compose with ``+``::

        combined = ExternalPotential(nfw) + ExternalPotential(disk)

    The result is a :class:`CompositeForce` that itself satisfies the
    :class:`Force` interface.
    """

    @abstractmethod
    def acc(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        """Acceleration on each particle, shape ``(N, 3)``, internal units."""
        ...

    def _eval_acc(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        return self.acc(pos, vel, mass, t)

    def _dedup_key(self):
        """Identity used by ``Sim.add_external_force`` to reject duplicates.

        Return a hashable/comparable key built from the force's meaningful
        configuration; two forces with equal, non-``None`` keys are treated as
        duplicates. The default is ``None`` which opts out of dedup entirely. 
        Never key on mutable state.
        """
        return None

    def __add__(self, other: "Force") -> "Force":
        from .CompositeForce import CompositeForce
        return CompositeForce([self, other])


class NullForce(Force):
    """No-op external force. Used as an empty external slot."""
    
    def acc(self, pos, vel, mass, t):
        return np.zeros_like(pos)

    def potential(self, pos, vel, mass, t):
        return np.zeros(pos.shape[0])

    def acc_and_potential(self, pos, vel, mass, t):
        return self.acc(pos, vel, mass, t), self.potential(pos, vel, mass, t)
