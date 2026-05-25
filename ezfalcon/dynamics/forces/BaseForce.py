from abc import ABC, abstractmethod
import numpy as np

class BaseForce(ABC):
    """Abstract base for any object that produces a force on N-body particles.

    Concrete subclasses must implement :meth:`force`. Forces that are the
    gradient of a potential should subclass :class:`ConservativeForceField`
    and additionally implement :meth:`potential`. Forces that depend on
    velocity (drag, dynamical friction) should subclass
    :class:`DissipativeForce`.

    Forces compose with ``+``::

        combined = ExternalGalpyPotential(nfw) + ExternalGalpyPotential(disk)

    The result is a :class:`CompositeForce` that itself satisfies the
    :class:`BaseForce` interface.
    """

    @abstractmethod
    def acc(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        """Acceleration on each particle, shape ``(N, 3)``, internal units."""
        ...
    def _eval_acc(self,  pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        return self.acc(pos, vel, mass, t)
    # def _c_handle(self):
    #     """Opt-in C entry point for the all-C integrator path.

    #     Forces with a C implementation override this to return a PyCapsule
    #     wrapping a function pointer. Forces without one return ``None`` and
    #     the integrator falls back to the Python loop.
    #     """
    #     return None

    def __add__(self, other: "BaseForce") -> "BaseForce":
        from .CompositeForce import CompositeForce
        return CompositeForce([self, other])
    
class NullBaseForce(BaseForce):
    """No-op self-gravity. Used when self-gravity is disabled."""
    def __init__(self):
        self._zeros_array_3d = None
        self._zeros_array_1d = None

    def acc(self, pos, vel, mass, t):
        if self._zeros_array_3d is None:
            self._zeros_array_3d = np.zeros_like(pos)
        return self._zeros_array_3d
    def potential(self, pos, vel, mass, t):
        if self._zeros_array_1d is None:
            self._zeros_array_1d = np.zeros(pos.shape[0])
        return self._zeros_array_1d
    def acc_and_potential(self, pos, vel, mass, t):
        return self.acc(pos, vel, mass, t), self.potential(pos, vel, mass, t)