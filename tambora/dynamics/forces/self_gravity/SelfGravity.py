"""Self-gravity solvers as :class:`InteractionForce` + :class:`Conservative` objects.

These wrap the C-extension gravity functions in the Force-object API used at
the integrator boundary. Construction params (eps, theta, kernel) are frozen
at instantiation time so the solver instance is reusable across substeps.

Until the Phase-4 C-wrapper split (gravity / gravity_force / gravity_pot) lands,
all three entry points call the existing single-pass ``gravity()`` and slice
the output. The Python-side API is forward-compatible: when the wrappers gain
dedicated force-only / pot-only paths, only the body of these methods changes.
"""

from typing import Tuple, Union
import numpy as np

from ..Force import Force
from ..Conservative import Conservative
from .falcON import _falcON_gravity
from .directSummation import _direct_summation_C, _direct_summation_py
from abc import abstractmethod


def _own_softening(eps):
    """A solver's own softening [kpc]: None, a number, or one value per particle."""
    if eps is None:
        return None
    arr = np.asarray(eps)
    if arr.ndim > 1 or arr.dtype.kind not in 'iuf':
        raise TypeError("A solver's own eps must be a number or one value per particle.")
    if not np.all(np.isfinite(arr) & (arr >= 0)):
        raise ValueError("A solver's own eps must be finite and non-negative.")
    return eps if arr.ndim == 0 else np.asarray(eps, dtype=np.float64)


class SelfGravityForce(Force, Conservative):
    """Base class for self-gravity solvers.

    ``eps``, the particles' softening, is passed to each call; a solver built with
    its own ``eps`` uses that instead.
    """
    eps = None

    @abstractmethod
    def acc(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray: ...
    @abstractmethod
    def potential(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray: ...

    def acc_and_potential(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> Tuple[np.ndarray, np.ndarray]:
        return self.acc(pos, mass, eps), self.potential(pos, mass, eps)

    def _softening(self, eps):
        """The solver's own softening if it has one, else ``eps``."""
        eps = eps if self.eps is None else self.eps
        if eps is None:
            raise ValueError("No softening: pass eps, or give the solver its own.")
        return eps

    def _eval_acc(self, pos, vel, mass, t):
        return self.acc(pos, mass)

    def _eval_potential(self, pos, vel, mass, t):
        return self.potential(pos, mass)

    def _eval_acc_and_potential(self, pos, vel, mass, t):
        return self.acc_and_potential(pos, mass)
    

class FalcONGravity(SelfGravityForce):
    """Self-gravity via the falcON fast-multipole tree.

    Parameters
    ----------
    theta : float, optional
        Tree opening angle (default 0.6). Smaller = more accurate but slower.
    kernel : int, optional
        Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.
    eps : float or (N,) array, optional
        Softening length(s) [kpc] to use instead of the ``eps`` passed to each call.
    """

    def __init__(self, *, theta: float = 0.6, kernel: int = 1,
                 eps: Union[float, np.ndarray, None] = None):
        self.theta = theta
        self.kernel = kernel
        self.eps = _own_softening(eps)

    def acc_and_potential(
        self, pos: np.ndarray, mass: np.ndarray, eps=None) -> Tuple[np.ndarray, np.ndarray]:
        return _falcON_gravity(
            pos, mass, self._softening(eps), self.theta, self.kernel,
            return_potential=True,
        )

    def acc(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray:
        return _falcON_gravity(
            pos, mass, self._softening(eps), self.theta, self.kernel,
            return_potential=False,
        )

    def potential(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray:
        return self.acc_and_potential(pos, mass, eps)[1]


class DirectSummationGravity(SelfGravityForce):
    """Self-gravity via direct O(N^2) pair summation.

    Parameters
    ----------
    use_C : bool, optional
        Use the C extension (default ``True``). Set to ``False`` for the
        pure-Python reference implementation.
    eps : float or (N,) array, optional
        Softening length(s) [kpc] to use instead of the ``eps`` passed to each call.
        With an array, pairwise softening uses the arithmetic mean
        ``(eps_i + eps_j) / 2``.
    """

    def __init__(self, *, use_C: bool = True, eps: Union[float, np.ndarray, None] = None):
        self.use_C = use_C
        self.eps = _own_softening(eps)

    def acc_and_potential(
        self, pos: np.ndarray, mass: np.ndarray, eps=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(pos, mass, self._softening(eps), return_potential=True)

    def acc(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(pos, mass, self._softening(eps), return_potential=False)

    def potential(self, pos: np.ndarray, mass: np.ndarray, eps=None) -> np.ndarray:
        return self.acc_and_potential(pos, mass, eps)[1]
    
class NullSelfGravity(SelfGravityForce):
    """No-op self-gravity. Used when self-gravity is disabled."""
    def __init__(self):
        self._zeros_array_3d = None
        self._zeros_array_1d = None

    # Shared read-only zeros, remade only for a new number of particles.
    def acc(self, pos, mass, eps=None):
        if self._zeros_array_3d is None or self._zeros_array_3d.shape != pos.shape:
            self._zeros_array_3d = np.zeros_like(pos)
            self._zeros_array_3d.flags.writeable = False
        return self._zeros_array_3d
    def potential(self, pos, mass, eps=None):
        if self._zeros_array_1d is None or len(self._zeros_array_1d) != len(pos):
            self._zeros_array_1d = np.zeros(len(pos))
            self._zeros_array_1d.flags.writeable = False
        return self._zeros_array_1d
    def acc_and_potential(self, pos, mass, eps=None):
        return self.acc(pos, mass), self.potential(pos, mass)
