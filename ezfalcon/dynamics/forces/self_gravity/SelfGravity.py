"""Self-gravity solvers as :class:`ConservativeForceField` objects.

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

from ..ConservativeForce import ConservativeForce
from .falcON import _falcON_gravity
from .directSummation import _direct_summation_C, _direct_summation_py
from abc import abstractmethod

class SelfGravityForce(ConservativeForce):
    @abstractmethod
    def acc(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def potential(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray: ...
    
    def acc_and_potential(self, pos: np.ndarray, mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.acc(pos, mass), self.potential(pos, mass)
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
    eps : float or (N,) array
        Gravitational softening length(s) [kpc].
    theta : float, optional
        Tree opening angle (default 0.6). Smaller = more accurate but slower.
    kernel : int, optional
        Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.
    """

    def __init__(
        self,
        eps: Union[float, np.ndarray],
        theta: float = 0.6,
        kernel: int = 1,
    ):
        self.eps = eps
        self.theta = theta
        self.kernel = kernel

    def acc_and_potential(
        self, pos: np.ndarray, mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _falcON_gravity(
            pos, mass, self.eps, self.theta, self.kernel,
            return_potential=True,
        )

    def acc(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        return _falcON_gravity(
            pos, mass, self.eps, self.theta, self.kernel,
            return_potential=False,
        )

    def potential(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        return self.force_and_potential(pos, mass)[1]


class DirectSummationGravity(SelfGravityForce):
    """Self-gravity via direct O(N^2) pair summation.

    Parameters
    ----------
    eps : float or (N,) array
        Gravitational softening length(s) [kpc]. If an array, pairwise
        softening uses the arithmetic mean ``(eps_i + eps_j) / 2``.
    use_C : bool, optional
        Use the C extension (default ``True``). Set to ``False`` for the
        pure-Python reference implementation.
    """

    def __init__(self, eps: Union[float, np.ndarray], use_C: bool = True):
        self.eps = eps
        self.use_C = use_C

    def acc_and_potential(
        self, pos: np.ndarray, mass: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(pos, mass, self.eps, return_potential=True)

    def acc(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(pos, mass, self.eps, return_potential=False)

    def potential(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        return self.acc_and_potential(pos, mass)[1]
    
class NullSelfGravity(SelfGravityForce):
    """No-op self-gravity. Used when self-gravity is disabled."""
    def __init__(self):
        self._zeros_array_3d = None
        self._zeros_array_1d = None

    def acc(self, pos, mass):
        if self._zeros_array_3d is None:
            self._zeros_array_3d = np.zeros_like(pos)
        return self._zeros_array_3d
    def potential(self, pos, mass):
        if self._zeros_array_1d is None:
            self._zeros_array_1d = np.zeros(pos.shape[0])
        return self._zeros_array_1d
    def acc_and_potential(self, pos, mass):
        return self.acc(pos, mass), self.potential(pos, mass)
