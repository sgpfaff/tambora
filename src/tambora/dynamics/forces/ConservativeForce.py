from abc import abstractmethod
from typing import Tuple
import numpy as np

from .BaseForce import BaseForce


class ConservativeForce(BaseForce):
    """Base class for forces that are the gradient of a potential.

    Concrete subclasses must implement both :meth:`force` and
    :meth:`potential`. Solvers that compute both arrays in a single shared
    pass (falcON, direct summation) should additionally override
    :meth:`force_and_potential` to skip the second sweep.
    """

    @abstractmethod
    def acc(self, pos: np.ndarray, mass: np.ndarray, t) -> np.ndarray: ...
    
    @abstractmethod
    def potential(self, pos: np.ndarray, mass: np.ndarray, t) -> np.ndarray: ...
    
    def acc_and_potential(self, pos: np.ndarray, mass: np.ndarray, t):
        """Default: call ``force`` and ``potential`` separately.

        Override for solvers that produce both arrays in one pass.
        """
        return self.acc(pos, mass, t), self.potential(pos, mass, t)
    
    def _eval_acc(self, pos, vel, mass, t):
        return self.acc(pos, mass, t)
    
    def _eval_potential(self, pos, vel, mass, t):
        return self.potential(pos, mass, t)
    
    def _eval_acc_and_potential(self, pos, vel, mass, t):
        return self.acc_and_potential(pos, mass, t)
