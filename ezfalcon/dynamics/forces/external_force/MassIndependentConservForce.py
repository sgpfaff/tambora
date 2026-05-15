from ..ConservativeForce import ConservativeForce
from abc import abstractmethod
from typing import Tuple
import numpy as np

class MassIndependentConservForce(ConservativeForce):
    @abstractmethod
    def acc(self, pos: np.ndarray, t) -> np.ndarray: ...
    @abstractmethod
    def potential(self, pos: np.ndarray, t) -> np.ndarray: ...
    
    def acc_and_potential(self, pos: np.ndarray, t) -> Tuple[np.ndarray, np.ndarray]:
        return self.acc(pos, t), self.potential(pos, t)
    def _eval_acc(self, pos, vel, mass, t):
        return self.acc(pos, t)
    def _eval_potential(self, pos, vel, mass, t):
        return self.potential(pos, t)
    def _eval_acc_and_potential(self, pos, vel, mass, t):
        return self.acc_and_potential(pos, t)
