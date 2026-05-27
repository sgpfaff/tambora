from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from ..forces.BaseForce import BaseForce
from ..forces.ConservativeForce import ConservativeForce
from ..forces.self_gravity.SelfGravity import SelfGravityForce
import numpy as np

@dataclass
class StepResult:
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    t: float
    self_acc: np.ndarray  # (N, 3) or None
    self_pot: np.ndarray  # (N,)   or None
    conserv_ext_acc:  np.ndarray  # (N, 3) or None
    base_ext_acc: np.ndarray # (N, 3) or None


class BaseIntegrator(ABC):
    """Abstract base for time integrators.

    A concrete integrator advances a State by one timestep given a
    self-gravity Force and an external Force (either may be None). It
    returns a StepResult containing the new state and per-component
    accelerations. The driver loop in _integrate() owns timing and output
    cadence — the integrator only does math.
    """

    @abstractmethod
    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        t: float,
        dt: float,
        self_gravity_force: SelfGravityForce,
        conserv_ext_force: ConservativeForce,
        base_external_force: BaseForce
    ) -> StepResult:
        ...
    
    def _eval_conservative(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        t,
        self_gravity_force: SelfGravityForce,
        cons_ext_force: ConservativeForce,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate self-gravity and conservative external forces.

        Returns ``(sg_acc, sg_pot, ext_acc)``. Self-gravity uses the one-pass
        acc+potential entry point so falcON / direct summation don't sweep twice.
        """
        sg_acc, sg_pot = self_gravity_force.acc_and_potential(pos, mass)
        ext_acc = cons_ext_force.acc(pos, mass, t)
        return sg_acc, sg_pot, ext_acc
    
    def reset(self):
        """Clear cached state (e.g. previous-step accelerations)."""
        pass