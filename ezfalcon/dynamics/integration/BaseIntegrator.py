from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..state import State
from ..forces.BaseForce import BaseForce


@dataclass
class StepResult:
    state: State
    self_acc: Optional[np.ndarray]  # (N, 3) or None
    self_pot: Optional[np.ndarray]  # (N,)   or None
    ext_acc:  Optional[np.ndarray]  # (N, 3) or None


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
        state: State,
        self_gravity_force: Optional[BaseForce],
        external_force: Optional[BaseForce],
        dt: float,
    ) -> StepResult:
        ...

    def _c_step_handle(self):
        """Optional capsule wrapping ezfalcon_integrator_t. None -> Python only."""
        return None

    def reset(self):
        """Clear cached state (e.g. previous-step accelerations)."""
        pass
