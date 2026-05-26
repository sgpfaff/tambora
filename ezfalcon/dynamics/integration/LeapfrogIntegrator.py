from typing import Optional
from ..forces.BaseForce import BaseForce
from ..forces.ConservativeForce import ConservativeForce
from ..forces.self_gravity import SelfGravityForce
from .BaseIntegrator import BaseIntegrator, StepResult
import dataclasses
import numpy as np

class LeapfrogIntegrator(BaseIntegrator):
    def __init__(self):
        self._last_conserv_acc = None
        self._last_base_acc = None

    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        t: float,
        dt: float,
        self_gravity_force: SelfGravityForce,
        conserv_ext_force: ConservativeForce,
        base_ext_force: BaseForce
    ) -> StepResult:
        
        # --- base force half-kick -------------------------------------------------------
        a_base_0 = self._initial_base_acc(pos, vel, mass, 0.0, base_ext_force)
        vel = vel + a_base_0 * dt / 2

        # --- conservative force half-kick -----------------------------------------------
        a_cons_0 = self._initial_conservative_acc(pos, mass, 0.0, self_gravity_force, conserv_ext_force)
        vel_half = vel + a_cons_0 * dt / 2

        # --- full step drift ------------------------------------------------------------
        pos_new = pos + vel_half * dt
        t_new = t + dt

        # --- conservative force half-kick -----------------------------------------------
        sg_acc, sg_pot, conserv_ext_acc = self._eval_conservative(pos_new, mass, t_new,
            self_gravity_force, conserv_ext_force
        )
        a_cons_1 = sg_acc + conserv_ext_acc
        self._last_conserv_acc = a_cons_1
        vel = vel_half + a_cons_1 * dt / 2

        # --- base force half-kick ------------------------------------------------------
        base_acc_new = base_ext_force.acc(pos_new, vel, mass, t_new)
        self._last_base_acc = base_acc_new
        vel_new = vel + base_acc_new * dt / 2
        
        return StepResult(pos_new, vel_new, mass, t_new, 
                          sg_acc, sg_pot, conserv_ext_acc, base_acc_new)

    def _initial_conservative_acc(self, pos, mass, t, sg_force, ext_force):
        """Acc for the first half-kick of a step. Lazily computed and cached."""
        if self._last_conserv_acc is None:
            sg_acc, _, ext_acc = self._eval_conservative(pos, mass, t, sg_force, ext_force)
            self._last_conserv_acc = sg_acc + ext_acc
        return self._last_conserv_acc

    def _initial_base_acc(self, pos, vel, mass, t, base_force):
        if self._last_base_acc is None:
            self._last_base_acc = base_force.acc(pos, vel, mass, t)
        return self._last_base_acc

    def reset(self):
        self._last_conserv_acc = None
        self._last_base_acc = None
