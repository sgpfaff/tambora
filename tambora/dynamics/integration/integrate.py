from ..forces import SelfGravityForce, ConservativeForce, BaseForce
from .BaseIntegrator import BaseIntegrator
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional
import warnings

def _runner(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, 
            integrator: BaseIntegrator, 
            self_gravity_force: Optional[SelfGravityForce], 
            conserv_ext_force: Optional[ConservativeForce],
            base_ext_force: Optional[BaseForce],
            t_end: float, dt: float, dt_out: float, 
            return_self_gravity_pot: bool = True, 
            return_self_gravity_acc: bool = True):
    '''
    Integrate particle trajectories optionally under 
    influence of self-gravity and external forces.

    Parameters
    ----------
    pos : ndarray
        The starting positions of particles.
    vel : ndarray
        The starting velocities of particles.
    mass : ndarray
        The mass of particles.
    integrator: BaseIntegrator
        Integrator class to use, inherited from BaseIntegrator.
    self_gravity_force : SelfGravity
        self-gravity solver class.
    conserv_ext_forces: ConservativeForce
        Conservative external forces.
    base_ext_forces : BaseForce
        Non-conservative external forces.
    t_end : float
        End time of integration.
        Units: `Gyr`
    dt : float
        Timestep for integration.
        Units: `Gyr`
    dt_out : float
        Output interval.
        Units: `Gyr`
    return_self_gravity_pot : bool, optional
        Whether to return the self-gravitational potential at each output snapshot. Default is True.
    return_self_gravity_acc : bool, optional
        Whether to return the self-gravitational acceleration at each output snapshot. Default is True.
    **kwargs
        Additional keyword arguments to pass to the self-gravity method.

    Returns
    -------
    positions : (nsnaps, N, 3) array
        Positions at each output snapshot.
        Units: `kpc`
    velocities : (nsnaps, N, 3) array
        Velocities at each output snapshot.
        Units: `kpc / Gyr`
    ts_out : (nsnaps,) array
        Times of each output snapshot.
        Units: `Gyr`
    self_gravity_acc : (nsnaps, N, 3) array or None
        Self-gravitational accelerations at each output snapshot. 
        Returns None if return_self_gravity is False.
        Units: `kpc / Gyr^2`
    self_gravity_pot : (nsnaps, N) array or None
        Self-gravitational potentials at each output snapshot.
        Returns None if return_self_potential is False.
        Units: `kpc^2 / Myr^2`
    '''
    _check_dt_dt_out(dt, dt_out, t_end)

    (ts_out, ts_integrate, 
    nsnaps, steps_per_output) = _make_time_arrays(dt, dt_out, t_end)

    positions, velocities = _make_pos_vel_arrays(pos, vel, mass, nsnaps)
    self_gravity_pot, self_gravity_acc = _make_self_gravity_arrays(pos, mass, self_gravity_force, 
                                                                   return_self_gravity_pot, return_self_gravity_acc, nsnaps)
    i_out = 1
    current_pos, current_vel = pos, vel
    current_t = 0
    integrator.reset()
    for step, t in enumerate(tqdm(ts_integrate[1:]), start=1):
        step_result = integrator.step(current_pos, current_vel, mass, current_t, dt,
                                      self_gravity_force, conserv_ext_force, base_ext_force)
        current_pos, current_vel, current_t = step_result.pos, step_result.vel, step_result.t
        if step % steps_per_output == 0 and i_out < nsnaps: # recording snapshot
            positions[i_out] = step_result.pos.copy()
            velocities[i_out] = step_result.vel.copy()
            if return_self_gravity_acc:
                self_gravity_acc[i_out] += step_result.self_acc.copy() if step_result.self_acc is not None else 0.0
            if return_self_gravity_pot:
                self_gravity_pot[i_out] += step_result.self_pot.copy() if step_result.self_pot is not None else 0.0
            i_out += 1

    return positions, velocities, ts_out, self_gravity_acc, self_gravity_pot


def _check_dt_dt_out(dt, dt_out, t_end):
    if dt <= 0 or dt_out <= 0 or t_end <= 0:
        raise ValueError("dt, dt_out, and t_end must be positive.")
    if dt_out < dt:
        raise ValueError("dt_out must be greater than or equal to dt.")
    if abs(dt_out / dt - round(dt_out / dt)) > 1e-9:
        raise ValueError("dt_out must be a multiple of dt.")
    if abs(t_end / dt - round(t_end / dt)) > 1e-9:
        actual_t_end = int(t_end / dt) * dt
        warnings.warn(f"t_end={t_end} Gyr is not an exact multiple of dt={dt} Gyr. "
                        f"The simulation will end before t_end.")
    if abs(t_end / dt_out - round(t_end / dt_out)) > 1e-9:
        n_steps_w = int(t_end / dt) if abs(t_end / dt - round(t_end / dt)) > 1e-9 else round(t_end / dt)
        steps_per_output = round(dt_out / dt)
        nsnaps = n_steps_w // steps_per_output
        actual_t_end = nsnaps * dt_out
        warnings.warn(f"t_end={t_end} Gyr is not an exact multiple of dt_out={dt_out} Gyr. "
                        f"Last output will be at t={actual_t_end:.10g} Gyr instead of t={t_end} Gyr.")

def _make_time_arrays(dt, dt_out, t_end):
    ratio_save = t_end / dt_out
    n_steps_save = round(ratio_save) if abs(ratio_save - round(ratio_save)) < 1e-9 else int(ratio_save)

    ratio_integrate = t_end / dt
    n_steps_integrate = round(ratio_integrate) if abs(ratio_integrate - round(ratio_integrate)) < 1e-9 else int(ratio_integrate)

    steps_per_output = round(dt_out / dt)
    nsnaps = n_steps_save + 1  # +1 for initial snapshot at t=0
    ts_out = np.arange(nsnaps, dtype=np.float64) * dt_out
    ts_integrate = np.arange(n_steps_integrate + 1, dtype=np.float64) * dt
    return ts_out, ts_integrate, nsnaps, steps_per_output


def _make_pos_vel_arrays(pos, vel, mass, nsnaps):
    n = mass.shape[0]
    positions = np.zeros((nsnaps, n, 3), dtype=np.float64)
    velocities = np.zeros((nsnaps, n, 3), dtype=np.float64)
    positions[0] = pos.copy()
    velocities[0] = vel.copy()
    return positions, velocities


def _make_self_gravity_arrays(pos, mass, self_gravity_force, return_self_gravity_pot,
                            return_self_gravity_acc, nsnaps):
    self_gravity_pot = None
    self_gravity_acc = None
    self_acc, self_pot = self_gravity_force.acc_and_potential(pos, mass)
    n = mass.shape[0]
    if return_self_gravity_pot:
        self_gravity_pot = np.zeros((nsnaps, n), dtype=np.float64)
        self_gravity_pot[0] += self_pot.copy() if self_pot is not None else 0.0
    if return_self_gravity_acc:
        self_gravity_acc = np.zeros((nsnaps, n, 3), dtype=np.float64)
        self_gravity_acc[0] = self_acc.copy() if self_acc is not None else 0.0
    return self_gravity_pot, self_gravity_acc