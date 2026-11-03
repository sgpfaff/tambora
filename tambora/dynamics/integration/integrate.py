from ..forces import SelfGravityForce, Conservative, Force
from .BaseIntegrator import BaseIntegrator, StepResult
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional
from .step_state import StepState, StepContext, _Progress
from .recorder import _MemoryRecorder
import warnings

def _runner(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, 
            integrator: BaseIntegrator, 
            self_gravity_force: Optional[SelfGravityForce], 
            conserv_ext_force: Optional[Conservative],
            base_ext_force: Optional[Force], t0: float,
            t_end: float, dt: float, dt_out: float,
            return_self_gravity_pot: bool = True,
            return_self_gravity_acc: bool = True,
            slices: Optional[dict] = None,
            hooks: tuple = (),
            progress: bool = True,
            eps=None,
            recorder=None):
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
    conserv_ext_forces: Conservative
        Conservative external forces.
    base_ext_forces : Force
        Non-conservative external forces.
    t0 : float
        Start time of integration. Default is 0.0 Gyr.
        Units: `Gyr`
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
    slices : dict, optional
        Mapping of component name to slice, used to expose components to hooks
        via ``StepState``. Default is None (no named components).
    hooks : sequence of (Hook, Cadence), optional
        Hooks to run during integration, each paired with the cadence that
        decides when it fires. Default is empty (no hooks).
    recorder : _Recorder, optional
        Receives each output as it is made (``begin``, then ``record`` per output);
        the return value is its ``outputs()``. Default: a new ``_MemoryRecorder``.
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
    _check_dt_dt_out(dt, dt_out, t0, t_end)
    _warn_uneven_end(dt, dt_out, t0, t_end)

    n_steps, nsnaps, steps_per_output = _count_steps(dt, dt_out, t0, t_end)

    recorder = _MemoryRecorder() if recorder is None else recorder
    recorder.begin(nsnaps, mass.shape[0], return_self_gravity_acc, return_self_gravity_pot)
    self_acc0, self_pot0 = self_gravity_force.acc_and_potential(pos, mass, eps)
    current_pos, current_vel = pos, vel
    current_t = t0
    integrator.reset()

    # Hook plumbing: one reusable view over the live state, refreshed each fire.
    # The progress bar is created up front so hooks can report to it (via
    # state.report) from the t0 fire onwards.
    pbar = tqdm(range(1, n_steps + 1), disable=not progress)
    ctx = StepContext(slices or {}, self_gravity_force, conserv_ext_force,
                      base_ext_force, progress=_Progress(pbar), eps=eps)
    state = StepState(None, ctx)

    # Each output is recorded after the hooks due at its step, t0 included.
    try:
        # Fire once on the initial state (t0) so hooks capture the starting point.
        if hooks:
            conserv_ext_acc0 = conserv_ext_force.acc(pos, mass, t0)
            base_ext_acc0 = base_ext_force.acc(pos, vel, mass, t0)
            initial_result = StepResult(pos=pos, vel=vel, mass=mass, t=t0,
                                        self_acc=self_acc0, self_pot=self_pot0,
                                        conserv_ext_acc=conserv_ext_acc0,
                                        base_ext_acc=base_ext_acc0, step=0)
            _fire(hooks, state, initial_result, 0, steps_per_output)
        recorder.record(0, t0, pos, vel, self_acc0, self_pot0)

        for step in pbar:
            step_result = integrator.step(current_pos, current_vel, mass, current_t, dt,
                                          self_gravity_force, conserv_ext_force, base_ext_force, eps,
                                          t_new=_time_at(t0, dt, step))
            step_result.step = step
            current_pos, current_vel, current_t = step_result.pos, step_result.vel, step_result.t
            if hooks:
                _fire(hooks, state, step_result, step, steps_per_output)
            if step % steps_per_output == 0:
                recorder.record(step, step_result.t, step_result.pos, step_result.vel,
                                step_result.self_acc, step_result.self_pot)
    finally:
        pbar.close()

    return recorder.outputs()


def _fire(hooks, state, result, step, steps_per_output):
    """
    Run any hooks whose cadence is due at this step.

    Refreshes the shared ``state`` view once (only when something fires), then
    calls each due hook on it.
    """
    due = [hook for hook, cadence in hooks if cadence.due(step, steps_per_output)]
    if not due:
        return
    state._update(result)
    for hook in due:
        hook(state)


def _check_dt_dt_out(dt, dt_out, t0, t_end):
    """Raise if dt, dt_out, t0 and t_end can't make a run."""
    if (t0 > t_end) and (dt > 0):
        raise ValueError("The end time (t_end) is less than the start time (t0), implying backwards integration. " \
        "dt must be negative for backwards integration.")
    elif (t0 < t_end) and (dt < 0):
        raise ValueError("The end time (t_end) is greater than the start time (t0), implying forwards integration. " \
        "dt must be positive for forwards integration.")
    if np.abs(dt_out) < np.abs(dt):
        raise ValueError("The absolute value of dt_out must be greater than or equal to dt.")
    if np.sign(dt_out) != np.sign(dt):
        raise ValueError("dt_out must have the same sign as dt.")
    if abs(dt_out / dt - round(dt_out / dt)) > 1e-9:
        raise ValueError("dt_out must be a multiple of dt.")


def _warn_uneven_end(dt, dt_out, t0, t_end):
    """Warn if the last step, or the last output, comes before t_end."""
    n_steps, nsnaps, steps_per_output = _count_steps(dt, dt_out, t0, t_end)
    last_output_step = (nsnaps - 1) * steps_per_output
    steps_fall_short = abs((t_end - t0) / dt - n_steps) >= 1e-9
    if steps_fall_short:
        warnings.warn(f"Simulation duration ({t_end - t0} Gyr) is not an exact multiple of dt={dt} Gyr. "
                      f"The run will end at t={_time_at(t0, dt, n_steps):.10g} Gyr instead of t={t_end} Gyr.")
    if steps_fall_short or last_output_step != n_steps:
        warnings.warn(f"Simulation duration ({t_end - t0} Gyr) is not an exact multiple of dt_out={dt_out} Gyr. "
                      f"Last output will be at t={_time_at(t0, dt, last_output_step):.10g} Gyr "
                      f"instead of t={t_end} Gyr.")

def _time_at(t0, dt, step):
    """Time after *step* steps (an int or int array), from the count: summing dt drifts."""
    return t0 + step * dt


def _count_steps(dt, dt_out, t0, t_end):
    """Steps in the run, outputs it makes (t0 included), and steps per output."""
    ratio = (t_end - t0) / dt
    n_steps = round(ratio) if abs(ratio - round(ratio)) < 1e-9 else int(ratio)
    steps_per_output = round(dt_out / dt)
    return n_steps, n_steps // steps_per_output + 1, steps_per_output
