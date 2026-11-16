"""A run that stops early keeps the outputs it recorded up to then."""

import warnings

import numpy as np
import pytest

from tambora.simulation import Sim
from tambora.dynamics.hooks import EveryStep
from tambora.dynamics.forces.external_force import ExternalConservativeForce
import tambora.dynamics.integration.integrate as integrate

RUN = dict(t_end=1.0, dt=0.01, dt_out=0.1, method='direct', progress=False, monitors=False)


class _InterruptAfter(ExternalConservativeForce):
    """A zero field that raises KeyboardInterrupt once evaluated after *t_stop*."""
    def __init__(self, t_stop):
        self.t_stop = t_stop

    def acc(self, pos, t):
        if t > self.t_stop:
            raise KeyboardInterrupt
        return np.zeros_like(pos)

    def potential(self, pos, t):
        return np.zeros(len(pos))


def _sim(t_stop=np.inf):
    rng = np.random.default_rng(0)
    sim = Sim()
    sim.add_particles(rng.normal(size=(50, 3)), rng.normal(size=(50, 3)) * 10, np.full(50, 1e5),
                      name='c', eps=0.05)
    sim.add_external_force(_InterruptAfter(t_stop))
    return sim


def _raise_at(step, exc, once=False):
    fired = []

    def hook(state):
        if state.step == step and not (once and fired):
            fired.append(step)
            raise exc
    return hook


@pytest.fixture(scope='module')
def full():
    sim = _sim()
    sim.run(**RUN)
    return sim


def _assert_first_outputs_match(sim, full, n):
    assert len(sim.times) == n
    np.testing.assert_array_equal(sim.times, full.times[:n])
    for name in ('_positions', '_velocities', '_cached_self_acc', '_cached_self_pot'):
        np.testing.assert_array_equal(getattr(sim, name), getattr(full, name)[:n])


@pytest.mark.parametrize("step, n_kept", [(35, 4), (40, 4)])     # at an output step, its hooks fire first
def test_an_interrupted_run_keeps_the_outputs_recorded_so_far(full, step, n_kept):
    sim = _sim()
    sim.add_hook(_raise_at(step, KeyboardInterrupt()), EveryStep())
    with pytest.warns(UserWarning, match=f"stopped early: the {n_kept} outputs up to t = 0.3 Gyr"):
        with pytest.raises(KeyboardInterrupt) as err:
            sim.run(**RUN)
    assert err.value.__context__ is None
    _assert_first_outputs_match(sim, full, n_kept)
    np.testing.assert_array_equal(sim.c.pos(t=0.3), full.c.pos(t=0.3))


def test_an_error_in_a_hook_keeps_the_outputs_and_is_raised_again(full):
    sim = _sim()
    sim.add_hook(_raise_at(55, ValueError("boom")), EveryStep())
    with pytest.warns(UserWarning, match="the 6 outputs up to t = 0.5 Gyr"):
        with pytest.raises(ValueError, match="boom"):
            sim.run(**RUN)
    _assert_first_outputs_match(sim, full, 6)


def test_an_interrupt_from_inside_a_force_keeps_the_outputs(full):
    sim = _sim(t_stop=0.45)
    with pytest.warns(UserWarning, match="the 5 outputs up to t = 0.4 Gyr"):
        with pytest.raises(KeyboardInterrupt):
            sim.run(**RUN)
    _assert_first_outputs_match(sim, full, 5)


def test_a_run_that_stops_before_its_first_output_keeps_nothing():
    sim = _sim()
    sim.add_hook(_raise_at(0, ValueError("at t0")), EveryStep())
    with warnings.catch_warnings():
        warnings.simplefilter('error')                 # no "stopped early" warning either
        with pytest.raises(ValueError, match="at t0"):
            sim.run(**RUN)
    assert not sim._has_run
    np.testing.assert_array_equal(sim.times, [0.0])


def test_a_rerun_after_an_interrupt_runs_normally(full):
    sim = _sim()
    sim.add_hook(_raise_at(35, KeyboardInterrupt(), once=True), EveryStep())
    with pytest.warns(UserWarning), pytest.raises(KeyboardInterrupt):
        sim.run(**RUN)
    sim.run(**RUN)
    _assert_first_outputs_match(sim, full, len(full.times))


def test_warnings_made_errors_dont_replace_the_interrupt():
    sim = _sim()
    sim.add_hook(_raise_at(35, KeyboardInterrupt()), EveryStep())
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(KeyboardInterrupt):
            sim.run(**RUN)
    assert len(sim.times) == 4


def test_the_progress_bar_is_closed_when_a_run_stops_early(monkeypatch):
    bars = []

    class FakeBar:
        def __init__(self, iterable, disable=False):
            self.iterable, self.closed = iterable, False
            bars.append(self)

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(integrate, 'tqdm', FakeBar)
    sim = _sim()
    sim.add_hook(_raise_at(35, ValueError("boom")), EveryStep())
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        sim.run(**RUN)
    assert bars and bars[-1].closed
