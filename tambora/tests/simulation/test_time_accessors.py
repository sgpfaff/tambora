"""Picking a snapshot on the Sim's and a component's accessors."""

import re
import warnings

import numpy as np
import pytest

from tambora.dynamics import DirectSum, ExternalConservativeForce
from tambora.simulation import Sim

VIEW_METHODS = ['pos', 'vel', 'x', 'y', 'z', 'r', 'phi', 'theta', 'cylR', 'vx', 'vy', 'vz', 'vr', 'vphi',
                'vtheta', 'cylvR', 'p', 'px', 'py', 'pz', 'L', 'Lx', 'Ly', 'Lz', 'com_pos', 'com_vel',
                'compute_external_pot', 'KE', 'external_acc', 'external_ax', 'external_ay', 'external_az',
                'self_potential', 'PE', 'energy', 'virial_ratio', 'self_gravity', 'self_ax', 'self_ay',
                'self_az']
SIM_METHODS = VIEW_METHODS + ['system_energy', 'dE']


class _GrowingField(ExternalConservativeForce):
    """A harmonic field that grows with time, so each snapshot's value differs."""
    def acc(self, pos, t):
        return -(1 + 100 * t) * pos

    def potential(self, pos, t):
        return 0.5 * (1 + 100 * t) * np.sum(pos**2, axis=-1)


def _ran():
    rng = np.random.default_rng(1)
    sim = Sim(solver=DirectSum())
    sim.add_particles(rng.normal(size=(12, 3)), rng.normal(size=(12, 3)) * 10, np.full(12, 1e5), name='a', eps=0.05)
    sim.add_particles(rng.normal(size=(8, 3)) + 3, rng.normal(size=(8, 3)) * 10, np.full(8, 2e5), name='b', eps=0.1)
    sim.add_force(_GrowingField())
    sim.run(t_end=0.04, dt=0.01, dt_out=0.01, progress=False, monitors=False)
    return sim


SIM = _ran()


def _call(view, name, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the external potential warns when computed for every snapshot
        return getattr(view, name)(**kwargs)


@pytest.mark.parametrize("index", [0, 2, -1])
@pytest.mark.parametrize("name", SIM_METHODS)
def test_index_picks_that_snapshot_of_every_one(name, index):
    np.testing.assert_array_equal(_call(SIM, name, index=index), _call(SIM, name)[index])


@pytest.mark.parametrize("index", [0, 2, -1])
@pytest.mark.parametrize("name", VIEW_METHODS)
def test_index_on_a_component(name, index):
    np.testing.assert_array_equal(_call(SIM.b, name, index=index), _call(SIM.b, name)[index])


def test_index_with_a_solver_computes_that_snapshot():
    acc = SIM.self_gravity(index=-1, solver=DirectSum(), return_internal=True)
    np.testing.assert_allclose(acc, SIM.self_gravity(index=-1, return_internal=True), rtol=1e-10)
    assert not np.allclose(acc, SIM.self_gravity(index=0, return_internal=True), rtol=1e-6)


def test_index_takes_a_numpy_int():
    np.testing.assert_array_equal(SIM.pos(index=np.int64(2)), SIM.pos()[2])


def test_t_and_index_together_are_refused():
    with pytest.raises(TypeError, match="Give t or index, not both."):
        SIM.pos(t=0.02, index=2)


def test_index_must_be_an_int():
    with pytest.raises(TypeError, match="index must be an int. Got float."):
        SIM.pos(index=2.0)


@pytest.mark.parametrize("index", [5, -6])
def test_index_out_of_range_is_refused(index):
    with pytest.raises(IndexError):
        SIM.pos(index=index)


def test_nearest_t_gives_the_nearest_output_time():
    assert SIM.nearest_t(0.013) == SIM.times[1]
    assert SIM.nearest_t(0.017) == SIM.times[2]
    assert type(SIM.nearest_t(0.017)) is float


def test_nearest_t_of_an_output_time_is_that_time():
    for t in SIM.times:
        assert SIM.nearest_t(t) == t


def test_nearest_t_takes_ints_and_numpy_numbers():
    assert SIM.nearest_t(0) == SIM.times[0]
    assert SIM.nearest_t(np.int64(0)) == SIM.times[0]
    assert SIM.nearest_t(np.float32(0.021)) == SIM.times[2]


def test_nearest_t_allows_round_off_at_the_ends():
    assert SIM.nearest_t(0.04 + 1e-15) == SIM.times[-1]
    assert SIM.nearest_t(-1e-15) == SIM.times[0]


@pytest.mark.parametrize("t", [0.05, -0.01])
def test_nearest_t_outside_the_run_is_refused(t):
    with pytest.raises(ValueError, match=re.escape(f"t={t} Gyr is outside the run, which covers 0 to 0.04 Gyr.")):
        SIM.nearest_t(t)


@pytest.mark.parametrize("t", [[0.01, 0.02], np.array([0.01, 0.02]), '0.01', True])
def test_nearest_t_takes_one_number(t):
    with pytest.raises(TypeError, match="t must be a number. Got"):
        SIM.nearest_t(t)


def test_nearest_t_on_a_run_backwards_from_t0():
    rng = np.random.default_rng(2)
    sim = Sim(solver=DirectSum(), t0=1.0)
    sim.add_particles(rng.normal(size=(4, 3)), rng.normal(size=(4, 3)), np.full(4, 1e5), eps=0.05)
    sim.run(t_end=0.97, dt=-0.01, dt_out=-0.01, progress=False, monitors=False)
    assert sim.nearest_t(0.982) == sim.times[2]
    with pytest.raises(ValueError, match="covers 0.97 to 1 Gyr"):
        sim.nearest_t(1.01)
