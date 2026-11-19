"""The ``solver=`` argument of the Sim's and a component's self-gravity analysis methods."""

import numpy as np
import pytest

from tambora.dynamics import DirectSum, FalcON
from tambora.simulation import Sim

SIM_METHODS = ['self_gravity', 'self_ax', 'self_ay', 'self_az', 'self_potential', 'PE', 'energy',
               'system_energy', 'virial_ratio']
COMPONENT_METHODS = [name for name in SIM_METHODS if name != 'system_energy']
EPS_A, EPS_B = 0.05, 0.1


def _sim(eps=True, **kwargs):
    rng = np.random.default_rng(0)
    sim = Sim(**kwargs)
    sim.add_particles(rng.normal(size=(30, 3)), rng.normal(size=(30, 3)) * 10, np.full(30, 1e5), name='a',
                      eps=EPS_A if eps else None)
    sim.add_particles(rng.normal(size=(20, 3)) + 3, rng.normal(size=(20, 3)) * 10, np.full(20, 2e5), name='b',
                      eps=EPS_B if eps else None)
    return sim


def _ran(**kwargs):
    sim = _sim(**kwargs)
    sim.run(t_end=0.03, dt=0.01, dt_out=0.01, progress=False)
    return sim


EPS = np.concatenate([np.full(30, EPS_A), np.full(20, EPS_B)])


@pytest.mark.parametrize("name", SIM_METHODS)
def test_a_solver_gives_what_its_method_name_and_keywords_gave(name):
    sim = _sim()
    np.testing.assert_array_equal(getattr(sim, name)(t=0, solver=DirectSum(use_C=False)),
                                  getattr(sim, name)(t=0, method='direct'))
    np.testing.assert_array_equal(getattr(sim, name)(t=0, solver=FalcON(theta=0.4)),
                                  getattr(sim, name)(t=0, method='falcON', theta=0.4))


@pytest.mark.parametrize("name", COMPONENT_METHODS)
@pytest.mark.parametrize("include_all_components", [True, False])
def test_a_solver_gives_a_component_what_its_method_name_gave(name, include_all_components):
    sim = _sim()
    np.testing.assert_array_equal(
        getattr(sim.b, name)(t=0, solver=DirectSum(use_C=False), include_all_components=include_all_components),
        getattr(sim.b, name)(t=0, method='direct', include_all_components=include_all_components))


def test_a_solver_computes_with_the_particles_softening():
    sim, solver = _sim(), DirectSum()
    pos = sim.pos(t=0, return_internal=True)
    acc, phi = solver.acc_and_potential(pos, sim.mass, EPS)
    np.testing.assert_array_equal(sim.self_gravity(t=0, solver=solver, return_internal=True), acc)
    np.testing.assert_array_equal(sim.self_potential(t=0, solver=solver, return_internal=True), sim.mass * phi)
    alone = solver.acc(pos[30:], sim.mass[30:], EPS_B)
    np.testing.assert_array_equal(
        sim.b.self_gravity(t=0, solver=solver, include_all_components=False, return_internal=True), alone)
    np.testing.assert_array_equal(
        sim.b.self_gravity(t=0, solver=solver, include_all_components=True, return_internal=True), acc[30:])


def test_a_solvers_own_softening_replaces_the_particles_and_they_need_none():
    solver = DirectSum(eps=0.3)
    for sim in (_sim(), _sim(eps=False)):
        expected = solver.acc(sim.pos(t=0, return_internal=True), sim.mass)
        np.testing.assert_array_equal(sim.self_gravity(t=0, solver=solver, return_internal=True), expected)
    with pytest.raises(ValueError, match="Softening isn't set for every particle here"):
        _sim(eps=False).self_gravity(t=0, solver=DirectSum())


def test_after_a_run_a_solver_computes_afresh_and_saves_nothing():
    sim, solver = _ran(), DirectSum()
    saved = sim._cached_self_acc.copy()
    fresh = sim.self_gravity(t=2, solver=solver, return_internal=True)
    np.testing.assert_array_equal(fresh, solver.acc(sim.pos(t=2, return_internal=True), sim.mass, EPS))
    assert not np.array_equal(fresh, saved[2])          # the run's falcON values
    np.testing.assert_array_equal(sim._cached_self_acc, saved)
    np.testing.assert_array_equal(sim.self_gravity(t=2, return_internal=True), saved[2])


def test_a_solver_computes_one_snapshot_at_a_time_except_for_dE():
    sim, solver = _ran(), DirectSum()
    with pytest.raises(TypeError, match="Cannot compute on-the-fly for all times"):
        sim.self_gravity(solver=solver)
    energies = np.array([sim.system_energy(t=i, solver=solver, return_internal=True) for i in range(4)])
    np.testing.assert_array_equal(sim.dE(solver=solver), np.abs((energies - energies[0]) / energies[0]))


@pytest.mark.parametrize("kwargs, error, message", [
    (dict(solver=DirectSum(), method='direct'), ValueError, "Give solver or method, not both."),
    (dict(solver=DirectSum(), use_cached=True), ValueError,
     "A solver computes afresh, so use_cached can't be True with solver=."),
    (dict(solver=DirectSum(), eps=0.1), TypeError, "eps can't be given with solver=. Give the solver what it needs."),
    (dict(solver=FalcON(), theta=0.3, kernel=0), TypeError, "kernel, theta can't be given with solver="),
    (dict(solver='direct'), TypeError, r"solver must be a self-gravity solver, e.g. DirectSum\(\). Got 'str'."),
    (dict(solver=DirectSum), TypeError, "Got 'ABCMeta'"),
], ids=['with method', 'with use_cached', 'with eps', 'with the method keywords', 'a method name', 'a solver class'])
@pytest.mark.parametrize("of", ['the Sim', 'a component'])
@pytest.mark.parametrize("name", ['self_gravity', 'self_potential', 'energy'])
def test_a_solver_is_refused(kwargs, error, message, of, name):
    sim = _ran()
    view = sim if of == 'the Sim' else sim.b
    with pytest.raises(error, match=message):
        getattr(view, name)(t=1, **kwargs)


def test_a_loaded_sim_takes_a_solver(tmp_path):
    ran, solver = _ran(output_dir=tmp_path / 'run'), DirectSum()
    sim = Sim.load(tmp_path / 'run')
    np.testing.assert_array_equal(sim.self_gravity(t=1, solver=solver), ran.self_gravity(t=1, solver=solver))
    np.testing.assert_array_equal(sim.b.energy(t=1, solver=solver, include_all_components=False),
                                  ran.b.energy(t=1, solver=solver, include_all_components=False))
