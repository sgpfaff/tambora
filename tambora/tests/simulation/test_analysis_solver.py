"""Which solver the Sim's and a component's self-gravity analysis methods compute with."""

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


def _ran(cache=True, **kwargs):
    sim = _sim(**kwargs)
    sim.run(t_end=0.03, dt=0.01, dt_out=0.01, progress=False, cache_self_gravity_acc=cache, cache_self_gravity_pot=cache)
    return sim


EPS = np.concatenate([np.full(30, EPS_A), np.full(20, EPS_B)])


@pytest.mark.parametrize("name", SIM_METHODS)
@pytest.mark.parametrize("solver", [DirectSum(), FalcON(theta=0.4)], ids=['direct', 'falcON'])
def test_without_a_solver_the_sims_own_is_used(name, solver):
    """Before a run nothing is saved, so the Sim's solver computes."""
    own, other = _sim(solver=solver), _sim(solver=None)
    np.testing.assert_array_equal(getattr(own, name)(t=0), getattr(other, name)(t=0, solver=solver))
    assert np.any(getattr(own, name)(t=0) != getattr(other, name)(t=0))     # the other Sim has no self-gravity


@pytest.mark.parametrize("name", COMPONENT_METHODS)
@pytest.mark.parametrize("include_all_components", [True, False])
def test_without_a_solver_a_component_uses_the_sims_own(name, include_all_components):
    own, other = _sim(solver=DirectSum()), _sim(solver=None)
    np.testing.assert_array_equal(
        getattr(own.b, name)(t=0, include_all_components=include_all_components),
        getattr(other.b, name)(t=0, include_all_components=include_all_components, solver=DirectSum()))


@pytest.mark.parametrize("name", SIM_METHODS)
def test_after_a_run_the_saved_values_are_used(name):
    """Only saved values can give every snapshot at once."""
    sim = _ran()
    every = getattr(sim, name)()
    assert np.shape(every) == (4,) + np.shape(getattr(sim, name)(t=2))
    np.testing.assert_array_equal(every[2], getattr(sim, name)(t=2))
    np.testing.assert_allclose(every[2], getattr(sim, name)(t=2, solver=FalcON()), rtol=1e-6)


def test_dE_at_one_time_is_measured_from_the_first_snapshot():
    sim = _ran()
    assert sim.dE(t=0) == 0 and sim.dE(t=2) == sim.dE()[2] > 0
    energies = [sim.system_energy(t=i, solver=DirectSum(), return_internal=True) for i in (0, 2)]
    assert sim.dE(t=2, solver=DirectSum()) == abs((energies[1] - energies[0]) / energies[0])


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


@pytest.mark.parametrize("kwargs, message", [
    (dict(solver='direct'), r"solver must be a self-gravity solver, e.g. DirectSum\(\). Got 'str'."),
    (dict(solver=DirectSum), "Got 'ABCMeta'"),
    (dict(method='direct'), "unexpected keyword argument 'method'"),
    (dict(use_cached=True), "unexpected keyword argument 'use_cached'"),
    (dict(solver=DirectSum(), eps=0.1), "unexpected keyword argument 'eps'"),
    (dict(theta=0.3), "unexpected keyword argument 'theta'"),
], ids=['a method name', 'a solver class', 'method', 'use_cached', 'eps', 'a solver setting'])
@pytest.mark.parametrize("of", ['the Sim', 'a component'])
@pytest.mark.parametrize("name", ['self_gravity', 'self_potential', 'energy'])
def test_what_is_not_a_solver_is_refused_and_so_are_the_old_arguments(kwargs, message, of, name):
    sim = _ran()
    view = sim if of == 'the Sim' else sim.b
    with pytest.raises(TypeError, match=message):
        getattr(view, name)(t=1, **kwargs)


def test_a_sim_whose_solver_couldnt_be_loaded_says_so_and_takes_one(tmp_path, monkeypatch):
    import tambora.simulation.simulation as simulation_module
    _ran(output_dir=tmp_path / 'run', cache=False)
    read = simulation_module.read_run

    def unknown_solver(directory):
        run = read(directory)
        run['settings']['run']['self_gravity']['class'] = 'NoSuchSolver'
        return run
    monkeypatch.setattr(simulation_module, 'read_run', unknown_solver)
    with pytest.warns(UserWarning, match="self-gravity solver couldn't be loaded"):
        sim = Sim.load(tmp_path / 'run')
    for view in (sim, sim.b):
        with pytest.raises(RuntimeError, match="The run's self-gravity solver couldn't be loaded. Pass solver="):
            view.self_gravity(t=1)
        assert view.self_gravity(t=1, solver=DirectSum()).shape[-1] == 3


def test_a_loaded_sim_computes_with_the_runs_solver_or_a_given_one(tmp_path):
    ran, solver = _ran(output_dir=tmp_path / 'run', solver=DirectSum(use_C=False), cache=False), DirectSum()
    sim = Sim.load(tmp_path / 'run')
    assert sim._self_gravity_force is not ran._self_gravity_force
    np.testing.assert_array_equal(sim.self_gravity(t=1), ran.self_gravity(t=1))
    np.testing.assert_array_equal(sim.self_gravity(t=1), ran.self_gravity(t=1, solver=DirectSum(use_C=False)))
    np.testing.assert_array_equal(sim.self_gravity(t=1, solver=solver), ran.self_gravity(t=1, solver=solver))
    np.testing.assert_array_equal(sim.b.energy(t=1, solver=solver, include_all_components=False),
                                  ran.b.energy(t=1, solver=solver, include_all_components=False))
