"""
Tests for acc and ax/ay/az: the acceleration on a Sim's or a component's particles from
chosen sources (particles, external forces), from the run's values or a given solver.

Expected values come from direct summation over the sources with non-source targets
massless, which is exact, and from the external forces evaluated by hand.
"""

import numpy as np
import pytest

from tambora.simulation import Sim, _view
from tambora.dynamics import DirectSummationGravity, ExternalConservativeForce, FalcONGravity, Force
from tambora.units import KPCGYR_TO_KMS

N_A, N_B = 30, 20
EPS_B = np.linspace(0.05, 0.1, N_B)


class _Harmonic(ExternalConservativeForce):
    def acc(self, pos, t):
        return -(1 + t) * pos

    def potential(self, pos, t):
        return 0.5 * (1 + t) * np.sum(pos**2, axis=-1)


class _Drag(Force):
    def acc(self, pos, vel, mass, t):
        return -0.1 * vel


def _sim(method='direct', run=True, cache=True, eps_b=EPS_B):
    rng = np.random.default_rng(1)
    sim = Sim()
    sim.add_particles(rng.normal(size=(N_A, 3)), rng.normal(size=(N_A, 3)) * 10, np.full(N_A, 1e7),
                      name='a', eps=0.1)
    sim.add_particles(rng.normal(size=(N_B, 3)) + 3, rng.normal(size=(N_B, 3)) * 10, np.full(N_B, 1e6),
                      name='b', eps=eps_b)
    sim.add_external_force(_Harmonic())
    sim.add_external_force(_Drag())
    if run:
        sim.run(t_end=0.02, dt=0.01, dt_out=0.01, method=method, progress=False, monitors=False,
                cache_self_gravity_acc=cache)
    return sim


def _mask(sim, *views):
    m = np.zeros(len(sim.mass), dtype=bool)
    for v in views:
        m[v._sl] = True
    return m


def _gravity(sim, i, target, *sources, eps=None):
    """Direct summation of the sources' gravity on *target* at snapshot i [internal units]."""
    src, tgt = _mask(sim, *sources), _mask(sim, target)
    solved = src | tgt
    mass = np.where(src, sim.mass, 0.0)[solved]
    eps = np.broadcast_to(sim.eps if eps is None else eps, sim.mass.shape)[solved]
    acc = DirectSummationGravity().acc(sim._positions[i][solved], mass, eps=eps)
    return acc[tgt[solved]]


def _external(sim, i, target):
    t, pos, vel = sim.times[i], sim._positions[i][target._sl], sim._velocities[i][target._sl]
    return -(1 + t) * pos - 0.1 * vel


def _views(sim):
    return {'sim': sim, 'a': sim.a, 'b': sim.b}


@pytest.mark.parametrize("view", ['sim', 'a', 'b'])
def test_by_default_the_acceleration_is_everything_the_run_integrated(view):
    sim = _sim()
    v = _views(sim)[view]
    expected = [_gravity(sim, i, v, sim.particles) + _external(sim, i, v) for i in range(3)]
    for i in range(3):
        np.testing.assert_allclose(v.acc(t=i, return_internal=True), expected[i], rtol=1e-12)
    np.testing.assert_allclose(v.acc(return_internal=True), np.stack(expected), rtol=1e-12)


def test_each_kind_of_source_alone():
    sim = _sim()
    b = sim.b
    get = lambda sources: b.acc(t=1, sources=sources, return_internal=True)
    np.testing.assert_allclose(get(sim.particles), _gravity(sim, 1, b, sim.particles), rtol=1e-12)
    np.testing.assert_allclose(get(sim.external), _external(sim, 1, b), rtol=1e-12)
    np.testing.assert_allclose(get(sim.a), _gravity(sim, 1, b, sim.a), rtol=1e-12)
    np.testing.assert_allclose(get(sim.b), _gravity(sim, 1, b, sim.b), rtol=1e-12)


@pytest.mark.parametrize("view", ['sim', 'a', 'b'])
def test_the_sources_add_up_to_everything(view):
    sim = _sim()
    v = _views(sim)[view]
    parts = sum(v.acc(sources=s, return_internal=True) for s in (sim.a, sim.b, sim.external))
    np.testing.assert_allclose(parts, v.acc(return_internal=True), rtol=1e-11)


def test_a_list_of_sources_counts_each_particle_once():
    sim = _sim()
    b, solver = sim.b, DirectSummationGravity()
    get = lambda sources, **kw: b.acc(t=1, sources=sources, return_internal=True, **kw)
    np.testing.assert_array_equal(get([sim.a, sim.a]), get(sim.a))
    np.testing.assert_array_equal(get([sim.a, sim.b], solver=solver), get(sim.particles, solver=solver))
    np.testing.assert_array_equal(get((sim.particles, sim.external)), get(None))


def test_falcON_feels_the_sources_on_particles_that_arent_sources():
    sim = _sim()
    acc = sim.b.acc(t=1, sources=sim.a, solver=FalcONGravity(), return_internal=True)
    exact = _gravity(sim, 1, sim.b, sim.a)
    assert np.isfinite(acc).all()
    assert np.median(np.abs(acc - exact) / np.abs(exact)) < 0.05


def test_a_given_solver_and_its_own_softening_replace_the_runs():
    sim = _sim()
    get = lambda sources, solver: sim.b.acc(t=1, sources=sources, solver=solver, return_internal=True)
    np.testing.assert_allclose(get(sim.particles, DirectSummationGravity(eps=0.3)),
                               _gravity(sim, 1, sim.b, sim.particles, eps=0.3), rtol=1e-12)
    eps = np.linspace(0.2, 0.4, N_A + N_B)
    np.testing.assert_allclose(get(sim.a, DirectSummationGravity(eps=eps)),
                               _gravity(sim, 1, sim.b, sim.a, eps=eps), rtol=1e-12)
    np.testing.assert_allclose(get(sim.b, DirectSummationGravity(eps=eps)),
                               _gravity(sim, 1, sim.b, sim.b, eps=eps), rtol=1e-12)
    np.testing.assert_allclose(get(sim.particles, DirectSummationGravity()),
                               _gravity(sim, 1, sim.b, sim.particles), rtol=1e-12)
    with pytest.raises(ValueError, match=r"needs one per particle \(50\)"):
        get(sim.a, DirectSummationGravity(eps=eps[:N_B]))


def test_without_the_runs_values_the_runs_solver_computes_them():
    sim = _sim(cache=False)
    for i in range(3):
        np.testing.assert_allclose(sim.acc(t=i, sources=sim.particles, return_internal=True),
                                   _gravity(sim, i, sim, sim.particles), rtol=1e-12)


def test_recomputing_the_run_gives_its_values_bit_for_bit():
    # Above ~50 particles, falcON's results differ in the last bits between one number and an
    # array of equal values.
    def run(cache):
        rng = np.random.default_rng(2)
        sim = Sim()
        sim.add_particles(rng.normal(size=(200, 3)), np.zeros((200, 3)), np.full(200, 1e7), eps=0.1)
        sim.run(t_end=0.02, dt=0.01, dt_out=0.01, method='falcON', progress=False, monitors=False,
                cache_self_gravity_acc=cache)
        return sim
    cached, recomputed = run(True), run(False)
    np.testing.assert_array_equal(recomputed.acc(sources=recomputed.particles),
                                  cached.acc(sources=cached.particles))


def test_before_a_run_a_solver_is_needed_for_the_particles():
    sim = _sim(run=False)
    with pytest.raises(ValueError, match="No run yet"):
        sim.acc(t=0)
    np.testing.assert_allclose(sim.b.acc(t=0, sources=sim.external, return_internal=True),
                               _external(sim, 0, sim.b), rtol=1e-12)
    np.testing.assert_allclose(sim.b.acc(t=0, sources=sim.a, solver=DirectSummationGravity(),
                                         return_internal=True),
                               _gravity(sim, 0, sim.b, sim.a), rtol=1e-12)


def test_a_run_without_self_gravity_gives_none():
    sim = _sim(method=None, eps_b=None)              # b's softening isn't needed
    np.testing.assert_array_equal(sim.b.acc(sources=sim.b), np.zeros((3, N_B, 3)))
    np.testing.assert_allclose(sim.b.acc(t=2, return_internal=True), _external(sim, 2, sim.b), rtol=1e-12)


def test_every_snapshot_at_once_matches_each_in_turn():
    sim = _sim()
    every = sim.b.acc(sources=sim.a, return_internal=True)
    assert every.shape == (3, N_B, 3)
    for i in range(3):
        np.testing.assert_array_equal(every[i], sim.b.acc(t=i, sources=sim.a, return_internal=True))


def test_only_a_slow_loop_over_snapshots_shows_its_progress(monkeypatch, capsys):
    sim = _sim()
    sim.b.acc(sources=sim.a)
    assert capsys.readouterr().err == ''
    monkeypatch.setattr(_view, '_PROGRESS_DELAY', 0)
    sim.b.acc(sources=sim.a)
    assert 'snapshots' in capsys.readouterr().err


@pytest.mark.parametrize("col, name", enumerate(['ax', 'ay', 'az']))
def test_the_components_are_columns_of_acc_in_the_same_units(col, name):
    sim = _sim()
    acc = sim.b.acc(sources=sim.a)
    np.testing.assert_array_equal(getattr(sim.b, name)(sources=sim.a), acc[..., col])
    np.testing.assert_allclose(acc, sim.b.acc(sources=sim.a, return_internal=True) * KPCGYR_TO_KMS, rtol=1e-15)


def test_all_the_particles_and_external_forces_are_views_of_the_sim():
    sim = _sim(run=False)
    assert sim.particles.name == 'particles'
    np.testing.assert_array_equal(sim.particles.mass, sim.mass)
    for name in ('particles', 'external', 'acc'):
        with pytest.raises(ValueError, match="already means something else"):
            sim.add_particles(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1), name=name)


def _other_sim():
    return _sim(run=False)


@pytest.mark.parametrize("make_sources, solver, error, match", [
    pytest.param(lambda sim: sim, None, ValueError, "ambiguous", id='the_sim'),
    pytest.param(lambda sim: _other_sim().a, None, ValueError, "same Sim", id='another_sims'),
    pytest.param(lambda sim: [], None, ValueError, "empty", id='empty'),
    pytest.param(lambda sim: 'a', None, TypeError, "sources must be", id='a_name'),
    pytest.param(lambda sim: sim.external, DirectSummationGravity(), ValueError, "no particles", id='solver_unused'),
])
def test_unusable_sources_are_refused(make_sources, solver, error, match):
    sim = _sim()
    with pytest.raises(error, match=match):
        sim.b.acc(t=0, sources=make_sources(sim), solver=solver)


def test_a_solve_needs_every_solved_particles_softening():
    sim = _sim(method=None, eps_b=None)
    with pytest.raises(ValueError, match="Softening isn't set"):
        sim.a.acc(t=0, sources=sim.b, solver=DirectSummationGravity())
    sim.a.acc(t=0, sources=sim.b, solver=DirectSummationGravity(eps=0.1))
