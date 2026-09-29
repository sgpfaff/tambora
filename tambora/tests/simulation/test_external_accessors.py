"""
Cross-class contract for the external-force accessors.

``Sim``, ``Component`` and ``StepState`` each expose the external field, and they
must agree.


Testing Approach
----------------

Two properties checked against a real ``Sim``:

* **Shape is independent of configuration.** With no external force at all, every
  accessor must return zeros of the natural shape.
* **The three views agree.** A component's slice must equal ``Sim``'s over the
  same particles, ``StepState``'s live value must equal ``Sim``'s at the same
  time, and the all-snapshots path must equal the single-snapshot path stacked.
"""

import numpy as np
import pytest

from tambora.simulation import Sim

galpy = pytest.importorskip("galpy")
from galpy.potential import MWPotential2014                     # noqa: E402
from tambora.dynamics.forces import ExternalPotential      # noqa: E402


N_C, N_D = 4, 2
N = N_C + N_D


def _sim(external=False):
    """Two components at ~8 kpc, run for 3 snapshots, with or without a host."""
    rng = np.random.default_rng(0)
    s = Sim()
    s.add_particles('c', rng.random((N_C, 3)) + 8.0, np.zeros((N_C, 3)), np.ones(N_C))
    s.add_particles('d', rng.random((N_D, 3)) + 8.0, np.zeros((N_D, 3)), np.ones(N_D))
    if external:
        s.add_external_force(ExternalPotential(MWPotential2014))
    s.run(t_end=0.2, dt=0.1, dt_out=0.1, method=None, progress=False)
    return s


@pytest.fixture(params=[False, True], ids=['no_external_force', 'with_galpy_host'])
def sim(request):
    return _sim(external=request.param)


# --- shape does not depend on whether an external force was added -------------

@pytest.mark.parametrize("name, width", [
    ('external_acc', 3), ('external_ax', None),
    ('external_ay', None), ('external_az', None),
])
def test_sim_acceleration_accessors_have_a_shape_independent_of_configuration(sim, name, width):
    out = getattr(sim, name)(t=0)
    assert out.shape == ((N, 3) if width else (N,))


@pytest.mark.parametrize("name, width", [
    ('external_acc', 3), ('external_ax', None),
    ('external_ay', None), ('external_az', None),
])
def test_component_acceleration_accessors_have_a_shape_independent_of_configuration(sim, name, width):
    out = getattr(sim.c, name)(t=0)
    assert out.shape == ((N_C, 3) if width else (N_C,))


def test_external_pot_shape_is_independent_of_configuration(sim):
    assert sim.compute_external_pot(t=0).shape == (N,)
    assert sim.c.compute_external_pot(t=0).shape == (N_C,)


def test_external_pot_over_all_snapshots_does_not_raise(sim):
    # Regression: Component passed the (n_snap, N, 3) stack straight to
    # potential(), which returns one value per SNAPSHOT, so `mass * ext_pot`
    # failed to broadcast --> (4,) against (3,). 
    n_snap = len(sim.times)
    assert sim.compute_external_pot().shape == (n_snap, N)
    assert sim.c.compute_external_pot().shape == (n_snap, N_C)
    assert sim.d.compute_external_pot().shape == (n_snap, N_D)


def test_no_external_force_really_means_zero():
    # The other half of the contract: the shapes above must be zeros here, or
    # they would be passing for the wrong reason.
    s = _sim(external=False)
    np.testing.assert_array_equal(s.external_acc(t=0), np.zeros((N, 3)))
    np.testing.assert_array_equal(s.c.external_acc(t=0), np.zeros((N_C, 3)))
    np.testing.assert_array_equal(s.compute_external_pot(t=0), np.zeros(N))
    np.testing.assert_array_equal(s.c.compute_external_pot(), np.zeros((len(s.times), N_C)))


def test_a_real_external_force_is_not_silently_zero():
    # ... and with a host present the accessors must actually report it, or every
    # zeros-assertion above would be vacuous.
    s = _sim(external=True)
    assert np.abs(s.external_acc(t=0)).sum() > 0
    assert np.abs(s.compute_external_pot(t=0)).sum() > 0
    assert np.abs(s.c.compute_external_pot()).sum() > 0


# --- the three views agree ----------------------------------------------------

def test_component_slices_reassemble_the_sim(sim):
    np.testing.assert_allclose(
        np.vstack([sim.c.external_acc(t=0), sim.d.external_acc(t=0)]),
        sim.external_acc(t=0))
    np.testing.assert_allclose(
        np.concatenate([sim.c.compute_external_pot(t=0), sim.d.compute_external_pot(t=0)]),
        sim.compute_external_pot(t=0))


def test_all_snapshot_path_matches_the_single_snapshot_path(sim):
    # The two branches of compute_external_pot must not disagree.
    stacked = np.vstack([sim.c.compute_external_pot(t=i) for i in range(len(sim.times))])
    np.testing.assert_allclose(sim.c.compute_external_pot(), stacked)


def test_axis_accessors_are_columns_of_external_acc(sim):
    acc = sim.c.external_acc(t=0)
    for i, name in enumerate(('external_ax', 'external_ay', 'external_az')):
        np.testing.assert_allclose(getattr(sim.c, name)(t=0), acc[..., i])


@pytest.mark.parametrize("external", [False, True], ids=['no_external_force', 'with_galpy_host'])
def test_step_state_agrees_with_sim_at_the_same_time(external):
    # StepState is the live view, Sim the stored one: same field, same answer.
    # Hooks must be registered before run(), so this builds its own sim rather
    # than reusing the already-run fixture.
    rng = np.random.default_rng(0)
    s = Sim()
    s.add_particles('c', rng.random((N_C, 3)) + 8.0, np.zeros((N_C, 3)), np.ones(N_C))
    s.add_particles('d', rng.random((N_D, 3)) + 8.0, np.zeros((N_D, 3)), np.ones(N_D))
    if external:
        s.add_external_force(ExternalPotential(MWPotential2014))
    seen = {}
    s.add_hook(lambda st: seen.setdefault(
        st.t, (st.external_acc().copy(), st.c.external_acc().copy())))
    s.run(t_end=0.2, dt=0.1, dt_out=0.1, method=None, progress=False)

    # Shape contract holds at every fire, including t0.
    for t, (full, comp) in seen.items():
        assert full.shape == (N, 3)                   # never a scalar
        assert comp.shape == (N_C, 3)
        np.testing.assert_allclose(comp, full[:N_C])  # the component view slices it


def _run_capturing_external_acc(t_end=0.2, external=True):
    rng = np.random.default_rng(0)
    s = Sim()
    s.add_particles('c', rng.random((N_C, 3)) + 8.0, np.zeros((N_C, 3)), np.ones(N_C))
    if external:
        s.add_external_force(ExternalPotential(MWPotential2014))
    seen = {}
    s.add_hook(lambda st: seen.setdefault(st.t, st.external_acc().copy()))
    s.run(t_end=t_end, dt=0.1, dt_out=0.1, method=None, progress=False)
    return s, seen


def test_step_state_external_acc_is_populated_at_the_t0_fire():
    _, seen = _run_capturing_external_acc()
    assert np.abs(seen[min(seen)]).sum() > 0, \
        "external field is zero at t0 but the host is present"


def test_t0_external_acc_matches_the_field_at_the_initial_conditions():
    # Not merely non-zero: it must be the *right* value. Compares against Sim's
    # own stored accessor at the same snapshot, which is computed independently.
    s, seen = _run_capturing_external_acc()
    np.testing.assert_allclose(seen[min(seen)],
                               s.external_acc(t=0, return_internal=True), rtol=1e-10)


def test_t0_external_acc_is_still_zero_without_an_external_force():
    # The t0 evaluation must not invent a field that isn't there.
    _, seen = _run_capturing_external_acc(external=False)
    np.testing.assert_array_equal(seen[min(seen)], np.zeros((N_C, 3)))


def test_t0_external_acc_is_the_field_at_t0_not_a_copy_of_the_next_step():
    _, seen = _run_capturing_external_acc(t_end=0.2)
    ts = sorted(seen)
    assert not np.allclose(seen[ts[0]], seen[ts[1]])
