"""
Tests for the boundedness diagnostics in ``tambora/dynamics/diagnostics.py``.


Testing Approach
----------------

Tests are split into two groups:

* ``bound_mask``: the energy criterion (specific energy in the COM frame). These
  cover the physics it must get right (self-bound cores stay, fast escapers go,
  the answer is frame- and solver-independent).

* ``reconstruct_mask``: the pure replay of a transition log that backs every
  derived accessor on ``BoundednessHook`` (``mask_at``/``history``/``fraction``).

Blobs are built cold (uniform velocity) so ``v - v_com == 0`` and boundedness is
decided by the potential alone -- which makes the expected answer analytic.
"""

import numpy as np
import pytest

from tambora.dynamics.diagnostics import bound_mask, reconstruct_mask
from tambora.units import G_INTERNAL, KMS_TO_KPCGYR

R_GAL = 20.0      # kpc; an arbitrary offset from the origin to catch any
                  # accidental assumption that the cluster sits at (0,0,0)


def _cold_blob(m_cl, scale, N=300, n_far=2, seed=0):
    """A compact cold cluster at (R_GAL,0,0) plus `n_far` distant tracers along +/-x.

    Velocities are uniform (so ``v - v_com == 0``): boundedness is decided by
    potential alone.
    """
    rng = np.random.default_rng(seed)
    core = rng.normal(scale=scale / 4, size=(N, 3))
    far = np.array([[3 * scale, 0.0, 0.0], [-3 * scale, 0.0, 0.0]])[:n_far]
    pos = np.vstack([core, far]) + np.array([R_GAL, 0.0, 0.0])
    vel = np.zeros_like(pos)
    mass = np.full(len(pos), m_cl / len(pos))
    return pos, vel, mass


def _escape_speed(m_cl, r):
    """Speed needed to escape mass `m_cl` from radius `r` [kpc/Gyr]."""
    return np.sqrt(2 * G_INTERNAL * m_cl / r)


# --- the physics the criterion must get right ---------------------------------

def test_cold_isolated_cluster_is_entirely_bound():
    """Zero relative velocity => E = phi_self < 0 for every particle."""
    pos, vel, mass = _cold_blob(1e6, 0.3, N=200, n_far=0)
    assert bound_mask(pos, vel, mass, eps=0.01, method='direct').all()


def test_fast_tracer_is_unbound():
    """A particle above the escape speed is dropped; the cold core survives."""
    pos, vel, mass = _cold_blob(1e6, 0.3, N=200, n_far=0)
    # One extra particle just outside the core, moving at 10x escape speed.
    r = 1.0
    pos = np.vstack([pos, [R_GAL + r, 0.0, 0.0]])
    vel = np.vstack([vel, [10 * _escape_speed(1e6, r), 0.0, 0.0]])
    mass = np.append(mass, mass[0])

    b = bound_mask(pos, vel, mass, eps=0.01, method='direct')
    assert not b[-1]            # the escaper is gone
    assert b[:-1].all()         # the core is untouched


def test_mask_is_invariant_under_a_velocity_boost():
    """Binding is measured in the COM frame, so a bulk boost changes nothing."""
    pos, vel, mass = _cold_blob(1e6, 0.3, N=150, n_far=0)
    r = 1.0
    pos = np.vstack([pos, [R_GAL + r, 0.0, 0.0]])
    vel = np.vstack([vel, [10 * _escape_speed(1e6, r), 0.0, 0.0]])
    mass = np.append(mass, mass[0])

    rest = bound_mask(pos, vel, mass, eps=0.01, method='direct')
    boosted = bound_mask(pos, vel + np.array([10*_escape_speed(1e6, r), -10 * _escape_speed(1e6, r), 5*_escape_speed(1e6, r)]), mass,
                         eps=0.01, method='direct')
    np.testing.assert_array_equal(rest, boosted)


def test_all_unbound_returns_all_false():
    """A blob flying apart everywhere unbinds completely without dividing by zero."""
    rng = np.random.default_rng(3)
    N = 30
    pos = rng.normal(scale=0.3, size=(N, 3)) + np.array([R_GAL, 0.0, 0.0])
    m_cl = 1e3                                   # negligible self-gravity ...
    # ... against a large isotropic velocity spread, so nothing can hold together.
    vel = rng.normal(scale=1e3, size=(N, 3))
    mass = np.full(N, m_cl / N)
    assert not bound_mask(pos, vel, mass, eps=0.01, method='direct').any()


def test_direct_and_falcON_agree():
    """The criterion is a property of the physics, not of the solver."""
    pos, vel, mass = _cold_blob(1e6, 0.3, N=200, n_far=0)
    r = 1.0
    pos = np.vstack([pos, [R_GAL + r, 0.0, 0.0]])
    vel = np.vstack([vel, [10 * _escape_speed(1e6, r), 0.0, 0.0]])
    mass = np.append(mass, mass[0])

    np.testing.assert_array_equal(
        bound_mask(pos, vel, mass, eps=0.05, method='direct'),
        bound_mask(pos, vel, mass, eps=0.05, method='falcON', theta=0.4))


# --- the COM-frame seed -------------------------------------------------------

def _cluster_plus_stream(f_stream, n_total=600, m_cl=1e6, scale=0.1, seed=0):
    """A cold bound cluster moving at `V`, plus a 'wrapped stream' around it.

    The stream's velocities point in every direction (as a real wrapped stream's
    do), so its mean is ~0 while the cluster moves at V. That drags the *global*
    mean velocity far from the cluster's -- which is exactly the situation the
    most-bound seed exists to survive. Truth is known: the cluster particles.
    """
    rng = np.random.default_rng(seed)
    V = 150.0 * KMS_TO_KPCGYR                    # orbital speed, internal units
    n_c = int(round(n_total * (1 - f_stream)))
    n_s = n_total - n_c

    pos_c = rng.normal(scale=scale, size=(n_c, 3)) + np.array([R_GAL, 0.0, 0.0])
    vel_c = np.tile([0.0, V, 0.0], (n_c, 1))     # cold: bound by potential alone

    # n_s == 0 needs no special case: the empty arrays below flow through
    # column_stack/vstack unchanged, giving the bare cluster.
    ang = rng.uniform(0, 2 * np.pi, n_s)
    pos_s = np.column_stack([R_GAL * np.cos(ang), R_GAL * np.sin(ang),
                             rng.normal(0, 0.5, n_s)])
    d = rng.normal(size=(n_s, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    pos = np.vstack([pos_c, pos_s])
    vel = np.vstack([vel_c, V * d])
    mass = np.full(n_total, m_cl / n_c)
    truth = np.zeros(n_total, bool)
    truth[:n_c] = True
    return pos, vel, mass, truth


@pytest.mark.parametrize("f_stream", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_remnant_survives_a_stream_that_drags_the_global_mean_velocity(f_stream):
    """Regression: seeding the COM frame from the global mean unbinds everything.

    The stream pulls the mass-weighted mean velocity away from the remnant's by
    more than the remnant's escape speed, so on the first pass every cluster
    particle looks unbound and unbinding is monotonic, so it never recovers.
    """
    pos, vel, mass, truth = _cluster_plus_stream(f_stream)
    b = bound_mask(pos, vel, mass, eps=0.01, method='direct')

    assert b[truth].mean() > 0.99      # the remnant survives
    assert not b[~truth].any()         # and no stream particle is kept


def test_seed_is_robust_to_the_global_mean_being_useless():
    """The seed must key on binding energy, not on the remnant being a majority."""
    # 90% stream: the global mean velocity is nowhere near the remnant's.
    scale = 0.1
    pos, vel, mass, truth = _cluster_plus_stream(0.9, scale=scale)
    v_com_global = (mass[:, None] * vel).sum(0) / mass.sum()
    v_com_true = (mass[truth, None] * vel[truth]).sum(0) / mass[truth].sum()
    m_cl = mass[truth].sum()
    # Point-mass escape speed at the cluster's scale radius. Only ~20% of m_cl lies within `scale` of a Gaussian blob),
    # which is fine: this is an order-of-magnitude precondition, and the frame
    # error clears it by ~15x.
    v_esc = np.sqrt(2 * G_INTERNAL * m_cl / scale)

    # The global mean really is off by more than the cluster can hold on to.
    assert np.linalg.norm(v_com_global - v_com_true) > v_esc

    assert bound_mask(pos, vel, mass, eps=0.01, method='direct')[truth].mean() > 0.99


# =============================================================================
# reconstruct_mask -- pure replay of a transition log
# =============================================================================
#
# An event is (idx, time, direction, *payload); direction +1 => bound, -1 =>
# unbound. reconstruct_mask replays every event with time <= t onto a copy of
# the initial mask. It assumes events are in chronological order (which the hook
# guarantees). Indices below are component-local, matching the hook.

def _initial():
    return np.array([True, True, True, True])


def test_replays_unbind_and_rebind_events():
    events = [(0, 1.0, -1),          # particle 0 unbinds at t=1
              (2, 2.0, -1),          # particle 2 unbinds at t=2
              (0, 3.0, +1)]          # particle 0 rebinds at t=3
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 0.5),
                                  [True, True, True, True])   # before any event
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.5),
                                  [False, True, True, True])  # 0 gone
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 2.5),
                                  [False, True, False, True]) # 0 and 2 gone
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 3.5),
                                  [True, True, False, True])  # 0 rebound


def test_query_time_boundary_is_inclusive():
    events = [(0, 1.0, -1)]
    # An event exactly at the query time has already happened (time <= t).
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.0),
                                  [False, True, True, True])
    # Just before it, the flip has not yet applied.
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 0.999),
                                  [True, True, True, True])


def test_does_not_mutate_initial_mask():
    initial = _initial()
    before = initial.copy()
    out = reconstruct_mask(initial, [(0, 1.0, -1)], t=2.0)
    np.testing.assert_array_equal(initial, before)    # input untouched
    assert out is not initial                          # returns a fresh array


def test_ignores_trailing_payload():
    # Events may carry captured pos/vel after (idx, time, direction); replay
    # must give the same result as the bare 3-tuples.
    bare = [(0, 1.0, -1), (2, 2.0, +1)]
    with_payload = [(0, 1.0, -1, np.array([1.0, 2.0, 3.0])),
                    (2, 2.0, +1, np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0]))]
    np.testing.assert_array_equal(
        reconstruct_mask(_initial(), bare, 5.0),
        reconstruct_mask(_initial(), with_payload, 5.0))


def test_ordered_double_flip_resolves_to_last_event():
    # Two flips on the same particle, in chronological order: the later one wins.
    events = [(1, 1.0, -1), (1, 2.0, +1)]
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.5),
                                  [True, False, True, True])  # after unbind only
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 2.5),
                                  [True, True, True, True])   # after rebind


def test_assumes_chronological_order():
    # reconstruct_mask replays events in list order, not by timestamp, so it
    # relies on the log being chronological (as the hook always produces it).
    # Fed the same two flips out of order, it returns the wrong state -- this
    # documents the precondition rather than endorsing unordered input.
    ordered = [(1, 1.0, -1), (1, 2.0, +1)]      # correct: ends bound at t>=2
    shuffled = [(1, 2.0, +1), (1, 1.0, -1)]     # same events, wrong order
    np.testing.assert_array_equal(reconstruct_mask(_initial(), ordered, 5.0),
                                  [True, True, True, True])
    np.testing.assert_array_equal(reconstruct_mask(_initial(), shuffled, 5.0),
                                  [True, False, True, True])  # last-in-list wins


# Every helper above uses equal masses (np.full), which leaves the 
# *weighting* in bound_mask's v_com unconstrained; a plain
# vel.mean(0) would satisfy all of them. These pin it explicitly.

def _heavy_core_plus_light_escapers():
    """4 heavy particles at rest (99.7% of the mass) + 3 light ones escaping.

    The mass-weighted frame sits on the heavy core, so the light particles are
    correctly seen to escape. An unweighted mean is dragged a third of the way
    to the escapers' velocity and keeps everything bound.
    """
    mass = np.array([250.] * 4 + [1.] * 3)
    pos = np.random.default_rng(3).normal(scale=0.01, size=(7, 3))
    vel = np.zeros((7, 3))
    vel[4:, 0] = 0.8            # ~1.15x the light particles' escape speed
    return pos, vel, mass


def test_v_com_is_mass_weighted_not_a_plain_mean():
    pos, vel, mass = _heavy_core_plus_light_escapers()
    b = bound_mask(pos, vel, mass, eps=0.005, method='direct')
    np.testing.assert_array_equal(b, [True] * 4 + [False] * 3)


def test_mass_weighted_com_frame_survives_a_boost():
    # Same system, boosted: the weighting must hold in any frame.
    pos, vel, mass = _heavy_core_plus_light_escapers()
    ref = bound_mask(pos, vel, mass, eps=0.005, method='direct')
    boosted = bound_mask(pos, vel + [30., -12, 7], mass, eps=0.005, method='direct')
    np.testing.assert_array_equal(boosted, ref)


def test_heavy_core_dominates_the_frame_regardless_of_particle_count():
    # Regression: the seed selects its core on binding energy (mass*phi), not on
    # the specific potential. Selecting on phi alone fills the core with the
    # LIGHT particles nearest the centre (a 1 Msun particle beside a 250 Msun one
    # has the deeper specific potential), which here seeds the frame on the
    # escapers and unbinds the entire massive remnant.
    rng = np.random.default_rng(3)
    mass = np.concatenate([np.full(4, 250.), np.full(60, 1.)])
    pos = rng.normal(scale=0.01, size=(64, 3))
    vel = np.zeros((64, 3))
    vel[4:, 0] = 0.8
    b = bound_mask(pos, vel, mass, eps=0.005, method='direct')
    assert b[:4].all()          # heavy core kept
    assert not b[4:].any()      # every light escaper dropped
