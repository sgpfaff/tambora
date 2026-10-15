"""
Tests for ``BoundednessHook`` in ``tambora/dynamics/hooks/boundedness.py``.


Testing Approach
----------------

Tests are split into two groups, by what they take on trust:

* The **accessors** (``release_time``, ``transitions``/``transition_times``,
  ``mask_at``/``history``/``n_bound``/``n_unbound``/``fraction``) are driven from
  a hand-assembled event log. That isolates their own logic (direction
  filtering, the rebinding rules, the ``fraction`` denominator, etc.) from the
  question of whether the hook recorded the log correctly.
* ``__call__`` is driven with a minimal ``StepState`` stub (as
  ``test_conservation.py`` does) and checked on the log it *builds*. 

The replay these accessors sit on (``reconstruct_mask``) and the physics they
report (``bound_mask``) are tested in ``tests/dynamics/test_boundedness.py``.
"""

import warnings

import numpy as np
import pytest

from tambora.dynamics.hooks import BoundednessHook
from tambora.dynamics.hooks.boundedness import _bound_com, _bound_dispersion


def _initial():
    return np.array([True, True, True, True])


def _all(n):
    return np.ones(n, dtype=bool)

# =============================================================================
# BoundednessHook.release_time -- when did each unbound particle leave?
# =============================================================================
#
# Dates each particle's *last* unbinding, replaying the same log reconstruct_mask
# uses. The two must agree on who is unbound, or one of them is lying.

def _hook(initial, events, t):
    bh = BoundednessHook('sat', eps=0.01)
    bh.initial_mask, bh.events, bh.t = np.asarray(initial), events, t
    return bh


def test_release_time_dates_the_last_unbinding_not_the_first():
    # Particle 0 chatters at the boundary: out, back, out again. It was released
    # at t=3, not t=1. It was still part of the remnant in between.
    bh = _hook(_initial(), [(0, 1.0, -1), (0, 2.0, +1), (0, 3.0, -1)], [0., 4.])
    assert bh.release_time(t=4.0)[0] == 3.0


def test_release_time_is_nan_for_particles_bound_at_query_time():
    bh = _hook(_initial(), [(0, 1.0, -1), (0, 3.0, +1)], [0., 4.])
    assert bh.release_time(t=2.0)[0] == 1.0        # unbound at t=2 ...
    assert np.isnan(bh.release_time(t=4.0)[0])     # ... rebound by t=4


def test_release_time_agrees_with_mask_at():
    events = [(0, 1.0, -1), (2, 2.0, -1), (0, 3.0, +1)]
    bh = _hook(_initial(), events, [0., 4.])
    for t in (0.5, 1.5, 2.5, 3.5):
        np.testing.assert_array_equal(np.isnan(bh.release_time(t=t)), bh.mask_at(t))


def test_release_time_marks_particles_unbound_before_the_first_fire():
    # Already gone when the hook started: unbound, but the log cannot date it.
    bh = _hook([False, True, True, True], [(1, 1.0, -1)], [0., 2.])
    out = bh.release_time(t=2.0)
    assert out[0] == -np.inf
    assert out[1] == 1.0
    assert np.isnan(out[2])


def test_release_time_defaults_to_the_last_fire():
    bh = _hook(_initial(), [(0, 1.0, -1), (2, 5.0, -1)], [0., 2.0])
    np.testing.assert_array_equal(bh.release_time(), bh.release_time(t=2.0))
    assert np.isnan(bh.release_time()[2])          # t=5 event is after the last fire


# =============================================================================
# BoundednessHook.transitions -- direction filtering by name
# =============================================================================
#
# The log stores +/-1; the query API takes strings. These pin that seam.

def test_transitions_filters_by_direction_name():
    events = [(0, 1.0, -1), (2, 2.0, -1), (0, 3.0, +1)]
    bh = _hook(_initial(), events, [0., 4.])
    assert bh.transitions('unbound') == [(0, 1.0, -1), (2, 2.0, -1)]
    assert bh.transitions('bound') == [(0, 3.0, +1)]
    assert bh.transitions() == events            # None -> both, chronological


def test_transitions_rejects_the_raw_integer_codes():
    # Passing -1 straight from the log is the obvious slip; it must not silently
    # return everything (which a truthiness check would do).
    bh = _hook(_initial(), [(0, 1.0, -1)], [0., 2.])
    for bad in (-1, +1, 'stripped'):
        with pytest.raises(ValueError, match='direction'):
            bh.transitions(bad)


def test_transition_times_tracks_transitions():
    events = [(0, 1.0, -1), (2, 1.0, -1), (0, 3.0, +1)]
    bh = _hook(_initial(), events, [0., 4.])
    np.testing.assert_array_equal(bh.transition_times('unbound'), [1.0, 1.0])
    np.testing.assert_array_equal(bh.transition_times(), [1.0, 1.0, 3.0])
    assert bh.transition_times('bound').tolist() == [3.0]


def test_transition_times_is_empty_when_nothing_flipped():
    bh = _hook(_initial(), [], [0., 1.])
    assert bh.transition_times('unbound').size == 0


# =============================================================================
# BoundednessHook.__call__ -- building the log from live state
# =============================================================================
#
# The accessor tests above hand-assemble bh.events, so they say nothing about
# whether the hook records the right thing in the first place. These drive it
# with a minimal stub (as test_conservation.py does) and check the log it builds.

class _FakeComponent:
    """Minimal stand-in for a Component: just what BoundednessHook reads."""

    def __init__(self, pos, vel, mass):
        self._pos, self._vel, self.mass = pos, vel, mass

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel


class _FakeState:
    """Minimal stand-in for StepState, serving one pre-set mask."""

    def __init__(self, t, mask, comp):
        self.t, self._mask, self._comp = t, mask, comp
        self.reported = {}

    def component(self, name):
        return self._comp

    def bound_mask(self, component, **kw):
        return self._mask

    def report(self, **fields):
        self.reported.update(fields)


def _drive(masks, times=None, comp=None, **hook_kw):
    """Fire a hook once per mask; return the hook and the last state."""
    masks = [np.asarray(m) for m in masks]
    n = masks[0].size
    times = range(len(masks)) if times is None else times
    if comp is None:
        comp = _FakeComponent(np.zeros((n, 3)), np.zeros((n, 3)), np.ones(n))
    bh = BoundednessHook('sat', eps=0.01, **hook_kw)
    state = None
    for t, m in zip(times, masks):
        state = _FakeState(float(t), m, comp)
        bh(state)
    return bh, state


def test_first_fire_seeds_the_initial_mask_and_logs_nothing():
    bh, _ = _drive([[True, True, False]])
    np.testing.assert_array_equal(bh.initial_mask, [True, True, False])
    assert bh.events == []          # nothing to compare against yet


def test_call_logs_one_event_per_particle_that_flips():
    bh, _ = _drive([[True, True, True],
                    [True, False, True],       # 1 unbinds at t=1
                    [True, False, False]])     # 2 unbinds at t=2
    assert bh.events == [(1, 1.0, -1), (2, 2.0, -1)]
    assert bh.t == [0.0, 1.0, 2.0]


def test_call_logs_rebinding_with_a_positive_direction():
    bh, _ = _drive([[True, True], [False, True], [True, True]])
    assert bh.events == [(0, 1.0, -1), (0, 2.0, +1)]


def test_call_logs_nothing_when_the_mask_is_unchanged():
    bh, _ = _drive([[True, False]] * 4)
    assert bh.events == []


def test_logged_events_reconstruct_the_masks_they_came_from():
    # The round trip that matters: whatever __call__ recorded, mask_at must
    # hand back the original masks at the fire times.
    masks = [[True, True, True, True],
             [True, False, True, True],
             [True, False, False, True],
             [True, True, False, True]]        # 1 rebinds
    bh, _ = _drive(masks)
    np.testing.assert_array_equal(bh.history(), np.array(masks))


def test_call_reports_the_bound_count_to_the_progress_bar():
    _, state = _drive([[True, True, False]])
    assert state.reported == {'n_bound(sat)': 2}


def test_initial_mask_is_not_a_live_view_of_the_solver_output():
    # __call__ must copy: a solver reusing its output buffer would otherwise
    # rewrite the history in place.
    mask = np.array([True, True, True])
    comp = _FakeComponent(np.zeros((3, 3)), np.zeros((3, 3)), np.ones(3))
    bh = BoundednessHook('sat', eps=0.01)
    bh(_FakeState(0.0, mask, comp))
    mask[:] = False                            # solver reuses the buffer
    np.testing.assert_array_equal(bh.initial_mask, [True, True, True])


def test_reset_clears_the_records():
    bh, state = _drive([[True, True], [False, True]], track=('com',))
    bh.reset()
    bh(_FakeState(0., np.array([True, False]), state._comp))
    assert bh.t == [0.]
    np.testing.assert_array_equal(bh.initial_mask, [True, False])
    assert bh.events == []
    assert len(bh.com) == 1


# --- tracked reductions and captured payloads -------------------------------

def test_track_records_one_reduction_per_fire():
    pos = np.array([[0., 0, 0], [2, 0, 0], [99, 0, 0]])
    comp = _FakeComponent(pos, np.zeros((3, 3)), np.ones(3))
    bh, _ = _drive([[True, True, False]] * 2, comp=comp, track=('com',))
    assert len(bh.com) == 2                    # aligned with bh.t
    np.testing.assert_allclose(bh.com[0], [1., 0, 0])   # unbound particle excluded


def test_untracked_reductions_are_not_attributes():
    bh, _ = _drive([[True, True]], track=('com',))
    assert not hasattr(bh, 'dispersion')


def test_capture_transitions_appends_payload_in_declared_order():
    pos = np.array([[1., 2, 3], [0, 0, 0]])
    vel = np.array([[4., 5, 6], [0, 0, 0]])
    comp = _FakeComponent(pos, vel, np.ones(2))
    bh, _ = _drive([[True, True], [False, True]], comp=comp,
                   capture_transitions=('pos', 'vel'))
    (idx, t, direction, p, v), = bh.events
    assert (idx, t, direction) == (0, 1.0, -1)
    np.testing.assert_array_equal(p, [1, 2, 3])
    np.testing.assert_array_equal(v, [4, 5, 6])


def test_events_carry_no_payload_by_default():
    bh, _ = _drive([[True, True], [False, True]])
    assert len(bh.events[0]) == 3


def test_captured_payload_is_a_snapshot_not_a_live_reference():
    pos = np.array([[1., 2, 3], [0, 0, 0]])
    comp = _FakeComponent(pos, np.zeros((2, 3)), np.ones(2))
    bh, _ = _drive([[True, True], [False, True]], comp=comp,
                   capture_transitions=('pos',))
    pos[0] = [9, 9, 9]                         # particle moves on
    np.testing.assert_array_equal(bh.events[0][3], [1, 2, 3])


# --- construction -----------------------------------------------------------

def test_unknown_track_quantity_raises():
    with pytest.raises(ValueError, match='track quantity'):
        BoundednessHook('sat', eps=0.01, track=('com', 'nonsense'))


def test_unknown_capture_raises():
    with pytest.raises(ValueError, match='capture'):
        BoundednessHook('sat', eps=0.01, capture_transitions=('acc',))


def test_dedup_key_distinguishes_meaningful_config():
    base = BoundednessHook('sat', eps=0.01)
    assert base._dedup_key() == BoundednessHook('sat', eps=0.01)._dedup_key()
    for other in (BoundednessHook('host', eps=0.01),
                  BoundednessHook('sat', eps=0.02),
                  BoundednessHook('sat', eps=0.01, track=('com',)),
                  BoundednessHook('sat', eps=0.01, capture_transitions=('pos',)),
                  BoundednessHook('sat', eps=0.01, method='direct')):
        assert base._dedup_key() != other._dedup_key()


# =============================================================================
# BoundednessHook counting accessors -- history / n_bound / n_unbound / fraction
# =============================================================================
#
# Thin wrappers over the log, but each has a convention worth pinning: what the
# default `times` is, and what fraction divides by.

def test_history_defaults_to_the_fire_times():
    bh, _ = _drive([[True, True], [True, False]], times=[0.0, 0.5])
    np.testing.assert_array_equal(bh.history(), [[True, True], [True, False]])
    np.testing.assert_array_equal(bh.history(), bh.history(bh.t))


def test_history_resolves_times_between_fires():
    # Piecewise-constant: a time between fires reports the last fire's state.
    bh, _ = _drive([[True, True], [True, False]], times=[0.0, 1.0])
    np.testing.assert_array_equal(bh.history([0.5, 1.5]),
                                  [[True, True], [True, False]])


def test_history_accepts_times_outside_the_fired_range():
    bh, _ = _drive([[True, True], [True, False]], times=[1.0, 2.0])
    np.testing.assert_array_equal(bh.history([0.0]), [[True, True]])   # pre-first
    np.testing.assert_array_equal(bh.history([99.0]), [[True, False]]) # post-last


def test_n_bound_counts_per_time():
    bh, _ = _drive([[True, True, True], [True, False, True], [False, False, True]])
    np.testing.assert_array_equal(bh.n_bound(), [3, 2, 1])


def test_n_bound_and_n_unbound_sum_to_the_component_size():
    bh, _ = _drive([[True, True, True], [True, False, True], [False, False, False]])
    np.testing.assert_array_equal(bh.n_bound() + bh.n_unbound(), 3)
    np.testing.assert_array_equal(bh.n_unbound(), [0, 1, 3])


def test_fraction_is_the_bound_count_over_the_component_size():
    bh, _ = _drive([[True, True, True, True], [True, True, False, False]])
    np.testing.assert_allclose(bh.fraction(), [1.0, 0.5])


def test_fraction_does_not_start_at_one_when_the_component_starts_unbound():
    # Documented: the denominator is the component's size, NOT the number bound
    # at the first fire -- so this curve starts below 1 rather than rebasing.
    bh, _ = _drive([[True, True, True, False], [True, True, False, False]])
    np.testing.assert_allclose(bh.fraction(), [0.75, 0.5])


def test_counting_accessors_accept_explicit_times():
    bh, _ = _drive([[True, True], [True, False]], times=[0.0, 1.0])
    np.testing.assert_array_equal(bh.n_bound([1.5]), [1])
    np.testing.assert_allclose(bh.fraction([1.5]), [0.5])
    np.testing.assert_array_equal(bh.n_unbound([1.5]), [1])


# =============================================================================
# Tracked reductions -- the mass-weighted arithmetic itself
# =============================================================================
#
# _bound_com is a weighted mean; _bound_dispersion is the mass-weighted RMS
# speed about that mean. Both are restricted to the bound set, which is the part
# worth pinning: a reduction that quietly includes stripped particles would
# track the stream, not the remnant.

def test_bound_com_is_mass_weighted():
    pos = np.array([[0., 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(_bound_com(pos, np.array([1., 1]), _all(2)), [1, 0, 0])
    np.testing.assert_allclose(_bound_com(pos, np.array([3., 1]), _all(2)), [0.5, 0, 0])


def test_bound_com_ignores_unbound_particles():
    pos = np.array([[0., 0, 0], [2, 0, 0], [1e6, 0, 0]])
    mask = np.array([True, True, False])
    np.testing.assert_allclose(_bound_com(pos, np.ones(3), mask), [1, 0, 0])


def test_bound_com_matches_numpy_average():
    rng = np.random.default_rng(0)
    pos, mass = rng.normal(size=(20, 3)), rng.uniform(0.5, 2, 20)
    mask = rng.random(20) > 0.3
    np.testing.assert_allclose(
        _bound_com(pos, mass, mask),
        np.average(pos[mask], axis=0, weights=mass[mask]))


def test_bound_dispersion_is_zero_for_a_cold_set():
    vel = np.tile([3., -1, 2], (5, 1))          # every particle identical
    assert _bound_dispersion(vel, np.ones(5), _all(5)) == pytest.approx(0.0)


def test_bound_dispersion_is_the_rms_speed_about_the_com():
    vel = np.array([[1., 0, 0], [-1, 0, 0]])    # v_com = 0; |v| = 1 each
    assert _bound_dispersion(vel, np.ones(2), _all(2)) == pytest.approx(1.0)


def test_bound_dispersion_is_invariant_under_a_velocity_boost():
    rng = np.random.default_rng(1)
    vel, mass = rng.normal(size=(30, 3)), rng.uniform(0.5, 2, 30)
    ref = _bound_dispersion(vel, mass, _all(30))
    boosted = _bound_dispersion(vel + [100., -50, 30], mass, _all(30))
    assert boosted == pytest.approx(ref)


def test_bound_dispersion_ignores_unbound_particles():
    vel = np.array([[1., 0, 0], [-1, 0, 0], [1e6, 0, 0]])
    mask = np.array([True, True, False])
    assert _bound_dispersion(vel, np.ones(3), mask) == pytest.approx(1.0)


def test_reductions_of_an_empty_bound_set_are_nan_without_warning():
    # bound_mask is documented to return all-False when a component fully
    # dissolves. The reductions must return nan (keeping bh.com aligned with bh.t) rather than raise -- and must do it
    # deliberately, not as 0/0 debris behind a RuntimeWarning.
    pos, mass, none_bound = np.zeros((2, 3)), np.ones(2), np.zeros(2, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any RuntimeWarning fails here
        assert np.isnan(_bound_com(pos, mass, none_bound)).all()
        assert np.isnan(_bound_dispersion(pos, mass, none_bound))


def test_reductions_are_nan_when_the_bound_set_is_entirely_massless():
    # Same 0/0, reachable a second way: massless tracers.
    pos, massless = np.zeros((2, 3)), np.zeros(2)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.isnan(_bound_com(pos, massless, _all(2))).all()
        assert np.isnan(_bound_dispersion(pos, massless, _all(2)))


def test_a_dissolved_component_tracks_nan_rather_than_failing():
    # End to end: the hook must survive total dissolution. The first fire is the
    # control.
    comp = _FakeComponent(np.array([[0., 0, 0], [2, 0, 0]]),
                          np.array([[1., 0, 0], [-1, 0, 0]]),
                          np.ones(2))
    bh, _ = _drive([[True, True], [False, False]], comp=comp,
                   track=('com', 'dispersion'))
    np.testing.assert_allclose(bh.com[0], [1., 0, 0])       # both bound
    assert bh.dispersion[0] == pytest.approx(1.0)
    assert np.isnan(bh.com[1]).all()                        # dissolved
    assert np.isnan(bh.dispersion[1])
