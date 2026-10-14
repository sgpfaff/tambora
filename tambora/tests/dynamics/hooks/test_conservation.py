"""
Tests for ``ConservationMonitor`` (``tambora/dynamics/hooks/conservation.py``).


Testing Approach
----------------

* The monitor only touches three things on the state it is handed: ``t``,
``system_energy()``, and ``report()``. These tests drive it with a minimal stub 
rather than a real ``StepState``. That keeps them focused on the hook's own arithmetic 
(reference capture, the drift formula, the reported label) instead of re-testing ``StepState``. 
"""

import numpy as np
import pytest

from tambora.dynamics.hooks import ConservationMonitor
from tambora.dynamics.hooks.cadence import EveryOutput


class _FakeState:
    """Minimal stand-in for StepState: just what ConservationMonitor reads."""

    def __init__(self, t, energy):
        self.t = t
        self._energy = energy
        self.reported = {}

    def system_energy(self):
        return self._energy

    def report(self, **fields):
        self.reported.update(fields)


def _drive(energies, times=None):
    """Fire a monitor once per scripted energy; return (monitor, last_state)."""
    times = times if times is not None else list(range(len(energies)))
    mon = ConservationMonitor()
    state = None
    for t, e in zip(times, energies):
        state = _FakeState(t, e)
        mon(state)
    return mon, state


# --- construction / validation -------------------------------------------------

def test_default_track_is_energy():
    assert ConservationMonitor().track == ('energy',)


def test_unknown_quantity_raises():
    with pytest.raises(ValueError, match="Unknown conserved quantity 'mass'"):
        ConservationMonitor(track=('mass',))


def test_empty_track_raises():
    with pytest.raises(ValueError, match="at least one conserved quantity"):
        ConservationMonitor(track=())


def test_track_is_normalised_to_a_tuple():
    # A list in, a tuple out -- so the dedup key stays hashable/comparable.
    assert ConservationMonitor(track=['energy']).track == ('energy',)


def test_default_cadence_is_every_output():
    assert isinstance(ConservationMonitor().default_cadence, EveryOutput)


# --- dedup identity ------------------------------------------------------------

def test_dedup_key_matches_for_same_track():
    assert ConservationMonitor()._dedup_key() == ConservationMonitor()._dedup_key()


def test_dedup_key_includes_track():
    # Distinct tracked sets are different diagnostics, so must not collide.
    a = ConservationMonitor(track=('energy',))._dedup_key()
    assert a == (ConservationMonitor, ('energy',))


# --- the drift arithmetic ------------------------------------------------------

def test_first_fire_has_zero_drift():
    mon, _ = _drive([-100.0])
    assert mon.drift['energy'] == [0.0]


def test_drift_is_relative_energy_change():
    # E0 = -100, then -90 -> |(-90 - -100)/-100| = 0.1; then -50 -> 0.5
    mon, _ = _drive([-100.0, -90.0, -50.0])
    np.testing.assert_allclose(mon.drift['energy'], [0.0, 0.1, 0.5])


def test_reference_is_captured_once_not_rebased():
    # Drift is always measured against the FIRST value, not the previous one.
    mon, _ = _drive([-100.0, -90.0, -100.0])
    np.testing.assert_allclose(mon.drift['energy'], [0.0, 0.1, 0.0])


def test_drift_is_sign_insensitive():
    # A positive excursion of the same size gives the same magnitude.
    mon, _ = _drive([-100.0, -110.0])
    np.testing.assert_allclose(mon.drift['energy'], [0.0, 0.1])


def test_records_times_and_raw_values():
    mon, _ = _drive([-100.0, -90.0], times=[0.0, 0.5])
    assert mon.t == [0.0, 0.5]
    np.testing.assert_allclose(mon.values['energy'], [-100.0, -90.0])


def test_reports_labelled_drift_to_progress_bar():
    mon, state = _drive([-100.0, -90.0])
    assert '|dE/E0|' in state.reported
    assert state.reported['|dE/E0|'] == f"{0.1:.2e}"


def test_reset_clears_the_records_and_the_reference():
    mon, _ = _drive([10., 11., 12.])
    mon.reset()
    mon(_FakeState(0., 20.))
    assert mon.t == [0.]
    assert mon.values['energy'] == [20.]
    assert mon.drift['energy'] == [0.]          # measured from the new first fire, not from 10
