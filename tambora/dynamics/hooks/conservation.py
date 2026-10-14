"""
Conservation diagnostics as an integration hook.

``ConservationMonitor`` answers one question -- *is the integrator conserving
this?* -- for one or more conserved quantities, selected by name via
``track=(...)``. Each quantity supplies a **reduction** (what the quantity is)
and a **scale** (what "small" means for it), so the drift arithmetic is shared:

    drift = ||value - value0|| / scale

``value0`` and ``scale`` are captured once, at the first fire. For a scalar this
collapses to the familiar relative drift; for a vector quantity the norm keeps
it meaningful even when the vector's total is near zero (the usual case for
momentum in a centre-of-mass frame, where dividing by ``|value0|`` would be
undefined).
"""

import numpy as np

from .base import Hook
from .cadence import EveryOutput


# -- per-quantity reductions and their scales --------------------------------

def _energy(state):
    """Total system energy [internal units] -- a scalar."""
    return state.system_energy()


def _energy_scale(value0, state):
    """|E0|: the standard denominator for relative energy drift."""
    return abs(float(value0))


# name -> (reduce, scale, progress-bar label).
#
# `reduce(state) -> value`      the conserved quantity (scalar or vector)
# `scale(value0, state) -> float`  fixed denominator, captured at the first fire
#
# Adding a quantity is one entry here (plus a note on when it is *valid* -- e.g.
# momentum is only conserved without external forces).
_CONSERVED = {
    'energy': (_energy, _energy_scale, '|dE/E0|'),
}


class ConservationMonitor(Hook):
    """
    Track the drift of conserved quantities over a run.

    Parameters
    ----------
    track : sequence of str, optional
        Conserved quantities to monitor. Currently available: ``'energy'``.
        Default ``('energy',)``.

    Attributes
    ----------
    t : list of float
        Times at which the hook fired.
    values : dict of str -> list
        The raw conserved quantity at each fire, keyed by name.
    drift : dict of str -> list of float
        ``||value - value0|| / scale`` at each fire, keyed by name. For
        ``'energy'`` this is the familiar ``|(E - E0) / E0|``.

    Examples
    --------
    >>> mon = ConservationMonitor()               # doctest: +SKIP
    >>> sim.add_hook(mon)                         # doctest: +SKIP
    >>> sim.run(t_end=1., dt=0.01, dt_out=0.1)    # doctest: +SKIP
    >>> mon.drift['energy'][-1]                   # doctest: +SKIP
    """

    default_cadence = EveryOutput()

    def __init__(self, track=('energy',)):
        track = tuple(track)
        if not track:
            raise ValueError("track must name at least one conserved quantity.")
        for name in track:
            if name not in _CONSERVED:
                raise ValueError(
                    f"Unknown conserved quantity {name!r}. "
                    f"Available: {sorted(_CONSERVED)}")
        self.track = track
        self.reset()

    def reset(self):
        self.t = []
        self.values = {name: [] for name in self.track}
        self.drift = {name: [] for name in self.track}
        self._ref = {}          # name -> (value0, scale), captured on first fire

    def _dedup_key(self):
        # Two monitors tracking the same set are duplicates; different sets are not.
        return (type(self), self.track)

    def __call__(self, state):
        self.t.append(state.t)
        for name in self.track:
            reduce_fn, scale_fn, label = _CONSERVED[name]
            value = reduce_fn(state)
            if name not in self._ref:
                self._ref[name] = (value, scale_fn(value, state))
            value0, scale = self._ref[name]
            drift = float(np.linalg.norm(
                np.atleast_1d(np.asarray(value) - np.asarray(value0))) / scale)
            self.values[name].append(value)
            self.drift[name].append(drift)
            state.report(**{label: f"{drift:.2e}"})
