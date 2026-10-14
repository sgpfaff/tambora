"""
Boundedness diagnostics as integration hooks.

``BoundednessHook`` computes the self-bound mask of a component once per fire
(via the cached ``StepState.bound_mask``) and, from it, always maintains a
memory-light transition log. 

Boolean quantities (bound fraction, counts, full mask history) are *derived* 
from that log on demand. Coordinate-dependent reductions (COM, dispersion) can't 
be derived from booleans, so they are opted into with ``track=(...)`` and stored 
per fire.
"""

import numpy as np
from .base import Hook
from .cadence import EveryOutput
from ..diagnostics import reconstruct_mask


# User-facing names for the two flip directions, mapped to the integer codes the
# event log stores. The log stays numeric (reconstruct_mask replays on the sign);
# only the query API speaks strings.
_DIRECTIONS = {'unbound': -1, 'bound': +1}


# -- coordinate-dependent reductions (must be tracked; not derivable from deltas) --

def _bound_com(x, mass, mask):
    # nan, deliberately: the COM of nothing is undefined.
    m = mass[mask]
    if m.sum() == 0:
        return np.full(x.shape[1], np.nan)
    return (m[:, None] * x[mask]).sum(0) / m.sum()


def _bound_dispersion(vel, mass, mask):
    m = mass[mask]
    if m.sum() == 0:
        return np.nan                       # see _bound_com
    v_com = (m[:, None] * vel[mask]).sum(0) / m.sum()
    return np.sqrt((m * np.sum((vel[mask] - v_com) ** 2, axis=-1)).sum() / m.sum())


_REDUCTIONS = {
    'com':        lambda c, mask: _bound_com(c.pos(), c.mass, mask),
    'com_vel':    lambda c, mask: _bound_com(c.vel(), c.mass, mask),
    'dispersion': lambda c, mask: _bound_dispersion(c.vel(), c.mass, mask),
}


class BoundednessHook(Hook):
    """Track boundedness of a component over time.

    A particle is bound if its specific energy in the component's COM frame is
    negative (i.e. bound to the component *in isolation*).

    Parameters
    ----------
    component : str
        Component to test for boundedness.
    eps : float
        Softening length [kpc].
    track : sequence of str, optional
        Coordinate reductions to store each fire. Available: ``'com'``,
        ``'com_vel'``, ``'dispersion'``. Each becomes a list attribute of the
        same name (aligned with ``self.t``).
    capture_transitions : sequence of str, optional
        Per-flip payload to attach to each event. Subset of ``('pos', 'vel')``.
    method, theta, max_iter
        Passed to the iterative unbinding solver (must match across hooks to
        share the cached ``bound_mask``).

    Attributes
    ----------
    t : list of float
        Times at which the hook fired.
    initial_mask : (n,) bool array
        Bound mask at the first fire.
    events : list of tuple
        ``(idx, time, direction, *payload)`` per flip; ``direction`` is +1
        (became bound) or -1 (became unbound). Indices are component-local.
        Prefer the ``transitions`` / ``release_time`` accessors over reading this
        directly; they take readable direction names (``'unbound'`` /
        ``'bound'``) and handle rebinding correctly.
    """

    default_cadence = EveryOutput()

    def __init__(self, component, eps, track=(), capture_transitions=(),
                 method='falcON', theta=0.6, max_iter=50):
        for name in track:
            if name not in _REDUCTIONS:
                raise ValueError(
                    f"Unknown track quantity {name!r}. Available: {sorted(_REDUCTIONS)}")
        for name in capture_transitions:
            if name not in ('pos', 'vel'):
                raise ValueError(
                    f"Unknown capture {name!r}. Available: ('pos', 'vel')")

        self.component = component
        self.eps = eps
        self.track = tuple(track)
        self.capture_transitions = tuple(capture_transitions)
        self.method = method
        self.theta = theta
        self.max_iter = max_iter
        self.reset()

    def reset(self):
        self.t = []
        self.initial_mask = None
        self.events = []
        self._prev = None
        for name in self.track:                 # tracked reductions -> list attributes
            setattr(self, name, [])

    def _dedup_key(self):
        # Full intentional config: only an exactly-identical hook is a duplicate.
        return (type(self), self.component, self.eps, self.track,
                self.capture_transitions, self.method, self.theta, self.max_iter)

    def __call__(self, state):
        c = state.component(self.component)
        mask = state.bound_mask(self.component, eps=self.eps, method=self.method,
                                theta=self.theta, max_iter=self.max_iter)
        self.t.append(state.t)

        # transition log
        if self._prev is None:
            self.initial_mask = mask.copy()
        else:
            for i in np.flatnonzero(mask != self._prev):
                direction = 1 if mask[i] else -1
                payload = []
                if 'pos' in self.capture_transitions:
                    payload.append(c.pos()[i].copy())
                if 'vel' in self.capture_transitions:
                    payload.append(c.vel()[i].copy())
                self.events.append((int(i), state.t, direction, *payload))
        self._prev = mask.copy()

        # coordinate reductions (stored; not delta-derivable)
        for name in self.track:
            getattr(self, name).append(_REDUCTIONS[name](c, mask))

        state.report(**{f"n_bound({self.component})": int(mask.sum())})

    # --- derived quantities (computed from the transition log, not stored) --- #

    def mask_at(self, t):
        """
        Bound mask at time *t*.

        Parameters
        ----------
        t : float
            Absolute simulation time [Gyr]. Times before the first fire return
            ``initial_mask``; times after the last return the final state.

        Returns
        -------
        mask : (n,) bool array
            True for particles bound at *t*, indexed component-locally, so it
            lines up with the component's own accessors::

                sim.sat.pos(t)[bh.mask_at(t)]      # the bound particles at t

        Notes
        -----
        * The mask is built by replaying a chronological transition log, so 
        arbitrary times are allowed. However, note that boundedness can change
        between fires so use the fire times (`bh.t`) when possible.

        """
        return reconstruct_mask(self.initial_mask, self.events, t)

    def history(self, times=None):
        """
        Bound mask at each of several times.

        Parameters
        ----------
        times : sequence of float, optional
            Absolute simulation times [Gyr]. Defaults to ``self.t``, the times
            the hook fired. Arbitrary times are allowed; see ``mask_at`` for how they are resolved.

        Returns
        -------
        history : (len(times), n) bool array
            Row *i* is the bound mask at ``times[i]``.

        Notes
        -----
        Each row replays the whole event log, so this costs
        ``O(len(times) x len(events))`` which can be costly
        for dense logs. Prefer ``n_bound``/``fraction`` when you only need a count, or
        ``mask_at`` when you only need one time.
        """
        times = self.t if times is None else times
        return np.array([self.mask_at(t) for t in times])

    def n_bound(self, times=None):
        """
        Number of bound particles at each of several times.

        Parameters
        ----------
        times : sequence of float, optional
            Absolute simulation times [Gyr]. Defaults to ``self.t``.

        Returns
        -------
        n : (len(times),) int array
            Bound particle count at each time. 
            
        Notes
        -----
        * To convert to a bound *mass*, weight by the component's masses via ``mask_at``.
        """
        return self.history(times).sum(axis=1)

    def n_unbound(self, times=None):
        """
        Number of unbound particles at each of several times.

        Parameters
        ----------
        times : sequence of float, optional
            Absolute simulation times [Gyr]. Defaults to ``self.t``.

        Returns
        -------
        n : (len(times),) int array
            Unbound particle count at each time.
        """
        return self.initial_mask.size - self.n_bound(times)

    def fraction(self, times=None):
        """
        Fraction of the component that is bound at each of several times.

        Parameters
        ----------
        times : sequence of float, optional
            Absolute simulation times [Gyr]. Defaults to ``self.t``.

        Returns
        -------
        f : (len(times),) float array
            Bound fraction in [0, 1], relative to the component's **initial
            size** (the number of particles it was created with, not the
            number bound at the first fire).

        Notes
        -----
        If the component is not completely bound to start with, the fraction
        will not begin at 1.

        Examples
        --------
        Visualize the mass-loss as the fraction
        of bound particles with respect to time::

            plt.plot(bh.t, bh.fraction())
        """
        return self.n_bound(times) / self.initial_mask.size

    def release_time(self, t=None):
        """
        Time of each particle's most recent unbinding as of time *t*.


        Parameters
        ----------
        t : float, optional
            Query time. Defaults to the last fire.

        Returns
        -------
        t_release : (n,) float array
            Absolute simulation time [Gyr] at which each particle was
            released. ``nan`` for particles bound at *t*. Particles already unbound at the
            start of the simulation get ``-inf``.
        """
        t = self.t[-1] if t is None else t
        out = np.full(self.initial_mask.size, np.nan)
        out[~self.initial_mask] = -np.inf     # released before we were watching
        for e in self.events:
            idx, te, direction = e[0], e[1], e[2]
            if te <= t:
                out[idx] = te if direction == -1 else np.nan
        return out

    def transition_times(self, direction=None):
        """
        Times at which bound<->unbound transitions occurred.

        Parameters
        ----------
        direction : {'unbound', 'bound'}, optional
            Keep only transitions in this direction. Default (``None``) keeps
            all. See ``transitions`` for the meaning of each.

        Returns
        -------
        times : (n_events,) float array
            Absolute simulation times [Gyr], in chronological order. A time
            appears once per transition, so it repeats when several particles
            flip at the same fire. Empty if nothing flipped.

        Notes
        -----
        * Resolution is the hook's cadence: a flip is dated to the fire that first
        *saw* it, so times are late by up to one fire interval and are drawn from
        ``self.t``.
            
        Examples
        --------
        The mass-loss history represented as a histogram 
        of stripping times::

            plt.hist(bh.transition_times('unbound'), bins=50)
        """
        return np.array([e[1] for e in self.transitions(direction)])

    def transitions(self, direction=None):
        """
        Every recorded boundedness flip, optionally filtered by direction.

        Parameters
        ----------
        direction : {'unbound', 'bound'}, optional
            Keep only transitions in this direction:

            * ``'unbound'``: when particles **became unbound** (the stripping log).
            * ``'bound'``: when particles **became bound** (rejoined the remnant).

            Default (``None``) returns both, interleaved chronologically.

        Returns
        -------
        events : list of tuple
            ``(idx, time, direction, *payload)`` in chronological order:

            * ``idx`` (*int*) **->** Component-local particle index, i.e. an index
              into ``sim.<component>.pos(t)``, *not* into the full system.
            * ``time`` (*float*) **->** Absolute simulation time [Gyr] of the fire
              that observed the flip.
            * ``direction`` (*int*) **->** ``-1`` became unbound, ``+1`` became bound.
            * ``*payload`` **->** the arrays requested by ``capture_transitions`` in
            the order it was provided.

        Notes
        -----
        * A particle may appear many times. To ask "when was each particle
        released, once, accounting for rebinding", use ``release_time``.

        * A transition is dated to the fire that first *saw* it, so times are 
        late by up to one fire interval and are drawn from ``self.t``.

        * Transition velocities are returned in internal units (`[kpc/Gyr]`).

        Examples
        --------
        Where each particle was when it was released, given
        ``capture_transitions=('pos',)``::

            released = bh.transitions('unbound')
            xyz = np.array([e[3] for e in released])
        """
        if direction is None:
            return list(self.events)
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"Unknown direction {direction!r}. Available: "
                f"{sorted(_DIRECTIONS)}, or None for both.")
        want = _DIRECTIONS[direction]
        return [e for e in self.events if e[2] == want]

