"""
Pure diagnostic computations shared by the live hooks (``StepState``) and the
post-run accessors (``Sim``).
"""

import numpy as np

# The COM frame for the first unbinding pass is seeded from this many of the
# most-bound particles (a fraction of the component, clamped).
_SEED_FRAC = 0.01
_SEED_MIN, _SEED_MAX = 8, 128


def center_of_mass(values, mass):
    """Mass-weighted mean of ``values`` over the particles.

    Parameters
    ----------
    values : (..., N, 3) array
        Positions or velocities of N particles, at one snapshot or several.
    mass : (N,) array
        Masses of the particles.

    Returns
    -------
    (..., 3) array
        The center of mass, or its velocity, at each snapshot.
    """
    return np.sum(mass[:, None] * values, axis=-2) / mass.sum()


def virial_ratio(pos, vel, mass, acc):
    r"""Virial ratio :math:`2K/|W|` about the center of mass: 1 in equilibrium.

    .. math::

        K = \frac{1}{2} \sum_i m_i |v_i - v_{\rm cm}|^2, \qquad
        W = \sum_i m_i (x_i - x_{\rm cm}) \cdot a_i.

    Parameters
    ----------
    pos, vel, acc : (..., N, 3) array
        Positions, velocities and accelerations of N particles, at one snapshot or
        several, in consistent units (e.g. kpc, kpc/Gyr and kpc/Gyr^2).
    mass : (N,) array
        Masses of the particles.

    Returns
    -------
    float or (...,) array
        The virial ratio at each snapshot.
    """
    m = mass[:, None]
    x = pos - center_of_mass(pos, mass)[..., None, :]
    v = vel - center_of_mass(vel, mass)[..., None, :]
    K = 0.5 * np.sum(m * v**2, axis=(-2, -1))
    W = np.sum(m * x * acc, axis=(-2, -1))
    return 2 * K / np.abs(W)


def bound_mask(pos, vel, mass, eps, method='falcON', theta=0.6, max_iter=50):
    """Boolean mask of self-bound particles via iterative unbinding.

    A particle is bound if its specific energy in the cluster's center-of-mass
    frame is negative,

        0.5 |v - v_com|^2 + phi_self < 0,

    where ``phi_self`` is the *specific* self-gravity potential of the currently
    bound set. Particles failing the test are dropped, the COM frame and
    potential are recomputed from those that remain, and the test repeats until
    the set stops shrinking.


    Parameters
    ----------
    pos : (N, 3) array
        Positions [kpc].
    vel : (N, 3) array
        Velocities [kpc / Gyr] (internal units).
    mass : (N,) array
        Masses [Msun].
    eps : float
        Softening length [kpc].
    method : str, optional
        Self-gravity solver: ``'falcON'`` (default), ``'direct'``, ``'direct_C'``.
    theta : float, optional
        falcON opening angle (ignored by the direct methods).
    max_iter : int, optional
        Maximum unbinding iterations.

    Returns
    -------
    bound : (N,) bool array
        True for bound particles.

    Notes
    -----
    The "center" velocity for the first iteration is computed 
    using the most bound particles only to avoid stripped stars
    biasing the result.
    """
    from .forces.self_gravity import self_gravity

    mass = np.asarray(mass)
    kw = dict(eps=eps, theta=theta) if method == 'falcON' else dict(eps=eps)

    bound = np.ones(len(mass), dtype=bool)
    for i in range(max_iter):
        mb = mass[bound]
        vb = vel[bound]
        _, phi = self_gravity(pos[bound], mb, method=method, **kw)   # specific potential

        if i == 0:
            # Seed from the most-bound core.
            k = int(np.clip(len(mb) * _SEED_FRAC, _SEED_MIN, _SEED_MAX))
            k = min(k, len(mb))
            core = np.argpartition(mb * phi, k - 1)[:k]     # energy, not phi
            v_com = (mb[core, None] * vb[core]).sum(0) / mb[core].sum()
        else:
            v_com = (mb[:, None] * vb).sum(0) / mb.sum()

        E = 0.5 * np.sum((vb - v_com) ** 2, axis=-1) + phi
        keep = E < 0
        if keep.all():
            break                          # converged: nothing new unbound
        idx = np.flatnonzero(bound)
        bound[idx[~keep]] = False          # monotonic: once unbound, stays unbound
        if not bound.any():
            break

    return bound


def reconstruct_mask(initial_mask, events, t):
    """Reconstruct a bound mask at time *t* from an initial mask + transitions.

    Parameters
    ----------
    initial_mask : (N,) bool array
        Bound mask at the first fire.
    events : sequence of (idx, time, direction, *payload)
        Transition events; ``direction`` is +1 (became bound) or -1 (unbound).
        Any trailing payload (e.g. captured position) is ignored here.
    t : float
        Query time. The state is reconstructed by replaying every event with
        ``time <= t``.

    Returns
    -------
    mask : (N,) bool array
        Bound mask at time *t* (piecewise-constant between fires).
    """
    m = initial_mask.copy()
    for e in events:
        idx, te, direction = e[0], e[1], e[2]
        if te <= t:
            m[idx] = (direction == +1)
    return m
