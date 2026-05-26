"""Procedural self-gravity API.

Thin wrapper around the :class:`FalcONGravity` / :class:`DirectSummationGravity`
solver classes for one-shot use (e.g. recomputing energy with a different
solver after a run). The integrator uses the solver classes directly.
"""

import numpy as np
from .SelfGravity import FalcONGravity, DirectSummationGravity


SELF_GRAVITY_METHODS = ['direct', 'direct_C', 'falcON']


def _check_eps_array_length(eps, mass):
    if isinstance(eps, np.ndarray) and len(eps) != len(mass):
        raise ValueError(
            f"If 'eps' is an array, it must have the same length as 'mass'. "
            f"Got len(eps)={len(eps)} and len(mass)={len(mass)}."
        )


def _make_self_gravity(method: str, mass, **kwargs):
    """Construct a self-gravity solver from a method-string + kwargs.

    Validates kwargs against the method and raises a clear error on
    unknown / missing ones, mirroring the previous procedural API.
    """
    if method not in SELF_GRAVITY_METHODS:
        raise ValueError(
            f"Unknown method '{method}' for self-gravity. "
            f"Supported methods: {SELF_GRAVITY_METHODS}"
        )

    if method == 'falcON':
        if 'eps' not in kwargs and 'theta' not in kwargs:
            raise ValueError(
                "Must provide 'eps' and 'theta' keyword arguments for falcON method."
            )
        if 'eps' not in kwargs:
            raise ValueError(
                "Must provide 'eps' keyword argument for falcON method."
            )
        extra = set(kwargs) - {'eps', 'theta', 'kernel'}
        if extra:
            raise ValueError(
                f"{extra} is (are) invalid kwarg(s) for 'falcON' self-gravity method. "
                f"Only kwargs for self-gravity methods are allowed."
            )
        eps = kwargs['eps']
        _check_eps_array_length(eps, mass)
        return FalcONGravity(
            eps=eps,
            theta=kwargs.get('theta', 0.6),
            kernel=kwargs.get('kernel', 1),
        )

    # 'direct' or 'direct_C'
    method_label = 'direct' if method == 'direct' else 'direct_C'
    if 'eps' not in kwargs:
        nice_name = 'direct summation' if method == 'direct' else 'direct_C summation'
        raise ValueError(
            f"Must provide 'eps' keyword argument for {nice_name} method."
        )
    extra = set(kwargs) - {'eps'}
    if extra:
        raise ValueError(
            f"{extra} is (are) invalid kwarg(s) for '{method_label}' self-gravity method. "
            f"Only kwargs for self-gravity methods are allowed."
        )
    eps = kwargs['eps']
    _check_eps_array_length(eps, mass)
    return DirectSummationGravity(eps=eps, use_C=(method == 'direct_C'))


def self_gravity(pos, mass, method='falcON', return_potential=True, **kwargs):
    """Compute accelerations and potentials from self-gravity.

    Parameters
    ----------
    pos : (N, 3) array
        Particle positions [kpc].
    mass : (N,) array
        Particle masses [Msun].
    method : str, optional
        ``'falcON'`` (default), ``'direct_C'``, or ``'direct'``.
    return_potential : bool, optional
        If ``True`` (default), return ``(acc, pot)``; else just ``acc``.
    **kwargs
        Solver-specific. For ``'falcON'``: ``eps``, ``theta``, ``kernel``.
        For ``'direct'`` / ``'direct_C'``: ``eps``.

    Returns
    -------
    acc : (N, 3) array
        Accelerations [kpc / Gyr^2].
    pot : (N,) array, optional
        Specific gravitational potential [kpc^2 / Gyr^2]. Only returned if
        ``return_potential`` is ``True``.
    """
    solver = _make_self_gravity(method, mass, **kwargs)
    if return_potential:
        return solver.acc_and_potential(pos, mass)
    return solver.acc(pos, mass)