"""Sampling particles from a model: :func:`sample`."""

import operator
from typing import Optional

import numpy as np

from .particles import ParticleSet
from ..interop import sampler_backend_for


def sample(model, n: int, *, seed: Optional[int] = None, backend: Optional[str] = None) -> ParticleSet:
    """Draw ``n`` particles from ``model``.

    Each particle gets an equal share of the model's mass, so the set has the model's
    total mass whatever ``n`` is.

    Parameters
    ----------
    model : object
        A model from a supported package: a galpy potential, whose density is drawn in
        equilibrium in its own potential, or a galpy distribution function such as
        ``galpy.df.kingdf(...)``.
    n : int
        Number of particles.
    seed : int, optional
        Seed in ``[0, 2**32)``; the same seed gives the same particles. By default a
        fresh one is drawn. Either way it is recorded in ``meta['seed']``.
    backend : str, optional
        Name of the backend to use, e.g. ``'galpy'``. By default it is chosen from ``model``.

    Returns
    -------
    ParticleSet
        With ``meta`` recording the backend, the model and the seed.
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    elif not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    sampler = sampler_backend_for(model, backend)
    pos, vel = sampler.draw(n, int(seed))
    mass = np.full(n, sampler.total_mass / n)
    return ParticleSet(pos, vel, mass,
                       meta={'backend': sampler.name, 'model': sampler.describe(), 'seed': int(seed)})
