"""Sampling particles from a model: :func:`sample`, and :func:`sample_components` for several
models in equilibrium together."""

import dataclasses
import operator
from typing import Optional

import numpy as np

from .particles import ParticleSet
from ..interop import sampler_backend_for


def _check_seed(seed) -> int:
    """``seed`` as an int, or a fresh one if it's None."""
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    return int(seed)


def sample(model, n: int, *, potential=None, seed: Optional[int] = None) -> ParticleSet:
    """Draw ``n`` particles from ``model``.

    Each particle gets an equal share of the model's mass, so the set has the model's
    total mass whatever ``n`` is.

    Parameters
    ----------
    model : object
        The model to sample density from. One of tambora's profiles (:class:`Plummer`,
        :class:`Hernquist`, :class:`King`, :class:`TruncatedNFW`), a galpy potential, or a
        galpy distribution function.
    n : int
        Number of particles.
    potential : object, optional
        The potential to draw the ``model``'s density in, if not its own. One of tambora's
        profiles, a galpy potential, or a list of them.
    seed : int, optional
        Seed to use for the draws, in ``[0, 2**32)``. Default: a fresh seed.

    Returns
    -------
    ParticleSet
        With ``meta`` recording the backend, the model and the seed.

    Notes
    -----
    Drawing in another ``potential``, e.g. a star cluster in the potential of itself and
    its dark halo, needs a fix to galpy made after galpy 1.12.0. To use a development
    version of galpy that has it, before a release does, set the environment variable
    ``TAMBORA_GALPY_DEV_TRACERS=1``. A galpy distribution function already has its
    potential, and a :class:`King` is defined by its own, so neither can take one.
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    seed = _check_seed(seed)
    sampler = sampler_backend_for(model, potential=potential)
    pos, vel = sampler.draw(n, seed)
    mass = np.full(n, sampler.total_mass / n)
    return ParticleSet(pos, vel, mass,
                       meta={'backend': sampler.name, 'model': sampler.describe(), 'seed': seed})


def sample_components(components, n, *, seed: Optional[int] = None) -> tuple:
    """Draw several components, each in equilibrium in the potential of them all.

    Parameters
    ----------
    components : list
        The models to draw, each as for :func:`sample`.
    n : list of int
        Number of particles of each component, in the same order.
    seed : int, optional
        Seed in ``[0, 2**32)`` that each component's own seed is drawn from.
        Default: a fresh seed.

    Returns
    -------
    tuple
        A :class:`ParticleSet` for each component, in the order of ``components``. Each
        records ``seed`` in ``meta['parent_seed']``, and its own seed, which redraws it with
        :func:`sample`, in ``meta['seed']``.

    Notes
    -----
    Needs the same galpy as :func:`sample` with a ``potential``.

    Examples
    --------
    The stars and dark matter of a dwarf galaxy::

        stars, dm = ic.sample_components(
            [ic.Plummer(M=1e6, rscale=0.3), ic.Hernquist(M=1e9, rscale=3.)],
            n=[20_000, 100_000], seed=1)
    """
    if not isinstance(components, (list, tuple)):
        raise TypeError(f"components must be a list of models, got {type(components).__name__}")
    if not components:
        raise ValueError("components is empty")
    if not isinstance(n, (list, tuple)):
        raise TypeError(f"n must be a list of particle numbers, one per component, got {type(n).__name__}")
    if len(n) != len(components):
        raise ValueError(f"n must give one number per component: got {len(n)} for {len(components)} components")
    seed = _check_seed(seed)
    total = list(components)
    children = np.random.SeedSequence(seed).spawn(len(components))
    parts = []
    for model, n_model, child in zip(components, n, children):
        ps = sample(model, n_model, potential=total, seed=int(child.generate_state(1)[0]))
        parts.append(dataclasses.replace(ps, meta={**ps.meta, 'parent_seed': seed}))
    return tuple(parts)
