"""Initial conditions: particles sampled from a model, as a :class:`ParticleSet`."""

from .particles import ParticleSet
from .sampler import sample

__all__ = ['ParticleSet', 'sample']
