"""The particles a sampler returns, ready for :meth:`Sim.add_particles`."""

import dataclasses
from typing import Any, Mapping

import numpy as np


# eq=False: the generated __eq__ would compare arrays and fail with "truth value is ambiguous".
@dataclasses.dataclass(frozen=True, eq=False)
class ParticleSet:
    """Positions, velocities and masses of N particles, and where they came from.
    Parameters
    ----------
    pos : (N, 3) array
        Positions [kpc].
    vel : (N, 3) array
        Velocities [km/s].
    mass : (N,) array
        Masses [Msun].
    meta : mapping, optional
        Where the particles came from, e.g. the sampler, its settings and the seed.
        Default: empty.
    """
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        pos = np.array(self.pos, dtype=float)       # copies, so the caller's arrays stay theirs
        vel = np.array(self.vel, dtype=float)
        mass = np.array(self.mass, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(f"vel must have the same shape as pos, {pos.shape}, got {vel.shape}")
        if mass.shape != (len(pos),):
            raise ValueError(f"mass must have shape ({len(pos)},), got {mass.shape}")
        # The class is frozen, so the checked values go in through object.__setattr__.
        for name, value in (('pos', pos), ('vel', vel), ('mass', mass)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'meta', dict(self.meta) if self.meta is not None else {})

    def __iter__(self):
        yield from (self.pos, self.vel, self.mass)

    def __len__(self):
        return len(self.mass)

    def __repr__(self):
        return f"ParticleSet(n={len(self)}, total mass={self.mass.sum():.4g} Msun, meta={self.meta!r})"

    def shifted(self, pos=(0., 0., 0.), vel=(0., 0., 0.)):
        """A copy moved by ``pos`` [kpc] and ``vel`` [km/s], e.g. to place a cluster on its orbit."""
        pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
        if pos.shape != (3,) or vel.shape != (3,):
            raise ValueError(f"pos and vel must each be one 3-vector, got shapes {pos.shape} and {vel.shape}")
        return dataclasses.replace(self, pos=self.pos + pos, vel=self.vel + vel)
