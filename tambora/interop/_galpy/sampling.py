"""galpy distribution functions as a :class:`~tambora.interop.SamplerBackend`."""

import inspect

import numpy as np
from galpy import df as _df
from galpy.potential import mass as _mass
from galpy.util.conversion import get_physical, mass_in_msol

from .bridge import _check_physical
from .._backend import SamplerBackend
from ...units import UnitSystem


def _sampling_rmin(d):
    """The inner radius [natural units] inside which a galpy DF's ``sample()`` draws nothing.

    From galpy 1.11, eddingtondf, osipkovmerrittdf and constantbetadf take an ``rmin`` when
    built, and choose one themselves for a potential that is infinitely deep at the centre.
    Their ``sample()`` then uses it unless given another. Other DFs default to 0.
    """
    default = inspect.signature(d.sample).parameters['rmin'].default
    return d._rmin if default is None else default


class GalpySampler(SamplerBackend):
    """A galpy spherical distribution function, e.g. ``kingdf``, ``isotropicPlummerdf``, or
    ``eddingtondf`` (which can also sample a tracer density in a different total potential)."""

    name = 'galpy'

    @classmethod
    def accepts(cls, obj) -> bool:
        return isinstance(obj, _df.sphericaldf)

    def __init__(self, obj):
        _check_physical(obj)
        self.obj = obj
        phys = get_physical(obj)
        ro, vo = phys['ro'], phys['vo']
        self.units = UnitSystem(length_kpc=ro, velocity_kms=vo, mass_msun=mass_in_msol(vo, ro))
        if isinstance(obj, _df.kingdf):
            # galpy 1.9 integrates the King density to a mass 0.8% off the M it was given.
            mass_nat = obj.M
        else:
            mass_nat = _mass(obj._denspot, obj._rmax, use_physical=False)
            rmin = _sampling_rmin(obj)
            if rmin > 0:        # nothing is drawn inside rmin, so its mass isn't handed out
                mass_nat -= _mass(obj._denspot, rmin, use_physical=False)
        self._total_mass = float(mass_nat) * self.units.mass_msun

    @property
    def total_mass(self):
        return self._total_mass

    def draw(self, n, seed):
        state = np.random.get_state()        # galpy draws from numpy's global generator:
        np.random.seed(seed)                 # seed it, then give the user theirs back
        try:
            o = self.obj.sample(n=n, return_orbit=True)
        finally:
            np.random.set_state(state)
        # Natural units, scaled here, so galpy's astropy-units setting can't change the result.
        pos = np.column_stack([o.x(use_physical=False), o.y(use_physical=False), o.z(use_physical=False)])
        vel = np.column_stack([o.vx(use_physical=False), o.vy(use_physical=False), o.vz(use_physical=False)])
        return pos * self.units.length_kpc, vel * self.units.velocity_kms
