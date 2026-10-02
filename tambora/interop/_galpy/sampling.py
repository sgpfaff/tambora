"""galpy distribution functions and potentials, and tambora's profiles, as a
:class:`~tambora.interop.SamplerBackend`."""

import inspect
import warnings

import numpy as np
from galpy import df as _df
from galpy import potential as _gp
from galpy.potential import mass as _mass
from galpy.util.conversion import get_physical, mass_in_msol

from .bridge import _check_physical, _ensure_pot, _get_ro_vo, _iter_components
from .potential import _flatten
from .._backend import SamplerBackend
from ...ic.profiles import PROFILES, Hernquist, King, Plummer
from ...units import G_KPC_KMS, UnitSystem

# galpy's exact isotropic DFs, used when a potential of exactly this type is sampled in itself.
_EXACT_DFS = {
    _gp.PlummerPotential: _df.isotropicPlummerdf,
    _gp.HernquistPotential: _df.isotropicHernquistdf,
}

_RADII = np.logspace(-8, 4, 121)    # [natural units]; 1e4 is eddingtondf's default rmax
_FAR = 1e8                          # [natural units]: "infinity", for the mass check


def _sampling_rmin(d):
    """The inner radius [natural units] inside which a galpy DF's ``sample()`` draws nothing.

    From galpy 1.11, eddingtondf, osipkovmerrittdf and constantbetadf take an ``rmin`` when
    built, and choose one themselves for a potential that is infinitely deep at the centre.
    Their ``sample()`` then uses it unless given another. Other DFs default to 0.
    """
    default = inspect.signature(d.sample).parameters['rmin'].default
    return d._rmin if default is None else default


def _names(pot):
    return '+'.join(type(p).__name__ for p in _iter_components(pot))


def _as_potential(obj):
    """``obj``, a galpy Potential or a (nested) list/tuple of them, as one galpy potential."""
    if not isinstance(obj, (list, tuple)):
        return obj
    items = list(_flatten(obj))
    for i, p in enumerate(items):   # before combining: galpy's `+` recurses forever on a non-potential
        if not isinstance(p, _gp.Potential):
            raise TypeError(f"Expected a galpy Potential (element {i} of the list), "
                            f"got {type(p).__name__}.")
    return items[0] if len(items) == 1 else _ensure_pot(items)


def _rmax(dens):
    """Where to stop sampling ``dens`` [natural units]: once all but 1e-10 of its mass is inside."""
    m = np.array([_mass(dens, r, use_physical=False) for r in _RADII])
    m_far = _mass(dens, _FAR, use_physical=False)
    if not np.isfinite(m_far) or m_far > 1.01 * m[-1]:
        raise ValueError(
            f"Can't sample a {_names(dens)}: its mass is infinite, or so spread out that more "
            f"than 1% of it is beyond R={_RADII[-1]:g} natural units. Use a density with a finite "
            f"mass, or build a galpy DF with an rmax yourself and sample that.")
    converged = 1 - m / m_far < 1e-10
    return _RADII[np.argmax(converged)] if converged.any() else _RADII[-1]


def _df_for(pot):
    """A galpy DF drawing ``pot``'s density in its own potential, and a label for it."""
    pot = _as_potential(pot)
    for p in _iter_components(pot):
        name = type(p).__name__
        if isinstance(p, _gp.KingPotential):
            raise TypeError(f"Can't sample a {name}: it doesn't keep the W0 its distribution "
                            f"function needs. Pass galpy's df.kingdf(W0=..., M=..., rt=...) to "
                            f"ic.sample instead.")
        if isinstance(p, _gp.KeplerPotential):
            raise TypeError(f"Can't sample a {name}: it's a point mass, with no extended "
                            f"density to draw particles from.")
        # galpy's Eddington inversion needs the density's first two radial derivatives.
        if not (hasattr(p, '_ddensdr') and hasattr(p, '_d2densdr2')):
            raise TypeError(
                f"Can't sample a {name}: galpy can only draw from spherical densities that "
                f"define their first two radial derivatives (_ddensdr and _d2densdr2).")
        _check_physical(p)
    ro, vo = _get_ro_vo(pot)
    exact = _EXACT_DFS.get(type(pot))
    if exact is not None:
        return exact(pot=pot, ro=ro, vo=vo), f'{exact.__name__}({_names(pot)})'
    return _df.eddingtondf(pot=pot, rmax=_rmax(pot), ro=ro, vo=vo), f'eddingtondf({_names(pot)})'


def _df_for_profile(profile):
    """The galpy DF for one of tambora's profiles.

    galpy's units are set by the profile itself (its scale radius and the speed
    sqrt(G M / radius)), so galpy works with numbers of order 1 whatever its size.
    """
    ro = profile.rt if isinstance(profile, King) else profile.rscale
    vo = np.sqrt(G_KPC_KMS * profile.M / ro)    # makes M one galpy mass unit
    if isinstance(profile, Plummer):
        return _df.isotropicPlummerdf(pot=_gp.PlummerPotential(amp=1., b=1., ro=ro, vo=vo), ro=ro, vo=vo)
    if isinstance(profile, Hernquist):          # galpy's amp is twice the mass
        return _df.isotropicHernquistdf(pot=_gp.HernquistPotential(amp=2., a=1., ro=ro, vo=vo), ro=ro, vo=vo)
    return _df.kingdf(W0=profile.W0, M=1., rt=1., ro=ro, vo=vo)


class GalpySampler(SamplerBackend):
    """A galpy spherical DF (e.g. ``kingdf``, ``eddingtondf``); a spherical galpy potential,
    whose density is drawn in its own potential, with galpy's exact DF for a Plummer or
    Hernquist and an Eddington-inversion DF otherwise; or one of tambora's profiles
    (:class:`~tambora.ic.Plummer`, :class:`~tambora.ic.Hernquist`, :class:`~tambora.ic.King`)."""

    name = 'galpy'

    @classmethod
    def accepts(cls, obj) -> bool:
        if isinstance(obj, (list, tuple)):
            return any(isinstance(p, _gp.Potential) for p in _flatten(obj))
        return isinstance(obj, (_df.sphericaldf, _gp.Potential) + PROFILES)

    def __init__(self, obj):
        if isinstance(obj, _df.sphericaldf):
            _check_physical(obj)
            if obj._denspot is not obj._pot:
                warnings.warn(
                    f"This {type(obj).__name__} draws a tracer ({_names(obj._denspot)}) in a "
                    f"different potential ({_names(obj._pot)}). galpy's sampler can give such "
                    f"tracers speeds that are too high when the potential is much deeper than "
                    f"the tracer's own. Check the particles' "
                    f"virial ratio in the total potential.")
            self.df, self._label = obj, type(obj).__name__
        elif isinstance(obj, PROFILES):
            self.df, self._label = _df_for_profile(obj), obj.describe()
        else:
            self.df, self._label = _df_for(obj)
        self.obj = obj
        d = self.df
        phys = get_physical(d)
        ro, vo = phys['ro'], phys['vo']
        self.units = UnitSystem(length_kpc=ro, velocity_kms=vo, mass_msun=mass_in_msol(vo, ro))
        if isinstance(d, _df.kingdf):
            # galpy 1.9 integrates the King density to a mass 0.8% off the M it was given.
            mass_nat = d.M
        else:
            mass_nat = _mass(d._denspot, d._rmax, use_physical=False)
            rmin = _sampling_rmin(d)
            if rmin > 0:        # nothing is drawn inside rmin, so its mass isn't handed out
                mass_nat -= _mass(d._denspot, rmin, use_physical=False)
        self._total_mass = float(mass_nat) * self.units.mass_msun

    @property
    def total_mass(self):
        return self._total_mass

    def draw(self, n, seed):
        state = np.random.get_state()        # galpy draws from numpy's global generator:
        np.random.seed(seed)                 # seed it, then give the user theirs back
        try:
            o = self.df.sample(n=n, return_orbit=True)
        finally:
            np.random.set_state(state)
        # Natural units, scaled here, so galpy's astropy-units setting can't change the result.
        pos = np.column_stack([o.x(use_physical=False), o.y(use_physical=False), o.z(use_physical=False)])
        vel = np.column_stack([o.vx(use_physical=False), o.vy(use_physical=False), o.vz(use_physical=False)])
        return pos * self.units.length_kpc, vel * self.units.velocity_kms

    def describe(self) -> str:
        return self._label
