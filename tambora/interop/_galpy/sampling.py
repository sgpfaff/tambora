"""galpy distribution functions and potentials, and tambora's profiles, as a
:class:`~tambora.interop.SamplerBackend`, including densities drawn in another potential."""

import inspect
import os
import warnings

import numpy as np
from galpy import __version__ as _galpy_version
from galpy import df as _df
from galpy import potential as _gp
from galpy.potential import mass as _mass
from galpy.potential.SphericalPotential import SphericalPotential as _SphericalPotential
from galpy.util.conversion import get_physical, mass_in_msol
from packaging.version import parse as _parse_version

from .bridge import _check_physical, _ensure_pot, _iter_components
from .potential import _flatten
from .._backend import SamplerBackend
from ...ic.profiles import PROFILES, Hernquist, King, Plummer, TruncatedNFW
from ...units import G_KPC_KMS, UnitSystem

# galpy's exact isotropic DFs, used when a potential of exactly this type is sampled in itself.
_EXACT_DFS = {
    _gp.PlummerPotential: _df.isotropicPlummerdf,
    _gp.HernquistPotential: _df.isotropicHernquistdf,
}

# Spherical potentials that aren't SphericalPotential subclasses in galpy: these, and any
# SphericalPotential, can be the potential a density is drawn in.
_ALSO_SPHERICAL = (
    _gp.PlummerPotential, _gp.TwoPowerSphericalPotential, _gp.PowerSphericalPotential,
    _gp.PowerSphericalPotentialwCutoff, _gp.IsochronePotential, _gp.HomogeneousSpherePotential,
    _gp.PseudoIsothermalPotential,
)

# Up to 1.12.0, galpy's sampler draws a density in a much deeper potential than its own far too
# fast (virial ratios up to 1.6). galpy's main branch has part of the fix (#1568) and the rest is
# to follow; this is the first galpy release with all of it, None until there is one.
_TRACERS_RELEASE = None

# galpy's development versions don't say whether they have a fix (main reads 1.12.1.dev0 before
# and after it), so tambora draws tracers with one only if this environment variable is 1.
_DEV_TRACERS_ENV = 'TAMBORA_GALPY_DEV_TRACERS'

# Since galpy 1.10, its King models fail for W0 above 45.7 ("x must be increasing").
_KING_MAX_W0 = 45.

_RADII = np.logspace(-8, 4, 121)    # [natural units]; 1e4 is eddingtondf's default rmax
_FAR = 1e8                          # [natural units]: "infinity", for the mass check


def _galpy_has(release):
    """Whether this galpy is ``release`` or later (never, while ``release`` is None)."""
    return release is not None and _parse_version(_galpy_version) >= _parse_version(release)


def _dev_galpy():
    """Whether this galpy is a development version after 1.12.0, which may have the fix."""
    version = _parse_version(_galpy_version)
    return version.is_devrelease and version > _parse_version("1.12.0")


def _tracers_fixed():
    """Whether galpy draws a density in another potential right, as far as tambora knows: a
    release with the fix does, and so may a development version the user vouches for."""
    if _galpy_has(_TRACERS_RELEASE):
        return True
    return _dev_galpy() and os.environ.get(_DEV_TRACERS_ENV) == '1'


def _warn_if_unchecked():
    """Warn, if galpy is a development version the user vouches for, that tambora can't check it."""
    if _tracers_fixed() and not _galpy_has(_TRACERS_RELEASE):
        warnings.warn(f"Drawing a density in another potential with a development version of galpy "
                      f"({_galpy_version}), as {_DEV_TRACERS_ENV}=1 asks. tambora can't tell whether "
                      f"this galpy has the fix; without it, the particles are too fast. Check their "
                      f"virial ratio in the total potential.")


def _tracers_unfixed():
    """The error for a galpy that can't draw a density in another potential."""
    needs = (f"galpy {_TRACERS_RELEASE} or later" if _TRACERS_RELEASE is not None
             else "a fix to galpy that's in no release yet")
    hint = (f" This development version of galpy may have the fix: set {_DEV_TRACERS_ENV}=1 to "
            f"use it." if _dev_galpy() else "")
    return ImportError(f"Drawing a density in another potential needs {needs} (this is galpy "
                       f"{_galpy_version}).{hint}")


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


def _indexed(obj, path=()):
    """Each item of ``obj``, one or a nested list/tuple of them, with its indices in it."""
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _indexed(item, path + (i,))
    else:
        yield path, obj


def _where(path):
    """Which element of a list the indices ``path`` point to: 'element 1', 'element [1][0]'."""
    return f"element {path[0] if len(path) == 1 else ''.join(f'[{i}]' for i in path)} of the list"


def _as_potential(obj):
    """``obj``, a galpy Potential or a (nested) list/tuple of them, as one galpy potential."""
    if not isinstance(obj, (list, tuple)):
        return obj
    for path, p in _indexed(obj):   # before combining: galpy's `+` recurses forever on a non-potential
        if isinstance(p, PROFILES):
            raise TypeError(f"Can't draw a list of models with tambora's profiles in its own "
                            f"potential ({_where(path)} is a {type(p).__name__}). To draw "
                            f"several models in equilibrium together, use ic.sample_components.")
        if not isinstance(p, _gp.Potential):
            raise TypeError(f"Expected a galpy Potential ({_where(path)}), got {type(p).__name__}.")
    items = list(_flatten(obj))
    _common_units(items)    # before combining, which galpy checks only with an assert
    return items[0] if len(items) == 1 else _ensure_pot(items)


def _rmax(dens):
    """Where to stop sampling ``dens`` [natural units]: once all but 1e-10 of its mass is inside."""
    m = np.array([_mass(dens, r, use_physical=False) for r in _RADII])
    m_far = _mass(dens, _FAR, use_physical=False)
    if not np.isfinite(m_far) or m_far > 1.01 * m[-1]:
        hint = ("For an NFW halo, use ic.TruncatedNFW, or truncate yours with galpy's "
                "ExpTruncNFWPotential.from_nfw(nfw, rc=...) (galpy 1.12 or later). "
                if any(isinstance(p, _gp.NFWPotential) for p in _iter_components(dens)) else "")
        raise ValueError(
            f"Can't sample a {_names(dens)}: its mass is infinite, or so spread out that more "
            f"than 1% of it is beyond R={_RADII[-1]:g} natural units. {hint}Use a density with a "
            f"finite mass.")
    converged = 1 - m / m_far < 1e-10
    return _RADII[np.argmax(converged)] if converged.any() else _RADII[-1]


def _check_density(p):
    """Raise unless galpy can draw particles from the density of the potential ``p``."""
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


def _df_for(pot):
    """A galpy DF drawing ``pot``'s density in its own potential, and a label for it."""
    pot = _as_potential(pot)
    for p in _iter_components(pot):
        _check_density(p)
        _check_physical(p)
    ro, vo = _common_units([pot])
    exact = _EXACT_DFS.get(type(pot))
    if exact is not None:
        return exact(pot=pot, ro=ro, vo=vo), f'{exact.__name__}({_names(pot)})'
    return _df.eddingtondf(pot=pot, rmax=_rmax(pot), ro=ro, vo=vo), f'eddingtondf({_names(pot)})'


def _profile_units(profile):
    """galpy units [kpc, km/s] set by a profile itself: its radius, and sqrt(G M / radius),
    which makes M one galpy mass unit. galpy then works with numbers of order 1, whatever
    the profile's size."""
    ro = profile.rt if isinstance(profile, King) else profile.rscale
    return ro, float(np.sqrt(G_KPC_KMS * profile.M / ro))


def _profile_potential(profile, ro, vo):
    """The galpy potential of one of tambora's profiles, in galpy units ``ro`` [kpc], ``vo`` [km/s]."""
    M = profile.M / mass_in_msol(vo, ro)
    if isinstance(profile, Plummer):
        return _gp.PlummerPotential(amp=M, b=profile.rscale / ro, ro=ro, vo=vo)
    if isinstance(profile, Hernquist):          # galpy's amp is twice the mass
        return _gp.HernquistPotential(amp=2. * M, a=profile.rscale / ro, ro=ro, vo=vo)
    if isinstance(profile, TruncatedNFW):
        if not hasattr(_gp, 'ExpTruncNFWPotential'):
            raise ImportError(f"Sampling a TruncatedNFW needs galpy 1.12 or later, for its "
                              f"ExpTruncNFWPotential; this is galpy {_galpy_version}.")
        return _gp.ExpTruncNFWPotential(mass=M, a=profile.rscale / ro, rc=profile.rtrunc / ro, ro=ro, vo=vo)
    _check_king(profile)
    return _gp.KingPotential(W0=profile.W0, M=M, rt=profile.rt / ro, ro=ro, vo=vo)


def _check_king(profile):
    """Raise if galpy can't build the King model ``profile``."""
    if profile.W0 > _KING_MAX_W0:
        raise ValueError(f"Can't use {profile.describe()}: galpy's King models fail for W0 above "
                         f"about {_KING_MAX_W0:g} (since galpy 1.10).")


def _check_profile(profile):
    """Raise if galpy can't draw particles from one of tambora's profiles."""
    if isinstance(profile, TruncatedNFW):
        ratio = profile.rtrunc / profile.rscale
        if ratio < 0.2:     # measured: galpy fails, or samples out of equilibrium, below this
            raise ValueError(f"Can't sample {profile.describe()}: galpy's sampler is unreliable "
                             f"when rtrunc is under a fifth of rscale (here {ratio:.3g} of it).")


def _df_for_profile(profile):
    """The galpy DF for one of tambora's profiles, in its own potential."""
    ro, vo = _profile_units(profile)
    if isinstance(profile, King):
        _check_king(profile)
        return _df.kingdf(W0=profile.W0, M=1., rt=1., ro=ro, vo=vo)
    pot = _profile_potential(profile, ro, vo)
    _check_profile(profile)
    return _df_for(pot)[0]


def _parts(obj):
    return list(_flatten(obj)) if isinstance(obj, (list, tuple)) else [obj]


def _label(obj):
    return ' + '.join(p.describe() if isinstance(p, PROFILES) else _names(p) for p in _parts(obj))


def _members(obj, what):
    """The galpy potentials and tambora profiles in ``obj`` (one, or a nested list of them)."""
    items = _parts(obj)
    if not items:
        raise ValueError(f"{what} is an empty list.")
    for path, p in _indexed(obj):
        if not isinstance(p, (_gp.Potential,) + PROFILES):
            where = f" ({_where(path)})" if path else ""
            raise TypeError(f"{what} takes galpy potentials and tambora's profiles, got "
                            f"{type(p).__name__}{where}.")
    return items


def _common_units(natives):
    """The galpy units [kpc, km/s] shared by galpy potentials, which must all have the same ones.

    galpy checks this when combining potentials, but with an assert, which ``python -O`` drops,
    and galpy before 1.11 doesn't combine a list, so doesn't check it at all.
    """
    units = {(get_physical(leaf)['ro'], get_physical(leaf)['vo'])
             for p in natives for leaf in _iter_components(p)}
    if len(units) > 1:
        raise ValueError(f"The galpy potentials have different units (ro, vo): {sorted(units)}. "
                         f"Give them all the same ro and vo.")
    return units.pop()


def _same(a, b):
    return a is b or (isinstance(a, PROFILES) and a == b)


def _is_model(potential, model):
    """Whether ``potential`` is just ``model`` itself, in any order, so the model is in its own
    potential."""
    rest = _parts(model)
    for p in _parts(potential):
        match = next((i for i, q in enumerate(rest) if _same(p, q)), None)
        if match is None:
            return False
        del rest[match]
    return not rest


def _tracer_df(model, potential):
    """A galpy DF drawing ``model``'s density in equilibrium in ``potential``, and a label for it.

    ``model`` and ``potential`` may mix tambora's profiles and galpy potentials. Profiles are
    built in the galpy potentials' units if there are any, and otherwise in the model's own.
    """
    if isinstance(model, King):
        raise TypeError("A King model is defined by its own potential, so it can't be drawn in "
                        "another one.")
    parts = _members(model, "The model")
    members = _members(potential, "The potential")
    natives = [p for p in parts + members if isinstance(p, _gp.Potential)]
    for p in natives:
        for leaf in _iter_components(p):
            _check_physical(leaf)
    ro, vo = _common_units(natives) if natives else _profile_units(parts[0])
    as_galpy = lambda p: _profile_potential(p, ro, vo) if isinstance(p, PROFILES) else p
    dens = _as_potential([as_galpy(p) for p in parts])
    total = _as_potential([as_galpy(p) for p in members])
    for p in parts:
        if isinstance(p, PROFILES):
            _check_profile(p)
    for p in _iter_components(dens):
        _check_density(p)
    for p in _iter_components(total):
        if not isinstance(p, (_SphericalPotential,) + _ALSO_SPHERICAL):
            raise TypeError(f"Can't sample in a {type(p).__name__}: galpy's samplers need a "
                            f"spherical potential.")
    rmax = _rmax(dens)
    if not _tracers_fixed():    # last, so that a mistake isn't taken for a need for newer galpy
        raise _tracers_unfixed()
    _warn_if_unchecked()
    d = _df.eddingtondf(pot=total, denspot=dens, rmax=rmax, ro=ro, vo=vo)
    return d, f"{_label(model)} in {_label(potential)}"


class GalpySampler(SamplerBackend):
    """A galpy spherical DF (e.g. ``kingdf``, ``eddingtondf``); a spherical galpy potential,
    whose density is drawn in its own potential, with galpy's exact DF for a Plummer or
    Hernquist and an Eddington-inversion DF otherwise; or one of tambora's profiles
    (:class:`~tambora.ic.Plummer`, :class:`~tambora.ic.Hernquist`, :class:`~tambora.ic.King`,
    :class:`~tambora.ic.TruncatedNFW`).

    Given a ``potential``, the model's density is drawn in it instead, with an Eddington DF.
    This needs a galpy release with the fix made after 1.12.0, or a development version of
    galpy that has it and ``TAMBORA_GALPY_DEV_TRACERS=1`` in the environment.
    """

    name = 'galpy'

    @classmethod
    def accepts(cls, obj) -> bool:
        if isinstance(obj, (list, tuple)):
            return any(isinstance(p, (_gp.Potential,) + PROFILES) for p in _flatten(obj))
        return isinstance(obj, (_df.sphericaldf, _gp.Potential) + PROFILES)

    def __init__(self, obj, potential=None):
        if potential is not None and _is_model(potential, obj):
            potential = None
        if isinstance(obj, _df.sphericaldf):
            if potential is not None:
                raise TypeError("A galpy DF has its potential already, so it can't be drawn in "
                                "another one. A galpy potential's or a profile's density can.")
            _check_physical(obj)
            if obj._denspot is not obj._pot:
                if not _tracers_fixed():
                    warnings.warn(
                        f"This {type(obj).__name__} draws a tracer ({_names(obj._denspot)}) in a "
                        f"different potential ({_names(obj._pot)}). galpy {_galpy_version} can give "
                        f"such tracers speeds that are too high when the potential is much deeper "
                        f"than the tracer's own. Check the particles' virial ratio in the total "
                        f"potential.")
                _warn_if_unchecked()
            self.df, self._label = obj, type(obj).__name__
        elif potential is not None:
            self.df, self._label = _tracer_df(obj, potential)
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
