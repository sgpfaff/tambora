'''galpy bridge functions.

Converts galpy potentials into ezfalcon internal-unit callables
using galpy's module-level ``evaluate*`` functions in natural units.
This removes the astropy dependency from the hot path and works
uniformly across all galpy versions.

On galpy >= 1.11, list inputs are converted to ``CompositePotential``
via the ``+`` operator.  On older versions, lists are passed through
as-is (the expected API at that time).
'''

from functools import reduce
import operator

from galpy.util.coords import rect_to_cyl, cyl_to_rect_vec
from galpy.util.conversion import get_physical
from galpy import potential
from galpy import __version__ as galpy_version
from galpy.potential.WrapperPotential import WrapperPotential as _WrapperPotentialCls
from packaging.version import parse as parse_version
from .units import KMS_TO_KPCGYR
import numpy as np
import warnings

_has_composite = hasattr(potential, 'CompositePotential')

# galpy physical units --> ezfalcon internal units conversion factors
FROM_GALPY_TO_INTERNAL = {
    'pos': 1.0,                 # kpc --> kpc
    'vel': KMS_TO_KPCGYR,       # km/s --> kpc/Gyr 
    'mass': 1.0,                # Msun --> Msun
    'time' : 1.0,               # Gyr --> Gyr
    'pot' : KMS_TO_KPCGYR**2,   # (km/s)^2 --> (kpc/Gyr)^2
    'acc' : KMS_TO_KPCGYR,      # km/s/Gyr --> kpc/Gyr^2
}

def _opt(name):
    """
    Return getattr(potential, name) or None if it 
    doesn't exist in this galpy version.
    """
    return getattr(potential, name, None)

# EllipsoidalPotential subclasses are vectorized in galpy > 1.11.2
_ELLIPSOIDAL_POTENTIALS = (
    potential.PerfectEllipsoidPotential,
    potential.PowerTriaxialPotential,
    potential.TwoPowerTriaxialPotential,
    potential.TriaxialGaussianPotential,
    potential.TriaxialJaffePotential,
    potential.TriaxialHernquistPotential,
    potential.TriaxialNFWPotential,
)

_galpy_has_vectorized_ellipsoidal = parse_version(galpy_version) > parse_version("1.11.2")

VECTORIZED_POTENTIALS = tuple(p for p in (
    # SPHERICAL POTENTIALS
    potential.BurkertPotential,
    potential.DehnenCoreSphericalPotential,
    potential.DehnenSphericalPotential,
    _opt('EinastoPotential'),
    potential.HernquistPotential,
    potential.interpSphericalPotential,
    potential.IsochronePotential,
    potential.JaffePotential,
    potential.KeplerPotential,
    potential.KingPotential,
    potential.NFWPotential,
    potential.PlummerPotential,
    potential.PowerSphericalPotential,
    potential.PowerSphericalPotentialwCutoff,
    potential.PseudoIsothermalPotential,
    potential.TwoPowerSphericalPotential,
    # AXISYMMETRIC POTENTIALS
    potential.FlattenedPowerPotential,
    potential.KuzminDiskPotential,
    potential.KuzminKutuzovStaeckelPotential,
    potential.LogarithmicHaloPotential,
    potential.MiyamotoNagaiPotential,
    potential.MN3ExponentialDiskPotential,
    potential.RingPotential,
    # TRIAXIAL POTENTIALS
    potential.DehnenBarPotential,
    potential.SpiralArmsPotential,
    potential.interpRZPotential,
) if p is not None) + (
    # EllipsoidalPotentials: vectorized in galpy > 1.11.2
    _ELLIPSOIDAL_POTENTIALS if _galpy_has_vectorized_ellipsoidal else ()
)

UNVECTORIZED_POTENTIALS = (
    potential.HomogeneousSpherePotential,
    potential.SphericalShellPotential,
    potential.DoubleExponentialDiskPotential,
    potential.RazorThinExponentialDiskPotential,
    potential.FerrersPotential,
    potential.NullPotential,
    potential.SoftenedNeedleBarPotential,
) + (
    () if _galpy_has_vectorized_ellipsoidal else _ELLIPSOIDAL_POTENTIALS
)

SUPPORTED_WRAPPERS = tuple(p for p in (
    potential.DehnenSmoothWrapperPotential,
    potential.GaussianAmplitudeWrapperPotential,
    potential.SolidBodyRotationWrapperPotential,
    potential.CorotatingRotationWrapperPotential,
    _opt('TimeDependentAmplitudeWrapperPotential'),
    _opt('KuzminLikeWrapperPotential'),
) if p is not None)

UNVECTORIZED_WRAPPERS = tuple(p for p in (
    _opt('RotateAndTiltWrapperPotential'),
) if p is not None)

ALL_SUPPORTED_WRAPPERS = SUPPORTED_WRAPPERS + UNVECTORIZED_WRAPPERS

ALL_SUPPORTED_POTENTIALS = VECTORIZED_POTENTIALS + UNVECTORIZED_POTENTIALS

def _ensure_pot(pot):
    '''Ensure ``pot`` is in the form accepted by galpy's ``evaluate*`` functions.

    - Single ``Potential`` or ``CompositePotential``: returned as-is.
    - ``list``: on galpy >= 1.11 converted to ``CompositePotential`` via
      ``reduce(operator.add, ...)``.  On older galpy, returned as-is
      (lists were the standard composite API).
    '''
    if isinstance(pot, list):
        if _has_composite:
            return reduce(operator.add, pot)
        return pot
    return pot

def _iter_components(pot):
    '''Yield the individual component potentials of *pot*.

    For a single Potential, yields just that potential.
    For a CompositePotential (galpy >= 1.11), yields each member.
    For a list (old galpy), yields each element.
    '''
    if isinstance(pot, list):
        yield from pot
    elif _has_composite and isinstance(pot, potential.CompositePotential):
        yield from pot
    else:
        yield pot

def _unwrap_pot(pot):
    '''Recursively extract the leaf (non-wrapper) potentials from a galpy potential.

    Wrappers store the inner potential in ``_pot``.  This descends through
    nested wrappers and CompositePotentials / lists so that only concrete
    leaf potentials are returned.
    '''
    if isinstance(pot, _WrapperPotentialCls):
        inner = pot._pot
        for p in _iter_components(inner):
            yield from _unwrap_pot(p)
    else:
        yield pot

def _check_supported_pot(pot):
    '''Validate that a galpy potential (or composite) is supported by ezfalcon.

    Wrapper potentials are accepted; their inner (leaf) potentials are
    validated recursively.
    '''
    for p in _iter_components(pot):
        if isinstance(p, _WrapperPotentialCls):
            # Reject unknown wrappers
            if not isinstance(p, tuple(w for w in ALL_SUPPORTED_WRAPPERS if w is not None)):
                raise TypeError(
                    f"{type(p).__name__} is not supported by ezfalcon."
                )
            # Warn for known-but-unvectorized wrappers
            if isinstance(p, tuple(w for w in UNVECTORIZED_WRAPPERS if w is not None)):
                warnings.warn(
                    f"{type(p).__name__} is supported by ezfalcon but not vectorized. "
                    f"Performance may be poor."
                )
            # Validate inner (leaf) potentials
            for leaf in _unwrap_pot(p):
                _check_supported_leaf(leaf)
        else:
            _check_supported_leaf(p)

def _check_supported_leaf(p):
    '''Validate a single non-wrapper galpy potential.'''
    if isinstance(p, UNVECTORIZED_POTENTIALS):
        warnings.warn(
            f"{type(p).__name__} is supported by ezfalcon but not vectorized. "
            f"Performance may be poor."
        )
    elif not isinstance(p, ALL_SUPPORTED_POTENTIALS):
        raise TypeError(
            f"{type(p).__name__} is not supported by ezfalcon. "
            f"Supported potentials: {', '.join(p.__name__ for p in ALL_SUPPORTED_POTENTIALS)}"
        )

def _check_physical(obj):
    '''Warn if a galpy object does not have physical units explicitly set.'''
    if not obj._roSet and not obj._voSet:
        warnings.warn(
            "The provided galpy object does not have physical units explicitly set. "
            "Using galpy defaults (ro=8.0 kpc, vo=220.0 km/s). "
            "Set them explicitly with turn_physical_on(ro=..., vo=...)"
        )

def _get_ro_vo(pot):
    '''Extract ro/vo from a potential.  Warns on inconsistent values.'''
    components = list(_iter_components(pot))
    phys = get_physical(components[0])
    ro, vo = phys['ro'], phys['vo']
    for p in components[1:]:
        pp = get_physical(p)
        if not np.isclose(pp['ro'], ro) or not np.isclose(pp['vo'], vo):
            warnings.warn(
                f"Potential {type(p).__name__} has ro={pp['ro']}, vo={pp['vo']} "
                f"which differs from the first potential (ro={ro}, vo={vo}). "
                f"Using the first potential's values."
            )
            break
    return ro, vo

def _needs_scalar_loop(pot):
    '''Check if any component of pot requires scalar-only evaluation.

    Handles wrappers: an unvectorized wrapper (e.g. RotateAndTilt) or
    a wrapper around an unvectorized inner potential both trigger scalar mode.
    '''
    for p in _iter_components(pot):
        if isinstance(p, _WrapperPotentialCls):
            if isinstance(p, UNVECTORIZED_WRAPPERS):
                return True
            # Check inner leaves
            if any(isinstance(leaf, UNVECTORIZED_POTENTIALS)
                   for leaf in _unwrap_pot(p)):
                return True
        elif isinstance(p, UNVECTORIZED_POTENTIALS):
            return True
    return False

def _galpy_pot_to_pot_fn(pot):
    '''
    Convert a galpy potential to a function that 
    returns potentials in ezfalcon internal units.
    '''
    pot = _ensure_pot(pot)
    ro, vo = _get_ro_vo(pot)
    vo_int = vo * KMS_TO_KPCGYR  # kpc/Gyr
    scalar = _needs_scalar_loop(pot)

    def pot_fn(pos, t):
        R, phi, z = rect_to_cyl(*np.array(pos).T)
        R_nat = R / ro
        z_nat = z / ro
        t_nat = t * vo_int / ro
        if scalar:
            pv = np.array([
                potential.evaluatePotentials(
                    pot, Ri, zi, phi=pi, t=t_nat, use_physical=False
                ) for Ri, zi, pi in zip(R_nat, z_nat, phi)
            ])
        else:
            pv = np.asarray(potential.evaluatePotentials(
                pot, R_nat, z_nat, phi=phi, t=t_nat, use_physical=False
            ))
        return pv * vo_int**2
    return pot_fn

def _galpy_pot_to_acc_fn(pot):
    '''
    Convert a galpy potential to a function that 
    returns accelerations in ezfalcon internal units.
    
    Parameters
    ----------
    pot : galpy potential
        A single galpy Potential, a list of Potentials, or a
        CompositePotential (galpy >=1.11).  Physical units need not
        be turned on; ``get_physical()`` is used to determine ro/vo.
    
    Returns
    -------
    acc_fn : function
        A function ``acc_fn(pos, t)`` that takes Cartesian positions
        ``(N, 3)`` in kpc and time in Gyr, and returns accelerations
        ``(N, 3)`` in kpc/Gyr^2.
    '''
    pot = _ensure_pot(pot)
    ro, vo = _get_ro_vo(pot)
    vo_int = vo * KMS_TO_KPCGYR  # kpc/Gyr
    scalar = _needs_scalar_loop(pot)

    def acc_fn(pos, t):
        R, phi, z = rect_to_cyl(*np.array(pos).T)
        R_nat = R / ro
        z_nat = z / ro
        t_nat = t * vo_int / ro
        kw = dict(phi=phi, t=t_nat, use_physical=False)

        if scalar:
            results = np.array([
                [potential.evaluateRforces(pot, Ri, zi, phi=pi, t=t_nat, use_physical=False),
                 potential.evaluatezforces(pot, Ri, zi, phi=pi, t=t_nat, use_physical=False),
                 potential.evaluatephitorques(pot, Ri, zi, phi=pi, t=t_nat, use_physical=False)]
                for Ri, zi, pi in zip(R_nat, z_nat, phi)
            ])
            Rf = results[:, 0]
            zf = results[:, 1]
            pt = results[:, 2]
        else:
            Rf = np.asarray(potential.evaluateRforces(pot, R_nat, z_nat, **kw))
            zf = np.asarray(potential.evaluatezforces(pot, R_nat, z_nat, **kw))
            pt = np.asarray(potential.evaluatephitorques(pot, R_nat, z_nat, **kw))

        aR = Rf * vo_int**2 / ro          # kpc/Gyr^2
        az = zf * vo_int**2 / ro
        aphi = pt * vo_int**2 / R          # phitorque (energy) / R = force

        ax, ay, az = cyl_to_rect_vec(aR, aphi, az, phi)
        return np.array([ax, ay, az]).T
    return acc_fn
