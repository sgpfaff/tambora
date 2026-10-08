'''galpy bridge functions.

Converts galpy potentials into tambora internal-unit callables
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
from ...units import KMS_TO_KPCGYR
import numpy as np
import warnings

_has_composite = hasattr(potential, 'CompositePotential')

# galpy physical units --> tambora internal units conversion factors
FROM_GALPY_TO_INTERNAL = {
    'pos': 1.0,                 # kpc --> kpc
    'vel': KMS_TO_KPCGYR,       # km/s --> kpc/Gyr 
    'mass': 1.0,                # Msun --> Msun
    'time' : 1.0,               # Gyr --> Gyr
    'pot' : KMS_TO_KPCGYR**2,   # (km/s)^2 --> (kpc/Gyr)^2
    'acc' : KMS_TO_KPCGYR,      # km/s/Gyr --> kpc/Gyr^2
}

# Where tambora asks, once, whether galpy evaluates a potential on arrays of points as it does
# one point at a time [natural units].
_PROBE_R = np.array([0.3, 0.9, 1.7, 4.0])
_PROBE_Z = np.array([0.1, -0.4, 0.8, -2.0])
_PROBE_PHI = np.array([0.3, 1.9, 4.0, 5.5])

_FORCES = (potential.evaluateRforces, potential.evaluatezforces, potential.evaluatephitorques)

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

def _takes_arrays(p):
    """Whether galpy evaluates the potential ``p`` on arrays of points as it does one point at a time.

    Asked once, at t = 0, at a few points: if galpy's array call fails, or gives other values
    (NaNs counting as equal), ``p`` is evaluated one point at a time. An error evaluating a
    single point is the potential's own, and is raised.
    """
    kw = dict(t=0., use_physical=False)
    for f in (potential.evaluatePotentials, potential.evaluateRforces, potential.evaluatezforces,
              potential.evaluatephitorques):
        one = np.array([f(p, R, z, phi=phi, **kw) for R, z, phi in zip(_PROBE_R, _PROBE_Z, _PROBE_PHI)],
                       dtype=float)
        try:    # an axisymmetric potential's torque is a single 0, for any number of points
            many = np.broadcast_to(np.asarray(f(p, _PROBE_R, _PROBE_Z, phi=_PROBE_PHI, **kw), dtype=float),
                                   one.shape)
        except Exception:
            return False
        if not np.allclose(many, one, rtol=1e-10, atol=0., equal_nan=True):
            return False
    return True


def _galpy_pot_to_fns(pot):
    '''
    Convert a galpy potential to functions that return its accelerations and potentials in
    tambora internal units: ``acc_fn(pos, t)`` and ``pot_fn(pos, t)``, for Cartesian
    positions ``(N, 3)`` in kpc and a time in Gyr.

    Each component that galpy evaluates on arrays of points is evaluated that way, together
    with the others that are; the rest are evaluated one point at a time, with a warning.
    '''
    pot = _ensure_pot(pot)
    ro, vo = _get_ro_vo(pot)
    vo_int = vo * KMS_TO_KPCGYR  # kpc/Gyr
    arrays, points = [], []
    for p in _iter_components(pot):
        if _takes_arrays(p):
            arrays.append(p)
        else:
            warnings.warn(f"galpy can't evaluate {type(p).__name__} on arrays of points, so tambora "
                          f"evaluates it one point at a time, which can be slow.")
            points.append(p)
    together = _ensure_pot(arrays) if arrays else None

    def evaluate(fs, R, z, phi, t):
        '''galpy's functions ``fs`` summed over the components, one row each [natural units].'''
        total = np.zeros((len(fs),) + np.shape(R))
        if together is not None:
            for row, f in zip(total, fs):
                row += np.asarray(f(together, R, z, phi=phi, t=t, use_physical=False))
        for p in points:
            total += np.array([[f(p, Ri, zi, phi=pi, t=t, use_physical=False) for f in fs]
                               for Ri, zi, pi in zip(R, z, phi)]).reshape(-1, len(fs)).T
        return total

    def natural(pos, t):
        R, phi, z = rect_to_cyl(*np.array(pos).T)
        return R, R / ro, z / ro, phi, t * vo_int / ro

    def pot_fn(pos, t):
        _, R_nat, z_nat, phi, t_nat = natural(pos, t)
        return evaluate((potential.evaluatePotentials,), R_nat, z_nat, phi, t_nat)[0] * vo_int**2

    def acc_fn(pos, t):
        R, R_nat, z_nat, phi, t_nat = natural(pos, t)
        Rf, zf, pt = evaluate(_FORCES, R_nat, z_nat, phi, t_nat)

        aR = Rf * vo_int**2 / ro          # kpc/Gyr^2
        az = zf * vo_int**2 / ro
        on_axis = R == 0
        aphi = pt * vo_int**2 / np.where(on_axis, 1.0, R)
        if np.any(on_axis):
            aphi[on_axis] = evaluate((potential.evaluateRforces,), R_nat[on_axis], z_nat[on_axis],
                                     phi[on_axis] + np.pi / 2, t_nat)[0] * vo_int**2 / ro

        ax, ay, az = cyl_to_rect_vec(aR, aphi, az, phi)
        return np.array([ax, ay, az]).T

    return acc_fn, pot_fn


def _galpy_pot_to_pot_fn(pot):
    '''A function that returns ``pot``'s potentials in tambora internal units; see
    :func:`_galpy_pot_to_fns`.'''
    return _galpy_pot_to_fns(pot)[1]


def _galpy_pot_to_acc_fn(pot):
    '''A function that returns ``pot``'s accelerations in tambora internal units; see
    :func:`_galpy_pot_to_fns`.'''
    return _galpy_pot_to_fns(pot)[0]
