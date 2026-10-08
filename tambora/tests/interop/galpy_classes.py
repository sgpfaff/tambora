"""Every Potential class galpy has, for tests that should hold for all of them."""

import inspect

import numpy as np
from galpy import potential
from galpy.orbit import Orbit


def satellite_orbit():
    o = Orbit([1., 0.1, 1.1, 0.1, 0., 0.3])
    o.integrate(np.linspace(-1., 1., 101), potential.MWPotential2014)
    return o


# What galpy's defaults don't build, with arguments that do.
BUILT_WITH = {
    'MovingObjectPotential': lambda: potential.MovingObjectPotential(
        satellite_orbit(), pot=potential.PlummerPotential(amp=0.1, b=0.1)),
    'interpRZPotential': lambda: potential.interpRZPotential(
        RZPot=potential.MWPotential2014, rgrid=(0.01, 2., 51), zgrid=(0., 1., 51),
        interpPot=True, interpRforce=True, interpzforce=True),
    'interpSphericalPotential': lambda: potential.interpSphericalPotential(
        rforce=potential.NFWPotential(), rgrid=np.geomspace(0.01, 20., 101)),
    'AdiabaticContractionWrapperPotential': lambda: potential.AdiabaticContractionWrapperPotential(
        pot=potential.NFWPotential(amp=2.), baryonpot=potential.HernquistPotential(amp=0.3, a=0.2)),
    'KuzminLikeWrapperPotential': lambda: potential.KuzminLikeWrapperPotential(
        pot=potential.PlummerPotential(), a=0.5, b=0.1),
    'RotateAndTiltWrapperPotential': lambda: potential.RotateAndTiltWrapperPotential(
        pot=potential.MiyamotoNagaiPotential(), zvec=[0., 0.3, 1.]),
}
# The snapshot potentials need a pynbody snapshot; a CompositePotential is tested as a sum.
NOT_BUILT = {'SnapshotRZPotential', 'InterpSnapshotRZPotential', 'CompositePotential'}

EVERY_GALPY_POTENTIAL = sorted(name for name, c in inspect.getmembers(potential, inspect.isclass)
                               if issubclass(c, potential.Potential) and c is not potential.Potential
                               and name not in NOT_BUILT)


def build(name):
    """galpy's potential ``name``, with its defaults or the arguments in ``BUILT_WITH``."""
    try:
        return BUILT_WITH.get(name, getattr(potential, name))()
    except Exception as e:
        raise AssertionError(f"galpy's {name} doesn't build with its defaults ({type(e).__name__}: {e}); "
                             f"add arguments that do to BUILT_WITH in {__name__}") from e
