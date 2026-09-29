"""
Tests for ``TidalTensorGalpyForce``.


Testing Approach
----------------

A point-mass (Kepler) host is the one case with a closed-form tidal tensor, so
it pins down the sign convention -- the thing most likely to be silently
backwards -- against an exact answer rather than against the code's own output.
"""

import numpy as np
import astropy.units as u
from galpy.potential import KeplerPotential

from tambora.dynamics.forces.external_force import TidalTensorGalpyForce
from tambora.units import G_INTERNAL

# Point-mass host: analytic tidal tensor at galactocentric radius R along +x is
# diag(2GM/R^3, -GM/R^3, -GM/R^3) -- radial stretching, transverse compression.
M_GAL = 1e11      # Msun
R_GAL = 20.0      # kpc


def _kepler_force(R=R_GAL, M=M_GAL):
    pot = KeplerPotential(amp=M * u.Msun)
    pot.turn_physical_on()
    return TidalTensorGalpyForce(pot, center=[R, 0.0, 0.0])


def test_tidal_tensor_sign_convention_kepler():
    """Largest eigenvalue is the +2GM/R^3 radial stretching mode."""
    force = _kepler_force()
    T = force.tidal_tensor(np.array([[R_GAL, 0.0, 0.0]]))
    lam = np.linalg.eigvalsh(T)                      # ascending
    np.testing.assert_allclose(lam[-1], 2 * G_INTERNAL * M_GAL / R_GAL**3, rtol=1e-6)
    np.testing.assert_allclose(lam[:2], -G_INTERNAL * M_GAL / R_GAL**3, rtol=1e-6)


def test_potential_is_consistent_with_acc():
    """-grad(phi) == acc, checked by central difference about the centre."""
    force = _kepler_force()
    h = 1e-5
    base = np.array([R_GAL + 0.3, 0.1, -0.2])        # offset from the centre
    grad = np.empty(3)
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        phi_p = force.potential(np.atleast_2d(base + step), 0.0)[0]
        phi_m = force.potential(np.atleast_2d(base - step), 0.0)[0]
        grad[j] = (phi_p - phi_m) / (2 * h)
    acc = force.acc(np.atleast_2d(base), 0.0)[0]
    np.testing.assert_allclose(-grad, acc, rtol=1e-6)
