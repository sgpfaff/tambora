#-----------------------#
#    pyfalcon tests     #
#-----------------------#

import pytest
import numpy as np
from src.tambora.dynamics import FalcONGravity, DirectSummationGravity
from src.tambora.dynamics.forces.self_gravity.falcON.falcON import _falcON_gravity
from src.tambora.util.units import G_INTERNAL

np.random.seed(42)
falcON_gravity = FalcONGravity(eps=0.0, theta=0.5, kernel=0)
direct_gravity = DirectSummationGravity(eps=0.0)

def test_newtons_third_law():
    '''
    Test that falcON correctly computes that
    the force on each particle is equal and opposite. This ensures that the
    direct summation method is correctly implementing Newton's third law.
    ''' 
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    mass = np.array([1.0, 10.0])
    acc = falcON_gravity.acc(pos, mass)
    assert np.allclose(mass[0] * acc[0], -mass[1] * acc[1], rtol=1e-15)

def test_two_body_acceleration():
    '''
    Test that falcON correctly computes the acceleration
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    acc = falcON_gravity.acc(pos, mass)
    acc_analytic_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])**3
    acc_analytic_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])**3
    np.testing.assert_allclose(acc[0], acc_analytic_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytic_1, rtol=1e-15)

def test_two_body_potential():
    '''
    Test that falcON correctly computes the potential
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    pot = falcON_gravity.potential(pos, mass)
    pot_analytic_0 = -G_INTERNAL * mass[1] / np.linalg.norm(pos[1] - pos[0])
    pot_analytic_1 = -G_INTERNAL * mass[0] / np.linalg.norm(pos[0] - pos[1])
    np.testing.assert_allclose(pot[0], pot_analytic_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytic_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test if returns zero acceleration for a single particle.
    '''
    pos = np.random.normal(size=(1,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(1,))
    acc, pot = falcON_gravity.acc_and_potential(pos, mass)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)

def test_acceleration_with_scalar_softening_length():
    '''
    Test that falcON correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    softened_falcON = FalcONGravity(eps=eps, kernel=0)
    acc = softened_falcON.acc(pos, mass)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / (np.linalg.norm(pos[1] - pos[0])**2 + eps**2)**1.5
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / (np.linalg.norm(pos[0] - pos[1])**2 + eps**2)**1.5
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_acceleration_with_vector_softening_length():
    '''
    Test that falcON correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = np.array([2.0, 0.0])
    mean_eps = np.mean(eps)
    softened_falcON = FalcONGravity(eps=eps, kernel=0)
    acc = softened_falcON.acc(pos, mass)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / (np.linalg.norm(pos[1] - pos[0])**2 + mean_eps**2)**1.5
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / (np.linalg.norm(pos[0] - pos[1])**2 + mean_eps**2)**1.5
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_potential_with_scalar_softening_length():
    '''
    Test that falcON correctly implements gravitational softening for the potential.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    softened_falcON = FalcONGravity(eps=eps, kernel=0)
    pot = softened_falcON.potential(pos, mass)
    pot_analytical_0 = - G_INTERNAL * mass[1] / np.sqrt(np.linalg.norm(pos[1] - pos[0])**2 + eps**2)
    pot_analytical_1 = - G_INTERNAL * mass[0] / np.sqrt(np.linalg.norm(pos[0] - pos[1])**2 + eps**2)
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_potential_with_vector_softening_length():
    '''
    Test that falcON method correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = np.array([2.0, 0.0])
    mean_eps = np.mean(eps)
    softened_falcON = FalcONGravity(eps=eps, kernel=0)
    pot = softened_falcON.potential(pos, mass)
    pot_analytic_0 = - G_INTERNAL * mass[1] / (np.linalg.norm(pos[1] - pos[0])**2 + mean_eps**2)**0.5
    pot_analytic_1 = - G_INTERNAL * mass[0] / (np.linalg.norm(pos[0] - pos[1])**2 + mean_eps**2)**0.5
    np.testing.assert_allclose(pot[0], pot_analytic_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytic_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test that falcON correctly returns zero acceleration for a single particle.
    '''
    pos = np.random.normal(size=(1,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(1,))
    acc, pot = falcON_gravity.acc_and_potential(pos, mass)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)

def test_acc_and_pot_shapes():
    '''
    Test that falcON returns acceleration and potential arrays of the correct shape.
    '''
    pos = np.random.normal(size=(3,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(3,))
    acc, pot = falcON_gravity.acc_and_potential(pos, mass)
    assert acc.shape == (3, 3)
    assert pot.shape == (3,)

# --- _direct_summation_py and _direct_summation_C -----------------------------------------------------------------------------

def test_falcON_gravity_returns_returns_tuple_for_potential_true():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _falcON_gravity(pos, mass, eps=0.0, theta=0.5, kernel=0, return_potential=True)
    assert isinstance(out, tuple)

def test_falcON_gravity_returns_acc_only_when_false():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _falcON_gravity(pos, mass, eps=0.0, theta=0.5, kernel=0, return_potential=False)
    assert isinstance(out, np.ndarray)

### Test against Direct Summation ------------------------------------------------------------------------------ #

def test_acceleration_against_direct():
    '''
    Test that the falcON method correctly computes the acceleration
    against direct summation.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    acc = falcON_gravity.acc(pos, mass)
    acc_direct = direct_gravity.acc(pos, mass)
    np.testing.assert_allclose(acc, acc_direct, rtol=1e-15)

def test_potential_against_direct():
    '''
    Test that the falcON method correctly computes the potential
    against direct summation.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    pot = falcON_gravity.potential(pos, mass)
    pot_direct = direct_gravity.potential(pos, mass)
    np.testing.assert_allclose(pot, pot_direct, rtol=1e-15)