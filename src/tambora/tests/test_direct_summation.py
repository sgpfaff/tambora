import pytest
import numpy as np
from src.tambora.dynamics import DirectSummationGravity
from src.tambora.dynamics.forces.self_gravity.directSummation.directSummation import _direct_summation_py, _direct_summation_C
from src.tambora.util.units import G_INTERNAL

np.random.seed(42)
direct_gravity = DirectSummationGravity(eps=0.0)

def test_newtons_third_law():
    '''
    Test that the direct summation method correctly computes that
    the force on each particle is equal and opposite. This ensures that the
    direct summation method is correctly implementing Newton's third law.
    ''' 
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    mass = np.array([1.0, 10.0])
    acc = direct_gravity.acc(pos, mass)
    assert np.allclose(mass[0] * acc[0], -mass[1] * acc[1], rtol=1e-15)

def test_two_body_acceleration():
    '''
    Test that the direct summation method correctly computes the acceleration
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    acc = direct_gravity.acc(pos, mass)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])**3
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])**3
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_two_body_potential():
    '''
    Test that the direct summation method correctly computes the potential
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    pot = direct_gravity.potential(pos, mass)
    pot_analytical_0 = -G_INTERNAL * mass[1] / np.linalg.norm(pos[1] - pos[0])
    pot_analytical_1 = -G_INTERNAL * mass[0] / np.linalg.norm(pos[0] - pos[1])
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test if returns zero acceleration for a single particle.
    '''
    pos = np.random.normal(size=(1,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(1,))
    acc, pot = direct_gravity.acc_and_potential(pos, mass)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)

def test_acceleration_with_scalar_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    softened_direct_gravity = DirectSummationGravity(eps=eps)
    acc = softened_direct_gravity.acc(pos, mass)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / (np.linalg.norm(pos[1] - pos[0])**2 + eps**2)**1.5
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / (np.linalg.norm(pos[0] - pos[1])**2 + eps**2)**1.5
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_acceleration_with_vector_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = np.array([2.0, 0.0])
    mean_eps = np.mean(eps)
    softened_direct_gravity = DirectSummationGravity(eps=eps)
    acc = softened_direct_gravity.acc(pos, mass)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / (np.linalg.norm(pos[1] - pos[0])**2 + mean_eps**2)**1.5
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / (np.linalg.norm(pos[0] - pos[1])**2 + mean_eps**2)**1.5
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_potential_with_scalar_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening for the potential.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    softened_direct_gravity = DirectSummationGravity(eps=eps)
    pot = softened_direct_gravity.potential(pos, mass)
    pot_analytical_0 = - G_INTERNAL * mass[1] / np.sqrt(np.linalg.norm(pos[1] - pos[0])**2 + eps**2)
    pot_analytical_1 = - G_INTERNAL * mass[0] / np.sqrt(np.linalg.norm(pos[0] - pos[1])**2 + eps**2)
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_potential_with_vector_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = np.array([2.0, 0.0])
    mean_eps = np.mean(eps)
    softened_direct_gravity = DirectSummationGravity(eps=eps)
    pot = softened_direct_gravity.potential(pos, mass)
    pot_analytic_0 = - G_INTERNAL * mass[1] / (np.linalg.norm(pos[1] - pos[0])**2 + mean_eps**2)**0.5
    pot_analytic_1 = - G_INTERNAL * mass[0] / (np.linalg.norm(pos[0] - pos[1])**2 + mean_eps**2)**0.5
    np.testing.assert_allclose(pot[0], pot_analytic_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytic_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test that the direct summation method correctly returns zero acceleration for a single particle.
    '''
    pos = np.random.normal(size=(1,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(1,))
    acc, pot = direct_gravity.acc_and_potential(pos, mass)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)

def test_acc_and_pot_shapes():
    '''
    Test that the direct summation method returns acceleration and potential arrays of the correct shape.
    '''
    pos = np.random.normal(size=(3,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(3,))
    acc, pot = direct_gravity.acc_and_potential(pos, mass)
    assert acc.shape == (3, 3)
    assert pot.shape == (3,)

def test_Direct_Summation_python_matches_C():
    '''
    Test that the direct summation method correctly implements gravitational softening for the potential.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    direct_gravity_py = DirectSummationGravity(eps=eps, use_C=False)
    direct_gravity_C = DirectSummationGravity(eps=eps, use_C=True)
    acc_py, pot_py = direct_gravity_py.acc_and_potential(pos, mass)
    acc_C, pot_C = direct_gravity_C.acc_and_potential(pos, mass)
    np.testing.assert_allclose(acc_py, acc_C, rtol=1e-15)
    np.testing.assert_allclose(pot_py, pot_C, rtol=1e-15)
    pot_analytical_0 = -G_INTERNAL * mass[1] / np.sqrt(np.linalg.norm(pos[1] - pos[0])**2 + eps**2)
    pot_analytical_1 = -G_INTERNAL * mass[0] / np.sqrt(np.linalg.norm(pos[0] - pos[1])**2 + eps**2)
    print(pot_py)
    np.testing.assert_allclose(pot_py[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot_py[1], pot_analytical_1, rtol=1e-15)

# --- _direct_summation_py and _direct_summation_C -----------------------------------------------------------------------------

def test_direct_summation_py_returns_returns_tuple_for_potential_true():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation_py(pos, mass, eps=0.0, return_potential=True)
    assert isinstance(out, tuple)

def test_direct_summation_py_returns_acc_only_when_false():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation_py(pos, mass, eps=0.0, return_potential=False)
    assert isinstance(out, np.ndarray)

def test_direct_summation_C_returns_returns_tuple_for_potential_true():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation_C(pos, mass, eps=0.0, return_potential=True)
    assert isinstance(out, tuple)

def test_direct_summation_C_returns_acc_only_when_false():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation_C(pos, mass, eps=0.0, return_potential=False)
    assert isinstance(out, np.ndarray)