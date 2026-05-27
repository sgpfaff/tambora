from tambora.dynamics.forces import ConservativeForce, BaseForce
from src.tambora.dynamics.forces.CompositeForce import _CompositePlain, _CompositeConservative
import numpy as np
import pytest

class ExampleBaseForce(BaseForce):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def acc(self, pos, vel, mass, t):
        return 2 * pos * vel * self.multiplier

class ExampleConservativeForce(ConservativeForce):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def acc(self, pos, mass, t):
        return 2 * pos * self.multiplier
    def potential(self, pos, mass, t):
        return np.sum(pos**2) * self.multiplier

# --- Composite of BaseForces ------------------------------------------------------ #

def test_sum_of_base_forces_makes_CompositePlain():
    baseForce1 = ExampleBaseForce(multiplier=1)
    baseForce2 = ExampleBaseForce(multiplier=2)
    composite = baseForce1 + baseForce2
    assert isinstance(composite, _CompositePlain)

def test_base_CompositeForce_requires_pos_vel_mass_t():
    baseForce1 = ExampleBaseForce(multiplier=1)
    baseForce2 = ExampleBaseForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    # Should not raise an error
    composite.acc(pos, vel, mass, t)
    pytest.raises(TypeError, composite.acc, pos, mass, t)  # missing vel
    pytest.raises(TypeError, composite.acc, pos, vel, t)  # missing mass
    pytest.raises(TypeError, composite.acc, pos, vel, mass)  # missing t

def test_sum_of_base_forces_equals_composite():
    baseForce1 = ExampleBaseForce(multiplier=1)
    baseForce2 = ExampleBaseForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = baseForce1.acc(pos, vel, mass, t) + baseForce2.acc(pos, vel, mass, t)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

def test_composite_of_base_forces_equals_analytic():
    baseForce1 = ExampleBaseForce(multiplier=1)
    baseForce2 = ExampleBaseForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * vel * (baseForce1.multiplier + baseForce2.multiplier)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

# --- Composite of ConservativeForces ------------------------------------------------------ #

def test_sum_of_conservative_forces_makes_CompositeConservative():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    assert isinstance(composite, _CompositeConservative)

def test_sum_of_conservative_forces_equals_composite():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = conservativeForce1.acc(pos, mass, t) + conservativeForce2.acc(pos, mass, t)
    assert np.allclose(composite.acc(pos, mass, t), expected_acc)

def test_composite_of_conservative_forces_equals_analytic():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * (conservativeForce1.multiplier + conservativeForce2.multiplier)
    assert np.allclose(composite.acc(pos, mass, t), expected_acc)

def test_composite_potential_equals_sum_of_potentials():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_potential = conservativeForce1.potential(pos, mass, t) + conservativeForce2.potential(pos, mass, t)
    assert np.allclose(composite.potential(pos, mass, t), expected_potential)

# --- Composite of BaseForces and ConservativeForces ---------------------------------------------- #

def test_sum_of_mixed_forces_makes_CompositePlain():
    baseForce1 = ExampleBaseForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    assert isinstance(composite, _CompositePlain)

def test_sum_of_mixed_forces_equals_composite():
    baseForce1 = ExampleBaseForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = baseForce1.acc(pos, vel, mass, t) + conservativeForce1.acc(pos, mass, t)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

def test_composite_of_mixed_forces_equals_analytic():
    baseForce1 = ExampleBaseForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * (baseForce1.multiplier * vel + conservativeForce1.multiplier)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)