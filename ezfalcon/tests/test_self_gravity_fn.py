'''
Test the self_gravity function
'''

import pytest
from ezfalcon.dynamics import self_gravity
import numpy as np


SELF_GRAVITY_METHODS = ['direct', 'falcON']

def test_raises_error_for_unknown_method():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        self_gravity(pos, mass, method='unknown_method')

def test_passes_additional_kwargs_to_falcON():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    # Just test that it doesn't raise an error when we pass additional kwargs
    self_gravity(pos, mass, method='falcON', eps=0.1, theta=0.5)

def test_raises_error_for_no_eps_with_direct():
    '''
    Test that the direct summation method raises an error if eps is not provided.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' keyword argument for direct summation method."):
        self_gravity(pos, mass, method='direct')

def test_raises_error_for_no_theta_with_falcon():
    '''
    Test that falcON defaults theta when only eps is provided (no error).
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    # Should NOT raise — theta defaults to 0.6
    acc, pot = self_gravity(pos, mass, method='falcON', eps=0.1)
    assert acc.shape == (2, 3)
    assert pot.shape == (2,)

def test_raises_error_for_no_eps_with_falcon():
    '''
    Test that error is raised if eps is not provided for falcON.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' keyword argument for falcON method."):
        self_gravity(pos, mass, method='falcON', theta=0.5)

def test_raises_error_for_no_eps_and_theta_with_falcon():
    '''
    Test that error is raised if neither eps nor theta is provided for falcON.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' and 'theta' keyword arguments for falcON method."):
        self_gravity(pos, mass, method='falcON')

# --- return_potential flag -----------------------------------------------------------------------------

def test_return_potential_true_returns_tuple():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = self_gravity(pos, mass, method='direct',eps=0.0, return_potential=True)
    assert isinstance(out, tuple)

def test_return_potential_false_returns_acc_only():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = self_gravity(pos, mass, method='direct',eps=0.0, return_potential=False)
    assert isinstance(out, np.ndarray)

def test_rejects_invalid_kwargs():
    '''
    Test that only kwargs for self-gravity methods are allowed.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    with pytest.raises(ValueError, match="{'invalid_kwarg'} is \(are\) invalid kwarg\(s\) for 'direct' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        self_gravity(pos, mass, method='direct', eps=0.1, invalid_kwarg=123)
    with pytest.raises(ValueError, match="\{'invalid_kwarg'\} is \(are\) invalid kwarg\(s\) for 'falcON' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        self_gravity(pos, mass, method='falcON', eps=0.1, theta=0.5, invalid_kwarg=123)

def test_rejects_unknown_method():
    '''
    Test that an error is raised if an unknown method is provided.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    with pytest.raises(ValueError, match="Unknown method 'unknown_method' for self-gravity. Supported methods: \['direct', 'direct_C', 'falcON'\]"):
        self_gravity(pos, mass, method='unknown_method', eps=0.1)

# --- direct_C method -------------------------------------------------------------------------- #

def test_direct_C_no_eps_raises():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' keyword argument for direct_C summation method."):
        self_gravity(pos, mass, method='direct_C')

def test_direct_C_invalid_kwargs_raises():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="invalid kwarg"):
        self_gravity(pos, mass, method='direct_C', eps=0.1, bad_kwarg=1)

def test_direct_C_eps_array_wrong_length_raises():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="same length as 'mass'"):
        self_gravity(pos, mass, method='direct_C', eps=np.array([0.1, 0.2, 0.3]))

def test_falcon_eps_array_wrong_length_raises():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="same length as 'mass'"):
        self_gravity(pos, mass, method='falcON', eps=np.array([0.1]), theta=0.5)

def test_direct_C_runs_and_matches_direct():
    pos = np.random.normal(size=(5, 3))
    mass = np.abs(np.random.normal(loc=1e9, scale=1e8, size=5))
    acc_c, pot_c = self_gravity(pos, mass, method='direct_C', eps=0.05)
    acc_d, pot_d = self_gravity(pos, mass, method='direct', eps=0.05)
    np.testing.assert_allclose(acc_c, acc_d, rtol=1e-10)
    np.testing.assert_allclose(pot_c, pot_d, rtol=1e-10)