"""
Tests for the ``SelfGravityForce`` base class's adapter shims.
"""

import numpy as np
import pytest

from tambora.dynamics.forces.self_gravity.SelfGravity import SelfGravityForce


_POS = np.array([[1., 2., 3.], [4., 5., 6.]])
_MASS = np.array([10., 20.])


class _CountingSolver(SelfGravityForce):
    """Minimal SelfGravityForce: implements only the two required methods.

    Deliberately does NOT override acc_and_potential, so the base class's
    two-sweep fallback is what runs. Counts calls so the fallback's cost is
    visible to the tests.
    """

    def __init__(self):
        self.acc_calls = 0
        self.pot_calls = 0

    def acc(self, pos, mass, eps=None):
        self.acc_calls += 1
        return pos * mass[:, None]

    def potential(self, pos, mass, eps=None):
        self.pot_calls += 1
        return np.sum(pos, axis=-1) * mass


def test_the_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SelfGravityForce()


def test_acc_and_potential_defaults_to_calling_both():
    # The unoptimised fallback: correct, at the cost of two sweeps.
    s = _CountingSolver()
    a, p = s.acc_and_potential(_POS, _MASS)
    np.testing.assert_array_equal(a, s.acc(_POS, _MASS))
    np.testing.assert_array_equal(p, s.potential(_POS, _MASS))
    assert s.acc_calls == 2
    assert s.pot_calls == 2


def test_the_default_acc_and_potential_really_does_sweep_twice():
    # Documents why the shipped solvers override it: the base class has no way
    # to share work between acc and potential.
    s = _CountingSolver()
    s.acc_and_potential(_POS, _MASS)
    assert (s.acc_calls, s.pot_calls) == (1, 1)


def test_eval_acc_drops_vel_and_t():
    # The integrator always calls _eval_acc(pos, vel, mass, t); self-gravity
    # depends on neither vel nor t, so the adapter must discard them rather than
    # forward them into acc(pos, mass) and raise TypeError.
    s = _CountingSolver()
    np.testing.assert_array_equal(
        s._eval_acc(_POS, vel=np.ones_like(_POS), mass=_MASS, t=99.0),
        s.acc(_POS, _MASS))


def test_eval_potential_drops_vel_and_t():
    s = _CountingSolver()
    np.testing.assert_array_equal(
        s._eval_potential(_POS, vel=np.ones_like(_POS), mass=_MASS, t=99.0),
        s.potential(_POS, _MASS))


def test_eval_acc_and_potential_drops_vel_and_t():
    s = _CountingSolver()
    a, p = s._eval_acc_and_potential(_POS, vel=np.ones_like(_POS), mass=_MASS, t=99.0)
    np.testing.assert_array_equal(a, s.acc(_POS, _MASS))
    np.testing.assert_array_equal(p, s.potential(_POS, _MASS))


@pytest.mark.parametrize("vel, t", [
    (None, 0.0),
    (np.full((2, 3), 1e6), 1e6),      # absurd values: they must not reach acc()
])
def test_the_adapters_ignore_whatever_vel_and_t_they_are_given(vel, t):
    # The point of the shims: self-gravity is velocity- and time-independent, so
    # the answer must not depend on either argument.
    s = _CountingSolver()
    np.testing.assert_array_equal(s._eval_acc(_POS, vel, _MASS, t), s.acc(_POS, _MASS))
    np.testing.assert_array_equal(s._eval_potential(_POS, vel, _MASS, t),
                                  s.potential(_POS, _MASS))
