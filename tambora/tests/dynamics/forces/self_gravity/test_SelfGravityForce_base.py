"""
Tests for the ``SelfGravityForce`` base class.
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


@pytest.mark.parametrize("name", ['_eval_acc', '_eval_potential', '_eval_acc_and_potential'])
def test_a_solver_refuses_the_external_force_calls(name):
    with pytest.raises(TypeError, match="^A self-gravity solver isn't an external force.$"):
        getattr(_CountingSolver(), name)(_POS, np.ones_like(_POS), _MASS, 0.0)
