"""
Tests for how the self-gravity solvers get their softening: passed to each call, unless
the solver was built with its own.
"""

import numpy as np
import pytest

from tambora.dynamics import DirectSummationGravity, FalcONGravity, NullSelfGravity

_rng = np.random.default_rng(2)
POS = _rng.normal(size=(40, 3))
MASS = _rng.uniform(1e5, 1e6, size=40)
EPS = _rng.uniform(0.05, 0.2, size=40)

SOLVERS = [
    pytest.param(lambda **kw: DirectSummationGravity(**kw), id='direct'),
    pytest.param(lambda **kw: FalcONGravity(theta=0.5, **kw), id='falcON'),
]


@pytest.mark.parametrize("make", SOLVERS)
@pytest.mark.parametrize("eps", [0.1, EPS], ids=['one_value', 'per_particle'])
def test_a_solver_without_its_own_softening_uses_the_calls(make, eps):
    own = make(eps=eps).acc_and_potential(POS, MASS)
    passed = make().acc_and_potential(POS, MASS, eps=eps)
    for a, b in zip(own, passed):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(make().acc(POS, MASS, eps=eps), own[0])
    np.testing.assert_array_equal(make().potential(POS, MASS, eps=eps), own[1])
    assert not np.allclose(make().acc(POS, MASS, eps=0.5), own[0])


@pytest.mark.parametrize("make", SOLVERS)
def test_a_solvers_own_softening_replaces_the_calls(make):
    np.testing.assert_array_equal(make(eps=0.1).acc(POS, MASS, eps=EPS), make(eps=0.1).acc(POS, MASS))


@pytest.mark.parametrize("make", SOLVERS)
def test_without_any_softening_a_solver_refuses(make):
    with pytest.raises(ValueError, match="No softening"):
        make().acc(POS, MASS)


@pytest.mark.parametrize("cls", [DirectSummationGravity, FalcONGravity])
def test_a_solvers_arguments_must_be_named(cls):
    with pytest.raises(TypeError):
        cls(0.1)


def test_no_self_gravity_takes_a_softening_and_ignores_it():
    np.testing.assert_array_equal(NullSelfGravity().acc(POS, MASS, eps=0.1), np.zeros_like(POS))
