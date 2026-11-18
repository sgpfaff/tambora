"""
Tests for the ``Force`` root class and ``NullForce``.
"""

import numpy as np
import pytest

from tambora.dynamics.forces import Force, NullForce
from tambora.dynamics.forces.CompositeForce import _CompositePlain


_POS = np.array([[1., 2., 3.], [4., 5., 6.]])
_VEL = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
_MASS = np.array([10., 20.])
_T = 0.5


class _Constant(Force):
    """Minimal concrete Force: a fixed acceleration per particle."""

    def __init__(self, value=1.0):
        self.value = value

    def acc(self, pos, vel, mass, t):
        return np.full_like(pos, self.value)


# --- Force: the base-class behaviour ------------------------------------------

def test_force_cannot_be_instantiated():
    # acc is abstract, so the root class is not usable on its own.
    with pytest.raises(TypeError):
        Force()


def test_eval_acc_forwards_the_full_state_to_acc():
    # The integrator boundary always calls _eval_acc(pos, vel, mass, t); the base
    # adapter passes it straight through to acc.
    f = _Constant(2.0)
    np.testing.assert_array_equal(f._eval_acc(_POS, _VEL, _MASS, _T),
                                  f.acc(_POS, _VEL, _MASS, _T))


def test_dedup_key_defaults_to_none_so_forces_opt_out():
    # None means "never a duplicate": add_force allows any number.
    assert _Constant()._dedup_key() is None


def test_a_force_isnt_saved_unless_its_class_says_how():
    f = _Constant()
    assert f.save() is None and f.save_problem() is None
    with pytest.raises(NotImplementedError, match="_Constant forces can't be loaded"):
        _Constant.load({})


def test_a_force_class_can_say_how_it_is_saved_and_loaded():
    class Saved(_Constant):
        def save(self):
            return {'value': self.value}

        @classmethod
        def load(cls, record):
            return cls(record['value'])

    loaded = Saved.load(Saved(3.0).save())
    assert type(loaded) is Saved and loaded.value == 3.0


def test_adding_two_forces_makes_a_composite():
    combined = _Constant(1.0) + _Constant(2.0)
    assert isinstance(combined, Force)
    np.testing.assert_allclose(combined.acc(_POS, _VEL, _MASS, _T),
                               np.full_like(_POS, 3.0))


# --- NullForce ----------------------------------------------------------------

def test_null_force_is_zero_everywhere():
    f = NullForce()
    np.testing.assert_array_equal(f.acc(_POS, _VEL, _MASS, _T), np.zeros_like(_POS))
    np.testing.assert_array_equal(f.potential(_POS, _VEL, _MASS, _T), np.zeros(len(_POS)))


def test_null_force_shapes_follow_pos():
    f = NullForce()
    assert f.acc(_POS, _VEL, _MASS, _T).shape == _POS.shape
    assert f.potential(_POS, _VEL, _MASS, _T).shape == (len(_POS),)


def test_null_force_acc_and_potential_agrees_with_the_separate_calls():
    f = NullForce()
    a, p = f.acc_and_potential(_POS, _VEL, _MASS, _T)
    np.testing.assert_array_equal(a, f.acc(_POS, _VEL, _MASS, _T))
    np.testing.assert_array_equal(p, f.potential(_POS, _VEL, _MASS, _T))


@pytest.mark.parametrize("method, shape_of", [
    ('acc', lambda pos: pos.shape),
    ('potential', lambda pos: (len(pos),)),
])
def test_null_force_tracks_a_changing_particle_count(method, shape_of):
    # Regression: the zero arrays were cached on first call with no key, so the
    # second call here returned the FIRST call's shape. One NullForce reused
    # across components of different sizes silently produced wrong-shaped output.
    f = NullForce()
    for pos in (np.zeros((4, 3)), np.zeros((10, 3)), np.zeros((1, 3))):
        out = getattr(f, method)(pos, None, None, 0.0)
        assert out.shape == shape_of(pos)


@pytest.mark.parametrize("method", ['acc', 'potential'])
def test_null_force_does_not_hand_out_the_same_mutable_array(method):
    # Regression: every call returned the same cached object, so a caller doing
    # `acc(...) += x` turned NullForce into a non-zero force for everyone after.
    f = NullForce()
    first = getattr(f, method)(_POS, _VEL, _MASS, _T)
    first += 5.0                                    # a caller writes in place
    second = getattr(f, method)(_POS, _VEL, _MASS, _T)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(second, np.zeros_like(second))


def test_null_force_matches_an_empty_composite():
    # NullForce and _CompositePlain([]) do the same job; Sim uses the latter.
    # They must not disagree, or which one you got would be observable.
    np.testing.assert_array_equal(
        NullForce().acc(_POS, _VEL, _MASS, _T),
        _CompositePlain([]).acc(_POS, _VEL, _MASS, _T))
