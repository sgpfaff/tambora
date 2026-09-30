"""
Tests for ``ParticleSet``.
"""

import dataclasses

import numpy as np
import pytest

from tambora.ic import ParticleSet


def _ps(n=4, meta=None):
    rng = np.random.default_rng(0)
    return ParticleSet(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), np.full(n, 2.5),
                       meta if meta is not None else {'seed': 7})


def test_it_unpacks_like_the_old_tuple():
    ps = _ps()
    pos, vel, mass = ps
    assert pos is ps.pos and vel is ps.vel and mass is ps.mass
    assert len(ps) == 4


def test_inputs_become_float_arrays():
    ps = ParticleSet([[1, 2, 3]], [[4, 5, 6]], [7])
    assert ps.pos.dtype == ps.vel.dtype == ps.mass.dtype == np.float64
    assert ps.pos.shape == ps.vel.shape == (1, 3) and ps.mass.shape == (1,)
    assert ps.meta == {}


@pytest.mark.parametrize("pos, vel, mass, match", [
    pytest.param(np.zeros((4, 2)), np.zeros((4, 3)), np.zeros(4), r"pos must have shape \(N, 3\)", id='pos'),
    pytest.param(np.zeros((4, 3)), np.zeros((3, 3)), np.zeros(4), r"vel must have the same shape as pos", id='vel'),
    pytest.param(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(3), r"mass must have shape \(4,\)", id='mass'),
    pytest.param(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 1)), r"mass must have shape \(4,\)", id='mass_2d'),
])
def test_inconsistent_shapes_are_rejected(pos, vel, mass, match):
    with pytest.raises(ValueError, match=match):
        ParticleSet(pos, vel, mass)


def test_it_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _ps().pos = np.zeros((4, 3))


def test_meta_is_copied_from_the_caller():
    meta = {'seed': 7}
    ps = _ps(meta=meta)
    meta['seed'] = 8
    assert ps.meta == {'seed': 7}


def test_comparing_two_sets_does_not_raise():
    # A generated __eq__ would compare the arrays and raise "truth value is ambiguous".
    a = _ps()
    assert a == a
    assert a != _ps()


def test_repr_summarises_instead_of_printing_arrays():
    assert repr(_ps()) == "ParticleSet(n=4, total mass=10 Msun, meta={'seed': 7})"


def test_shifted_moves_positions_and_velocities_and_keeps_the_rest():
    ps = _ps()
    moved = ps.shifted(pos=[1., 2., 3.], vel=[-10., 0., 10.])
    np.testing.assert_array_equal(moved.pos, ps.pos + [1., 2., 3.])
    np.testing.assert_array_equal(moved.vel, ps.vel + [-10., 0., 10.])
    np.testing.assert_array_equal(moved.mass, ps.mass)
    assert moved.meta == ps.meta
    assert moved.pos is not ps.pos        # the original is untouched


def test_shifted_takes_one_vector_not_one_per_particle():
    with pytest.raises(ValueError, match="one 3-vector"):
        _ps().shifted(pos=np.zeros((4, 3)))
