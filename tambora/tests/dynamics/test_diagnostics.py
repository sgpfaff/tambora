"""
Tests for ``center_of_mass`` and ``virial_ratio`` in ``tambora.dynamics.diagnostics``, on plain arrays.
"""

import numpy as np
import pytest

from tambora.dynamics.diagnostics import center_of_mass, virial_ratio
from tambora.units import G_INTERNAL


def test_the_center_of_mass_is_the_mass_weighted_mean_at_each_snapshot():
    rng = np.random.default_rng(1)
    values, mass = rng.normal(size=(4, 10, 3)), rng.uniform(1., 5., size=10)
    expected = [np.average(v, axis=0, weights=mass) for v in values]
    np.testing.assert_allclose(center_of_mass(values, mass), expected, rtol=1e-13)
    np.testing.assert_allclose(center_of_mass(values[2], mass), expected[2], rtol=1e-13)


def _binary(speed):
    """Masses 1 and 3 a unit distance apart, at ``speed`` times a circular orbit's speeds."""
    mass = np.array([1., 3.])
    r = mass[::-1] / mass.sum()  # each one's distance from the center of mass
    pos = np.array([[r[0], 0., 0.], [-r[1], 0., 0.]]) + [5., -2., 1.]
    vel = speed * np.sqrt(G_INTERNAL * mass.sum()) * np.array([[0., r[0], 0.], [0., -r[1], 0.]]) + [10., 0., 3.]
    acc = G_INTERNAL * np.array([[-mass[1], 0., 0.], [mass[0], 0., 0.]])
    return pos, vel, mass, acc


def test_a_binarys_virial_ratio_is_the_square_of_its_speed_in_circular_speeds():
    snapshots = [_binary(speed) for speed in (1., 0.5, 1.5)]
    pos, vel, acc = (np.stack([s[i] for s in snapshots]) for i in (0, 1, 3))
    np.testing.assert_allclose(virial_ratio(pos, vel, snapshots[0][2], acc), [1., 0.25, 2.25], rtol=1e-12)
    assert virial_ratio(*_binary(0.5)) == pytest.approx(0.25, rel=1e-12)
