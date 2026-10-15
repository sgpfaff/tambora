"""
Tests for ``com_pos`` and ``com_vel`` of ``Sim`` and ``Component``.
"""

import numpy as np
import pytest

from tambora.simulation import Sim
from tambora.units import KMS_TO_KPCGYR

_rng = np.random.default_rng(3)
STARS = dict(pos=_rng.normal(size=(20, 3)), vel=_rng.normal(scale=10., size=(20, 3)),
             mass=_rng.uniform(1e5, 1e6, size=20))
DM = dict(pos=_rng.normal(size=(30, 3)) + [5., 0., 0.], vel=_rng.normal(scale=30., size=(30, 3)),
          mass=_rng.uniform(1e6, 1e7, size=30))


def _sim():
    sim = Sim()
    sim.add_particles('stars', **STARS)
    sim.add_particles('dm', **DM)
    sim.eps = 0.1
    sim.run(t_end=0.03, dt=0.01, dt_out=0.01, method='direct', progress=False, monitors=False)
    return sim


SIM = _sim()


def _weighted_mean(values, mass):
    return np.stack([np.average(v, axis=0, weights=mass) for v in values])


@pytest.mark.parametrize("component", ['stars', 'dm', None])
def test_it_is_the_mass_weighted_mean_at_every_snapshot(component):
    obj = SIM if component is None else getattr(SIM, component)
    assert obj.com_pos().shape == obj.com_vel().shape == (len(SIM.times), 3)
    np.testing.assert_allclose(obj.com_pos(), _weighted_mean(obj.pos(), obj.mass), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(obj.com_vel(), _weighted_mean(obj.vel(), obj.mass), rtol=1e-12, atol=1e-12)


def test_one_snapshot_gives_one_point():
    assert SIM.com_pos(t=1).shape == SIM.stars.com_vel(t=0.02).shape == (3,)
    np.testing.assert_array_equal(SIM.com_pos(t=1), SIM.com_pos()[1])
    np.testing.assert_array_equal(SIM.stars.com_vel(t=0.02), SIM.stars.com_vel()[2])


def test_the_velocity_is_in_kms_unless_internal_units_are_asked_for():
    expected = np.average(STARS['vel'], axis=0, weights=STARS['mass'])
    np.testing.assert_allclose(SIM.stars.com_vel(t=0), expected, rtol=1e-12)
    np.testing.assert_allclose(SIM.stars.com_vel(t=0, return_internal=True), expected * KMS_TO_KPCGYR, rtol=1e-12)
