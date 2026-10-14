"""
Tests for ``Sim.virial_ratio`` and ``Component.virial_ratio``, on systems whose ratio is known
exactly: a binary, and a ring of particles around a point mass.
"""

import numpy as np
import pytest

from tambora.dynamics import DirectSummationGravity, ExternalConservativeForce
from tambora.simulation import Sim
from tambora.units import G_INTERNAL, KMS_TO_KPCGYR

M = 1e9  # Msun
_DIRECT = DirectSummationGravity(eps=0.)


def _run(sim, t_end=0.01, dt=0.01, dt_out=0.01):
    sim.run(t_end=t_end, dt=dt, dt_out=dt_out, method='direct', eps=0., progress=False, monitors=False)
    return sim


def _binary(speed=1., offset=(0., 0., 0.), boost=(0., 0., 0.)):
    """Masses M and 3M 1 kpc apart, at ``speed`` times the speeds of a circular orbit."""
    mass = np.array([M, 3 * M])
    r = mass[::-1] / mass.sum()  # each one's distance from the center of mass
    omega = np.sqrt(G_INTERNAL * mass.sum())
    pos = np.array([[r[0], 0., 0.], [-r[1], 0., 0.]]) + offset
    vel = speed * omega * np.array([[0., r[0], 0.], [0., -r[1], 0.]]) / KMS_TO_KPCGYR + boost
    sim = Sim()
    sim.add_particles('binary', pos, vel, mass)
    return sim


class _PointMass(ExternalConservativeForce):
    def __init__(self, M):
        self.M = M

    def acc(self, pos, t):
        return -G_INTERNAL * self.M * pos / np.linalg.norm(pos, axis=1, keepdims=True)**3

    def potential(self, pos, t):
        return -G_INTERNAL * self.M / np.linalg.norm(pos, axis=1)


class _UniformField(ExternalConservativeForce):
    def __init__(self, g):
        self.g = np.asarray(g, dtype=float)

    def acc(self, pos, t):
        return np.broadcast_to(self.g, pos.shape).copy()

    def potential(self, pos, t):
        return -pos @ self.g


def _ring(n=8, R=2.):
    """n particles of mass M/n, evenly spaced on a circle of radius R in the x-y plane."""
    phi = 2 * np.pi * np.arange(n) / n
    return R * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n)]), np.full(n, M / n)


def _circular(pos, acc):
    """Velocities [km/s] of circular orbits about the origin, given each particle's acceleration."""
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    speed = np.sqrt(r * np.linalg.norm(acc, axis=1, keepdims=True))
    return speed * np.column_stack([-pos[:, 1], pos[:, 0], np.zeros(len(pos))]) / r / KMS_TO_KPCGYR


def _ring_around_a_point_mass():
    """A ring on circular orbits in its own gravity and an external point mass M."""
    pos, mass = _ring()
    host = _PointMass(M)
    sim = Sim()
    sim.add_external_force(host)
    sim.add_particles('ring', pos, _circular(pos, host.acc(pos, 0.) + _DIRECT.acc(pos, mass)), mass)
    return sim


@pytest.mark.parametrize("speed", [0., 0.5, 1., 1.5])
def test_a_binarys_ratio_is_the_square_of_its_speed_in_circular_speeds(speed):
    sim = _run(_binary(speed))
    assert sim.virial_ratio(t=0) == pytest.approx(speed**2, rel=1e-12)
    assert sim.binary.virial_ratio(t=0) == pytest.approx(speed**2, rel=1e-12)


def test_the_ratio_is_taken_about_the_center_of_mass():
    sim = _run(_binary(offset=(8., -3., 1.), boost=(100., 50., -20.)))
    assert sim.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)
    assert sim.binary.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)


def test_initial_conditions_can_be_checked_before_a_run():
    sim = _binary()
    assert sim.virial_ratio(t=0, method='direct', eps=0.) == pytest.approx(1., rel=1e-12)
    assert sim.binary.virial_ratio(t=0, method='direct', eps=0.) == pytest.approx(1., rel=1e-12)


def test_self_gravity_computed_again_gives_the_cached_ratio():
    sim = _run(_binary(0.7))
    assert sim.virial_ratio(t=0, method='direct', eps=0.) == pytest.approx(sim.virial_ratio(t=0), rel=1e-12)


def test_a_uniform_external_field_leaves_the_ratio_alone():
    # It accelerates every particle alike, so the motion about the center of mass is the same.
    sim = _binary(offset=(8., -3., 1.))
    sim.add_external_force(_UniformField([-500., 200., 100.]))
    _run(sim)
    assert sim.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)
    assert sim.binary.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)


def test_external_forces_count():
    sim = _run(_ring_around_a_point_mass())
    assert sim.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)
    assert sim.ring.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)


def test_a_components_ratio_counts_the_other_components_unless_told_not_to():
    pos, mass = _ring()
    acc = _DIRECT.acc(np.vstack([pos, np.zeros((1, 3))]), np.append(mass, M))[:-1]
    sim = Sim()
    sim.add_particles('ring', pos, _circular(pos, acc), mass)
    sim.add_particles('host', np.zeros((1, 3)), np.zeros((1, 3)), np.array([M]))
    _run(sim)
    assert sim.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)
    assert sim.ring.virial_ratio(t=0) == pytest.approx(1., rel=1e-12)
    # In its own gravity alone, the ring's ratio is its total acceleration over its own.
    alone = np.linalg.norm(acc[0]) / np.linalg.norm(_DIRECT.acc(pos, mass)[0])
    assert sim.ring.virial_ratio(t=0, include_all_components=False, method='direct', eps=0.) == \
        pytest.approx(alone, rel=1e-12)


def test_at_every_snapshot_it_is_the_ratio_at_each():
    sim = _run(_ring_around_a_point_mass(), t_end=0.05, dt=0.001, dt_out=0.01)
    n_snap = len(sim.times)
    for obj in (sim, sim.ring):
        every = obj.virial_ratio()
        assert every.shape == (n_snap,)
        np.testing.assert_allclose(every, [obj.virial_ratio(t=i) for i in range(n_snap)], rtol=1e-14)
        # The ring turns by about 80 degrees, staying on its circular orbits.
        np.testing.assert_allclose(every, 1., rtol=1e-3)
