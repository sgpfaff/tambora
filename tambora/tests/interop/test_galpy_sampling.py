"""
Tests for the galpy sampler backend.
"""

import inspect

import numpy as np
import pytest

galpy = pytest.importorskip("galpy")
import astropy.units as u                                             # noqa: E402
from galpy import df, potential                                       # noqa: E402

from tambora.interop._galpy.sampling import GalpySampler               # noqa: E402
from tambora.units import G_KPC_KMS                                    # noqa: E402

RO, VO = 9., 230.       # not galpy's defaults, so a unit taken from the wrong place shows


def _plummer_pot(M=1e5, b=0.01):
    return potential.PlummerPotential(amp=M * u.Msun, b=b * u.kpc, ro=RO, vo=VO)


def _king():
    return df.kingdf(W0=3., M=1e4 * u.Msun, rt=0.03 * u.kpc, ro=RO, vo=VO)


def _plummer():
    return df.isotropicPlummerdf(pot=_plummer_pot(), ro=RO, vo=VO)


_TRACER_RMAX = 1.       # kpc


def _tracer():
    # Stars in the total potential of the stars and a dark halo 100 times as massive. The
    # default rmax is 10^7 of the stars' scale radii, where their enclosed mass has stopped
    # growing in floating point and galpy's sampler fails ("x must be increasing").
    stars = _plummer_pot()
    halo = potential.NFWPotential(amp=1e7 * u.Msun, a=0.5 * u.kpc, ro=RO, vo=VO)
    return df.eddingtondf(pot=stars + halo, denspot=stars, rmax=_TRACER_RMAX * u.kpc, ro=RO, vo=VO)


DFS = [pytest.param(_king, id='kingdf'), pytest.param(_plummer, id='isotropicPlummerdf'),
       pytest.param(_tracer, id='tracer_eddingtondf')]


@pytest.mark.parametrize("make", DFS)
def test_draw_matches_galpys_own_physical_output(make):
    d = make()
    pos, vel = GalpySampler(d).draw(500, 11)
    np.random.seed(11)
    o = d.sample(n=500, return_orbit=True)
    q = dict(use_physical=True, quantity=False)
    np.testing.assert_allclose(pos, np.c_[o.x(**q), o.y(**q), o.z(**q)], rtol=1e-12)
    np.testing.assert_allclose(vel, np.c_[o.vx(**q), o.vy(**q), o.vz(**q)], rtol=1e-12)


@pytest.mark.parametrize("make, mass", [
    pytest.param(_king, 1e4, id='kingdf'),
    pytest.param(_plummer, 1e5, id='isotropicPlummerdf'),
    # The stars' mass inside rmax, which is what gets sampled; not the halo's.
    pytest.param(_tracer, 1e5 * _TRACER_RMAX**3 / (_TRACER_RMAX**2 + 0.01**2)**1.5, id='tracer_eddingtondf'),
])
def test_the_total_mass_is_the_sampled_models(make, mass):
    assert GalpySampler(make()).total_mass == pytest.approx(mass, rel=1e-9)


@pytest.mark.skipif('rmin' not in inspect.signature(df.eddingtondf.__init__).parameters,
                    reason="galpy DFs take an rmin when built from galpy 1.11")
def test_the_total_mass_leaves_out_what_is_inside_the_dfs_rmin():
    # Built with rmin = its scale radius, a Hernquist DF draws nothing inside it, where a
    # quarter of its mass is; those particles must not share that quarter.
    M, a = 1e10, 2.
    pot = potential.HernquistPotential(amp=2 * M * u.Msun, a=a * u.kpc, ro=RO, vo=VO)   # galpy's amp is 2M
    sampler = GalpySampler(df.eddingtondf(pot=pot, rmin=a * u.kpc, ro=RO, vo=VO))
    pos, _ = sampler.draw(2000, 1)
    assert np.linalg.norm(pos, axis=1).min() > a          # guard: galpy really leaves it out
    rmax = 1e4 * RO                                       # eddingtondf's default rmax, in kpc
    enclosed = lambda r: M * r**2 / (r + a)**2            # Hernquist
    assert sampler.total_mass == pytest.approx(enclosed(rmax) - enclosed(a), rel=1e-9)


def test_the_units_are_the_dfs_and_have_g_equal_to_one():
    units = GalpySampler(_king()).units
    assert (units.length_kpc, units.velocity_kms) == (RO, VO)
    assert G_KPC_KMS * units.mass_msun / (units.length_kpc * units.velocity_kms**2) == pytest.approx(1., rel=1e-12)


def test_the_same_seed_gives_the_same_particles():
    s = GalpySampler(_king())
    a, b, c = s.draw(200, 1), s.draw(200, 1), s.draw(200, 2)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_drawing_leaves_numpys_global_generator_as_it_was():
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    GalpySampler(_king()).draw(100, 7)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_the_half_mass_radius_is_plummers():
    b = 0.01
    pos, _ = GalpySampler(_plummer()).draw(100_000, 3)
    r_half = b / np.sqrt(2**(2 / 3) - 1)
    assert np.median(np.linalg.norm(pos, axis=1)) == pytest.approx(r_half, rel=0.01)


@pytest.mark.xfail(raises=AssertionError, strict=True,
                   reason="galpy #1343: sampled speeds are too slow, <v^2> about 1.5% low")
def test_the_mean_square_speed_is_plummers_virial_value():
    M, b = 1e5, 0.01
    _, vel = GalpySampler(_plummer()).draw(100_000, 3)
    virial = 3 * np.pi * G_KPC_KMS * M / (32 * b)
    assert np.mean(np.sum(vel**2, axis=1)) == pytest.approx(virial, rel=0.005)


@pytest.mark.parametrize("obj", [
    pytest.param(_plummer_pot(), id='a_potential'),
    pytest.param('kingdf', id='string'),
])
def test_only_spherical_dfs_are_accepted(obj):
    assert not GalpySampler.accepts(obj)


@pytest.mark.parametrize("make", DFS)
def test_spherical_dfs_are_accepted(make):
    assert GalpySampler.accepts(make())


def test_a_df_without_physical_units_warns():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        GalpySampler(df.kingdf(W0=3., M=2.3, rt=1.4))


def test_describe_names_the_df():
    assert GalpySampler(_king()).describe() == 'kingdf'
