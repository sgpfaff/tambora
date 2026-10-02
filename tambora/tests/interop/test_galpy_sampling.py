"""
Tests for the galpy sampler backend.
"""

import inspect
import warnings

import numpy as np
import pytest
from scipy import stats

galpy = pytest.importorskip("galpy")
import astropy.units as u                                             # noqa: E402
from galpy import df, potential                                       # noqa: E402
from galpy.potential.SphericalPotential import SphericalPotential     # noqa: E402

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


# galpy's speeds can be too high for a tracer; the warning about it is tested on its own.
_TRACER_WARNING = "ignore:This eddingtondf draws a tracer:UserWarning"

DFS = [pytest.param(_king, id='kingdf'), pytest.param(_plummer, id='isotropicPlummerdf'),
       pytest.param(_tracer, id='tracer_eddingtondf', marks=pytest.mark.filterwarnings(_TRACER_WARNING))]


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
    pytest.param(_tracer, 1e5 * _TRACER_RMAX**3 / (_TRACER_RMAX**2 + 0.01**2)**1.5, id='tracer_eddingtondf',
                 marks=pytest.mark.filterwarnings(_TRACER_WARNING)),
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
    pytest.param('kingdf', id='string'),
    pytest.param(['a', 'b'], id='list_of_strings'),
    pytest.param(None, id='none'),
])
def test_only_galpy_dfs_and_potentials_are_accepted(obj):
    assert not GalpySampler.accepts(obj)


@pytest.mark.parametrize("make", DFS + [
    pytest.param(_plummer_pot, id='potential'),
    pytest.param(lambda: [_plummer_pot(), _plummer_pot(M=3e5, b=0.05)], id='list'),
])
def test_galpy_dfs_and_potentials_are_accepted(make):
    assert GalpySampler.accepts(make())


def test_a_df_without_physical_units_warns():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        GalpySampler(df.kingdf(W0=3., M=2.3, rt=1.4))


def test_describe_names_the_df():
    assert GalpySampler(_king()).describe() == 'kingdf'


def test_a_tracer_df_warns_that_its_speeds_can_be_too_high():
    with pytest.warns(UserWarning, match=r"draws a tracer \(PlummerPotential\) in a different "
                                         r"potential \(PlummerPotential\+NFWPotential\)"):
        GalpySampler(_tracer())


@pytest.mark.parametrize("make", [
    pytest.param(_king, id='kingdf'),
    pytest.param(_plummer, id='isotropicPlummerdf'),
    pytest.param(lambda: df.eddingtondf(pot=_plummer_pot(), rmax=1. * u.kpc, ro=RO, vo=VO), id='eddingtondf'),
])
def test_a_self_consistent_df_doesnt_warn(make):
    d = make()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        GalpySampler(d)


# --- sampling a potential's own density ---------------------------------------------------

def _hernquist_pot(M=1e10, a=2.):
    return potential.HernquistPotential(amp=2 * M * u.Msun, a=a * u.kpc, ro=RO, vo=VO)   # galpy's amp is 2M


def _dehnen_pot(M=1e9, a=1.):
    # galpy's amp is (3 - alpha) M
    return potential.DehnenSphericalPotential(amp=1.5 * M * u.Msun, a=a * u.kpc, alpha=1.5, ro=RO, vo=VO)


def _dehnen_enclosed(r, M=1e9, a=1.):
    return M * (r / (r + a))**1.5


_EDDINGTON_RMAX = 1e4 * RO      # kpc: where the Dehnen is cut, since its mass converges slowly


@pytest.mark.parametrize("make, df_type", [
    pytest.param(_plummer_pot, df.isotropicPlummerdf, id='Plummer'),
    pytest.param(_hernquist_pot, df.isotropicHernquistdf, id='Hernquist'),
    pytest.param(_dehnen_pot, df.eddingtondf, id='Dehnen'),
])
def test_a_potential_gets_galpys_exact_df_if_it_has_one_else_an_eddington_one(make, df_type):
    pot = make()
    d = GalpySampler(pot).df
    assert type(d) is df_type
    assert d._pot is pot


@pytest.mark.parametrize("make, mass", [
    pytest.param(_plummer_pot, 1e5, id='Plummer'),
    pytest.param(_hernquist_pot, 1e10, id='Hernquist'),
    pytest.param(_dehnen_pot, _dehnen_enclosed(_EDDINGTON_RMAX), id='Dehnen'),
    pytest.param(lambda: [_plummer_pot(), _dehnen_pot()],
                 1e5 + _dehnen_enclosed(_EDDINGTON_RMAX), id='list'),
])
def test_a_potential_is_sampled_with_its_own_mass(make, mass):
    assert GalpySampler(make()).total_mass == pytest.approx(mass, rel=1e-9)


def _plummer_cdf(r, b=0.01):
    return r**3 / (r**2 + b**2)**1.5


# Models sampled from potentials, with the fraction of their sampled mass inside r [kpc].
PROFILES = [
    pytest.param(_plummer_pot, _plummer_cdf, id='Plummer'),
    pytest.param(_hernquist_pot, lambda r: r**2 / (r + 2.)**2, id='Hernquist'),
    pytest.param(_dehnen_pot, lambda r: _dehnen_enclosed(r) / _dehnen_enclosed(_EDDINGTON_RMAX), id='Dehnen'),
    pytest.param(lambda: [_plummer_pot(), _plummer_pot(M=3e5, b=0.05)],
                 lambda r: (_plummer_cdf(r) + 3 * _plummer_cdf(r, b=0.05)) / 4, id='two_Plummers'),
]


@pytest.mark.parametrize("make, cdf", PROFILES)
def test_the_sampled_radii_follow_the_potentials_density(make, cdf):
    pos, _ = GalpySampler(make()).draw(10_000, 3)
    # Kolmogorov-Smirnov over the whole profile. With 10^4 particles it catches a length scale
    # off by ~10% (e.g. ro mixed up); the exact match with galpy's own output pins the units.
    assert stats.kstest(np.linalg.norm(pos, axis=1), cdf).pvalue > 1e-3


def _virial_ratio(sampler, pot, n=10_000, seed=3):
    """2K / -sum(m x.F) for the particles in pot, which is 1 in equilibrium."""
    pos, vel = sampler.draw(n, seed)
    r = np.linalg.norm(pos, axis=1) / RO
    rforce = potential.evaluaterforces(pot, r, 0 * r, use_physical=False)
    return np.sum(vel**2) / VO**2 / -np.sum(r * rforce)


@pytest.mark.parametrize("make, cdf", PROFILES)
def test_a_sampled_potential_is_close_to_equilibrium(make, cdf):
    # galpy #1343 makes the speeds about 1.5% slow (the strict xfail above), so allow 5%.
    # A velocity unit of 220 instead of 230 km/s gives 0.92.
    sampler = GalpySampler(make())
    assert _virial_ratio(sampler, sampler.df._pot) == pytest.approx(1., rel=0.05)


@pytest.mark.filterwarnings(_TRACER_WARNING)
def test_a_tracer_is_close_to_equilibrium_in_the_total_potential():
    # Sampled in the stars' own potential instead, the ratio here would be 0.90.
    d = _tracer()
    assert _virial_ratio(GalpySampler(d), d._pot) == pytest.approx(1., rel=0.05)


def _power_law_with_cutoff():
    return potential.PowerSphericalPotentialwCutoff(amp=1., alpha=1.8, rc=1.9 * u.kpc, ro=RO, vo=VO)


def _exp_truncated_nfw():
    return potential.ExpTruncNFWPotential(amp=1e12 * u.Msun, a=20 * u.kpc, rc=200 * u.kpc, ro=RO, vo=VO)


@pytest.mark.parametrize("make", [
    pytest.param(_power_law_with_cutoff, id='PowerSphericalPotentialwCutoff'),
    pytest.param(_exp_truncated_nfw, id='ExpTruncNFWPotential', marks=pytest.mark.skipif(
        not hasattr(potential, 'ExpTruncNFWPotential'), reason="new in galpy 1.12")),
])
def test_a_density_with_a_cutoff_is_sampled_whole(make):
    # With eddingtondf's default rmax, these stop gaining mass in floating point long before
    # it and galpy's sampler fails ("x must be increasing").
    pot = make()
    sampler = GalpySampler(pot)
    pos, _ = sampler.draw(1000, 1)
    assert np.isfinite(pos).all()
    everything = potential.mass(pot, 1e8, use_physical=False) * sampler.units.mass_msun
    assert sampler.total_mass == pytest.approx(everything, rel=1e-9)


def test_a_potentials_units_are_its_own():
    units = GalpySampler(_plummer_pot()).units
    assert (units.length_kpc, units.velocity_kms) == (RO, VO)


def test_ic_sample_draws_a_potential():
    from tambora import ic
    ps = ic.sample(_plummer_pot(), 1000, seed=2)
    assert ps.mass.sum() == pytest.approx(1e5, rel=1e-12)
    assert ps.meta == {'backend': 'galpy', 'model': 'isotropicPlummerdf(PlummerPotential)', 'seed': 2}
    np.testing.assert_array_equal(ps.pos, GalpySampler(_plummer_pot()).draw(1000, 2)[0])


def test_describe_names_the_df_and_the_potentials():
    assert GalpySampler([_plummer_pot(), _dehnen_pot()]).describe() == \
        'eddingtondf(PlummerPotential+DehnenSphericalPotential)'


@pytest.mark.parametrize("make, error, match", [
    pytest.param(lambda: potential.NFWPotential(amp=1e12 * u.Msun, a=20 * u.kpc, ro=RO, vo=VO),
                 ValueError, "Can't sample a NFWPotential: its mass is infinite", id='infinite_mass'),
    pytest.param(lambda: potential.BurkertPotential(amp=1., a=0.5 * u.kpc, ro=RO, vo=VO),
                 TypeError, r"Can't sample a BurkertPotential: galpy can only draw from spherical densities "
                            r"that define their first two radial derivatives \(_ddensdr and _d2densdr2\)",
                 id='no_density_derivatives'),
    pytest.param(lambda: potential.KingPotential(W0=3., M=1e4 * u.Msun, rt=0.03 * u.kpc, ro=RO, vo=VO),
                 TypeError, r"Can't sample a KingPotential: .* Pass galpy's df.kingdf\(W0=..., M=..., rt=...\)",
                 id='King'),
    pytest.param(lambda: potential.KeplerPotential(amp=1e4 * u.Msun, ro=RO, vo=VO),
                 TypeError, "Can't sample a KeplerPotential: it's a point mass", id='point_mass'),
    pytest.param(lambda: potential.MiyamotoNagaiPotential(amp=1e10 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc,
                                                          ro=RO, vo=VO),
                 TypeError, "Can't sample a MiyamotoNagaiPotential: galpy can only draw from spherical",
                 id='not_spherical'),
    pytest.param(lambda: [_plummer_pot(), 'halo'],
                 TypeError, r"Expected a galpy Potential \(element 1 of the list\), got str", id='bad_list'),
])
def test_a_potential_that_cant_be_sampled_says_why(make, error, match):
    with pytest.raises(error, match=match):
        GalpySampler(make())


def test_a_potential_without_physical_units_warns():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        GalpySampler(potential.PlummerPotential())


class _CustomPlummer(SphericalPotential):
    """A Plummer written the way galpy's docs describe a custom spherical potential."""
    def __init__(self, amp, b, ro=None, vo=None):
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units='mass')
        self._b2 = (b.to_value(u.kpc) / self._ro)**2

    def _revaluate(self, r, t=0.):
        return -1. / np.sqrt(r**2 + self._b2)

    def _rforce(self, r, t=0.):
        return -r / (r**2 + self._b2)**1.5

    def _r2deriv(self, r, t=0.):
        return (self._b2 - 2 * r**2) / (r**2 + self._b2)**2.5

    def _rdens(self, r, t=0.):
        return 3 * self._b2 / (4 * np.pi) / (r**2 + self._b2)**2.5


class _CustomPlummerWithDerivatives(_CustomPlummer):
    def _ddensdr(self, r, t=0.):
        return self._amp * -15 * self._b2 / (4 * np.pi) * r / (r**2 + self._b2)**3.5

    def _d2densdr2(self, r, t=0.):
        return self._amp * -15 * self._b2 / (4 * np.pi) * (self._b2 - 6 * r**2) / (r**2 + self._b2)**4.5


def test_a_custom_potential_with_its_density_derivatives_is_sampled_like_a_built_in_one():
    custom = GalpySampler(_CustomPlummerWithDerivatives(1e5 * u.Msun, 0.01 * u.kpc, ro=RO, vo=VO))
    assert custom.describe() == 'eddingtondf(_CustomPlummerWithDerivatives)'
    assert custom.total_mass == pytest.approx(1e5, rel=1e-9)
    # Same seed, so the same quantiles; galpy only grids the custom one more coarsely (it has
    # no _scale), which moves the innermost particles a little.
    (pos, vel), (pos0, vel0) = custom.draw(2000, 4), GalpySampler(_plummer_pot()).draw(2000, 4)
    quantiles = [10, 25, 50, 75, 90]
    for a, a0 in ((pos, pos0), (vel, vel0)):
        np.testing.assert_allclose(np.percentile(np.linalg.norm(a, axis=1), quantiles),
                                   np.percentile(np.linalg.norm(a0, axis=1), quantiles), rtol=1e-3)


class _CustomPlummerWithFirstDerivative(_CustomPlummer):
    _ddensdr = _CustomPlummerWithDerivatives._ddensdr


@pytest.mark.parametrize("cls", [_CustomPlummer, _CustomPlummerWithFirstDerivative])
def test_a_custom_potential_without_its_density_derivatives_says_what_to_add(cls):
    with pytest.raises(TypeError, match=rf"Can't sample a {cls.__name__}: .*\(_ddensdr and _d2densdr2\)"):
        GalpySampler(cls(1e5 * u.Msun, 0.01 * u.kpc, ro=RO, vo=VO))
