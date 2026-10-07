"""
Tests for the galpy sampler backend.
"""

import inspect
import re
import warnings

import numpy as np
import pytest
from scipy import stats

galpy = pytest.importorskip("galpy")
import astropy.units as u                                             # noqa: E402
from galpy import df, potential                                       # noqa: E402
from galpy.potential.SphericalPotential import SphericalPotential     # noqa: E402
from galpy.util.conversion import mass_in_msol                        # noqa: E402

from tambora import ic                                                 # noqa: E402
from tambora.interop._galpy import sampling                            # noqa: E402
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


# galpy #1568 fixes #1343 on galpy's main branch after 1.12.0, so in its development versions
# since and in the release that draws tracers right; this test then passes, so it's only xfailed
# before.
_GALPY_SPEEDS_FIXED = sampling._dev_galpy() or sampling._galpy_has(sampling._TRACERS_RELEASE)


@pytest.mark.xfail(not _GALPY_SPEEDS_FIXED, raises=AssertionError, strict=True,
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
    pytest.param(lambda: [ic.Plummer(M=1e5, rscale=0.01), ic.Hernquist(M=1e9, rscale=1.)], id='list_of_profiles'),
])
def test_galpy_dfs_and_potentials_are_accepted(make):
    assert GalpySampler.accepts(make())


def test_a_df_without_physical_units_warns():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        GalpySampler(df.kingdf(W0=3., M=2.3, rt=1.4))


def test_describe_names_the_df():
    assert GalpySampler(_king()).describe() == 'kingdf'


# Up to 1.12.0, galpy draws a tracer deep in a much deeper potential far too fast. Until a
# release with the fix, or a development version vouched for, tambora refuses to draw one
# itself, and warns about a galpy DF that does.
_GALPY_DRAWS_TRACERS = sampling._tracers_fixed()


@pytest.mark.skipif(_GALPY_DRAWS_TRACERS, reason="this galpy draws tracers right")
def test_a_tracer_df_warns_that_its_speeds_can_be_too_high():
    with pytest.warns(UserWarning, match=r"draws a tracer \(PlummerPotential\) in a different "
                                         r"potential \(PlummerPotential\+NFWPotential\)"):
        GalpySampler(_tracer())


@pytest.mark.parametrize("make", [
    pytest.param(_king, id='kingdf'),
    pytest.param(_plummer, id='isotropicPlummerdf'),
    pytest.param(lambda: df.eddingtondf(pot=_plummer_pot(), rmax=1. * u.kpc, ro=RO, vo=VO), id='eddingtondf'),
    pytest.param(_tracer, id='tracer_eddingtondf', marks=pytest.mark.skipif(
        not sampling._galpy_has(sampling._TRACERS_RELEASE), reason="until a galpy release draws tracers right")),
])
def test_a_df_that_galpy_draws_right_doesnt_warn(make):
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
    r = np.linalg.norm(pos, axis=1) / sampler.units.length_kpc     # pot's natural units
    rforce = potential.evaluaterforces(pot, r, 0 * r, use_physical=False)
    return np.sum(vel**2) / sampler.units.velocity_kms**2 / -np.sum(r * rforce)


# Up to 1.12.0, galpy #1343 draws speeds about 1.6% slow (the strict xfail above). With 50,000
# particles the ratio scatters by 0.3% from seed to seed, so 1.5% is 5 sigma once that's fixed.
_EQUILIBRIUM_N = 50_000
_EQUILIBRIUM_RTOL = 0.015 if _GALPY_SPEEDS_FIXED else 0.03


@pytest.mark.parametrize("make, cdf", PROFILES)
def test_a_sampled_potential_is_close_to_equilibrium(make, cdf):
    # A velocity unit of 220 instead of 230 km/s gives 0.92.
    sampler = GalpySampler(make())
    assert _virial_ratio(sampler, sampler.df._pot, n=_EQUILIBRIUM_N) == pytest.approx(1., rel=_EQUILIBRIUM_RTOL)


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
    pytest.param(lambda: [_plummer_pot(), [_hernquist_pot(), 'halo']],
                 TypeError, r"Expected a galpy Potential \(element \[1\]\[1\] of the list\), got str",
                 id='bad_nested_list'),
    pytest.param(lambda: [ic.Plummer(M=1e5, rscale=0.01), ic.Hernquist(M=1e9, rscale=1.)], TypeError,
                 r"Can't draw a list of models with tambora's profiles in its own potential \(element 0 of the "
                 r"list is a Plummer\)\. To draw several models in equilibrium together, use ic\.sample_components",
                 id='list_of_profiles'),
    pytest.param(lambda: [_plummer_pot(), ic.Hernquist(M=1e9, rscale=1.)], TypeError,
                 r"\(element 1 of the list is a Hernquist\)", id='profile_among_potentials'),
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


# --- tambora's profiles --------------------------------------------------------------------


PROFILE_EQUIVALENTS = [
    pytest.param(lambda: ic.Plummer(M=1e5, rscale=0.01), _plummer_pot, id='Plummer'),
    pytest.param(lambda: ic.Hernquist(M=1e10, rscale=2.), _hernquist_pot, id='Hernquist'),
    pytest.param(lambda: ic.King(M=1e4, W0=6., rt=0.03),
                 lambda: df.kingdf(W0=6., M=1e4 * u.Msun, rt=0.03 * u.kpc, ro=RO, vo=VO), id='King'),
]


@pytest.mark.parametrize("make_profile, make_native", PROFILE_EQUIVALENTS)
def test_a_profile_samples_the_galpy_model_it_stands_for(make_profile, make_native):
    # The same seed draws the same particles, whatever units galpy works in, so this checks
    # the translation (e.g. a Hernquist's galpy amplitude is twice its mass) exactly.
    profile, native = GalpySampler(make_profile()), GalpySampler(make_native())
    assert profile.total_mass == pytest.approx(native.total_mass, rel=1e-12)
    for a, b in zip(profile.draw(2000, 4), native.draw(2000, 4)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6 * np.median(np.abs(b)))


@pytest.mark.parametrize("make_profile, make_native", PROFILE_EQUIVALENTS)
def test_a_profile_is_sampled_with_its_mass(make_profile, make_native):
    profile = make_profile()
    assert GalpySampler(profile).total_mass == pytest.approx(profile.M, rel=1e-12)


def test_a_profile_scales_exactly_with_its_mass_and_radius():
    # A Plummer of any size is the same model: positions go with rscale, speeds with sqrt(G M / rscale).
    small, big = ic.Plummer(M=1e2, rscale=1e-4), ic.Plummer(M=1e14, rscale=300.)
    (pos_s, vel_s), (pos_b, vel_b) = GalpySampler(small).draw(1000, 8), GalpySampler(big).draw(1000, 8)
    np.testing.assert_allclose(pos_s / small.rscale, pos_b / big.rscale, rtol=1e-10)
    v_unit = lambda p: np.sqrt(G_KPC_KMS * p.M / p.rscale)
    np.testing.assert_allclose(vel_s / v_unit(small), vel_b / v_unit(big), rtol=1e-10)


def test_ic_sample_draws_a_profile():
    ps = ic.sample(ic.Hernquist(M=1e10, rscale=2.), 1000, seed=2)
    assert ps.mass.sum() == pytest.approx(1e10, rel=1e-12)
    assert ps.meta == {'backend': 'galpy', 'model': 'Hernquist(M=1e+10, rscale=2)', 'seed': 2}


@pytest.mark.parametrize("make_profile", [p.values[0] for p in PROFILE_EQUIVALENTS]
                         + [lambda: ic.TruncatedNFW(M=1e12, rscale=20., rtrunc=200.)])
def test_profiles_are_accepted(make_profile):
    assert GalpySampler.accepts(make_profile())


_HAS_EXP_TRUNC_NFW = hasattr(potential, 'ExpTruncNFWPotential')
_needs_exp_trunc_nfw = pytest.mark.skipif(not _HAS_EXP_TRUNC_NFW, reason="ExpTruncNFWPotential is new in galpy 1.12")


@_needs_exp_trunc_nfw
def test_a_truncated_nfw_samples_the_galpy_model_it_stands_for():
    profile = GalpySampler(ic.TruncatedNFW(M=1e12, rscale=20., rtrunc=200.))
    native = GalpySampler(potential.ExpTruncNFWPotential(mass=1e12 * u.Msun, a=20 * u.kpc, rc=200 * u.kpc,
                                                         ro=RO, vo=VO))
    assert profile.total_mass == pytest.approx(1e12, rel=1e-8)
    assert profile.total_mass == pytest.approx(native.total_mass, rel=1e-8)
    (pos, vel), (pos0, vel0) = profile.draw(2000, 4), native.draw(2000, 4)
    # Same radii; the speeds come from a numerical DF, built in different units, so compare
    # their distribution.
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), np.linalg.norm(pos0, axis=1), rtol=1e-6)
    quantiles = [10, 25, 50, 75, 90]
    np.testing.assert_allclose(np.percentile(np.linalg.norm(vel, axis=1), quantiles),
                               np.percentile(np.linalg.norm(vel0, axis=1), quantiles), rtol=1e-3)


@_needs_exp_trunc_nfw
def test_a_truncated_nfw_is_close_to_equilibrium():
    sampler = GalpySampler(ic.TruncatedNFW(M=1e12, rscale=20., rtrunc=200.))
    assert _virial_ratio(sampler, sampler.df._pot, n=_EQUILIBRIUM_N) == pytest.approx(1., rel=_EQUILIBRIUM_RTOL)


@_needs_exp_trunc_nfw
def test_a_truncated_nfw_cut_off_far_inside_its_scale_radius_is_refused():
    with pytest.raises(ValueError, match=r"Can't sample TruncatedNFW\(M=1e\+12, rscale=20, rtrunc=3\): "
                                         r"galpy's sampler is unreliable when rtrunc is under a fifth of rscale"):
        GalpySampler(ic.TruncatedNFW(M=1e12, rscale=20., rtrunc=3.))


def test_a_truncated_nfw_says_it_needs_galpy_1_12(monkeypatch):
    # galpy before 1.12 has no ExpTruncNFWPotential; take it away so this runs on any galpy.
    monkeypatch.delattr(potential, 'ExpTruncNFWPotential', raising=False)
    with pytest.raises(ImportError, match=r"Sampling a TruncatedNFW needs galpy 1.12 or later"):
        GalpySampler(ic.TruncatedNFW(M=1e12, rscale=20., rtrunc=200.))


def test_an_nfw_points_to_its_truncated_versions():
    with pytest.raises(ValueError, match=r"For an NFW halo, use ic.TruncatedNFW, or truncate yours with "
                                         r"galpy's ExpTruncNFWPotential.from_nfw"):
        GalpySampler(potential.NFWPotential(amp=1e12 * u.Msun, a=20 * u.kpc, ro=RO, vo=VO))


def test_another_infinite_mass_doesnt_mention_nfw():
    with pytest.raises(ValueError, match="its mass is infinite") as caught:
        GalpySampler(potential.PowerSphericalPotential(amp=1., alpha=2., ro=RO, vo=VO))
    assert 'NFW' not in str(caught.value)


def _combined_then_given_other_units():
    # galpy checks that potentials it combines share their units only with an assert, which
    # python -O drops: this is what combining these would give then.
    pot = _plummer_pot() + _hernquist_pot()
    list(sampling._iter_components(pot))[1].turn_physical_on(ro=8., vo=VO)
    return pot


@pytest.mark.parametrize("make", [
    pytest.param(lambda: [_plummer_pot(), potential.HernquistPotential(amp=2e10 * u.Msun, a=2. * u.kpc, ro=8., vo=VO)],
                 id='list'),
    pytest.param(_combined_then_given_other_units, id='combined'),
])
def test_galpy_potentials_with_different_units_are_refused(make):
    # Taking the first one's units instead, as tambora did, gave a Hernquist 9/8 of its mass on galpy 1.9.
    with pytest.raises(ValueError, match=r"The galpy potentials have different units \(ro, vo\): "
                                         r"\[\(8\.0, 230\.0\), \(9\.0, 230\.0\)\]"):
        GalpySampler(make())


@_needs_exp_trunc_nfw
def test_galpys_own_truncation_of_an_nfw_samples_with_its_mass():
    # The route the NFW error points to.
    nfw = potential.NFWPotential(amp=1e12 * u.Msun, a=20 * u.kpc, ro=RO, vo=VO)
    truncated = potential.ExpTruncNFWPotential.from_nfw(nfw, rc=200 * u.kpc)
    ps = ic.sample(truncated, 1000, seed=1)
    expected = potential.mass(truncated, 1e8, use_physical=False) * GalpySampler(truncated).units.mass_msun
    assert ps.mass.sum() == pytest.approx(expected, rel=1e-8)


# --- a density drawn in another potential --------------------------------------------------

STARS = ic.Plummer(M=1e5, rscale=0.005)         # a star cluster...
HALO = ic.Hernquist(M=1e9, rscale=1.)           # ...at the centre of a dark halo
_needs_galpy_drawing_tracers = pytest.mark.skipif(
    not _GALPY_DRAWS_TRACERS, reason="this galpy may draw tracers too fast")


@pytest.fixture
def tracers_allowed(monkeypatch):
    """Lifts the galpy-version guard, for what doesn't depend on the tracers' speeds."""
    monkeypatch.setattr(sampling, '_TRACERS_RELEASE', galpy.__version__)


@pytest.mark.skipif(_GALPY_DRAWS_TRACERS, reason="this galpy draws tracers right")
@pytest.mark.parametrize("draw", [
    pytest.param(lambda: ic.sample(STARS, 10, potential=[STARS, HALO]), id='sample'),
    pytest.param(lambda: ic.sample_components([STARS, HALO], n=[10, 10]), id='sample_components'),
])
def test_older_galpy_refuses_to_draw_a_density_in_another_potential(draw):
    with pytest.raises(ImportError, match=r"Drawing a density in another potential needs .*\(this is galpy "):
        draw()


@pytest.fixture
def galpy_as(monkeypatch):
    """Makes tambora take galpy for ``version``, with the fix in ``release``, and ``vouch`` as
    the environment variable that vouches for a development version (None: unset)."""
    def take(version, release=None, vouch=None):
        monkeypatch.setattr(sampling, '_galpy_version', version)
        monkeypatch.setattr(sampling, '_TRACERS_RELEASE', release)
        if vouch is None:
            monkeypatch.delenv('TAMBORA_GALPY_DEV_TRACERS', raising=False)
        else:
            monkeypatch.setenv('TAMBORA_GALPY_DEV_TRACERS', vouch)
    return take


@pytest.mark.parametrize("version, release, vouch, fixed", [
    pytest.param('1.12.0', None, None, False, id='1.12.0'),
    pytest.param('1.12.1', None, None, False, id='a_later_release_without_the_fix'),
    pytest.param('1.12.1', None, '1', False, id='a_release_cant_be_vouched_for'),
    pytest.param('1.12.1.dev0', None, None, False, id='a_development_version'),
    pytest.param('1.12.1.dev0', None, '1', True, id='a_development_version_vouched_for'),
    pytest.param('1.12.1.dev0', None, 'true', False, id='only_1_vouches'),
    pytest.param('1.12.0.dev0', None, '1', False, id='a_development_version_of_1.12.0'),
    pytest.param('1.13.0', '1.13.0', None, True, id='the_release_with_the_fix'),
    pytest.param('1.13.2', '1.13.0', None, True, id='a_release_after_it'),
    pytest.param('1.12.1', '1.13.0', None, False, id='a_release_before_it'),
    pytest.param('1.13.0.dev0', '1.13.0', '1', True, id='its_development_version_vouched_for'),
])
def test_galpy_draws_a_density_in_another_potential_from_the_release_with_the_fix(galpy_as, version, release,
                                                                                  vouch, fixed):
    galpy_as(version, release, vouch)
    assert sampling._tracers_fixed() is fixed


@pytest.mark.parametrize("version, release, match", [
    pytest.param('1.12.0', None, r"needs a fix to galpy that's in no release yet \(this is galpy 1\.12\.0\)\.$",
                 id='no_release_yet'),
    pytest.param('1.12.1', '1.13.0', r"needs galpy 1\.13\.0 or later \(this is galpy 1\.12\.1\)\.$", id='release'),
    pytest.param('1.12.1.dev0', None, r"needs a fix to galpy that's in no release yet \(this is galpy "
                                      r"1\.12\.1\.dev0\)\. This development version of galpy may have the fix: "
                                      r"set TAMBORA_GALPY_DEV_TRACERS=1 to use it\.$",
                 id='development_version'),
])
def test_galpy_without_the_fix_says_what_it_needs(galpy_as, version, release, match):
    galpy_as(version, release)
    with pytest.raises(ImportError, match=r"^Drawing a density in another potential " + match):
        GalpySampler(STARS, potential=[STARS, HALO])


_UNCHECKED = (r"with a development version of galpy \(1\.12\.1\.dev0\), as TAMBORA_GALPY_DEV_TRACERS=1 "
              r"asks\. tambora can't tell whether this galpy has the fix")


def test_a_development_galpy_vouched_for_draws_with_a_warning(galpy_as):
    galpy_as('1.12.1.dev0', vouch='1')
    with pytest.warns(UserWarning, match=_UNCHECKED):
        ps = ic.sample(STARS, 10, potential=[STARS, HALO], seed=1)
    assert ps.meta['model'].startswith('Plummer(M=100000, rscale=0.005) in ')


def test_a_tracer_df_warns_that_a_development_galpy_vouched_for_is_unchecked(galpy_as):
    galpy_as('1.12.1.dev0', vouch='1')
    d = _tracer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        GalpySampler(d)
    assert [str(w.message) for w in caught if 'too high' in str(w.message)] == []
    assert any(re.search(_UNCHECKED, str(w.message)) for w in caught)


def test_a_tracer_df_doesnt_warn_with_the_release_with_the_fix(galpy_as):
    galpy_as('1.13.0', release='1.13.0')
    d = _tracer()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        GalpySampler(d)


@pytest.mark.parametrize("potential", [
    pytest.param(STARS, id='itself'),
    pytest.param([STARS], id='list_of_itself'),
    pytest.param(ic.Plummer(M=1e5, rscale=0.005), id='an_equal_profile'),
])
def test_a_model_in_its_own_potential_is_sampled_as_usual(potential):
    # On any galpy: this is no tracer.
    ps, usual = ic.sample(STARS, 500, potential=potential, seed=1), ic.sample(STARS, 500, seed=1)
    assert ps.meta == usual.meta
    np.testing.assert_array_equal(ps.pos, usual.pos)
    np.testing.assert_array_equal(ps.vel, usual.vel)


def test_a_galpy_potential_in_itself_is_sampled_as_usual():
    pot = _plummer_pot()
    assert ic.sample(pot, 10, potential=[pot], seed=1).meta['model'] == 'isotropicPlummerdf(PlummerPotential)'


def test_a_list_in_itself_in_another_order_is_sampled_as_usual():
    # On any galpy: this is no tracer either.
    a, b = _plummer_pot(), _hernquist_pot()
    ps, usual = ic.sample([a, b], 500, potential=[b, [a]], seed=1), ic.sample([a, b], 500, seed=1)
    assert ps.meta == usual.meta
    np.testing.assert_array_equal(ps.pos, usual.pos)


def test_one_component_is_sampled_as_usual():
    (stars,) = ic.sample_components([STARS], n=[500], seed=1)
    usual = ic.sample(STARS, 500, seed=stars.meta['seed'])
    np.testing.assert_array_equal(stars.pos, usual.pos)


@pytest.mark.usefixtures('tracers_allowed')
def test_a_density_in_another_potential_gets_an_eddington_df_of_the_total():
    sampler = GalpySampler(STARS, potential=[STARS, HALO])
    assert type(sampler.df) is df.eddingtondf
    assert sampling._names(sampler.df._denspot) == 'PlummerPotential'
    assert sampling._names(sampler.df._pot) == 'PlummerPotential+HernquistPotential'
    assert sampler.describe() == 'Plummer(M=100000, rscale=0.005) in Plummer(M=100000, rscale=0.005) + ' \
                                 'Hernquist(M=1e+09, rscale=1)'


@pytest.mark.usefixtures('tracers_allowed')
def test_ic_sample_draws_a_density_in_another_potential_with_its_own_mass():
    ps = ic.sample(STARS, 1000, potential=[STARS, HALO], seed=2)
    assert ps.mass.sum() == pytest.approx(1e5, rel=1e-6)    # what's beyond rmax is left out
    assert ps.meta == {'backend': 'galpy', 'model': GalpySampler(STARS, potential=[STARS, HALO]).describe(),
                       'seed': 2}


@pytest.mark.usefixtures('tracers_allowed')
def test_ic_sample_components_draws_each_with_its_own_mass():
    stars, dm = ic.sample_components([STARS, HALO], n=[1000, 2000], seed=4)
    assert stars.mass.sum() == pytest.approx(1e5, rel=1e-6)
    # A Hernquist is cut at 10^4 of its scale radii, which leaves out 0.02% of its mass.
    assert dm.mass.sum() == pytest.approx(1e9 * (1e4 / (1e4 + 1.))**2, rel=1e-6)
    assert dm.meta['model'].startswith('Hernquist(M=1e+09, rscale=1) in Plummer(')


@pytest.mark.usefixtures('tracers_allowed')
@pytest.mark.parametrize("profile, native", [
    pytest.param(ic.Plummer(M=1e6, rscale=0.01), lambda: _plummer_pot(M=1e6, b=0.01), id='Plummer'),
    pytest.param(HALO, lambda: _hernquist_pot(M=1e9, a=1.), id='Hernquist'),
    pytest.param(ic.King(M=1e6, W0=6., rt=0.05),
                 lambda: potential.KingPotential(W0=6., M=1e6 * u.Msun, rt=0.05 * u.kpc, ro=RO, vo=VO), id='King'),
    pytest.param(ic.TruncatedNFW(M=1e9, rscale=1., rtrunc=10.),
                 lambda: potential.ExpTruncNFWPotential(mass=1e9 * u.Msun, a=1. * u.kpc, rc=10. * u.kpc, ro=RO, vo=VO),
                 id='TruncatedNFW', marks=_needs_exp_trunc_nfw),
])
def test_a_profile_in_the_potential_is_the_galpy_potential_it_stands_for(profile, native):
    # The profile is built in the stars' units; compare its enclosed mass with the same model
    # built directly in galpy, in other units.
    sampler = GalpySampler(STARS, potential=[STARS, profile])
    _, member = sampling._iter_components(sampler.df._pot)
    radii = [0.003, 0.01, 0.03, 0.3, 3., 30.]     # kpc
    got = [potential.mass(member, r / sampler.units.length_kpc, use_physical=False) * sampler.units.mass_msun
           for r in radii]
    want = [potential.mass(native(), r / RO, use_physical=False) * mass_in_msol(VO, RO) for r in radii]
    np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.usefixtures('tracers_allowed')
def test_a_tracers_radii_follow_its_own_density():
    pos, _ = GalpySampler(STARS, potential=[STARS, HALO]).draw(10_000, 3)
    assert stats.kstest(np.linalg.norm(pos, axis=1), lambda r: _plummer_cdf(r, b=0.005)).pvalue > 1e-3


@pytest.mark.usefixtures('tracers_allowed')
def test_a_profile_among_galpy_potentials_takes_their_units():
    halo = _hernquist_pot(M=1e9, a=1.)
    mixed = GalpySampler(STARS, potential=[STARS, halo])
    assert (mixed.units.length_kpc, mixed.units.velocity_kms) == (RO, VO)
    assert mixed.total_mass == pytest.approx(1e5, rel=1e-6)
    # The same model as with the halo as a profile, built in other units. galpy tabulates it on
    # grids set by the units, so the radii agree to 1e-5 and the speeds' quantiles to 0.2%; a
    # velocity unit of 220 instead of 230 km/s would put them 4.5% apart.
    (pos, vel), (pos0, vel0) = mixed.draw(2000, 4), GalpySampler(STARS, potential=[STARS, HALO]).draw(2000, 4)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), np.linalg.norm(pos0, axis=1), rtol=1e-4)
    quantiles = [10, 25, 50, 75, 90]
    np.testing.assert_allclose(np.percentile(np.linalg.norm(vel, axis=1), quantiles),
                               np.percentile(np.linalg.norm(vel0, axis=1), quantiles), rtol=5e-3)


@pytest.mark.parametrize("tracers", ['allowed', 'refused'])
@pytest.mark.parametrize("model, potential, error, match", [
    pytest.param(lambda: ic.King(M=1e4, W0=6., rt=0.03), lambda: [HALO], TypeError,
                 "A King model is defined by its own potential", id='King'),
    pytest.param(lambda: STARS, lambda: [STARS, potential.MiyamotoNagaiPotential(
                     amp=1e10 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc, ro=RO, vo=VO)], TypeError,
                 "Can't sample in a MiyamotoNagaiPotential: galpy's samplers need a spherical potential",
                 id='not_spherical'),
    pytest.param(lambda: STARS, lambda: 'halo', TypeError,
                 r"The potential takes galpy potentials and tambora's profiles, got str\.$", id='typo'),
    pytest.param(lambda: STARS, lambda: [STARS, 'halo'], TypeError,
                 r"The potential takes galpy potentials and tambora's profiles, got str \(element 1 of the list\)",
                 id='not_a_potential'),
    pytest.param(lambda: STARS, lambda: [STARS, [HALO, 'halo']], TypeError,
                 r"got str \(element \[1\]\[1\] of the list\)", id='nested'),
    pytest.param(lambda: [STARS, 'halo'], lambda: [HALO], TypeError,
                 r"The model takes galpy potentials and tambora's profiles, got str \(element 1 of the list\)",
                 id='model_not_a_potential'),
    pytest.param(lambda: STARS, lambda: [], ValueError, "The potential is an empty list", id='empty'),
    pytest.param(lambda: potential.BurkertPotential(amp=1., a=0.5 * u.kpc, ro=RO, vo=VO), lambda: [HALO],
                 TypeError, r"Can't sample a BurkertPotential: .*\(_ddensdr and _d2densdr2\)", id='density'),
    pytest.param(lambda: _plummer_pot(), lambda: [_plummer_pot(), potential.HernquistPotential(ro=8., vo=VO)],
                 ValueError, r"The galpy potentials have different units \(ro, vo\)", id='mixed_units'),
    pytest.param(lambda: potential.NFWPotential(amp=1e9 * u.Msun, a=1. * u.kpc, ro=RO, vo=VO), lambda: [HALO],
                 ValueError, "Can't sample a NFWPotential: its mass is infinite", id='infinite_mass'),
    pytest.param(lambda: ic.TruncatedNFW(M=1e9, rscale=1., rtrunc=0.1), lambda: [STARS], ValueError,
                 "galpy's sampler is unreliable when rtrunc is under a fifth of rscale", id='truncated_nfw',
                 marks=_needs_exp_trunc_nfw),
])
def test_what_cant_be_drawn_in_another_potential_says_why(galpy_as, tracers, model, potential, error, match):
    # Mistakes are found before galpy's version, so they're never taken for a need for newer galpy.
    if tracers == 'allowed':
        galpy_as(galpy.__version__, release=galpy.__version__)
    else:
        galpy_as('1.12.0')
    with pytest.raises(error, match=match):
        GalpySampler(model(), potential=potential())


def test_a_component_that_isnt_a_model_is_refused_on_any_galpy():
    with pytest.raises(TypeError, match=r"The potential takes galpy potentials and tambora's profiles, "
                                        r"got str \(element 1 of the list\)"):
        ic.sample_components([STARS, 'halo'], n=[10, 10])


@pytest.mark.usefixtures('tracers_allowed')
def test_a_list_of_profiles_is_drawn_in_another_potential_as_one_model():
    ps = ic.sample([STARS, HALO], 1000, potential=[STARS, HALO, _hernquist_pot()], seed=1)
    assert ps.mass.sum() == pytest.approx(1e5 + 1e9, rel=1e-4)
    assert ps.meta['model'] == ('Plummer(M=100000, rscale=0.005) + Hernquist(M=1e+09, rscale=1) in '
                                'Plummer(M=100000, rscale=0.005) + Hernquist(M=1e+09, rscale=1) + HernquistPotential')


@pytest.mark.usefixtures('tracers_allowed')
def test_a_galpy_potential_without_physical_units_warns_in_the_potential_too():
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        GalpySampler(STARS, potential=[STARS, potential.HernquistPotential()])


def test_a_galpy_df_cant_take_another_potential():
    with pytest.raises(TypeError, match="A galpy DF has its potential already"):
        GalpySampler(_plummer(), potential=[_plummer_pot(), _hernquist_pot()])


# galpy 1.12.0 gives the stars here a virial ratio of 1.4, and galpy main before the
# follow-up to #1568 1.05; drawn right, it's 1.005 +/- 0.005 with 50,000 particles.
_HALOS = [pytest.param(HALO, id='Hernquist'),
          pytest.param(ic.TruncatedNFW(M=1e9, rscale=1., rtrunc=10.), id='TruncatedNFW', marks=_needs_exp_trunc_nfw)]


@_needs_galpy_drawing_tracers
@pytest.mark.parametrize("halo", _HALOS)
def test_stars_at_the_centre_of_a_dark_halo_are_in_equilibrium_in_the_total_potential(halo):
    sampler = GalpySampler(STARS, potential=[STARS, halo])
    assert _virial_ratio(sampler, sampler.df._pot, n=50_000) == pytest.approx(1., abs=0.025)


@_needs_galpy_drawing_tracers
@pytest.mark.parametrize("halo", _HALOS)
def test_the_dark_halo_around_the_stars_is_in_equilibrium_in_the_total_potential(halo):
    sampler = GalpySampler(halo, potential=[STARS, halo])
    assert _virial_ratio(sampler, sampler.df._pot, n=20_000) == pytest.approx(1., abs=0.025)
