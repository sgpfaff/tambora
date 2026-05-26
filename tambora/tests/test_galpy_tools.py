import pytest
from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential, PlummerPotential, NFWPotential
from ezfalcon.tools import galpydfsampler, galpy_orbit_to_ezfalcon, mkPlummer_galpy, mkKing_galpy, mkNFW_galpy
from ezfalcon.tools.galpy_tools import _check_df, galpysampler
import numpy as np
import astropy.units as u

@pytest.fixture
def hernquist_df():
    pot = HernquistPotential()
    pot.turn_physical_on()
    df = isotropicHernquistdf(pot=pot)
    df.turn_physical_on()
    return df

def test_sampler_shapes(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=100, m_total=1e6)
    assert pos.shape == (100, 3)
    assert vel.shape == (100, 3)
    assert masses.shape == (100,)

def test_sampler_masses(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=200, m_total=1e6)
    np.testing.assert_allclose(masses, 1e6 / 200)

def test_sampler_center_offset(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=500, m_total=1e6,
                                       center_pos=[100, 0, 0])
    assert np.mean(pos[:, 0]) > 50  # shifted well away from origin

def test_orbit_to_ezfalcon_vel_units():
    from galpy.orbit import Orbit
    # Orbit at R=8 kpc, vR=0, vT=220 km/s, z=0, vz=0, phi=0
    # (natural units: R=1, vT=1 with ro=8 kpc, vo=220 km/s)
    o = Orbit([1., 0., 1., 0., 0., 0.], ro=8., vo=220.)
    o.turn_physical_on()
    pos, vel = galpy_orbit_to_ezfalcon(o)
    expected_vT_kpcgyr = (220 * u.km/u.s).to(u.kpc/u.Gyr).value
    assert np.max(np.abs(vel)) < 300.0  # kpc/Gyr are ~km/s (~225)
    assert np.max(np.abs(vel)) > 1.0  # but not zero

def test_orbit_to_ezfalcon_shapes():
    from galpy.orbit import Orbit
    o = Orbit.from_name("LMC")
    o.turn_physical_on()
    pos, vel = galpy_orbit_to_ezfalcon(o)
    assert pos.shape == (1, 3)
    assert vel.shape == (1, 3)

# --- _check_df ----------------------------------------------------------------------------- #

def test_check_df_rejects_unsupported():
    with pytest.raises(ValueError, match="Unsupported galpy df type"):
        _check_df("not_a_df")

# --- galpysampler dispatch ------------------------------------------------------------------ #

def test_galpysampler_plummer():
    pot = PlummerPotential()
    pot.turn_physical_on()
    pos, vel, masses = galpysampler(pot, n=50, m_total=1e6)
    assert pos.shape == (50, 3)
    assert vel.shape == (50, 3)
    np.testing.assert_allclose(masses.sum(), 1e6)

def test_galpysampler_nfw():
    pot = NFWPotential()
    pot.turn_physical_on()
    pos, vel, masses = galpysampler(pot, n=50, m_total=1e6)
    assert pos.shape == (50, 3)

# --- mkPlummer_galpy ------------------------------------------------------------------------ #

def test_mkPlummer_shapes():
    pos, vel, masses = mkPlummer_galpy(m=1e6, b=1.0, n=50)
    assert pos.shape == (50, 3)
    assert vel.shape == (50, 3)
    assert masses.shape == (50,)
    np.testing.assert_allclose(masses.sum(), 1e6)

def test_mkPlummer_center_offset():
    pos, vel, masses = mkPlummer_galpy(m=1e6, b=1.0, n=200,
                                       center_pos=[50, 0, 0])
    assert np.mean(pos[:, 0]) > 20

# --- mkKing_galpy --------------------------------------------------------------------------- #

def test_mkKing_shapes():
    pos, vel, masses = mkKing_galpy(m=1e6, n=50, W0=3.0)
    assert pos.shape == (50, 3)
    assert vel.shape == (50, 3)
    assert masses.shape == (50,)
    np.testing.assert_allclose(masses.sum(), 1e6)

# --- mkNFW_galpy ---------------------------------------------------------------------------- #

def test_mkNFW_shapes():
    pos, vel, masses = mkNFW_galpy(m=1e6, n=50)
    assert pos.shape == (50, 3)
    assert vel.shape == (50, 3)
    assert masses.shape == (50,)
    np.testing.assert_allclose(masses.sum(), 1e6)

def test_mkNFW_with_df_kwargs():
    pos, vel, masses = mkNFW_galpy(m=1e6, n=50,
                                   nfw_df_kwargs={'widrow': True, 'rmax': 5.0})
    assert pos.shape == (50, 3)

# --- Radial density profile tests ----------------------------------------------------------- #

_GALPY_RO = 8.0

def test_mkPlummer_density_profile():
    """Sampled Plummer radii follow the analytical CDF: F(r) = r^3 / (r^2 + b^2)^(3/2)."""
    from scipy.stats import kstest
    b = 1.0  # kpc
    pos, vel, masses = mkPlummer_galpy(m=1e10, b=b, n=5000)
    r = np.linalg.norm(pos, axis=1)
    _, pvalue = kstest(r, lambda x: np.where(x > 0, x**3 / (x**2 + b**2)**1.5, 0.0))
    assert pvalue > 0.001, f"Plummer KS test failed: p={pvalue:.4f}"

def test_mkNFW_density_profile():
    """Sampled NFW radii follow the analytical enclosed-mass CDF."""
    from scipy.stats import kstest
    rmax = 10.0  # natural units (same as position units for default NFWPotential)
    # Default NFWPotential has a=1.0 (scale radius in natural units).
    # Without explicit turn_physical_on, positions come out in natural units.
    r_s = 1.0

    pos, vel, masses = mkNFW_galpy(m=1e10, n=5000,
                                   nfw_df_kwargs={'rmax': rmax})
    r = np.linalg.norm(pos, axis=1)

    # NFW CDF: M(<r)/M(<rmax), where M(r) propto ln(1+x) - x/(1+x), x = r/r_s
    x_max = rmax / r_s
    f_max = np.log(1 + x_max) - x_max / (1 + x_max)

    def nfw_cdf(r_val):
        x = np.asarray(r_val) / r_s
        return np.where(r_val > 0, (np.log(1 + x) - x / (1 + x)) / f_max, 0.0)

    _, pvalue = kstest(r, nfw_cdf)
    assert pvalue > 0.001, f"NFW KS test failed: p={pvalue:.4f}"

def test_mkKing_density_profile():
    """Sampled King radii follow the CDF derived from the King density profile."""
    from scipy.stats import kstest
    from scipy.interpolate import interp1d
    from scipy.integrate import cumulative_trapezoid
    from galpy.df import kingdf
    from galpy.util.conversion import mass_in_msol

    W0 = 3.0
    m = 1e6
    ro, vo = _GALPY_RO, 220.0

    pos, vel, masses = mkKing_galpy(m=m, n=5000, W0=W0)
    r_kpc = np.linalg.norm(pos, axis=1)
    r_nat = r_kpc / ro

    # Build theoretical CDF in natural units
    king = kingdf(W0=W0, M=m / mass_in_msol(vo, ro), ro=ro, vo=vo)
    r_grid = np.linspace(1e-6, king.rt * 0.9999, 2000)
    rho_grid = np.array([king.dens(ri) for ri in r_grid])
    dm_dr = 4 * np.pi * r_grid**2 * rho_grid
    m_enc = np.concatenate([[0], cumulative_trapezoid(dm_dr, r_grid)])
    cdf_interp = interp1d(r_grid, m_enc / m_enc[-1],
                          bounds_error=False, fill_value=(0, 1))

    _, pvalue = kstest(r_nat, cdf_interp)
    assert pvalue > 0.001, f"King KS test failed: p={pvalue:.4f}"