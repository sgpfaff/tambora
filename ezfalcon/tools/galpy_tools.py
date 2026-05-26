import numpy as np
from scipy.special import exp1
from scipy import integrate, interpolate
from ..util._galpy_bridge import _check_physical
from ..util.units import KMS_TO_KPCGYR
from galpy import df, potential
from galpy.util.conversion import mass_in_msol
from galpy.potential.SphericalPotential import SphericalPotential
lunit = "kpc"
# galpy default unit scales (used for natural-unit conversions)
_GALPY_RO = 8.0    # kpc
_GALPY_VO = 220.0  # km/s

# --- Interface tools ----------------------------------------------------------------


def galpy_orbit_to_ezfalcon(orb):
    r'''Convert a galpy orbit object to ezfalcon compatible pos and vel arrays.
     
     Parameters
     ----------
     orb : galpy.orbit.Orbit
         A galpy orbit object. You must turn physical on before passing it in, e.g
         with `orb.turn_physical_on()`.
         
     Returns
     -------
     pos : (N, 3) array
         Cartesian positions of sampled particles.
         Units: `kpc`
     vel : (N, 3) array
         Cartesian velocities of sampled particles.
         Units: `km / s`
     '''
    _check_physical(orb)
    pos = np.atleast_2d(np.array([orb.x(return_physical=True), 
                                  orb.y(return_physical=True), 
                                  orb.z(return_physical=True)]).T) # kpc
    vel = np.atleast_2d(np.array([orb.vx(return_physical=True), 
                                  orb.vy(return_physical=True), 
                                  orb.vz(return_physical=True)]).T) # km / s
    return pos, vel

# --- Sampling tools ----------------------------------------------------------------

SUPPORTED_GALPY_DFS = (
    df.isotropicHernquistdf,
    df.isotropicPlummerdf,
    df.isotropicNFWdf,
    df.kingdf,
    df.eddingtondf
)

def _check_df(df):
    '''Currently only spherical galpy dfs are supported.'''
    if not isinstance(df, SUPPORTED_GALPY_DFS):
        raise ValueError(f"Unsupported galpy df type: {type(df)}. Supported types: {SUPPORTED_GALPY_DFS}")

def galpydfsampler(df, n, m_total, rmin=0.0, center_pos=[0, 0, 0], 
                   center_vel=[0, 0, 0]):
    '''
    Sample from a galpy spherical distribution 
    function and return ezfalcon compatible 
    positions and velocities.

    Parameters
    ----------
    df : galpy.df
        A galpy distribution function object.
    n : int
        Number of particles to sample.
    m_total : float
        Total mass of the sampled component.
        Units: `Msun`
    rmin : float, optional
        Minimum radius at which to sample. Default is 0.
        Units: `kpc`
    center_pos : array-like, optional
        Position of the center of the sampled component.
        Units: `kpc`
    center_vel : array-like, optional
        Velocity of the center of the sampled component.
        Units: `km / s`

    Returns
    -------
    pos : (N, 3) array
        Cartesian positions of sampled particles.
        Units: `kpc`
    vel : (N, 3) array
        Cartesian velocities of sampled particles.
        Units: `km / s`
    masses : (N,) array
        Masses of sampled particles.
        Units: `Msun`
    '''
    _check_physical(df)
    _check_df(df)
    o = df.sample(n=n, rmin=rmin/_GALPY_RO, return_orbit=True)
    pos, vel = galpy_orbit_to_ezfalcon(o)
    pos += np.asarray(center_pos)[:,None].T
    vel += np.asarray(center_vel)[:,None].T
    return pos, vel, np.repeat(m_total / n, n)

def galpysampler(pot, n, m_total, rmin=0.0, 
                 center_pos=[0, 0, 0], center_vel=[0, 0, 0],
                 df_kwargs={}):
    '''
    Sample from a galpy potential and return ezfalcon compatible 
    positions and velocities. Only supports spherical potentials.

    Parameters
    ----------
    pot : galpy.potential
        A galpy potential object. You 
        must turn physical on before passing it in, e.g. with `pot.turn_physical_on()`.
    df_kwargs : dict, optional
        Additional keyword arguments to pass to the galpy DF constructor.

    Returns
    -------
    pos : (N, 3) array
        Cartesian positions of sampled particles.
        Units: `kpc`
    vel : (N, 3) array
        Cartesian velocities of sampled particles.
        Units: `km / s`
    masses : (N,) array
        Masses of sampled particles.
        Units: `Msun`

    '''
    _check_physical(pot)
    if isinstance(pot, potential.PlummerPotential):
        _df = df.isotropicPlummerdf(pot=pot, **df_kwargs)
    elif isinstance(pot, potential.HernquistPotential):
        _df = df.isotropicHernquistdf(pot=pot, **df_kwargs)
    elif isinstance(pot, potential.NFWPotential):
        _df = df.isotropicNFWdf(pot=pot, **df_kwargs)
    else:
        _df = df.eddingtondf(pot=pot, **df_kwargs)
    return galpydfsampler(_df, n=n, m_total=m_total, rmin=rmin, center_pos=center_pos, 
                            center_vel=center_vel)


def mkPlummer_galpy(m, b, n, center_pos=[0, 0, 0], center_vel=[0, 0, 0]):
    '''
    Generate the positions, velocities, and masses of 
    a Plummer sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the Plummer sphere.
        Units: `Msun`
    b : float
        Scale radius of the Plummer sphere.
        Units: `kpc`
    n : int
        Number of particles to sample.
    rmin : float, optional
        Minimum radius at which to sample. Default is 0.
        Units: `kpc`
    center_pos : array-like, optional
        Position of the center of the Plummer sphere.
        Units: `kpc`
    center_vel : array-like, optional
        Velocity of the center of the Plummer sphere.
        Units: `km / s`
    '''
    pot = potential.PlummerPotential(
        amp = m / mass_in_msol(_GALPY_VO, _GALPY_RO),
        b = b / _GALPY_RO,
        ro = _GALPY_RO, vo=_GALPY_VO,
    )
    return galpysampler(pot, n, m, center_pos=center_pos, center_vel=center_vel)


def mkKing_galpy(m:float, n, W0:float, rt=None, npts=None, rmin=0.0,
                 center_pos=[0, 0, 0], center_vel=[0, 0, 0]):
    '''
    Generate the positions, velocities, and masses of
    a King sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the King sphere.
        Units: `Msun`
    n : int
        Number of particles to sample.
    W0 : float
        Dimensionless central potential 
        :math:`W_0 = \\Psi(0)/\\sigma^2` (in practice, needs to be :math:`\\lesssim 200`, where the DF is essentially isothermal).
    rt : float, optional
        Tidal radius.
        Units: `kpc`
    npts : int
        Number of points to use to solve for :math:`\\Psi(r)`.
    rmin : float, optional
            Minimum radius at which to sample. Default is 0.
            Units: `kpc`
    center_pos : array-like, optional
        Position of the center of the King sphere.
        Units: `kpc`
    center_vel : array-like, optional
        Velocity of the center of the King sphere.
        Units: `km / s`

    Returns
    -------
    pos : (N, 3) array
        Cartesian positions of sampled particles.
        Units: `kpc`
    vel : (N, 3) array
        Cartesian velocities of sampled particles.
        Units: `km / s`
    masses : (N,) array
        Masses of sampled particles.
        Units: `Msun`
    '''
    df_kwargs = {}
    if rt is not None:
        df_kwargs['rt'] = rt / _GALPY_RO
    if npts is not None:
        df_kwargs['npt'] = npts
    sat_df = df.kingdf(
        W0=W0, M=m / mass_in_msol(_GALPY_VO, _GALPY_RO),
        ro=_GALPY_RO, vo=_GALPY_VO, **df_kwargs,
    )
    return galpydfsampler(sat_df, n, m, rmin=rmin, 
                          center_pos=center_pos, center_vel=center_vel)


def mkNFW_galpy(m, n, rmin=0.0, center_pos=[0, 0, 0], center_vel=[0, 0, 0],
                nfw_df_kwargs={}, nfw_kwargs={}):
    '''
    Generate the positions, velocities, and masses of
    a NFW sphere using galpy.

    Parameters
    ----------
    m : float
        Total mass of the NFW sphere.
        Units: `Msun`
    n : int
        Number of particles to sample.
    rmin : float, optional
        Minimum radius at which to sample. Default is 0.
        Units: `kpc`
    center_pos : array-like, optional
        Position of the center of the NFW sphere.
        Units: `kpc`
    center_vel : array-like, optional
        Velocity of the center of the NFW sphere.
        Units: `km / s`
    nfw_df_kwargs : keyword arguments to pass to the galpy isotropicNFWdf sampler.
         See galpy.df.isotropicNFWdf for details. Relevant kwargs include:
            - widrow (bool, optional):
                If True, use the approximate form from Widrow (2000), otherwise use improved fit that has <~1e-5 relative density errors
            - rmax (float or Quantity, optional):
                Maximum radius to consider; set to numpy.inf to evaluate NFW w/o cut-off
    nfw_kwargs : keyword arguments to pass to the galpy NFWPotential constructor. See galpy.potential.NFWPotential for details. Relevant kwargs include:
    
    Returns
    -------
    pos : (N, 3) array
        Cartesian positions of sampled particles.
        Units: `kpc`
    vel : (N, 3) array
        Cartesian velocities of sampled particles.
        Units: `km / s`
    masses : (N,) array
        Masses of sampled particles.
        Units: `Msun`
    '''
    pot = potential.NFWPotential(**nfw_kwargs)
    return galpysampler(pot, n, m, rmin=rmin, 
                        center_pos=center_pos, center_vel=center_vel,
                        df_kwargs=nfw_df_kwargs)
class TruncatedNFWPotential(SphericalPotential):
    r'''Exponentially-truncated NFW potential.

    Density:

    .. math::

        \rho(r) = \frac{\rho_s\,e^{-r/r_c}}{(r/a)\,(1 + r/a)^2}

    where :math:`\rho_s = \mathrm{amp}/(4\pi a^3)`, matching galpy's NFW
    amplitude convention. All methods are closed-form. Mass and outer
    potential integrals are expressible via the exponential integral
    :math:`E_1` (`scipy.special.exp1`); a small-:math:`r` Taylor fallback
    avoids catastrophic cancellation when :math:`r \ll a, r_c`.

    Parameters
    ----------
    amp : float
        Mass-scale amplitude :math:`4\pi\,\rho_s\,a^3`. galpy multiplies this
        externally for `pot.dens`, `pot(r)`, etc.
    a : float
        NFW scale radius.
    rc : float
        Exponential truncation radius.
    ro, vo : float, optional
        galpy unit scales (kpc, km/s).
    '''

    def __init__(self, amp=1.0, a=1.0, rc=2.0, ro=None, vo=None):
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._a = a
        self._rc = rc
        self._alpha = a / rc
        self._exp_alpha = np.exp(self._alpha)
        self._E1_alpha = exp1(self._alpha)
        self.hasC = False
        self.hasC_dxdv = False
        self._scale = a
        # Threshold below which closed-form M(<r) suffers cancellation; use
        # series expansion instead. r/min(a, rc) < eps gives roughly eps^2
        # relative truncation error in F(r).
        self._small_r_thresh = 1e-3 * min(a, rc)

    # F(r) = M(<r) / amp, dimensionless mass scale (so that
    # _rforce = -F(r)/r^2 in amp-units).
    def _F(self, r):
        # closed form
        # F(r) = exp(a)(1+a)[E1(a) - E1(b)] - 1 + (a_/(a_+r)) exp(-r/rc),
        # where a = alpha = a_/rc, b = (a_+r)/rc.
        # For r << a_, rc, both subtractions cancel; use Taylor in r.
        r = np.asarray(r, dtype=float)
        small = r < self._small_r_thresh
        out = np.empty_like(r) if r.ndim else np.array(0.0)
        if r.ndim == 0:
            return self._F_scalar(float(r))
        out[~small] = self._F_closed(r[~small])
        out[small] = self._F_series(r[small])
        return out

    def _F_scalar(self, r):
        if r < self._small_r_thresh:
            return self._F_series(r)
        return self._F_closed(r)

    def _F_closed(self, r):
        a, rc = self._a, self._rc
        beta = (a + r) / rc
        return (self._exp_alpha * (1.0 + self._alpha)
                * (self._E1_alpha - exp1(beta))
                - 1.0
                + a * np.exp(-r / rc) / (a + r))

    def _F_series(self, r):
        # F(r) = (1/a^2) * [ r^2/2 - (r^3/3)(1/rc + 2/a)
        #                  + (r^4/4)(1/(2 rc^2) + 2/(rc a) + 3/a^2)
        #                  - (r^5/5)(1/(6 rc^3) + 1/(rc^2 a)
        #                           + 3/(rc a^2) + 4/a^3) + ... ]
        a, rc = self._a, self._rc
        c2 = 0.5
        c3 = -(1.0 / rc + 2.0 / a) / 3.0
        c4 = (0.5 / rc / rc + 2.0 / (rc * a) + 3.0 / (a * a)) / 4.0
        c5 = -(1.0 / (6.0 * rc**3) + 1.0 / (rc * rc * a)
               + 3.0 / (rc * a * a) + 4.0 / a**3) / 5.0
        return (r * r / (a * a)) * (c2 + r * (c3 + r * (c4 + r * c5)))

    # G(r) := 4*pi * \int_r^infty rho(s) s ds / amp
    #      = exp(-r/rc)/(a + r) - exp(alpha) E1(beta) / rc
    # Used in the potential.
    def _G(self, r):
        a, rc = self._a, self._rc
        beta = (a + r) / rc
        return np.exp(-r / rc) / (a + r) - self._exp_alpha * exp1(beta) / rc

    # rho(r) / amp: galpy NFW convention places the 1/(4*pi a^3) factor here
    # so that the user-facing pot.dens(r) = rho_s * exp(-r/rc) / [(r/a)(1+r/a)^2].
    def _rdens(self, r, t=0.0):
        a = self._a
        return np.exp(-r / self._rc) / (
            4.0 * np.pi * a * a * r * (1.0 + r / a) ** 2)

    # _revaluate / _rforce / _r2deriv: amp is applied externally by Potential.
    # F(r) ~ r^2/(2 a^2) near the origin, so F/r and F/r^2 have finite
    # limits we substitute by hand. Without this, _evaluate(0)/_rforce(0)
    # return NaN; galpy's eddingtondf seeds its rphi spline with r=0 and
    # scipy>=1.13's strict monotonicity check then raises.
    def _revaluate(self, r, t=0.0):
        r_arr = np.asarray(r, dtype=float)
        if r_arr.ndim == 0:
            if r_arr == 0.0:
                return -self._G(0.0)
            return -(self._F(r_arr) / r_arr + self._G(r_arr))
        out = -self._G(r_arr)
        nz = r_arr != 0.0
        out[nz] -= self._F(r_arr[nz]) / r_arr[nz]
        return out

    def _rforce(self, r, t=0.0):
        r_arr = np.asarray(r, dtype=float)
        zero_limit = -0.5 / (self._a * self._a)
        if r_arr.ndim == 0:
            if r_arr == 0.0:
                return zero_limit
            return -self._F(r_arr) / (r_arr * r_arr)
        out = np.full_like(r_arr, zero_limit)
        nz = r_arr != 0.0
        out[nz] = -self._F(r_arr[nz]) / (r_arr[nz] * r_arr[nz])
        return out

    def _r2deriv(self, r, t=0.0):
        return 4.0 * np.pi * self._rdens(r) - 2.0 * self._F(r) / r ** 3

    # _ddensdr / _d2densdr2: galpy calls these directly (no amp wrapping),
    # so we bake in self._amp.
    def _ddensdr(self, r, t=0.0):
        rho_phys = self._amp * self._rdens(r)
        g = 1.0 / r + 2.0 / (self._a + r) + 1.0 / self._rc
        return -rho_phys * g

    def _d2densdr2(self, r, t=0.0):
        rho_phys = self._amp * self._rdens(r)
        g = 1.0 / r + 2.0 / (self._a + r) + 1.0 / self._rc
        gprime = -1.0 / (r * r) - 2.0 / (self._a + r) ** 2
        return rho_phys * (g * g - gprime)
