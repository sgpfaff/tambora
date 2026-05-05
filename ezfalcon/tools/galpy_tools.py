import numpy as np
from scipy.special import exp1
from scipy import integrate, interpolate
from ..util._galpy_bridge import _check_physical
from ..util.units import KMS_TO_KPCGYR
from galpy import df, potential
from galpy.potential.Potential import _evaluatePotentials
from galpy.potential.SphericalPotential import SphericalPotential
from galpy.util.conversion import mass_in_msol
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
    elif isinstance(pot, TruncatedNFWPotential):
        # For truncated NFW, use a default rmax of 60*rc if not provided
        if 'rmax' not in df_kwargs or df_kwargs['rmax'] is None:
            df_kwargs['rmax'] = 60.0 * pot._rc * _GALPY_RO
        _df = TruncatedNFWDF(pot=pot, **df_kwargs)
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


def mkHernquist_galpy(m, a, n, center_pos=[0, 0, 0], center_vel=[0, 0, 0]):
    '''
    Generate the positions, velocities, and masses of 
    a Hernquist sphere using galpy.

    The Hernquist profile has a cusp :math:`\\rho \\propto r^{-1}` at small radii
    (matching the NFW inner slope) and falls off as :math:`\\rho \\propto r^{-4}`
    at large radii, giving a finite total mass.  It admits an exact isotropic 
    distribution function and is self-consistent for self-gravitating N-body 
    simulations.

    Parameters
    ----------
    m : float
        Total mass of the Hernquist sphere.
        Units: `Msun`
    a : float
        Scale radius of the Hernquist sphere.
        Units: `kpc`
    n : int
        Number of particles to sample.
    center_pos : array-like, optional
        Position of the center of the Hernquist sphere.
        Units: `kpc`
    center_vel : array-like, optional
        Velocity of the center of the Hernquist sphere.
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
    # galpy HernquistPotential total mass = amp * mass_in_msol / 2, so amp = 2m / mass_in_msol
    pot = potential.HernquistPotential(
        amp = 2.0 * m / mass_in_msol(_GALPY_VO, _GALPY_RO),
        a = a / _GALPY_RO,
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
    def _revaluate(self, r, t=0.0):
        return -(self._F(r) / r + self._G(r))

    def _rforce(self, r, t=0.0):
        return -self._F(r) / (r * r)

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


class TruncatedNFWDF(df.eddingtondf):
    r'''Eddington DF specialized for our truncated NFW potential.

    galpy's default `eddingtondf` builds multiple interpolators that can
    fail numerically with truncated potentials due to flat outer regimes
    and spurious NaN values from r=0 divergences. This subclass:
    1. Constrains the r(phi) and xi/CMF radial grids to the truncation radius
    2. Fixes _Emin to avoid NaN enegy boundaries
    3. Robustifies f(E) sampling by pruning duplicate energy points
    '''

    def __init__(self, pot=None, denspot=None, rmax=None, scale=None,
                 ro=None, vo=None):
        if denspot is None:
            denspot = pot
        super().__init__(pot=pot, denspot=denspot, rmax=rmax,
                         scale=scale, ro=ro, vo=vo)
        # If the truncation radius is finite and Emin is not finite, set Emin
        # to the potential at the truncation radius to avoid NaN energies.
        if np.isfinite(self._rmax) and not np.isfinite(self._Emin):
            self._Emin = _evaluatePotentials(pot, self._rmax, 0)
        if not np.isfinite(self._Emin):
            rmin = 1e-6 * self._scale
            self._Emin = _evaluatePotentials(self._pot, rmin, 0)

    def _setup_rphi_interpolator(self, r_a_min=1e-6, r_a_max=1e6, nra=10001):
        if np.isfinite(self._rmax):
            r_a_max = min(r_a_max, self._rmax / self._scale)
            r_a_max = max(r_a_max, r_a_min * 10.0)
        phi_at_zero = _evaluatePotentials(self._pot, 0.0, 0)
        if np.isfinite(phi_at_zero):
            r_a_values = np.concatenate(
                (np.array([0.0]), np.geomspace(r_a_min, r_a_max, nra))
            )
        else:
            r_a_values = np.geomspace(r_a_min, r_a_max, nra)
        phis = np.array(
            [_evaluatePotentials(self._pot, r * self._scale, 0)
             for r in r_a_values]
        )
        if np.any(np.diff(phis) <= 0):
            phim = np.maximum.accumulate(phis)
            indx_rm = np.where(np.diff(phim) == 0)[0]
            phis = np.delete(phim, indx_rm)
            r_a_values = np.delete(r_a_values, indx_rm)
        return interpolate.InterpolatedUnivariateSpline(
            phis, r_a_values * self._scale, k=3
        )

    def sample(self, R=None, z=None, phi=None, n=1, return_orbit=True, rmin=0.0):
        '''Sample from the truncated NFW DF.

        Overrides the base eddingtondf.sample to robustify the f(E) interpolator
        construction by pruning duplicate or non-monotonic energy points.
        '''
        if not hasattr(self, "_fE_interp"):
            Es4interp = np.hstack(
                (
                    np.geomspace(1e-8, 0.5, 101, endpoint=False),
                    sorted(1.0 - np.geomspace(1e-4, 0.5, 101)),
                )
            )
            Es4interp = (Es4interp * (self._Emin - self._potInf) + self._potInf)[::-1]
            fE4interp = self.fE(Es4interp)
            iindx = np.isfinite(fE4interp)
            if np.sum(iindx) == 0:
                raise ValueError("All fE values are non-finite")
            Es_finite = Es4interp[iindx]
            fE_finite = fE4interp[iindx]
            if Es_finite.size < 2:
                raise ValueError("Too few finite energy points to build f(E) interpolator")
            # Prune duplicate or non-increasing energy points
            if np.any(np.diff(Es_finite) <= 0):
                Es_mono = np.maximum.accumulate(Es_finite)
                indx_rm = np.where(np.diff(Es_mono) <= 0)[0]
                Es_finite = np.delete(Es_mono, indx_rm)
                fE_finite = np.delete(fE_finite, indx_rm)
            k = 3 if Es_finite.size >= 4 else 1
            self._fE_interp = interpolate.InterpolatedUnivariateSpline(
                Es_finite, fE_finite, k=k, ext=3
            )
        return super().sample(
            R=R, z=z, phi=phi, n=n, return_orbit=return_orbit, rmin=rmin
        )

    def _make_cmf_interpolator(self):
        '''Cumulative mass fraction interpolator constrained to rmax.'''
        from galpy.potential.SCFPotential import _RToxi, _xiToR
        from galpy.potential import mass

        ximin = _RToxi(self._rmin_sampling, a=self._scale)
        ximax = _RToxi(self._rmax, a=self._scale)
        xis = np.arange(ximin, ximax, 1e-4)
        rs = _xiToR(xis, a=self._scale)
        try:
            ms = mass(self._denspot, rs, use_physical=False)
        except (ValueError, TypeError):
            ms = np.array([mass(self._denspot, r, use_physical=False) for r in rs])
        mnorm = mass(self._denspot, self._rmax, use_physical=False)
        if self._rmin_sampling > 0:
            ms -= mass(self._denspot, self._rmin_sampling, use_physical=False)
            mnorm -= mass(self._denspot, self._rmin_sampling, use_physical=False)
        ms /= mnorm
        if np.isinf(self._rmax):
            xis = np.append(xis, 1)
            ms = np.append(ms, 1)
        else:
            xis = np.append(xis, ximax)
            ms = np.append(ms, 1)
        if np.any(np.diff(ms) <= 0):
            msm = np.maximum.accumulate(ms)
            dup = np.where(np.diff(msm) == 0)[0]
            if dup.size > 0:
                ms = np.delete(msm, dup)
                xis = np.delete(xis, dup)
            else:
                ms = msm
        return interpolate.InterpolatedUnivariateSpline(ms, xis, k=1)
        # Ensure ms is strictly increasing (handle potential flatness at large r)
        if np.any(np.diff(ms) <= 0):
            msm = np.maximum.accumulate(ms)
            indx_rm = np.where(np.diff(msm) == 0)[0]
            ms = np.delete(msm, indx_rm)
            xis = np.delete(xis, indx_rm)
        return interpolate.InterpolatedUnivariateSpline(ms, xis, k=1)


# class InterpTruncatedNFWPotential(SphericalPotential):
#     r'''Exponentially-truncated NFW potential.

#     .. math::

#         \rho(r) = \frac{\mathrm{amp}}{(r/a)\,(1 + r/a)^2\,\exp(r/r_c)}

#     Density and density derivatives are evaluated analytically. The mass
#     profile and the outer potential integral
#     :math:`\int_r^\infty \rho(s)\,s\,ds` are precomputed once on a log-r grid
#     and splined, so per-call cost is a spline lookup plus a handful of
#     arithmetic ops.

#     Parameters
#     ----------
#     amp : float
#         Density normalization :math:`\rho_0` (in the chosen unit system).
#     a : float
#         NFW scale radius.
#     rc : float
#         Exponential truncation radius.
#     rgrid : array-like, optional
#         Log-spaced radii on which to precompute the mass profile and outer
#         potential integral. Must bracket all radii at which the potential
#         will be queried.
#     ro, vo : float, optional
#         galpy unit scales (kpc, km/s).
#     '''

#     def __init__(self, amp=1.0, a=1.0, rc=2.0,
#                  rgrid=None, ro=None, vo=None):
#         SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo)
#         if rgrid is None:
#             rgrid = np.logspace(-4, 3, 4001)
#         self._a = a
#         self._rc = rc
#         self.hasC = False
#         self.hasC_dxdv = False
#         self._scale = a
#         self._rgrid = rgrid
#         self._log_rmin = np.log(rgrid[0])
#         self._log_rmax = np.log(rgrid[-1])

#         integrand_m = lambda s: 4.0 * np.pi * s * s * self._dens_unit(s)
#         m_grid = np.empty_like(rgrid)
#         m_grid[0] = integrate.quad(integrand_m, 0.0, rgrid[0])[0]
#         for i in range(1, len(rgrid)):
#             m_grid[i] = m_grid[i - 1] + integrate.quad(
#                 integrand_m, rgrid[i - 1], rgrid[i])[0]
#         self._mtot = m_grid[-1] + integrate.quad(
#             integrand_m, rgrid[-1], np.inf)[0]
#         self._m_spline = interpolate.CubicSpline(
#             np.log(rgrid), m_grid, extrapolate=True)

#         integrand_i = lambda s: self._dens_unit(s) * s
#         i_grid = np.empty_like(rgrid)
#         i_grid[-1] = integrate.quad(integrand_i, rgrid[-1], np.inf)[0]
#         for i in range(len(rgrid) - 2, -1, -1):
#             i_grid[i] = i_grid[i + 1] + integrate.quad(
#                 integrand_i, rgrid[i], rgrid[i + 1])[0]
#         self._i_spline = interpolate.CubicSpline(
#             np.log(rgrid), i_grid, extrapolate=True)

#     def _dens_unit(self, r):
#         x = r / self._a
#         return np.exp(-r / self._rc) / (x * (1.0 + x) ** 2)

#     def _mass_enclosed_unit(self, r):
#         return self._m_spline(np.log(r))

#     def _outer_integral_unit(self, r):
#         return self._i_spline(np.log(r))

#     # _revaluate / _rforce / _r2deriv: amp is applied externally by Potential.
#     def _revaluate(self, r, t=0.0):
#         return -(self._mass_enclosed_unit(r) / r
#                  + 4.0 * np.pi * self._outer_integral_unit(r))

#     def _rforce(self, r, t=0.0):
#         return -self._mass_enclosed_unit(r) / (r * r)

#     def _r2deriv(self, r, t=0.0):
#         return (4.0 * np.pi * self._dens_unit(r)
#                 - 2.0 * self._mass_enclosed_unit(r) / r ** 3)

#     def _rdens(self, r, t=0.0):
#         return self._dens_unit(r)

#     # _ddensdr / _d2densdr2: galpy convention bakes in self._amp here, since
#     # eddingtondf calls these directly (no amp wrapper).
#     def _ddensdr(self, r, t=0.0):
#         rho = self._dens_unit(r)
#         g = 1.0 / r + 2.0 / (self._a + r) + 1.0 / self._rc
#         return -self._amp * rho * g

#     def _d2densdr2(self, r, t=0.0):
#         rho = self._dens_unit(r)
#         g = 1.0 / r + 2.0 / (self._a + r) + 1.0 / self._rc
#         gprime = -1.0 / (r * r) - 2.0 / (self._a + r) ** 2
#         return self._amp * rho * (g * g - gprime)
    
# def truncatedNFW_galpy(rho0=1.0, a=1.0, rc=2.0,
#                        ro=_GALPY_RO, vo=_GALPY_VO):
#     '''Returns an exponentially truncated NFW galpy potential.'''
#     amp = rho0 * 4.0 * np.pi * a ** 3
#     pot = TruncatedNFWPotential(amp=amp, a=a, rc=rc, ro=ro, vo=vo)
#     return pot

# def mkNFW_galpy(m, n, rmin=0.0, center_pos=[0, 0, 0], center_vel=[0, 0, 0],
#                 nfw_df_kwargs={}, nfw_kwargs={}):
#     '''
#     Generate the positions, velocities, and masses of
#     a NFW sphere using galpy.

#     Parameters
#     ----------
#     m : float
#         Total mass of the NFW sphere.
#         Units: `Msun`
#     n : int
#         Number of particles to sample.
#     rmin : float, optional
#         Minimum radius at which to sample. Default is 0.
#         Units: `kpc`
#     center_pos : array-like, optional
#         Position of the center of the NFW sphere.
#         Units: `kpc`
#     center_vel : array-like, optional
#         Velocity of the center of the NFW sphere.
#         Units: `km / s`
#     nfw_df_kwargs : keyword arguments to pass to the galpy isotropicNFWdf sampler.
#          See galpy.df.isotropicNFWdf for details. Relevant kwargs include:
#             - widrow (bool, optional):
#                 If True, use the approximate form from Widrow (2000), otherwise use improved fit that has <~1e-5 relative density errors
#             - rmax (float or Quantity, optional):
#                 Maximum radius to consider; set to numpy.inf to evaluate NFW w/o cut-off
#     nfw_kwargs : keyword arguments to pass to the galpy NFWPotential constructor. See galpy.potential.NFWPotential for details. Relevant kwargs include:
    
#     Returns
#     -------
#     pos : (N, 3) array
#         Cartesian positions of sampled particles.
#         Units: `kpc`
#     vel : (N, 3) array
#         Cartesian velocities of sampled particles.
#         Units: `km / s`
#     masses : (N,) array
#         Masses of sampled particles.
#         Units: `Msun`
#     '''
#     nfw_with_units = {'ro': _GALPY_RO, 'vo': _GALPY_VO}
#     nfw_with_units.update(nfw_kwargs)
#     pot = potential.NFWPotential(**nfw_with_units)
#     return galpysampler(pot, n, m, rmin=rmin, 
#                         center_pos=center_pos, center_vel=center_vel,
#                         df_kwargs=nfw_df_kwargs)
