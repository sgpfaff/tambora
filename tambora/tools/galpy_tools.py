import numpy as np
from ..util._galpy_bridge import _check_physical
from ..util.units import KMS_TO_KPCGYR
from galpy import df, potential
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
