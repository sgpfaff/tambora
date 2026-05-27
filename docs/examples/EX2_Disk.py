from src.tambora.simulation import Sim
import matplotlib.pyplot as plt
import numpy as np
from scipy.differentiate import derivative
from scipy.interpolate import interp1d
from scipy import special, integrate
from astropy import units as u, constants as const

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# from gravhopper:

def expdisk(sigma0=None, Rd=None, z0=None, sigmaR_Rd=None, external_rotcurve=None, N=None, \
    center_pos=None, center_vel=None, force_origin=True, seed=None):
    """Generates initial conditions of an exponential disk with a sech^2 vertical distribution that is
    in (very) approximate equilibrium: rho(R,z) = (sigma0 / 2 z0) exp(-R/Rd) sech^2(z/z0)
    
    Parameters
    ----------
    sigma0 : astropy Quantity with dimensions of surface density
        Central surface density
    Rd : astropy Quantity with dimensions of length
        Radial exponential scale length
    z0 : astropy Quantity with dimensions of length
        Vertical scale height
    sigmaR_Rd : astropy Quantity with dimensions of velocity
        Radial velocity dispersion at R=Rd
    external_rotcurve : function or None
        Function that returns the circular velocity of any external potential that contributes
        to the rotation curve aside from the disk itself. The function should accept input
        as an astropy Quantity of dimension length, and should return an astropy Quantity of
        dimension velocity.
    N : int
        Number of particles
    center_pos : 3 element array-like Quantity, optional
        Force the center of mass of the IC to be at this position
    center_vel : 3 element array-like Quantity, optional
        Force the mean velocity of the IC to have this velocity
    force_origin : bool
        Force the center of mass to be at the origin and the mean velocity to be zero;
        equivalent to setting center_pos=[0,0,0]*u.kpc and
        center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
        center_vel is set. If force_origin is True and only one of center_pos or
        center_vel is set, the other is set to zero.
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        Seed to initialize random number generator to enable repeatable ICs.
        
    Returns
    -------
    IC : dict
        Properties of new particles to add, which sample the given distribution function. Contains
        the following key/value pairs:
        
        * **pos:** an array of positions
        * **vel:** an array of velocities
        * **mass:** an array of masses
        
        Each are astropy Quantities, with shape (Np,3).
        
    Example
    -------
    To create an exponential disk that is in a background logarithmic halo potential that
    generates a flat rotation curve of 200 km/s::
    
        particles = IC.expdisk(N=10000, sigma0=200*u.Msun/u.pc**2, Rd=2*u.kpc,
            z0=0.5*u.kpc, sigmaR_Rd=10*u.km/u.s,
            external_rotcurve=lambda x: 200*u.km/u.s)
            
    """
            
    rng = np.random.default_rng(seed)

    Rd_kpc = Rd.to(u.kpc).value
    totmass = (np.pi * Rd**2 * sigma0).to(u.Msun)

    # cylindrical radius transformation to give an exponential
    Rax = np.arange(0.001*Rd_kpc, 10*Rd_kpc, 0.01*Rd_kpc)
    R_cumprob = Rd_kpc**2 - Rd_kpc*np.exp(-Rax/Rd_kpc)*(Rax+Rd_kpc)
    R_cumprob /= R_cumprob[-1]
    probtransform = interp1d(R_cumprob, Rax)   # reverse interpolation
    # now get the uniform random deviate and transform it
    R_xi = rng.uniform(0.0, 1.0, size=N)
    R = probtransform(R_xi) * u.kpc
    # use random azimuth
    phi = rng.uniform(0.0, 2.0*np.pi, size=N)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    # get z from uniform random deviate
    z_xi = rng.uniform(0, 1.0 - 1e-15, size=N)
    z = 2 * z0 * np.arctanh(z_xi)
    z *= (2 * (rng.uniform(0, 1, size=N) < 0.5)) - 1

    # the velocity dispersions go as:
    #  sigma_R = sigmaR_Rd * exp(-R/Rd)
    #  sigma2_phi = sigmaR^2 * kappa^2 / 4 Omega^2
    #  sigma2_z = pi G Sigma(R) z0 / 2
    # and the mean azimuthal velocity is
    #  <vphi> = vc
    

    def om2(rad):
        # Input must be in kpc but with units stripped, because it's passed into np.derivative.
        rad = np.maximum(rad, 1e-10)  # avoid division by zero when derivative probes R<=0
        y_R = rad/(2.*Rd_kpc)
        # Disk contribution
        omega2 = np.pi * const.G * sigma0 / Rd * (special.iv(0,y_R)*special.kv(0,y_R) -
                special.iv(1,y_R)*special.kv(1,y_R))

        # Halo contribution                   
        if external_rotcurve is not None:
            omega_halo = external_rotcurve(rad*u.kpc) / (rad*u.kpc)
            omega2 += omega_halo**2
            
        return omega2

    Omega2 = om2(R.to(u.kpc).value)#.to(u.s**-2)
    kappa2 = 4.*Omega2 + R * derivative(om2, R.to(u.kpc).value).df * Omega2.unit / (u.kpc )

    sigma_R = sigmaR_Rd * np.exp(0.25*(1-R/Rd))
    sigma2_phi = sigma_R**2 * kappa2 / (4 * Omega2)
    sigma2_z = np.pi * const.G * z0 * sigma0 * 0.5 * np.exp(-R/Rd)

    # Asymmetric drift correction (Binney & Tremaine eq. 4.228)
    # d ln(Sigma * sigma_R^2) / d ln R for Sigma ~ exp(-R/Rd), sigma_R^2 ~ exp(-R/(2Rd))
    dlnrho_dlnR = -3 * R / (2 * Rd)
    va2 = -sigma_R**2 * (dlnrho_dlnR + 1 - kappa2 / (4 * Omega2))
    vc2 = (R**2 * Omega2).to(u.km**2/u.s**2)
    vphi_mean = np.sqrt(np.maximum(vc2 - va2, 0*u.km**2/u.s**2)).to(u.km/u.s)

    vphi = np.sqrt(sigma2_phi).to(u.km/u.s) * rng.normal(size=N) + vphi_mean
    vR = sigma_R.to(u.km/u.s) * rng.normal(size=N)
    vx = -vphi * np.sin(phi) + vR * np.cos(phi)
    vy = vphi * np.cos(phi) + vR * np.sin(phi)
    vz = np.sqrt(sigma2_z).to(u.km/u.s) * rng.normal(size=N)

    m = np.ones((N)) * (totmass/N)
    
    # Force COM and/or COV
    positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
        center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
    
    outIC = {'pos':positions, 'vel':velocities, 'mass': m}
    return outIC

def force_centers(positions, velocities, center_pos=None, center_vel=None, force_origin=True):
    """Move positions and velocities to have the desired center of mass position and mean velocity.
    
    Parameters
    ----------
    positions : array of Quantities of dimension length
        (Np,3) array of particle positions
    velocities : array of Quantities of dimension velocity
        (Np,3) array of particle velocities
    center_pos : 3 element array-like Quantity, optional
        Force the center of mass of the IC to be at this position
    center_vel : 3 element array-like Quantity, optional
        Force the mean velocity of the IC to have this velocity
    force_origin : bool
        Force the center of mass to be at the origin and the mean velocity to be zero;
        equivalent to setting center_pos=np.array([0,0,0])*u.kpc and
        center_vel=np.array([0,0,0])*u.km/u.s. Default is True unless center_pos and
        center_vel is set. If force_origin is True and only one of center_pos or
        center_vel is set, the other is set to zero.
        
    Returns
    -------
    newpositions : array of Quantities of dimension length
        New shifted positions
    newvelocities : array of Quantities of dimension velocity
        New shifted velocities
    """
    
    newpos = positions
    newvel = velocities
    
    if force_origin:
        if center_pos is None:
            center_pos = np.array([0,0,0])*u.kpc
        if center_vel is None:
            center_vel = np.array([0,0,0])*u.km/u.s
        sampled_com = np.nanmean(positions, axis=0)
        sampled_com = np.mean(positions, axis=0)
        dpos = center_pos - sampled_com
        newpos += dpos
        sampled_cov = np.nanmean(velocities, axis=0)
        sampled_cov = np.mean(velocities, axis=0)
        dvel = center_vel - sampled_cov
        newvel += dvel 

    return (newpos, newvel)

Ndisk = 10000
vhalo = 150*u.km/u.s
rhalo = 100*u.kpc
vals = expdisk(N=Ndisk, sigma0=200*u.Msun/u.pc**2, Rd=2*u.kpc, z0=0.2*u.kpc, sigmaR_Rd=20*u.km/u.s,
    external_rotcurve=None)#lambda x: vhalo)

pos, vel, mass = vals['pos'], vals['vel'], vals['mass']

mask = (
    np.isfinite(pos).all(axis=1)
    & np.isfinite(vel).all(axis=1)
    & np.isfinite(mass)
)

pos, vel, mass = pos[mask], vel[mask], mass[mask]

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax[0].scatter(pos[:,0], pos[:,1], c='k', s=1, alpha=0.5)
ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].set_aspect('equal')

ax[1].scatter(pos[:,0], pos[:,2], c='k', s=1, alpha=0.5)
ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')
ax[1].set_aspect('equal')
plt.suptitle('Initial conditions')
plt.tight_layout()
plt.show()

# # Disk With No Halo
# 
# Demonstrate disk instability

disk = Sim()
disk.add_particles('stars', pos.value, vel.to(u.km/u.s).value, mass.value)

disk.run(t_end=500, dt=1.0, dt_out=10, eps=0.5, theta=0.6)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax[0].scatter(disk.x(t=-1), disk.y(t=-1), c='k', s=1, alpha=0.5)
ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].set_aspect('equal')

ax[1].scatter(disk.x(t=-1), disk.z(t=-1), c='k', s=1, alpha=0.5)
ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')
ax[1].set_aspect('equal')
plt.suptitle('Disk without halo — final snapshot')
plt.tight_layout()
plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

times = np.arange(0, 501, 10)  # matches run(t_end=500, dt_out=10)

def get_state(i):
    # Prefer snapshot index; fallback to physical time if needed
    try:
        return disk.x(t=i), disk.y(t=i), disk.z(t=i)
    except Exception:
        tt = times[i]
        return disk.x(t=tt), disk.y(t=tt), disk.z(t=tt)

# Precompute limits so axes stay fixed
all_x, all_y, all_z = [], [], []
for i in range(len(times)):
    x, y, z = get_state(i)
    all_x.append(x); all_y.append(y); all_z.append(z)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_z = np.concatenate(all_z)

pad = 0.5
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim_xy = (all_y.min() - pad, all_y.max() + pad)
ylim_xz = (all_z.min() - pad, all_z.max() + pad)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

sc_xy = ax[0].scatter([], [], c='k', s=1, alpha=0.5)
sc_xz = ax[1].scatter([], [], c='k', s=1, alpha=0.5)

ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].set_aspect('equal')
ax[0].set_xlim(*xlim)
ax[0].set_ylim(*ylim_xy)

ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')
ax[1].set_aspect('equal')
ax[1].set_xlim(*xlim)
ax[1].set_ylim(*ylim_xz)

title = fig.suptitle('')

def update(i):
    x, y, z = get_state(i)
    sc_xy.set_offsets(np.c_[x, y])
    sc_xz.set_offsets(np.c_[x, z])
    title.set_text(f't = {times[i]} Myr')
    return sc_xy, sc_xz, title

ani = FuncAnimation(fig, update, frames=len(times), interval=250, blit=True)
plt.close(fig)

HTML(ani.to_jshtml())

# # Now with a halo

from galpy.potential import NFWPotential
diskhalo = Sim()
halo = NFWPotential(vmax=vhalo, rmax=rhalo)
diskhalo.add_particles('stars', pos.value, vel.to(u.km/u.s).value, mass.value)
diskhalo.add_external_pot(halo)
diskhalo.run(t_end=500, dt=1.0, dt_out=10, eps=0.5, theta=0.6)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax[0].scatter(diskhalo.x(t=-1), diskhalo.y(t=-1), c='k', s=1, alpha=0.5)
ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].set_aspect('equal')

ax[1].scatter(diskhalo.x(t=-1), diskhalo.z(t=-1), c='k', s=1, alpha=0.5)
ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')
ax[1].set_aspect('equal')
plt.suptitle('Disk with NFW halo — final snapshot')
plt.tight_layout()
plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

times = np.arange(0, 501, 10)  # matches run(t_end=100, dt_out=10)

def get_state(i):
    # Prefer snapshot index; fallback to physical time if needed
    try:
        return diskhalo.x(t=i), diskhalo.y(t=i), diskhalo.z(t=i)
    except Exception:
        tt = times[i]
        return diskhalo.x(t=tt), diskhalo.y(t=tt), diskhalo.z(t=tt)

# Precompute limits so axes stay fixed
all_x, all_y, all_z = [], [], []
for i in range(len(times)):
    x, y, z = get_state(i)
    all_x.append(x); all_y.append(y); all_z.append(z)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_z = np.concatenate(all_z)

pad = 0.5
xlim = (all_x.min() - pad, all_x.max() + pad)
ylim_xy = (all_y.min() - pad, all_y.max() + pad)
ylim_xz = (all_z.min() - pad, all_z.max() + pad)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

sc_xy = ax[0].scatter([], [], c='k', s=1, alpha=0.5)
sc_xz = ax[1].scatter([], [], c='k', s=1, alpha=0.5)

ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].set_aspect('equal')
ax[0].set_xlim(*xlim)
ax[0].set_ylim(*ylim_xy)

ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('z [kpc]')
ax[1].set_aspect('equal')
ax[1].set_xlim(*xlim)
ax[1].set_ylim(*ylim_xz)

title = fig.suptitle('')

def update(i):
    x, y, z = get_state(i)
    sc_xy.set_offsets(np.c_[x, y])
    sc_xz.set_offsets(np.c_[x, z])
    title.set_text(f't = {times[i]} Myr')
    return sc_xy, sc_xz, title

ani = FuncAnimation(fig, update, frames=len(times), interval=250, blit=True)
plt.close(fig)

HTML(ani.to_jshtml())

diskhalo.plot_energy_diagnostic()

disk.plot_energy_diagnostic()
