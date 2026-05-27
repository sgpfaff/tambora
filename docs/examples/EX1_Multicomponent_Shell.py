from galpy.potential import PlummerPotential, NFWPotential
from galpy.df import isotropicPlummerdf, isotropicNFWdf
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from src.tambora.tools import galpydfsampler
from src.tambora.simulation import Sim
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

shell = Sim()

# Dark Matter Component
progpotDM = PlummerPotential(amp=1e11*u.Msun, b=3*u.kpc)
dfDM = isotropicPlummerdf(pot=progpotDM)
posDM, velDM, massesDM = galpydfsampler(dfDM, n=60000, m_total=1e11, center_pos=[15,0,0], center_vel=[0,0.066,0])
shell.add_particles('dm', posDM, velDM, massesDM)

# # Stellar Component
progpotStars = PlummerPotential(amp=1e9*u.Msun, b=2*u.kpc)
dfStars = isotropicPlummerdf(pot=progpotStars)
posStars, velStars, massesStars = galpydfsampler(dfStars, n=30000, m_total=1e9, center_pos=[15,0,0], center_vel=[0,0.066,0])
shell.add_particles('stars', posStars, velStars, massesStars)

host_pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
shell.add_external_pot(host_pot)

# acc = shell.external_acc()

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.quiver(shell.x(), shell.y(), acc[:, 0], acc[:, 1], alpha=0.5)
# plt.plot(0, 0, color='r', marker='+', markersize=10, label='Host center')
# ax.set_xlabel('x [kpc]')
# ax.set_ylabel('y [kpc]')
# ax.set_title('Acceleration field in the x-y plane')
# ax.set_aspect('equal')
# plt.tight_layout()
# plt.xlim(-15, 15)
# plt.ylim(-15, 15)

# plt.show()


self_acc = shell.self_gravity(use_cached=False, eps=0.1, theta=0.5, t=0)
ext_acc = shell.external_acc()
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(shell.x(t=0)[::100], shell.y(t=0)[::100], ext_acc[:, 0][::100], ext_acc[:, 1][::100], alpha=1, color='b', label='external field')
ax.quiver(shell.x(t=0)[::100], shell.y(t=0)[::100], self_acc[:, 0][::100], self_acc[:, 1][::100], alpha=1, color='r', label='self-gravity')
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title('Acceleration Vectors')
# ax.set_aspect('equal')
plt.tight_layout()
plt.xlim(13, 17)
plt.ylim(-2, 2)
plt.legend()

plt.show()

shell.run(t_end=400, dt=0.01, dt_out=3., eps=0.5, theta=0.5)

shell.plot_energy_diagnostic()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

n_frames = len(shell.stars.x())  # t_end=100, dt_out=10 -> snapshots at 0,10,...,100

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
sc = ax_anim.scatter(shell.stars.x(0), shell.stars.y(0), s=0.5, c='k', alpha=0.1)
ax_anim.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

ax_anim.set_xlim(-25, 25)
ax_anim.set_ylim(-25, 25)
ax_anim.set_xlabel('x [kpc]')
ax_anim.set_ylabel('y [kpc]')
title = ax_anim.set_title('t = 0')

def update(i):
    sc.set_offsets(list(zip(shell.stars.x(i), shell.stars.y(i))))
    title.set_text(f't = {shell.times[i]} Myr')
    return sc, title

anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=100, blit=False)
plt.close(fig_anim)
HTML(anim.to_jshtml())

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

n_frames = len(shell.dm.x())  # t_end=100, dt_out=10 -> snapshots at 0,10,...,100

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
sc = ax_anim.scatter(shell.dm.x(0), shell.dm.y(0), s=0.5, c='k', alpha=0.1)
ax_anim.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

ax_anim.set_xlim(-20, 20)
ax_anim.set_ylim(-20, 20)
ax_anim.set_xlabel('x [kpc]')
ax_anim.set_ylabel('y [kpc]')
title = ax_anim.set_title('t = 0')

def update(i):
    sc.set_offsets(list(zip(shell.dm.x(i), shell.dm.y(i))))
    title.set_text(f't = {shell.times[i]} Myr')
    return sc, title

anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=100, blit=False)
plt.close(fig_anim)
HTML(anim.to_jshtml())

# # Without Self-Gravity

sim_ext_only = Sim()
sim_ext_only.add_particles('stars', posStars, velStars, massesStars)
sim_ext_only.add_external_pot(host_pot)
sim_ext_only.turn_self_gravity_off()

sim_ext_only.run(t_end=200, dt=0.1, dt_out=2, eps=0.5, theta=0.6)

plt.figure(figsize=(6, 6))
plt.scatter(sim_ext_only.x(-1), sim_ext_only.y(-1), s=1, alpha=0.1, c='k')
plt.xlim(-25, 25)
plt.ylim(-25, 25)

sim_ext_only.plot_energy_diagnostic()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

n_frames = len(sim.x())  # t_end=100, dt_out=10 -> snapshots at 0,10,...,100

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
sc = ax_anim.scatter(sim_ext_only.x(0), sim_ext_only.y(0), s=1, c='k', alpha=0.1)
ax_anim.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

ax_anim.set_xlim(-20, 20)
ax_anim.set_ylim(-20, 20)
ax_anim.set_xlabel('x [kpc]')
ax_anim.set_ylabel('y [kpc]')
title = ax_anim.set_title('t = 0')

def update(i):
    sc.set_offsets(list(zip(sim_ext_only.x(i), sim_ext_only.y(i))))
    title.set_text(f't = {sim_ext_only.times[i]} Myr')
    return sc, title

anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=50, blit=False)
plt.close(fig_anim)
HTML(anim.to_jshtml())
