from galpy.df import isotropicPlummerdf
from galpy.potential import PlummerPotential, NFWPotential, TriaxialNFWPotential
from src.tambora.simulation import Sim
from src.tambora.tools import galpydfsampler
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import importlib
import yaml
import os
import shutil
import argparse

plt.rcParams.update({
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'font.size': 14, 'axes.labelsize': 16,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'ytick.right': True, 'xtick.top': True,
    'mathtext.fontset': 'stix',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

from matplotlib.animation import FuncAnimation

def create_shell(n, df, host_pot, m_total, center_pos, center_vel):
    shell = Sim()
    pos, vel, mass = galpydfsampler(df=df, n=n, m_total=m_total, center_pos=center_pos, center_vel=center_vel)
    shell.add_particles('all', pos=pos, vel=vel, mass=mass)
    shell.add_external_pot(host_pot)
    return shell

def create_prog_orbit(shell, host_pot):
    init_center_pos, init_center_vel = np.median(shell.pos(0), axis=0), np.median(shell.vel(0), axis=0)
    prog = Sim()
    prog.add_particles('prog', pos=init_center_pos[None,:], vel=init_center_vel[None,:], mass=np.array([1e9]))
    prog.add_external_pot(host_pot)
    prog.turn_self_gravity_off()
    return prog

def calculate_bound(pos, vel, mass, self_pot, center_pos, center_vel):
    '''
    Determine which stars are bound through iterative 
    calculation.
    '''
    f_xv = np.hstack((pos, vel))
    f_center = np.hstack((center_pos, center_vel))
    Rmax = 10.0
    use  = np.sum((pos - center_pos)**2, axis=1) < Rmax**2
    # iteratively refine the selection, retaining only bound particles (which have
    # negative total energy in the satellite-centered frame using its own potential only)
    prev_f_center = f_center
    for i in range(50):
        f_center = np.median(f_xv[use], axis=0)
        f_bound = self_pot + 0.5 * mass* np.sum((f_xv[:,3:6] - f_center[3:6])**2, axis=1) < 0
        if np.sum(f_bound)<=1 or all(f_center==prev_f_center): break
        use = f_bound * (np.sum((f_xv[:,0:3] - f_center[0:3])**2, axis=1) < Rmax**2)
    return f_bound

def calculate_star_mask(vel, mass, self_PE):
    center_vel = np.mean(vel, axis=0)
    binding = self_PE + 0.5 * mass* np.sum((vel - center_vel)**2, axis=1)
    sorted_binding_indices = np.argsort(binding)
    return sorted_binding_indices[:int(0.1*len(binding))]
    
def make_animation(x, y, times, filename, mask=None, label1='', label2=''):
    fig, ax = plt.subplots(figsize=(6, 6))

    def animate(i):
        ax.clear()
        if mask is not None:
            ax.scatter(x[i][~mask[i]], y[i][~mask[i]], s=0.01, c='r', alpha=0.1, label=label1)
            ax.scatter(x[i][mask[i]], y[i][mask[i]], s=0.01, alpha=0.1, c='k', label=label2)
            ax.legend(loc='upper right', markerscale=10)
        else:
            ax.scatter(x[i], y[i], s=0.05, alpha=0.1, c='k')

        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        ax.set_title(f't = {times[i]:.0f} Myr')
        

    anim = FuncAnimation(fig, animate, frames=len(times), interval=100)
    anim.save(f'{filename}.mp4', writer='ffmpeg', fps=10)
    plt.close(fig)

def load_class(path):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def parse_quantity(x):
    if isinstance(x, str):
        return u.Quantity(x)
    return x

def __main__(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Make host potential
    host_pot_cls = load_class(config["host"]["potential"]["class"])
    host_pot_params = {k: parse_quantity(v) for k, v in config["host"]["potential"]["params"].items()}
    host_pot = host_pot_cls(**host_pot_params)

    # Make satellite potential and df
    sat_pot_cls = load_class(config["satellite"]["potential"]["class"])
    sat_pot_params = {k: parse_quantity(v) for k, v in config["satellite"]["potential"]["params"].items()}
    sat_pot = sat_pot_cls(**sat_pot_params)
    sat_df_cls = load_class(config["satellite"]["df_class"])
    sat_df = sat_df_cls(pot=sat_pot)

    # Create output directories and save config
    sim_params = config["simulation"]
    output_dir = config["outputs"]["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    if config['outputs']['save_animation']:
        os.makedirs(f'{output_dir}/anims', exist_ok=True)
    if config['outputs']['save_data']:
        os.makedirs(f'{output_dir}/data', exist_ok=True)
    shutil.copy(config_path, f'{output_dir}/config.yaml')
    
    # create Sim object
    shell = create_shell(n=sim_params['n_particles'], df=sat_df, host_pot=host_pot, 
                         m_total=parse_quantity(config["satellite"]["m_total"]), 
                         center_pos=parse_quantity(config["satellite"]["center_pos"]), 
                         center_vel=parse_quantity(config["satellite"]["center_vel"]))
    prog = create_prog_orbit(shell, host_pot)

    # Run Simulation
    shell.run(t_end=sim_params['t_end'], dt=sim_params['dt'], dt_out=sim_params['dt_out'], eps=sim_params['eps'], theta=sim_params['theta'])
    prog.run(t_end=sim_params['t_end'], dt=sim_params['dt'], dt_out=sim_params['dt_out'], eps=sim_params['eps'], theta=sim_params['theta'])

    # Identify bound particles at each snapshot
    mask = []
    for i, t in enumerate(shell.times):
        mask.append(calculate_bound(shell.pos(i), shell.vel(i), shell.mass, shell.self_PE(i), prog.pos(i)[0], prog.vel(i)[0]))
        print(f'{np.sum(mask[i])} bound particles at time {t:.1f}')

    star_mask = calculate_star_mask(shell.vel(0), shell.mass, shell.self_PE(0))

    # save data
    if config['outputs']['save_data']:
        np.save(f'{output_dir}/data/bound_mask.npy', mask)
        np.save(f'{output_dir}/data/shell_pos.npy', shell.pos())
        np.save(f'{output_dir}/data/shell_vel.npy', shell.vel())
        np.save(f'{output_dir}/data/shell_mass.npy', shell.mass)
        np.save(f'{output_dir}/data/times.npy', shell.times)
        np.save(f'{output_dir}/data/self_PE.npy', shell.self_PE())
        np.save(f'{output_dir}/data/star_mask.npy', star_mask)

    # make diagnostic plot
    shell.plot_diagnostic(f'{output_dir}/energy_conservation.png')

    # make animations
    if config['outputs']['save_animation']:
        print('Creating animations...')
        make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, f'{output_dir}/anims/all_pts')
        print('Creating animations with bound/unbound separation...')
        make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, f'{output_dir}/anims/bound_unbound', mask=mask, label1='unbound', label2='bound')
        print('Creating animations with stars/DM separation...')
        make_animation(shell.pos()[:,:,0], shell.pos()[:,:,1], shell.times, f'{output_dir}/anims/stars_DM', mask=np.repeat(star_mask, len(shell.times), axis=0).reshape(-1, len(shell.times)).T, label1='DM', label2='stars')
        print('Creating animations with stars only...')
        make_animation(shell.pos()[:,star_mask,0], shell.pos()[:,star_mask,1], shell.times, f'{output_dir}/anims/stars_only')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    __main__(args.config)



