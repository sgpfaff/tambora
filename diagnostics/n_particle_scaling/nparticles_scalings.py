import numpy as np
import multiprocessing as mp
import subprocess
import time
import os
from functools import partial

'''
Python script to test how the runtime 
and accuracy of the N-body code scales 
with the number of particles.
'''

import matplotlib.pyplot as plt

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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

from galpy.potential import PlummerPotential
from galpy.df import isotropicPlummerdf
from tambora.util import galpydfsampler
from tambora.simulation import Sim
import astropy.units as u





def run_simulation(args):
    """Run a single N-body simulation with the given number of particles."""
    n_particles = args
    pot = PlummerPotential(amp=1e10*u.Msun, b=2*u.kpc)
    df = isotropicPlummerdf(pot=pot)
    pos, vel, masses = galpydfsampler(df, n=n_particles, m_total=1e10)
    
    print(f"Starting simulation with N={n_particles} particles...")
    
    # Generate initial conditions
    # ic = generate_plummer_sphere(n_particles)
    
    start_time = time.time()
    sim = Sim()
    sim.add_particles('stars', pos, vel, masses)
    sim.run(t_end=100, dt=1, dt_out=1, eps=0.1, theta=0.6)
    

    elapsed_time = time.time() - start_time
    energy_error = np.array([np.abs(sim.percent_dE(t2=t)) for t in np.arange(len(sim.times))])

    return n_particles, elapsed_time, energy_error, sim.times


def main():
    # Number of particles to test
    n_particles_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    # Create output directory
    # output_dir = "nparticles_scaling_output"
    # os.makedirs(output_dir, exist_ok=True)
    
    # Run simulations in parallel using multiprocessing
    n_workers = min(mp.cpu_count(), len(n_particles_list))
    print(f"Running {len(n_particles_list)} simulations using "
          f"{n_workers} workers...")
    
    # Generate ICs
    
    # Run simulations in parallel
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(run_simulation, n_particles_list)
    
    # Collect results
    n_values = np.array([r[0] for r in results])
    times = np.array([r[1] for r in results])
    energies = np.array([r[2] for r in results])
    int_ts = np.array([r[3] for r in results])
    
    # Sort by N
    sort_idx = np.argsort(n_values)
    n_values = n_values[sort_idx]
    times = times[sort_idx]
    energies = energies[sort_idx]
    
    # Fit power law to runtime scaling
    log_n = np.log10(n_values)
    log_t = np.log10(times)
    slope, intercept = np.polyfit(log_n, log_t, 1)
    print(f"\nRuntime scaling: T ~ N^{slope:.2f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Runtime scaling
    ax.loglog(n_values, times, 'ko-', linewidth=2, markersize=8, label='Measured')
    n_fit = np.logspace(np.log10(n_values[0]), np.log10(n_values[-1]), 100)
    t_fit = 10**intercept * n_fit**slope
    ax.loglog(n_fit, t_fit, 'k', linewidth=1.5, alpha=0.35, 
              label=f'Fit: T ~ $N^{{{slope:.2f}}}$')
    ax.set_xlabel('Number of Particles (N)', fontsize=20)
    ax.set_ylabel('Runtime (s)', fontsize=20)
    ax.set_title('Runtime Scaling', fontsize=20)
    ax.legend(fontsize=16)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plot_file = "output/nparticles_runtime_scaling.png"
    #plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")

    # Accuracy scaling
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / (len(n_values) - 1)) for i in range(len(n_values))]
    for (t, e_list, n, color) in zip(int_ts, energies, n_values, colors):
        ax.loglog(t, e_list, linewidth=2, markersize=8, label=f'N={n}', c=color)
    ax.set_xlabel('Time (Myr)', fontsize=20)
    ax.set_ylabel('Energy Error (%)', fontsize=20)
    ax.set_title('Accuracy Scaling', fontsize=20)
    ax.legend(ncol=2, fontsize=16)
    plt.tight_layout()
    plot_file = "output/nparticles_accuracy_scaling.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")

    # Accuracy scaling 2
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.Blues
    max_errors = np.array([e.max() for e in energies])
    ## Fit
    log_n = np.log10(n_values)
    log_e = np.log10(max_errors)
    slope, intercept = np.polyfit(log_n, log_e, 1)
    n_fit = np.logspace(np.log10(n_values[0]), np.log10(n_values[-1]), 100)
    e_fit = 10**intercept * n_fit**slope
    ax.loglog(n_fit, e_fit, 'k', linewidth=1.5, alpha=0.35, 
              label=f'Fit: Max Error ~ $N^{{{slope:.2f}}}$')
    ax.loglog(n_values, max_errors, 'ko-',linewidth=2, markersize=8, label='Max Error')
    ax.set_xlabel('Number of Particles (N)', fontsize=20)
    ax.set_ylabel('Max Energy Error (%)', fontsize=20)
    ax.set_title('Accuracy Scaling', fontsize=20)
    ax.legend(ncol=2, fontsize=16)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plot_file = "output/nparticles_max_error.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")




if __name__ == "__main__":
    main()