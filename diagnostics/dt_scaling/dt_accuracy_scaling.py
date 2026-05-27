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
from src.tambora.util import galpydfsampler
from src.tambora.simulation import Sim
import astropy.units as u


def run_simulation(args):
    """Run a single N-body simulation with the given time step."""
    dt = args
    n_particles = 8192  # Fixed number of particles for dt scaling
    pot = PlummerPotential(amp=1e10*u.Msun, b=2*u.kpc)
    df = isotropicPlummerdf(pot=pot)
    pos, vel, masses = galpydfsampler(df, n=n_particles, m_total=1e10)
    
    print(f"Starting simulation with N={n_particles} particles...")
    
    # Generate initial conditions
    # ic = generate_plummer_sphere(n_particles)
    
    start_time = time.time()
    sim = Sim()
    sim.add_particles('stars', pos, vel, masses)
    sim.run(t_end=100, dt=dt, dt_out=1.0, eps=0.1, theta=0.6)
    

    elapsed_time = time.time() - start_time
    energy_error = np.array([np.abs(sim.percent_dE(t2=t)) for t in np.arange(len(sim.times))])

    return n_particles, elapsed_time, energy_error, sim.times


def main():
    # Time steps to test
    dt_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # Create output directory
    # output_dir = "nparticles_scaling_output"
    # os.makedirs(output_dir, exist_ok=True)
    
    # Run simulations in parallel using multiprocessing
    n_workers = min(mp.cpu_count(), len(dt_list))
    print(f"Running {len(dt_list)} simulations using "
          f"{n_workers} workers...")
    
    # Generate ICs
    
    # Run simulations in parallel
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(run_simulation, dt_list)
    
    # Collect results
    n_values = np.array([r[0] for r in results])
    times = np.array([r[1] for r in results])
    energies = np.array([r[2] for r in results])
    int_ts = np.array([r[3] for r in results])
    
    # Sort by dt
    sort_idx = np.argsort(dt_list)
    dt_list = np.array(dt_list)[sort_idx]
    times = times[sort_idx]
    energies = energies[sort_idx]
    
    # Fit power law to runtime scaling
    log_dt = np.log10(dt_list)
    log_t = np.log10(times)
    slope, intercept = np.polyfit(log_dt, log_t, 1)
    print(f"\nRuntime scaling: T ~ dt^{slope:.2f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Accuracy scaling
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / (len(dt_list) - 1)) for i in range(len(dt_list))]
    for (t, e_list, dt, color) in zip(int_ts, energies, dt_list, colors):
        ax.loglog(t, e_list, linewidth=2, markersize=8, label=f'dt={dt}', c=color)
    ax.set_xlabel('Time (Myr)', fontsize=20)
    ax.set_ylabel('Energy Error (%)', fontsize=20)
    ax.set_title('Accuracy Scaling', fontsize=20)
    ax.legend(ncol=2, fontsize=16)
    plt.tight_layout()
    plot_file = "output/dt_accuracy_scaling.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")

    # Accuracy scaling 2
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.Blues
    max_errors = np.array([e.max() for e in energies])
    ## Fit
    log_dt = np.log10(dt_list)
    log_e = np.log10(max_errors)
    slope, intercept = np.polyfit(log_dt, log_e, 1)
    dt_fit = np.logspace(np.log10(dt_list[0]), np.log10(dt_list[-1]), 100)
    e_fit = 10**intercept * dt_fit**slope
    ax.loglog(dt_fit, e_fit, 'k', linewidth=1.5, alpha=0.35, 
              label=f'Fit: Max Error ~ $dt^{{{slope:.2f}}}$')
    ax.loglog(dt_list, max_errors, 'ko-',linewidth=2, markersize=8, label='Max Error')
    ax.set_xlabel('Time Step (dt)', fontsize=20)
    ax.set_ylabel('Max Energy Error (%)', fontsize=20)
    ax.set_title('Accuracy Scaling', fontsize=20)
    ax.legend(ncol=2, fontsize=16)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plot_file = "output/dt_max_error.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()