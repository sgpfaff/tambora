"""
Benchmark: external potential evaluation time vs falcON self-gravity
as a function of the number of particles.

Compares NFW (vectorized), TriaxialNFW (vectorized in galpy >=1.12),
HomogeneousSphere (scalar loop), and falcON self-gravity.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from galpy.potential import (
    NFWPotential,
    TriaxialNFWPotential,
    HomogeneousSpherePotential,
)
from galpy import potential as gp
from galpy.util.coords import rect_to_cyl
import astropy.units as u

from ezfalcon.dynamics.forces.self_gravity import self_gravity

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


def time_galpy_force(pot, pos, n_repeat=3):
    """Time galpy force evaluation on pos array, return median wall-clock time."""
    ro = 8.0  # kpc
    R, phi, z = rect_to_cyl(*pos.T)
    R_nat, z_nat = R / ro, z / ro
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        gp.evaluateRforces(pot, R_nat, z_nat, phi=phi, use_physical=False)
        gp.evaluatezforces(pot, R_nat, z_nat, phi=phi, use_physical=False)
        gp.evaluatephitorques(pot, R_nat, z_nat, phi=phi, use_physical=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def time_galpy_force_scalar(pot, pos, n_repeat=3):
    """Time galpy force evaluation particle-by-particle (scalar loop)."""
    ro = 8.0
    R, phi, z = rect_to_cyl(*pos.T)
    R_nat, z_nat = R / ro, z / ro
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for Ri, zi, pi in zip(R_nat, z_nat, phi):
            gp.evaluateRforces(pot, Ri, zi, phi=pi, use_physical=False)
            gp.evaluatezforces(pot, Ri, zi, phi=pi, use_physical=False)
            gp.evaluatephitorques(pot, Ri, zi, phi=pi, use_physical=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def time_falcon(pos, masses, n_repeat=3):
    """Time falcON self-gravity, return median wall-clock time."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        self_gravity(pos, masses, method='falcON',
                     return_potential=False, eps=0.1, theta=0.6)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def generate_particles(n):
    """Generate random particles in a Plummer-like distribution."""
    rng = np.random.default_rng(42)
    # Simple Plummer sphere sampling
    r = 2.0 / np.sqrt(rng.uniform(0, 1, n)**(-2.0/3.0) - 1)
    # Clip to physically reasonable range
    r = np.clip(r, 0.01, 100.0)
    theta = np.arccos(2 * rng.uniform(0, 1, n) - 1)
    phi = 2 * np.pi * rng.uniform(0, 1, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z_pos = r * np.cos(theta)
    pos = np.column_stack([x, y, z_pos])
    masses = np.full(n, 1e10 / n)
    return pos, masses


def main():
    # Setup potentials
    nfw = NFWPotential(amp=1e12 * u.Msun, a=16 * u.kpc)
    tnfw = TriaxialNFWPotential(amp=1e12 * u.Msun, a=16 * u.kpc, b=0.8, c=0.6)
    hsphere = HomogeneousSpherePotential(amp=2.5e6 * u.Msun / u.kpc**3, R=5 * u.kpc)

    n_particles_list = [100, 500, 1000, 5000]#, 10000, 50000, 200000]

    results = {
        'N': [],
        'nfw': [],
        'tnfw_vec': [],
        'tnfw_scalar': [],
        'hsphere': [],
        'falcon': [],
    }

    for n in n_particles_list:
        print(f"N = {n:>7d} ...", end=" ", flush=True)
        pos, masses = generate_particles(n)

        t_nfw = time_galpy_force(nfw, pos)
        t_tnfw_vec = time_galpy_force(tnfw, pos)
        # For small N, also run scalar for comparison; skip for large N
        if n <= 10000:
            t_tnfw_scalar = time_galpy_force_scalar(tnfw, pos, n_repeat=1)
        else:
            t_tnfw_scalar = np.nan
        t_hsphere = time_galpy_force_scalar(hsphere, pos, n_repeat=1 if n > 10000 else 3)
        t_falcon = time_falcon(pos, masses)

        results['N'].append(n)
        results['nfw'].append(t_nfw)
        results['tnfw_vec'].append(t_tnfw_vec)
        results['tnfw_scalar'].append(t_tnfw_scalar)
        results['hsphere'].append(t_hsphere)
        results['falcon'].append(t_falcon)
        print(f"NFW={t_nfw:.4f}s  TriaxNFW(vec)={t_tnfw_vec:.4f}s  "
              f"TriaxNFW(scalar)={t_tnfw_scalar:.4f}s  "
              f"HomSph={t_hsphere:.4f}s  falcON={t_falcon:.4f}s")

    N = np.array(results['N'])
    t_nfw = np.array(results['nfw'])
    t_tnfw_vec = np.array(results['tnfw_vec'])
    t_tnfw_scalar = np.array(results['tnfw_scalar'])
    t_hsphere = np.array(results['hsphere'])
    t_falcon = np.array(results['falcon'])

    # Save results
    np.savez('diagnostics_benchmark_results.npz',
             N=N, nfw=t_nfw, tnfw_vec=t_tnfw_vec,
             tnfw_scalar=t_tnfw_scalar, hsphere=t_hsphere, falcon=t_falcon)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: wall-clock time
    ax1.loglog(N, t_nfw, 'o-', color='tab:blue', lw=2, ms=7,
               label='NFW (vectorized)')
    ax1.loglog(N, t_tnfw_vec, 's-', color='tab:orange', lw=2, ms=7,
               label='TriaxialNFW (vectorized, new)')
    mask = ~np.isnan(t_tnfw_scalar)
    ax1.loglog(N[mask], t_tnfw_scalar[mask], 's--', color='tab:orange',
               lw=1.5, ms=7, alpha=0.5,
               label='TriaxialNFW (scalar loop, old)')
    ax1.loglog(N, t_hsphere, '^-', color='tab:green', lw=2, ms=7,
               label='HomogeneousSphere (scalar loop)')
    ax1.loglog(N, t_falcon, 'D-', color='black', lw=2, ms=7,
               label='falcON self-gravity')

    ax1.set_xlabel('Number of particles')
    ax1.set_ylabel('Wall-clock time [s]')
    ax1.set_title('External potential evaluation time')
    ax1.legend(fontsize=11, loc='upper left')

    # Right panel: ratio
    ax2.loglog(N, t_nfw / t_falcon, 'o-', color='tab:blue', lw=2, ms=7,
               label='NFW / falcON')
    ax2.loglog(N, t_tnfw_vec / t_falcon, 's-', color='tab:orange', lw=2, ms=7,
               label='TriaxialNFW (vec) / falcON')
    ax2.loglog(N, t_hsphere / t_falcon, '^-', color='tab:green', lw=2, ms=7,
               label='HomogeneousSphere / falcON')
    ax2.axhline(1.0, ls='--', color='gray', lw=1.5, label='parity')

    ax2.set_xlabel('Number of particles')
    ax2.set_ylabel('Ratio (ext. potential / self-gravity)')
    ax2.set_title('Cost relative to falcON self-gravity')
    ax2.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig('vectorized_benchmark.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to vectorized_benchmark.png")


if __name__ == "__main__":
    main()
