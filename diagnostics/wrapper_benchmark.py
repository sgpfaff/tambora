"""
Benchmark: overhead of galpy wrapper potentials on force evaluation time.

Compares bare NFW, bare TriaxialNFW (vectorized), and several wrappers
around each, plus falcON self-gravity for reference.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from galpy.potential import (
    NFWPotential,
    TriaxialNFWPotential,
    DehnenSmoothWrapperPotential,
    GaussianAmplitudeWrapperPotential,
    SolidBodyRotationWrapperPotential,
    CorotatingRotationWrapperPotential,
    TimeDependentAmplitudeWrapperPotential,
    KuzminLikeWrapperPotential,
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
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


def time_galpy_force(pot, pos, n_repeat=3):
    """Time galpy force evaluation (vectorized) on pos array, return median."""
    ro = 8.0
    R, phi, z = rect_to_cyl(*pos.T)
    R_nat, z_nat = R / ro, z / ro
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        gp.evaluateRforces(pot, R_nat, z_nat, phi=phi, t=0., use_physical=False)
        gp.evaluatezforces(pot, R_nat, z_nat, phi=phi, t=0., use_physical=False)
        gp.evaluatephitorques(pot, R_nat, z_nat, phi=phi, t=0., use_physical=False)
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
    r = 2.0 / np.sqrt(rng.uniform(0, 1, n)**(-2.0/3.0) - 1)
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
    # --- Base potentials ---
    nfw = NFWPotential(amp=1e12 * u.Msun, a=16 * u.kpc)
    tnfw = TriaxialNFWPotential(amp=1e12 * u.Msun, a=16 * u.kpc, b=0.8, c=0.6)

    # --- Wrappers around NFW ---
    nfw_wrappers = {
        'DehnenSmooth':       DehnenSmoothWrapperPotential(pot=nfw),
        'GaussianAmp':        GaussianAmplitudeWrapperPotential(pot=nfw),
        'SolidBodyRot':       SolidBodyRotationWrapperPotential(pot=nfw),
        'CorotatingRot':      CorotatingRotationWrapperPotential(pot=nfw),
        'TimeDependentAmp':   TimeDependentAmplitudeWrapperPotential(pot=nfw, A=lambda t: 1.0),
        'KuzminLike':         KuzminLikeWrapperPotential(pot=nfw, a=1.0 * u.kpc),
    }

    # --- Wrappers around TriaxialNFW (KuzminLike doesn't work for triaxial) ---
    tnfw_wrappers = {
        'DehnenSmooth':       DehnenSmoothWrapperPotential(pot=tnfw),
        'GaussianAmp':        GaussianAmplitudeWrapperPotential(pot=tnfw),
        'SolidBodyRot':       SolidBodyRotationWrapperPotential(pot=tnfw),
        'CorotatingRot':      CorotatingRotationWrapperPotential(pot=tnfw),
        'TimeDependentAmp':   TimeDependentAmplitudeWrapperPotential(pot=tnfw, A=lambda t: 1.0),
    }

    n_particles_list = [100, 500, 1000, 5000, 10000, 50000, 200000]

    # Collect results
    results = {'N': [], 'falcon': [], 'NFW_bare': [], 'TriaxNFW_bare': []}
    for wname in nfw_wrappers:
        results[f'NFW_{wname}'] = []
    for wname in tnfw_wrappers:
        results[f'TriaxNFW_{wname}'] = []

    for n in n_particles_list:
        print(f"N = {n:>7d} ...", flush=True)
        pos, masses = generate_particles(n)

        results['N'].append(n)
        results['falcon'].append(time_falcon(pos, masses))
        results['NFW_bare'].append(time_galpy_force(nfw, pos))
        results['TriaxNFW_bare'].append(time_galpy_force(tnfw, pos))

        for wname, wpot in nfw_wrappers.items():
            results[f'NFW_{wname}'].append(time_galpy_force(wpot, pos))
        for wname, wpot in tnfw_wrappers.items():
            results[f'TriaxNFW_{wname}'].append(time_galpy_force(wpot, pos))

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])
    N = results['N']

    # Save
    np.savez('wrapper_benchmark_results.npz', **results)

    # ---------- PLOT ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color scheme: bare = solid thick, wrappers = dashed thinner
    cmap_nfw = plt.cm.Blues
    cmap_tnfw = plt.cm.Oranges

    wrapper_names_nfw = list(nfw_wrappers.keys())
    wrapper_names_tnfw = list(tnfw_wrappers.keys())

    # --- Left panel: NFW + wrappers ---
    ax = axes[0]
    ax.loglog(N, results['falcon'], 'D-', color='black', lw=2, ms=6,
              label='falcON self-gravity')
    ax.loglog(N, results['NFW_bare'], 'o-', color='tab:blue', lw=2.5, ms=7,
              label='NFW (bare)')
    for i, wname in enumerate(wrapper_names_nfw):
        c = cmap_nfw(0.4 + 0.5 * i / max(len(wrapper_names_nfw) - 1, 1))
        ax.loglog(N, results[f'NFW_{wname}'], '--', color=c, lw=1.5, ms=5,
                  marker='s', label=f'NFW + {wname}')
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title('NFW + wrappers')
    ax.legend(fontsize=9, loc='upper left')

    # --- Right panel: TriaxialNFW + wrappers ---
    ax = axes[1]
    ax.loglog(N, results['falcon'], 'D-', color='black', lw=2, ms=6,
              label='falcON self-gravity')
    ax.loglog(N, results['TriaxNFW_bare'], 's-', color='tab:orange', lw=2.5, ms=7,
              label='TriaxialNFW (bare)')
    for i, wname in enumerate(wrapper_names_tnfw):
        c = cmap_tnfw(0.4 + 0.5 * i / max(len(wrapper_names_tnfw) - 1, 1))
        ax.loglog(N, results[f'TriaxNFW_{wname}'], '--', color=c, lw=1.5, ms=5,
                  marker='^', label=f'TriaxNFW + {wname}')
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title('TriaxialNFW + wrappers')
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig('wrapper_benchmark.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to wrapper_benchmark.png")

    # --- Print overhead summary at N=200k ---
    print("\n--- Wrapper overhead at N=200,000 ---")
    print(f"{'Potential':<35s}  {'Time [s]':>10s}  {'vs bare':>8s}")
    print("-" * 58)
    t_nfw = results['NFW_bare'][-1]
    print(f"{'NFW (bare)':<35s}  {t_nfw:10.4f}  {'1.00x':>8s}")
    for wname in wrapper_names_nfw:
        t = results[f'NFW_{wname}'][-1]
        print(f"{'NFW + ' + wname:<35s}  {t:10.4f}  {t/t_nfw:7.2f}x")
    print()
    t_tnfw = results['TriaxNFW_bare'][-1]
    print(f"{'TriaxialNFW (bare)':<35s}  {t_tnfw:10.4f}  {'1.00x':>8s}")
    for wname in wrapper_names_tnfw:
        t = results[f'TriaxNFW_{wname}'][-1]
        print(f"{'TriaxNFW + ' + wname:<35s}  {t:10.4f}  {t/t_tnfw:7.2f}x")


if __name__ == "__main__":
    main()
