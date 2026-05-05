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
    HernquistPotential,
    JaffePotential,
    PowerSphericalPotential,
    TwoPowerSphericalPotential,
    PerfectEllipsoidPotential,
    PowerTriaxialPotential,
    TwoPowerTriaxialPotential,
    TriaxialGaussianPotential,
    TriaxialJaffePotential,
    TriaxialHernquistPotential,
    TriaxialNFWPotential,
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
plt.rcParams['font.serif'] = ['STIXGeneral', 'Liberation Serif', 'DejaVu Serif']
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

    hern = HernquistPotential(amp=1e12 * u.Msun, a=16 * u.kpc)
    thern = TriaxialHernquistPotential(amp=1e12 * u.Msun, a=16 * u.kpc, b=0.8, c=0.6)

    jaffe = JaffePotential(amp=1e12 * u.Msun, a=16 * u.kpc)
    tjaffe = TriaxialJaffePotential(amp=1e12 * u.Msun, a=16 * u.kpc, b=0.8, c=0.6)

    perf_ellip = PerfectEllipsoidPotential(amp=1e12 * u.Msun, a=16 * u.kpc, b=0.8, c=0.6)
    
    power_spher = PowerSphericalPotential(amp=1e12 * u.Msun, alpha=2.5)
    power_triaxial = PowerTriaxialPotential(amp=1e12 * u.Msun, alpha=2.5, b=0.8, c=0.6)
    
    two_power_spher = TwoPowerSphericalPotential(amp=1e12 * u.Msun, alpha=2.5, beta=3.0)
    two_power_triaxial = TwoPowerTriaxialPotential(amp=1e12 * u.Msun, alpha=2.5, beta=3.0, b=0.8, c=0.6)
    
    triaxial_gaussian = TriaxialGaussianPotential(amp=1e12 * u.Msun, b=0.8, c=0.6)

    n_particles_list = [100, 500, 1000, 5000, 10000, 50000, 200000]

    results = {
        'N': [],
        'nfw': [],
        'tnfw_vec': [],
        'tnfw_scalar': [],
        'hern': [],
        'thern_vec': [],
        'thern_scalar': [],
        'jaffe': [],
        'tjaffe_vec': [],
        'tjaffe_scalar': [],
        'perf_ellip': [],
        'power_spher': [],
        'power_triaxial_vec': [],
        'power_triaxial_scalar': [],
        'two_power_spher': [],
        'two_power_triaxial_vec': [],
        'two_power_triaxial_scalar': [],
        'triaxial_gaussian_vec': [],
        'triaxial_gaussian_scalar': [],
        'perf_ellip_vec': [],
        'perf_ellip_scalar': [],
        'falcon': []
    }

    for n in n_particles_list:
        print(f"N = {n:>7d} ...", end=" ", flush=True)
        pos, masses = generate_particles(n)

        t_nfw = time_galpy_force(nfw, pos)
        t_tnfw_vec = time_galpy_force(tnfw, pos)

        t_hern = time_galpy_force(hern, pos)
        t_thern_vec = time_galpy_force(thern, pos)

        t_jaffe = time_galpy_force(jaffe, pos)
        t_tjaffe_vec = time_galpy_force(tjaffe, pos)

        t_power_spherical = time_galpy_force(power_spher, pos)
        t_power_triaxial_vec = time_galpy_force(power_triaxial, pos)

        t_two_power_spherical = time_galpy_force(two_power_spher, pos)
        t_two_power_triaxial_vec = time_galpy_force(two_power_triaxial, pos)

        t_triaxial_gaussian_vec = time_galpy_force(triaxial_gaussian, pos)
        t_perf_ellip_vec = time_galpy_force(perf_ellip, pos)
    

        # For small N, also run scalar for comparison; skip for large N
        if n <= 50000:
            t_tnfw_scalar = time_galpy_force_scalar(tnfw, pos, n_repeat=1)
            t_thern_scalar = time_galpy_force_scalar(thern, pos, n_repeat=1)
            t_tjaffe_scalar = time_galpy_force_scalar(jaffe, pos, n_repeat=1)
            t_power_triaxial_scalar = time_galpy_force_scalar(power_triaxial, pos, n_repeat=1)
            t_two_power_triaxial_scalar = time_galpy_force_scalar(two_power_triaxial, pos, n_repeat=1)
            t_triaxial_gaussian_scalar = time_galpy_force_scalar(triaxial_gaussian, pos, n_repeat=1)    
            t_perf_ellip_scalar = time_galpy_force_scalar(perf_ellip, pos, n_repeat=1)
                                                          

        else:
            t_tnfw_scalar = np.nan
            t_thern_scalar = np.nan
            t_tjaffe_scalar = np.nan
            t_power_triaxial_scalar = np.nan
            t_two_power_triaxial_scalar = np.nan
            t_triaxial_gaussian_scalar = np.nan
            t_perf_ellip_scalar = np.nan
        t_falcon = time_falcon(pos, masses)

        results['N'].append(n)
        results['nfw'].append(t_nfw)
        results['tnfw_vec'].append(t_tnfw_vec)
        results['tnfw_scalar'].append(t_tnfw_scalar)
        results['hern'].append(t_hern)
        results['thern_vec'].append(t_thern_vec)
        results['thern_scalar'].append(t_thern_scalar)
        results['jaffe'].append(t_jaffe)
        results['tjaffe_vec'].append(t_tjaffe_vec)
        results['tjaffe_scalar'].append(t_tjaffe_scalar)
        results['power_spher'].append(t_power_spherical)
        results['power_triaxial_vec'].append(t_power_triaxial_vec)
        results['power_triaxial_scalar'].append(t_power_triaxial_scalar)
        results['two_power_spher'].append(t_two_power_spherical)
        results['two_power_triaxial_vec'].append(t_two_power_triaxial_vec)
        results['two_power_triaxial_scalar'].append(t_two_power_triaxial_scalar)
        results['triaxial_gaussian_vec'].append(t_triaxial_gaussian_vec)
        results['triaxial_gaussian_scalar'].append(t_triaxial_gaussian_scalar)
        results['perf_ellip_vec'].append(t_perf_ellip_vec)
        results['perf_ellip_scalar'].append(t_perf_ellip_scalar)
        results['falcon'].append(t_falcon)
        print(f"NFW={t_nfw:.4f}s  TriaxNFW(vec)={t_tnfw_vec:.4f}s  "
              f"TriaxNFW(scalar)={t_tnfw_scalar:.4f}s  "
               f"falcON={t_falcon:.4f}s")

    N = np.array(results['N'])
    t_nfw = np.array(results['nfw'])
    t_tnfw_vec = np.array(results['tnfw_vec'])
    t_tnfw_scalar = np.array(results['tnfw_scalar'])
    t_hern = np.array(results['hern'])
    t_thern_vec = np.array(results['thern_vec'])
    t_thern_scalar = np.array(results['thern_scalar'])
    t_jaffe = np.array(results['jaffe'])
    t_tjaffe_vec = np.array(results['tjaffe_vec'])
    t_tjaffe_scalar = np.array(results['tjaffe_scalar'])
    t_power_spher = np.array(results['power_spher'])
    t_power_triaxial_vec = np.array(results['power_triaxial_vec'])
    t_power_triaxial_scalar = np.array(results['power_triaxial_scalar'])
    t_two_power_spher = np.array(results['two_power_spher'])
    t_two_power_triaxial_vec = np.array(results['two_power_triaxial_vec'])
    t_two_power_triaxial_scalar = np.array(results['two_power_triaxial_scalar'])
    t_triaxial_gaussian_vec = np.array(results['triaxial_gaussian_vec'])
    t_triaxial_gaussian_scalar = np.array(results['triaxial_gaussian_scalar'])
    t_perf_ellip_vec = np.array(results['perf_ellip_vec'])
    t_perf_ellip_scalar = np.array(results['perf_ellip_scalar'])

    t_falcon = np.array(results['falcon'])

    # Save results
    np.savez('diagnostics_ellipsoidal_potentials_benchmark_results.npz',
            N=N, 
            nfw=t_nfw, tnfw_vec=t_tnfw_vec, tnfw_scalar=t_tnfw_scalar, 
            hern=t_hern, thern_vec=t_thern_vec, thern_scalar=t_thern_scalar,
            jaffe=t_jaffe, tjaffe_vec=t_tjaffe_vec, tjaffe_scalar=t_tjaffe_scalar,
            power_spher=t_power_spher, power_triaxial_vec=t_power_triaxial_vec, power_triaxial_scalar=t_power_triaxial_scalar,
            two_power_spher=t_two_power_spher, two_power_triaxial_vec=t_two_power_triaxial_vec, two_power_triaxial_scalar=t_two_power_triaxial_scalar,
            triaxial_gaussian_vec=t_triaxial_gaussian_vec, triaxial_gaussian_scalar=t_triaxial_gaussian_scalar,
            perf_ellip_vec=t_perf_ellip_vec, perf_ellip_scalar=t_perf_ellip_scalar,
            falcon=t_falcon)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Left panel: wall-clock time
    ax1.loglog(N, t_nfw, 'o-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax1.loglog(N, t_tnfw_vec, 'o--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_tnfw_scalar)
    ax1.loglog(N[mask], t_tnfw_scalar[mask], 'o:', color='tab:purple',
               lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_hern, '^-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax1.loglog(N, t_thern_vec, '^--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_thern_scalar)
    ax1.loglog(N[mask], t_thern_scalar[mask], '^:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_jaffe, 's-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax1.loglog(N, t_tjaffe_vec, 's--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_tjaffe_scalar)
    ax1.loglog(N[mask], t_tjaffe_scalar[mask], 's:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_power_spher, 'P-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax1.loglog(N, t_power_triaxial_vec, 'P--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_power_triaxial_scalar)
    ax1.loglog(N[mask], t_power_triaxial_scalar[mask], 'P:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_two_power_spher, 'H-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax1.loglog(N, t_two_power_triaxial_vec, 'H--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_two_power_triaxial_scalar)
    ax1.loglog(N[mask], t_two_power_triaxial_scalar[mask], 'H:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_triaxial_gaussian_vec, 'd--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_triaxial_gaussian_scalar)
    ax1.loglog(N[mask], t_triaxial_gaussian_scalar[mask], 'd:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_perf_ellip_vec, 'x--', color='tab:green', lw=2, ms=7, alpha=0.5)
    mask = ~np.isnan(t_perf_ellip_scalar)
    ax1.loglog(N[mask], t_perf_ellip_scalar[mask], 'x:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax1.loglog(N, t_falcon, 'D-', color='black', lw=2, ms=7)

    ax1.set_xlabel('Number of particles')
    ax1.set_ylabel('Wall-clock time [s]')
    ax1.set_title('External potential evaluation time')

    # Right panel: ratio
    ax2.loglog(N, t_nfw / t_falcon, 'o-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_tnfw_vec / t_falcon, 'o--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_tnfw_scalar / t_falcon, 'o:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_hern / t_falcon, '^-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_thern_vec / t_falcon, '^--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_thern_scalar / t_falcon, '^:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_jaffe / t_falcon, 's-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_tjaffe_vec / t_falcon, 's--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_tjaffe_scalar / t_falcon, 's:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_power_spher / t_falcon, 'P-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_power_triaxial_vec / t_falcon, 'P--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_power_triaxial_scalar / t_falcon, 'P:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_two_power_spher / t_falcon, 'H-', color='tab:blue', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_two_power_triaxial_vec / t_falcon, 'H--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_two_power_triaxial_scalar / t_falcon, 'H:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_triaxial_gaussian_vec / t_falcon, 'd--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_triaxial_gaussian_scalar / t_falcon, 'd:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.loglog(N, t_perf_ellip_vec / t_falcon, 'x--', color='tab:green', lw=2, ms=7, alpha=0.5)
    ax2.loglog(N, t_perf_ellip_scalar / t_falcon, 'x:', color='tab:purple', lw=1.5, ms=7, alpha=0.5)

    ax2.axhline(1.0, ls='--', color='gray', lw=1.5)

    ax2.set_xlabel('Number of particles')
    ax2.set_ylabel('Ratio (ext. potential / self-gravity)')
    ax2.set_title('Cost relative to falcON self-gravity')

    # --- Build grouped legend with section headers ---
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def _section_header(title):
        """Return (handle, label) for a bold section header with no visible marker."""
        handle = Line2D([], [], linestyle='none', marker='none', color='none')
        return handle, f'$\\bf{{{title}}}$'

    def _entry(marker, ls, color, label, **kw):
        lw = kw.pop('lw', 2)
        ms = kw.pop('ms', 7)
        alpha = kw.pop('alpha', 0.5)
        return Line2D([], [], marker=marker, linestyle=ls, color=color,
                      lw=lw, ms=ms, alpha=alpha, **kw), label

    handles, labels = [], []

    # falcON
    h, l = _entry('D', '-', 'black', 'falcON self-gravity', alpha=1.0)
    handles.append(h); labels.append(l)
    # blank spacer
    handles.append(Line2D([], [], linestyle='none', color='none'))
    labels.append('')

    # NFW family
    h, l = _section_header('NFW'); handles.append(h); labels.append(l)
    h, l = _entry('o', '-', 'tab:blue', 'Spherical (vectorized)'); handles.append(h); labels.append(l)
    h, l = _entry('o', '--', 'tab:green', 'Triaxial (vectorized, new)'); handles.append(h); labels.append(l)
    h, l = _entry('o', ':', 'tab:purple', 'Triaxial (scalar loop, old)'); handles.append(h); labels.append(l)

    # Hernquist family
    h, l = _section_header('Hernquist'); handles.append(h); labels.append(l)
    h, l = _entry('^', '-', 'tab:blue', 'Spherical (vectorized)'); handles.append(h); labels.append(l)
    h, l = _entry('^', '--', 'tab:green', 'Triaxial (vectorized, new)'); handles.append(h); labels.append(l)
    h, l = _entry('^', ':', 'tab:purple', 'Triaxial (scalar loop, old)'); handles.append(h); labels.append(l)

    # Jaffe family
    h, l = _section_header('Jaffe'); handles.append(h); labels.append(l)
    h, l = _entry('s', '-', 'tab:blue', 'Spherical (vectorized)'); handles.append(h); labels.append(l)
    h, l = _entry('s', '--', 'tab:green', 'Triaxial (vectorized, new)'); handles.append(h); labels.append(l)
    h, l = _entry('s', ':', 'tab:purple', 'Triaxial (scalar loop, old)'); handles.append(h); labels.append(l)

    # PowerSpherical family
    h, l = _section_header('Power'); handles.append(h); labels.append(l)
    h, l = _entry('P', '-', 'tab:blue', 'Spherical (vectorized)'); handles.append(h); labels.append(l)
    h, l = _entry('P', '--', 'tab:green', 'Triaxial (vectorized, new)'); handles.append(h); labels.append(l)
    h, l = _entry('P', ':', 'tab:purple', 'Triaxial (scalar loop, old)'); handles.append(h); labels.append(l)

    # TwoPowerSpherical family
    h, l = _section_header('TwoPower'); handles.append(h); labels.append(l)
    h, l = _entry('H', '-', 'tab:blue', 'Spherical (vectorized)'); handles.append(h); labels.append(l)
    h, l = _entry('H', '--', 'tab:green', 'Triaxial (vectorized, new)'); handles.append(h); labels.append(l)
    h, l = _entry('H', ':', 'tab:purple', 'Triaxial (scalar loop, old)'); handles.append(h); labels.append(l)

    # TriaxialGaussian
    h, l = _section_header('TriaxialGaussian'); handles.append(h); labels.append(l)
    h, l = _entry('d', '--', 'tab:green', 'Vectorized (new)'); handles.append(h); labels.append(l)
    h, l = _entry('d', ':', 'tab:purple', 'Scalar loop (old)'); handles.append(h); labels.append(l)

    # PerfectEllipsoid
    h, l = _section_header('PerfectEllipsoid'); handles.append(h); labels.append(l)
    h, l = _entry('x', '--', 'green', 'Vectorized (new)'); handles.append(h); labels.append(l)
    h, l = _entry('x', ':', 'purple', 'Scalar loop (old)'); handles.append(h); labels.append(l)

    ax2.legend(handles, labels, fontsize=8, loc='center left',
               bbox_to_anchor=(1.02, 0.5), handletextpad=0.6,
               labelspacing=0.3, borderaxespad=0.5)

    plt.tight_layout()
    fig.subplots_adjust(right=0.72)
    plt.savefig('ellipsoidal_benchmark.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to ellipsoidal_benchmark.png")


if __name__ == "__main__":
    main()
