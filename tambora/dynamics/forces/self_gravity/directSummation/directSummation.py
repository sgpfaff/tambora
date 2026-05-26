from .....util import G_INTERNAL
from ._direct_summation import gravity
import numpy as np

def _direct_summation_py(pos, mass, eps, return_potential):
    '''
    Compute accelerations and potentials from self-gravity using direct summation.
    
    Parameters
    ----------
    pos : (N, 3) array
        Positions of particles.
        Unit: kpc
    mass : (N,) array
        Masses of particles.
        Unit: Msun
    eps : float or (N,) array
        Gravitational softening length(s). When an array, pairwise softening
        is the arithmetic mean: eps_ij = (eps_i + eps_j) / 2.
        Unit: kpc
    return_potential : bool
        Whether to return the self-gravitational potential. Default is True.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    '''
    N = len(mass)
    eps = np.broadcast_to(np.asarray(eps, dtype=float), (N,))
    acc = np.zeros_like(pos)
    if return_potential:
        pot = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        dx = pos[mask] - pos[i]  # (N-1, 3)
        eps_ij = 0.5 * (eps[i] + eps[mask])  # (N-1,) pairwise softening
        r2 = np.sum(dx**2, axis=1) + eps_ij**2  # (N-1,)
        inv_r3 = 1.0 / r2**1.5  # (N-1,)
        acc[i] = G_INTERNAL * np.sum(mass[mask, None] * dx * inv_r3[:, None], axis=0)
        if return_potential:
            pot[i] = -G_INTERNAL * np.sum(mass[mask] / np.sqrt(r2))
    if return_potential:
        return acc, pot
    else:
        return acc

def _direct_summation_C(pos, mass, eps, return_potential):
    '''
    Compute accelerations and potentials from self-gravity using direct summation.
    
    Parameters
    ----------
    pos : (N, 3) array
        Positions of particles.
        Unit: kpc
    mass : (N,) array
        Masses of particles.
        Unit: Msun
    eps : float or (N,) array
        Gravitational softening length(s). When an array, pairwise softening
        is the arithmetic mean: eps_ij = (eps_i + eps_j) / 2.
        Unit: kpc
    return_potential : bool
        Whether to return the self-gravitational potential. Default is True.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    '''
    if return_potential:
        return gravity(pos, mass * G_INTERNAL, eps)
    else:
        return gravity(pos, mass * G_INTERNAL, eps)[0]