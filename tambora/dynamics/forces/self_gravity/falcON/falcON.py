from ._falcon import gravity
from .....util import G_INTERNAL

def _falcON_gravity(pos, mass, eps, theta, kernel, return_potential):
    """
    Compute accelerations and potentials from self-gravity using pyfalcon.

    Parameters
    ----------
    pos : (N, 3) array
        Positions of particles.
        Unit: kpc
    mass : (N,) array
        Masses of particles.
        Unit: Msun
    eps : float or (N,) array
        Gravitational softening length.
        Unit: kpc
    theta : float, optional
        Tree opening angle (default 0.6). Smaller = more accurate but slower.
    kernel : int, optional
        Softening kernel to use.
    return_potential : bool
        Whether to return the self-gravitational potential.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    """
    if return_potential:
        return gravity(pos, mass * G_INTERNAL, eps, theta=theta, kernel=kernel)
    else:
        return gravity(pos, mass * G_INTERNAL, eps, theta=theta, kernel=kernel)[0]