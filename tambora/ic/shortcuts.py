"""Shortcuts that draw a profile in one call: :func:`mkPlummer`, :func:`mkHernquist`,
:func:`mkKing` and :func:`mkTruncNFW`. ``mkPlummer(M, rscale, n)`` is
``sample(Plummer(M, rscale), n)``."""

from typing import Optional

from .particles import ParticleSet
from .profiles import Hernquist, King, Plummer, TruncatedNFW
from .sampler import sample


def mkPlummer(M: float, rscale: float, n: int, *, potential=None,
              seed: Optional[int] = None) -> ParticleSet:
    """Draw ``n`` particles from a :class:`Plummer` sphere.

    Parameters
    ----------
    M : float
        Total mass [Msun].
    rscale : float
        Scale radius [kpc].
    n : int
        Number of particles.
    potential : object, optional
        The potential to draw the sphere's density in, if not its own, as for :func:`sample`.
    seed : int, optional
        Seed to use for the draws, in ``[0, 2**32)``. Default: a fresh seed.

    Returns
    -------
    ParticleSet
        ``sample(Plummer(M, rscale), n, potential=potential, seed=seed)``.
    """
    return sample(Plummer(M, rscale), n, potential=potential, seed=seed)


def mkHernquist(M: float, rscale: float, n: int, *, potential=None,
                seed: Optional[int] = None) -> ParticleSet:
    """Draw ``n`` particles from a :class:`Hernquist` sphere.

    Parameters
    ----------
    M : float
        Total mass [Msun].
    rscale : float
        Scale radius [kpc].
    n : int
        Number of particles.
    potential : object, optional
        The potential to draw the sphere's density in, if not its own, as for :func:`sample`.
    seed : int, optional
        Seed to use for the draws, in ``[0, 2**32)``. Default: a fresh seed.

    Returns
    -------
    ParticleSet
        ``sample(Hernquist(M, rscale), n, potential=potential, seed=seed)``.
    """
    return sample(Hernquist(M, rscale), n, potential=potential, seed=seed)


def mkKing(M: float, W0: float, rt: float, n: int, *, seed: Optional[int] = None) -> ParticleSet:
    r"""Draw ``n`` particles from a :class:`King` model.

    Parameters
    ----------
    M : float
        Total mass [Msun].
    W0 : float
        Dimensionless central potential, :math:`\Psi(0)/\sigma^2`. At most 45.
    rt : float
        Tidal radius [kpc].
    n : int
        Number of particles.
    seed : int, optional
        Seed to use for the draws, in ``[0, 2**32)``. Default: a fresh seed.

    Returns
    -------
    ParticleSet
        ``sample(King(M, W0, rt), n, seed=seed)``.
    """
    return sample(King(M, W0, rt), n, seed=seed)


def mkTruncNFW(M: float, rscale: float, rtrunc: float, n: int, *, potential=None,
               seed: Optional[int] = None) -> ParticleSet:
    """Draw ``n`` particles from a :class:`TruncatedNFW` halo.

    Parameters
    ----------
    M : float
        Total mass [Msun], including what lies beyond ``rtrunc``.
    rscale : float
        NFW scale radius [kpc].
    rtrunc : float
        Truncation radius [kpc], the scale of the exponential cutoff. At least
        ``rscale / 20``.
    n : int
        Number of particles.
    potential : object, optional
        The potential to draw the halo's density in, if not its own, as for :func:`sample`.
    seed : int, optional
        Seed to use for the draws, in ``[0, 2**32)``. Default: a fresh seed.

    Returns
    -------
    ParticleSet
        ``sample(TruncatedNFW(M, rscale, rtrunc), n, potential=potential, seed=seed)``.

    Notes
    -----
    Requires galpy 1.12 or later.
    """
    return sample(TruncatedNFW(M, rscale, rtrunc), n, potential=potential, seed=seed)
