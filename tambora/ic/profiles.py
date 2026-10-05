"""Common profiles, given by their total mass and scale: :class:`Plummer`, :class:`Hernquist`,
:class:`King` and :class:`TruncatedNFW`.

Pass one to :func:`tambora.ic.sample`. Each package's backend translates it into that package's
own model, so the parameters mean the same thing whichever backend samples it.
"""

import dataclasses
import math
import numbers


def _check(profile):
    for field in dataclasses.fields(profile):
        value = getattr(profile, field.name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{type(profile).__name__}'s {field.name} must be a number "
                            f"[{field.metadata['unit']}], got {type(value).__name__}")
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{type(profile).__name__}'s {field.name} must be positive and "
                             f"finite, got {value}")
        object.__setattr__(profile, field.name, float(value))


def _field(unit):
    return dataclasses.field(metadata={'unit': unit})


class _Profile:
    #: Marks tambora's own models: any installed backend may take one, even before its
    #: package has been imported (see tambora.interop._registry).
    _tambora_model = True

    def __post_init__(self):
        _check(self)

    def describe(self) -> str:
        """Short label for ``ParticleSet.meta``, e.g. ``'Plummer(M=1e+06, rscale=0.01)'``."""
        params = ', '.join(f'{f.name}={getattr(self, f.name):g}' for f in dataclasses.fields(self))
        return f'{type(self).__name__}({params})'


@dataclasses.dataclass(frozen=True)
class Plummer(_Profile):
    r"""A Plummer sphere, sampled with its exact isotropic distribution function.

    .. math::

        \rho(r) = \frac{3M}{4\pi b^3} \left(1 + \frac{r^2}{b^2}\right)^{-5/2},

    with :math:`b` = ``rscale``. Half the mass is inside :math:`1.305\,b`.

    Parameters
    ----------
    M : float
        Total mass :math:`M` [Msun].
    rscale : float
        Scale radius :math:`b` [kpc].

    References
    ----------
    Plummer, H. C. 1911, MNRAS, 71, 460.
    """
    M: float = _field('Msun')
    rscale: float = _field('kpc')


@dataclasses.dataclass(frozen=True)
class Hernquist(_Profile):
    r"""A Hernquist sphere, sampled with its exact isotropic distribution function.

    .. math::

        \rho(r) = \frac{M}{2\pi} \frac{a}{r\,(r + a)^3},

    with :math:`a` = ``rscale``. Half the mass is inside :math:`(1 + \sqrt{2})\,a`.

    Parameters
    ----------
    M : float
        Total mass :math:`M` [Msun].
    rscale : float
        Scale radius :math:`a` [kpc].

    References
    ----------
    Hernquist, L. 1990, ApJ, 356, 359.
    """
    M: float = _field('Msun')
    rscale: float = _field('kpc')


@dataclasses.dataclass(frozen=True)
class King(_Profile):
    r"""A King model: a lowered isothermal sphere whose density falls to zero at its tidal radius.

    Parameters
    ----------
    M : float
        Total mass [Msun].
    W0 : float
        Dimensionless central potential, :math:`\Psi(0)/\sigma^2`. Larger is more
        concentrated; galpy handles values up to about 200.
    rt : float
        Tidal radius [kpc].

    References
    ----------
    King, I. R. 1966, AJ, 71, 64.
    """
    M: float = _field('Msun')
    W0: float = _field('dimensionless')
    rt: float = _field('kpc')


@dataclasses.dataclass(frozen=True)
class TruncatedNFW(_Profile):
    r"""An exponentially truncated NFW halo.

    .. math::

        \rho(r) = \frac{\rho_s\, e^{-r/r_t}}{(r/r_s)\,(1 + r/r_s)^2},

    with :math:`r_s` = ``rscale``, :math:`r_t` = ``rtrunc``, and :math:`\rho_s` set by the
    total mass :math:`M`.

    To truncate an NFW halo you've already defined in galpy, pass galpy's
    ``ExpTruncNFWPotential.from_nfw(nfw, rc=...)`` to :func:`~tambora.ic.sample` instead.

    Parameters
    ----------
    M : float
        Total mass :math:`M` [Msun], including what lies beyond :math:`r_t`.
    rscale : float
        NFW scale radius :math:`r_s` [kpc].
    rtrunc : float
        Truncation radius :math:`r_t` [kpc], the scale of the exponential cutoff.

    Notes
    -----
    Requires galpy 1.12 or later.

    References
    ----------
    Navarro, J. F., Frenk, C. S. & White, S. D. M. 1996, ApJ, 462, 563.
    """
    M: float = _field('Msun')
    rscale: float = _field('kpc')
    rtrunc: float = _field('kpc')


PROFILES = (Plummer, Hernquist, King, TruncatedNFW)
