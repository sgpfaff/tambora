from .ExternalConservativeForce import ExternalConservativeForce
import numpy as np
from tambora.interop._galpy.bridge import (
                _check_physical,
                _ensure_pot, _iter_components, _get_ro_vo,
             )

from tambora.units import KMS_TO_KPCGYR
from galpy.util.coords import rect_to_cyl
from galpy import potential

class TidalTensorGalpyForce(ExternalConservativeForce):
    def __init__(self, pot, center=None):
        pot = _ensure_pot(pot)
        for p in _iter_components(pot):
            if not isinstance(p, potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        self._pot = pot
        self._center = None if center is None else np.asarray(center, float)
        self._ro, vo = _get_ro_vo(pot)
        self._vo_int = vo * KMS_TO_KPCGYR

    def _tensor_disp(self, pos):
        pos = np.atleast_2d(np.asarray(pos, float))
        c = np.median(pos, axis=0) if self._center is None else self._center
        Rc, phic, zc = rect_to_cyl(*np.atleast_2d(c).T)
        T = np.asarray(self._pot.ttensor(R=Rc/self._ro, phi=phic, z=zc/self._ro,
                                         t=0.0, use_physical=False)).reshape(3, 3)
        return T, pos - c

    def tidal_tensor(self, pos):
        """Tidal tensor at the cluster center, in internal units [1/Gyr^2].

        The center is ``self._center`` if set, else the median of ``pos``
        (matching ``acc``/``potential``). Symmetric (3, 3); its largest
        eigenvalue is the stretching (tidal) direction.
        """
        T, _ = self._tensor_disp(pos)
        return T * self._vo_int**2 / self._ro**2

    def acc(self, pos, t):
        T, d = self._tensor_disp(pos)
        return d @ T.T * self._vo_int**2 / self._ro**2

    def potential(self, pos, t):          # per unit mass; -grad = acc
        T, d = self._tensor_disp(pos)
        return -0.5 * np.einsum('ni,ij,nj->n', d, T, d) * self._vo_int**2 / self._ro**2
