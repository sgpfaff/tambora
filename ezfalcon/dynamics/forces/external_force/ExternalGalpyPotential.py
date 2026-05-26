from .MassIndependentConservForce import MassIndependentConservForce
import galpy
from ....util._galpy_bridge import (
                _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn,
                _check_physical, _check_supported_pot,
                _ensure_pot, _iter_components,
            )
import numpy as np

class ExternalGalpyPotential(MassIndependentConservForce):
    """Wrap a galpy ``Potential`` as an :class:`ExternalForce`.

    Conservative (has a potential) and not C-backed. Combine with other
    galpy potentials via ``+``::

        sim.add_external_force(ExternalGalpyPotential(nfw)
                               + ExternalGalpyPotential(disk))
    """

    def __init__(self, potential):
        potential = _ensure_pot(potential)
        for p in _iter_components(potential):
            if not isinstance(p, galpy.potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        _check_supported_pot(potential)
        self._acc_fn = _galpy_pot_to_acc_fn(potential)
        self._potential_fn = _galpy_pot_to_pot_fn(potential)

    def acc(self, pos: np.ndarray, t) -> np.ndarray:
        return self._acc_fn(pos, t)

    def potential(self,  pos: np.ndarray, t) -> np.ndarray:
        return self._potential_fn(pos, t)
    