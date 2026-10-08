"""galpy potentials as a :class:`~tambora.interop.PotentialBackend`."""

import pickle
import weakref

from galpy import potential as _gp

from .bridge import (
    _check_physical, _check_supported_pot, _ensure_pot, _galpy_pot_to_fns, _iter_components,
)
from .._backend import PotentialBackend


def _flatten(obj):
    """Items of possibly nested lists/tuples, in order. Anything else is one item.

    galpy's own ``flatten`` recurses forever on a string and skips tuples, so it can't
    be used before the items have been checked.
    """
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten(item)
    else:
        yield obj


# Each potential's dedup key is its pickle the first time tambora sees it. Some galpy
# potentials cache results in themselves when evaluated, so a later pickle would stop the
# same object, or an identical fresh copy, from matching. Kept per component, because
# wrapping a list builds a new CompositePotential every time.
_first_seen_keys = weakref.WeakKeyDictionary()


def _component_key(p):
    try:
        return _first_seen_keys[p]
    except (KeyError, TypeError):
        pass
    try:
        key = pickle.dumps(p)
    except Exception:           # unpicklable (e.g. holds a lambda): fall back to identity
        key = id(p)
    try:
        _first_seen_keys[p] = key
    except TypeError:           # can't be weakly referenced: recomputed next time
        pass
    return key


class GalpyPotential(PotentialBackend):
    """A galpy ``Potential``, a ``CompositePotential``, or a (nested) list/tuple of them."""

    name = 'galpy'

    @classmethod
    def accepts(cls, obj) -> bool:
        if isinstance(obj, (list, tuple)):
            # Claim the list if anything in it is galpy; __init__ then names the bad element.
            return any(isinstance(p, _gp.Potential) for p in _flatten(obj))
        return isinstance(obj, _gp.Potential)

    def __init__(self, obj):
        # Check every element *before* combining: galpy's `+` recurses forever on a non-potential.
        is_list = isinstance(obj, (list, tuple))
        items = list(_flatten(obj)) if is_list else [obj]
        nested = is_list and any(isinstance(p, (list, tuple)) for p in obj)
        for i, p in enumerate(items):
            if not isinstance(p, _gp.Potential):
                where = f" (element {i} of the {'flattened ' if nested else ''}list)" if is_list else ""
                raise TypeError(f"Expected a galpy Potential{where}, got {type(p).__name__}.")
        pot = _ensure_pot(items) if is_list else obj
        for p in _iter_components(pot):
            _check_physical(p)
        _check_supported_pot(pot)
        self.obj = pot
        self._acc_fn, self._pot_fn = _galpy_pot_to_fns(pot)

    def acc(self, pos, t):
        return self._acc_fn(pos, t)

    def potential(self, pos, t):
        return self._pot_fn(pos, t)

    def dedup_key(self):
        return tuple(_component_key(p) for p in _iter_components(self.obj))

    def describe(self) -> str:
        return '+'.join(type(p).__name__ for p in _iter_components(self.obj))
