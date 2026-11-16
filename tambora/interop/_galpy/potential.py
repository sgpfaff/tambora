"""galpy potentials as a :class:`~tambora.interop.PotentialBackend`."""

import pickle
import warnings
import weakref

import galpy
from galpy import potential as _gp

from .bridge import (
    _check_physical, _combined, _galpy_pot_to_fns, _iter_components, _units,
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
# wrapping a list builds a new CompositePotential every time. Kept with it: why that
# pickle failed, if it did.
_first_seen_keys = weakref.WeakKeyDictionary()


def _first_seen(p):
    """*p*'s dedup key, and why it couldn't be pickled (None if it could)."""
    try:
        return _first_seen_keys[p]
    except (KeyError, TypeError):
        pass
    try:
        seen = pickle.dumps(p), None
    except Exception as e:      # unpicklable (e.g. holds a lambda): fall back to identity
        seen = id(p), f"{type(e).__name__}: {e}"
    try:
        _first_seen_keys[p] = seen
    except TypeError:           # can't be weakly referenced: recomputed next time
        pass
    return seen


def _component_key(p):
    return _first_seen(p)[0]


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
        pot = _combined(items) if is_list else obj
        for p in _iter_components(pot):
            _check_physical(p)
        self._wrap(pot)

    def _wrap(self, pot):
        self.obj = pot
        self._components = list(_iter_components(pot))     # as converted
        self._units = [tuple(map(float, _units(p))) for p in self._components]  # [kpc, km/s], fixed now
        self._acc_fn, self._pot_fn = _galpy_pot_to_fns(pot)

    def acc(self, pos, t):
        return self._acc_fn(pos, t)

    def potential(self, pos, t):
        return self._pot_fn(pos, t)

    def dedup_key(self):
        return tuple(_component_key(p) for p in _iter_components(self.obj))

    def describe(self) -> str:
        return '+'.join(type(p).__name__ for p in _iter_components(self.obj))

    def save_problem(self):
        """Why pickling a component failed when it was added, or None if every one pickled."""
        return next((why for why in (_first_seen(p)[1] for p in self._components) if why), None)

    def save(self):
        """The components, pickled as they are now, with the units they're converted with.

        Raises if a component can't be pickled, e.g. because it holds a lambda.
        """
        return {'galpy_version': galpy.__version__,
                'data': pickle.dumps((self._components, self._units), protocol=4)}

    @classmethod
    def load(cls, record):
        if record['galpy_version'] != galpy.__version__:
            warnings.warn(f"A potential was saved with galpy {record['galpy_version']}, and this is galpy "
                          f"{galpy.__version__}. It may not load, or may give other forces.", stacklevel=3)
        components, units = pickle.loads(record['data'])
        for p, (ro, vo) in zip(components, units):
            if _units(p) != (ro, vo):       # changed after it was added
                p.turn_physical_on(ro=ro, vo=vo)
        backend = cls.__new__(cls)          # its units were checked when it was added
        backend._wrap(_combined(components))
        return backend
