"""
Tests for the backend registry and the ``PotentialBackend`` interface.
"""

import numpy as np
import pytest

from tambora.interop import PotentialBackend, available_potential_backends, potential_backend_for
from tambora.interop import _registry


# --- PotentialBackend: the ABC enforces the interface ---------------------------

class _Incomplete(PotentialBackend):
    name = 'incomplete'

    @classmethod
    def accepts(cls, obj):
        return True

    def __init__(self, obj):
        self.obj = obj

    def acc(self, pos, t):
        return np.zeros_like(pos)
    # no potential()


class _Minimal(_Incomplete):
    name = 'minimal'

    def potential(self, pos, t):
        return np.zeros(len(pos))


def test_a_backend_missing_a_method_cannot_be_instantiated():
    with pytest.raises(TypeError, match="potential"):
        _Incomplete(object())


def test_the_defaults_are_harmless():
    b = _Minimal(3.0)
    assert b.check_still_valid() is None
    assert b.dedup_key() is None                 # opts out of dedup
    assert b.describe() == 'float'


# --- dispatch ----------------------------------------------------------------

class _FakeEntry:
    """Registry entries pointing at a package that is *not* imported."""
    @staticmethod
    def make(name, package='tambora_no_such_package', module='tambora_no_such_module'):
        return _registry._Entry(name, package, module, 'Nope', 'https://example.org/install')


def test_a_backend_whose_package_was_never_imported_is_skipped(monkeypatch):
    # Its module path doesn't exist: if dispatch tried to import it, this would raise.
    entries = (_FakeEntry.make('ghost'),) + _registry._POTENTIAL_BACKENDS
    monkeypatch.setattr(_registry, "_POTENTIAL_BACKENDS", entries)
    with pytest.raises(TypeError, match="Can't use a str"):
        potential_backend_for('not a potential')


def test_naming_a_backend_whose_package_is_missing_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(_registry, "_POTENTIAL_BACKENDS", (_FakeEntry.make('ghost'),))
    with pytest.raises(ImportError, match=r"needs tambora_no_such_package.*https://example.org/install"):
        potential_backend_for(object(), backend='ghost')


def test_the_error_for_an_unusable_object_lists_the_supported_packages():
    with pytest.raises(TypeError, match=r"Supported packages: galpy"):
        potential_backend_for(object())


def test_the_available_backends_are_the_ones_whose_package_is_installed(monkeypatch):
    # numpy is always installed; the fake package never is.
    entries = (_FakeEntry.make('ghost'), _FakeEntry.make('present', package='numpy'))
    monkeypatch.setattr(_registry, "_POTENTIAL_BACKENDS", entries)
    assert available_potential_backends() == ('present',)


@pytest.mark.parametrize("entry", _registry._POTENTIAL_BACKENDS, ids=lambda e: e.name)
def test_backend_class_name_matches_its_registry_entry(entry):
    # The registry needs the name before importing the backend, and the class needs
    # it afterwards (repr, dedup key), so there are two copies. This keeps them equal,
    # and fails with AttributeError if a backend forgot to set `name` at all.
    pytest.importorskip(entry.package)
    assert _registry._load(entry).name == entry.name


def test_galpy_objects_go_to_the_galpy_backend():
    gp = pytest.importorskip("galpy.potential")
    b = potential_backend_for(gp.PlummerPotential(ro=8., vo=220.))
    assert b.name == 'galpy'


def test_nested_lists_of_galpy_potentials_go_to_the_galpy_backend():
    gp = pytest.importorskip("galpy.potential")
    p, q = gp.PlummerPotential(ro=8., vo=220.), gp.NFWPotential(ro=8., vo=220.)
    assert potential_backend_for([[p, q]]).name == 'galpy'


# --- naming the backend explicitly -------------------------------------------

def test_the_named_backend_is_used():
    gp = pytest.importorskip("galpy.potential")
    assert potential_backend_for(gp.PlummerPotential(ro=8., vo=220.), backend='galpy').name == 'galpy'


def test_an_unknown_backend_name_is_a_value_error_listing_the_known_ones():
    with pytest.raises(ValueError, match=r"Unknown potential backend 'nope'. Known backends: 'galpy'"):
        potential_backend_for(object(), backend='nope')


def test_a_named_backend_that_cannot_use_the_object_is_a_type_error():
    pytest.importorskip("galpy")
    with pytest.raises(TypeError, match=r"The 'galpy' backend can't use a str"):
        potential_backend_for('not a potential', backend='galpy')
