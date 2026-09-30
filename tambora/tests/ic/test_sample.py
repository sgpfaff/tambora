"""
Tests for ``tambora.ic.sample``, through a stand-in backend so they don't need a package.
"""

import numpy as np
import pytest

from tambora import ic
from tambora.interop import SamplerBackend, _registry


class _Model:
    pass


class _StandIn(SamplerBackend):
    name = 'stand_in'

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, _Model)

    def __init__(self, obj):
        self.obj = obj

    @property
    def total_mass(self):
        return 12.

    def draw(self, n, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 3)), rng.normal(size=(n, 3))


@pytest.fixture(autouse=True)
def only_the_stand_in(monkeypatch):
    entry = _registry._Entry('stand_in', 'numpy', __name__, '_StandIn', 'https://example.org/install')
    monkeypatch.setattr(_registry, "_SAMPLER_BACKENDS", (entry,))


def test_particles_share_the_models_mass():
    ps = ic.sample(_Model(), 7, seed=1)
    np.testing.assert_allclose(ps.mass, 12. / 7)
    assert ps.mass.sum() == pytest.approx(12.)
    assert ps.pos.shape == ps.vel.shape == (7, 3)


def test_meta_records_the_backend_model_and_seed():
    assert ic.sample(_Model(), 3, seed=5).meta == {'backend': 'stand_in', 'model': '_Model', 'seed': 5}


def test_the_same_seed_gives_the_same_particles():
    a, b = ic.sample(_Model(), 5, seed=9), ic.sample(_Model(), 5, seed=9)
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.vel, b.vel)


def test_without_a_seed_a_fresh_one_is_drawn_and_recorded():
    a, b = ic.sample(_Model(), 5), ic.sample(_Model(), 5)
    assert a.meta['seed'] != b.meta['seed']
    np.testing.assert_array_equal(ic.sample(_Model(), 5, seed=a.meta['seed']).pos, a.pos)


def test_a_numpy_integer_seed_is_accepted():
    assert ic.sample(_Model(), 2, seed=np.int64(4)).meta['seed'] == 4


@pytest.mark.parametrize("seed, error", [
    pytest.param(-1, ValueError, id='negative'),
    pytest.param(2**32, ValueError, id='too_large'),
    pytest.param(1.5, TypeError, id='float'),
    pytest.param('3', TypeError, id='string'),
    pytest.param(True, TypeError, id='bool'),
])
def test_a_bad_seed_is_rejected(seed, error):
    with pytest.raises(error, match="seed must"):
        ic.sample(_Model(), 2, seed=seed)


@pytest.mark.parametrize("n, error", [
    pytest.param(0, ValueError, id='zero'),
    pytest.param(-3, ValueError, id='negative'),
    pytest.param(2.5, TypeError, id='float'),
])
def test_a_bad_n_is_rejected(n, error):
    with pytest.raises(error):
        ic.sample(_Model(), n, seed=1)


def test_an_object_no_backend_accepts_is_a_type_error():
    with pytest.raises(TypeError, match="Can't sample a str. Supported packages: stand_in"):
        ic.sample('not a model', 5)


def test_the_backend_can_be_named():
    assert ic.sample(_Model(), 2, seed=1, backend='stand_in').meta['backend'] == 'stand_in'
    with pytest.raises(ValueError, match="Unknown sampler backend 'nope'"):
        ic.sample(_Model(), 2, seed=1, backend='nope')
