"""
Tests for ``tambora.ic.sample`` and ``tambora.ic.sample_components``, through a stand-in
backend so they don't need a package.
"""

import sys

import numpy as np
import pytest

from tambora import ic
from tambora.interop import SamplerBackend, _registry


class _Model:
    def __init__(self, name='_Model', mass=12.):
        self.name, self.mass = name, mass

    def __repr__(self):
        return self.name


class _StandIn(SamplerBackend):
    name = 'stand_in'

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, _Model)

    def __init__(self, obj, potential=None):
        self.obj, self.potential = obj, potential

    @property
    def total_mass(self):
        return self.obj.mass

    def draw(self, n, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 3)), rng.normal(size=(n, 3))

    def describe(self):
        return repr(self.obj) if self.potential is None else f'{self.obj!r} in {self.potential!r}'


@pytest.fixture(autouse=True)
def only_the_stand_in(monkeypatch):
    entry = _registry._Entry('stand_in', 'numpy', __name__, '_StandIn')
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


class _ProfileStandIn(_StandIn):
    name = 'profile_stand_in'

    @classmethod
    def accepts(cls, obj):
        return isinstance(obj, (_Model, ic.Plummer))

    @property
    def total_mass(self):
        return 12.


def _only_a_backend_whose_package_isnt_imported(monkeypatch):
    # 'wave' is installed (it's in the standard library) but nothing here imports it.
    monkeypatch.delitem(sys.modules, 'wave', raising=False)
    entry = _registry._Entry('profile_stand_in', 'wave', __name__, '_ProfileStandIn')
    monkeypatch.setattr(_registry, "_SAMPLER_BACKENDS", (entry,))


def test_a_profile_goes_to_an_installed_backend_even_before_its_package_is_imported(monkeypatch):
    _only_a_backend_whose_package_isnt_imported(monkeypatch)
    assert ic.sample(ic.Plummer(M=1., rscale=1.), 3, seed=1).meta['backend'] == 'profile_stand_in'


def test_another_packages_object_still_needs_its_package_imported(monkeypatch):
    # _Model stands for another package's object: it can't be one if that package was never imported.
    _only_a_backend_whose_package_isnt_imported(monkeypatch)
    with pytest.raises(TypeError, match="Can't sample a _Model"):
        ic.sample(_Model(), 3, seed=1)


def test_a_profile_skips_a_backend_whose_package_isnt_installed(monkeypatch):
    entry = _registry._Entry('profile_stand_in', 'no_such_package_anywhere', __name__, '_ProfileStandIn')
    monkeypatch.setattr(_registry, "_SAMPLER_BACKENDS", (entry,))
    with pytest.raises(TypeError, match=r"Can't sample a Plummer\. .*\(installed: none\)"):
        ic.sample(ic.Plummer(M=1., rscale=1.), 3, seed=1)


# --- a model in another potential, and several components together ------------------------

def test_the_potential_goes_to_the_backend():
    halo = _Model('halo')
    assert ic.sample(_Model(), 3, potential=halo, seed=1).meta['model'] == '_Model in halo'


STARS, DM = _Model('stars', mass=2.), _Model('dm', mass=50.)


def _components(**kw):
    return ic.sample_components([STARS, DM], n=[4, 10], **kw)


def test_each_component_is_drawn_in_the_potential_of_them_all():
    stars, dm = _components(seed=1)
    assert stars.meta['model'] == 'stars in [stars, dm]'
    assert dm.meta['model'] == 'dm in [stars, dm]'


def test_the_components_come_back_as_a_tuple_in_their_order():
    parts = _components(seed=1)
    assert type(parts) is tuple
    assert [ps.meta['model'].split()[0] for ps in parts] == ['stars', 'dm']
    assert ic.sample_components((STARS, DM), n=(4, 10), seed=1)[1].meta == parts[1].meta


def test_each_component_has_its_own_number_of_particles_and_mass():
    stars, dm = _components(seed=1)
    assert (len(stars), len(dm)) == (4, 10)
    assert stars.mass.sum() == pytest.approx(2.)
    assert dm.mass.sum() == pytest.approx(50.)


def test_each_component_gets_its_own_seed_from_the_one_given():
    stars, dm = _components(seed=7)
    assert stars.meta['seed'] != dm.meta['seed'] and 7 not in (stars.meta['seed'], dm.meta['seed'])
    assert stars.meta['parent_seed'] == dm.meta['parent_seed'] == 7
    # Its own seed redraws it with sample, in the potential of them all.
    again = ic.sample(DM, 10, potential=[STARS, DM], seed=dm.meta['seed'])
    np.testing.assert_array_equal(again.pos, dm.pos)


def test_the_same_seed_gives_the_same_components():
    for a, b in zip(_components(seed=3), _components(seed=3)):
        np.testing.assert_array_equal(a.pos, b.pos)
        np.testing.assert_array_equal(a.vel, b.vel)


def test_a_components_particles_dont_depend_on_how_many_the_others_have():
    a = ic.sample_components([STARS, DM], n=[4, 10], seed=3)
    b = ic.sample_components([STARS, DM], n=[4, 99], seed=3)
    np.testing.assert_array_equal(a[0].pos, b[0].pos)


def test_without_a_seed_a_fresh_parent_seed_is_drawn_and_recorded():
    a, b = _components(), _components()
    assert a[0].meta['parent_seed'] != b[0].meta['parent_seed']
    again = _components(seed=a[0].meta['parent_seed'])
    np.testing.assert_array_equal(again[1].pos, a[1].pos)


@pytest.mark.parametrize("components, n, error, match", [
    # A dict would unpack into its keys, so it's refused rather than read in its own order.
    pytest.param({'stars': STARS, 'dm': DM}, [4, 10], TypeError, "components must be a list of models, got dict",
                 id='a_dict'),
    pytest.param(STARS, [4], TypeError, "components must be a list of models, got _Model", id='one_model'),
    pytest.param([], [], ValueError, "components is empty", id='empty'),
    pytest.param([STARS, DM], 10, TypeError, "n must be a list of particle numbers, one per component, got int",
                 id='n_not_a_list'),
    pytest.param([STARS, DM], [4], ValueError, "n must give one number per component: got 1 for 2 components",
                 id='n_too_short'),
    pytest.param([STARS], [0], ValueError, "n must be at least 1", id='bad_n'),
])
def test_bad_components_or_numbers_are_rejected(components, n, error, match):
    with pytest.raises(error, match=match):
        ic.sample_components(components, n, seed=1)


def test_a_bad_parent_seed_is_rejected():
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*32\)"):
        _components(seed=-1)
