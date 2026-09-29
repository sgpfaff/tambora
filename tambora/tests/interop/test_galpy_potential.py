"""
Tests for the galpy potential backend on its own, without ExternalPotential.
"""

import pickle

import numpy as np
import pytest

gp = pytest.importorskip("galpy.potential")
from galpy.orbit import Orbit                                    # noqa: E402

from tambora.interop._galpy.potential import GalpyPotential      # noqa: E402
from tambora.tools.util import _galpy_bridge                     # noqa: E402

POS = np.array([[8., 0., 0.5], [3., -2., 1.], [0.5, 4., -2.]])  # kpc


def _plummer(b=0.5):
    return gp.PlummerPotential(b=b, ro=8., vo=220.)


def _nfw():
    return gp.NFWPotential(a=2., normalize=0.35, ro=8., vo=220.)


def _bar():
    return gp.DehnenBarPotential(ro=8., vo=220.)                  # rotates, so depends on t


def _triaxial():
    return gp.TriaxialNFWPotential(a=2., b=0.8, c=0.6, ro=8., vo=220.)


# --- values -----------------------------------------------------------------------

def test_acc_and_potential_are_the_bridge_values_at_the_given_time():
    pot, t = _bar(), 0.3                                         # Gyr
    b = GalpyPotential(pot)
    acc, phi = b.acc(POS, t), b.potential(POS, t)
    assert acc.shape == (3, 3) and phi.shape == (3,)
    np.testing.assert_array_equal(acc, _galpy_bridge._galpy_pot_to_acc_fn(pot)(POS, t))
    np.testing.assert_array_equal(phi, _galpy_bridge._galpy_pot_to_pot_fn(pot)(POS, t))


def test_the_time_is_passed_through():
    # Guard for the test above: the bar must actually differ between the two times.
    b = GalpyPotential(_bar())
    assert not np.allclose(b.acc(POS, 0.3), b.acc(POS, 0.0))


@pytest.mark.parametrize("container", [list, tuple])
def test_a_list_or_tuple_is_the_sum_of_its_potentials(container):
    b = GalpyPotential(container([_plummer(), _nfw()]))
    expected = GalpyPotential(_plummer()).acc(POS, 0.) + GalpyPotential(_nfw()).acc(POS, 0.)
    np.testing.assert_allclose(b.acc(POS, 0.), expected, rtol=1e-14)


def test_a_nested_list_is_flattened():
    p, q = _plummer(), _nfw()
    np.testing.assert_array_equal(GalpyPotential([[p], (q,)]).acc(POS, 0.),
                                  GalpyPotential([p, q]).acc(POS, 0.))


# --- what it accepts, and the errors for what it doesn't ---------------------------

@pytest.mark.parametrize("obj, accepted", [
    pytest.param(_plummer(), True, id='potential'),
    pytest.param(_plummer() + _nfw(), True, id='composite'),
    pytest.param([_plummer(), 'x'], True, id='list_with_a_potential'),
    pytest.param([['x', _plummer()]], True, id='nested_list_with_a_potential'),
    pytest.param(['x', 3], False, id='list_without_one'),
    pytest.param('MWPotential2014', False, id='string'),
])
def test_accepts_claims_anything_containing_a_galpy_potential(obj, accepted):
    assert GalpyPotential.accepts(obj) is accepted


def test_a_bad_element_is_named():
    with pytest.raises(TypeError, match=r"element 1 of the list\), got str"):
        GalpyPotential([_plummer(), 'x'])


def test_a_bad_element_in_a_nested_list_is_named_in_the_flattened_list():
    with pytest.raises(TypeError, match=r"element 1 of the flattened list\), got int"):
        GalpyPotential([[_plummer()], 3])


def test_describe_names_each_component():
    assert GalpyPotential([_plummer(), _nfw()]).describe() == 'PlummerPotential+NFWPotential'


# --- dedup keys --------------------------------------------------------------------

def test_the_triaxial_halo_changes_its_pickle_when_evaluated():
    # Guard for the tests below: without this caching they would pass trivially.
    p = _triaxial()
    before = pickle.dumps(p)
    GalpyPotential(p).acc(POS, 0.)
    assert pickle.dumps(p) != before


def test_the_same_object_keeps_its_key_after_being_evaluated():
    p = _triaxial()
    key = GalpyPotential(p).dedup_key()
    GalpyPotential(p).acc(POS, 0.)
    assert GalpyPotential(p).dedup_key() == key


def test_a_fresh_copy_matches_one_that_has_been_evaluated():
    used = _triaxial()
    GalpyPotential(used).dedup_key()
    GalpyPotential(used).acc(POS, 0.)
    assert GalpyPotential(_triaxial()).dedup_key() == GalpyPotential(used).dedup_key()


def test_a_list_rewrapped_after_evaluation_keeps_its_key():
    # Each wrap builds a new CompositePotential, so the key is kept per component.
    parts = [_triaxial(), _plummer()]
    key = GalpyPotential(parts).dedup_key()
    GalpyPotential(parts).acc(POS, 0.)
    assert GalpyPotential(list(parts)).dedup_key() == key
    assert GalpyPotential(tuple(parts)).dedup_key() == key


def test_different_parameters_give_different_keys():
    assert GalpyPotential(_plummer(b=0.5)).dedup_key() != GalpyPotential(_plummer(b=0.9)).dedup_key()


def test_an_unpicklable_potential_is_keyed_by_identity():
    p = _plummer()
    p._gotcha = lambda: None                                     # lambdas can't be pickled
    assert GalpyPotential(p).dedup_key() == (id(p),)


class _UnhashablePlummer(gp.PlummerPotential):
    """Like a user's own subclass that defines __eq__, which makes it unhashable."""
    __hash__ = None


def test_a_potential_that_cannot_be_remembered_still_gets_a_key():
    # Its key can't be stored by identity, so it is recomputed each time. Two identical
    # fresh ones still match, and wrapping one doesn't fail.
    a, b = _UnhashablePlummer(b=0.5, ro=8., vo=220.), _UnhashablePlummer(b=0.5, ro=8., vo=220.)
    assert GalpyPotential(a).dedup_key() == GalpyPotential(b).dedup_key()


# --- a satellite ---------------------------------------------------------------------

def test_a_moving_object_goes_through_the_backend():
    o = Orbit([1., 0.1, 1.1, 0.1, 0., 0.3], ro=8., vo=220.)
    o.integrate(np.linspace(-1., 1., 101), gp.MWPotential2014)
    sat = gp.MovingObjectPotential(o, pot=gp.PlummerPotential(amp=0.1, b=0.1), ro=8., vo=220.)
    b = GalpyPotential(sat)
    np.testing.assert_array_equal(b.acc(POS, 0.), _galpy_bridge._galpy_pot_to_acc_fn(sat)(POS, 0.))
    assert b.describe() == 'MovingObjectPotential'
