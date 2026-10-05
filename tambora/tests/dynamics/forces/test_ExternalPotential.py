"""
Tests for ``ExternalPotential`` with the galpy backend.
"""

import pickle

import numpy as np
import pytest

galpy = pytest.importorskip("galpy")
import astropy.units as u                                              # noqa: E402
from galpy.potential import (                                          # noqa: E402
    DehnenBarPotential, NFWPotential, PlummerPotential, RotateAndTiltWrapperPotential,
    TriaxialNFWPotential,
)

from tambora.dynamics.forces import ExternalPotential                   # noqa: E402
from tambora.dynamics.forces.external_force import ExternalConservativeForce  # noqa: E402
from tambora.simulation import Sim                                     # noqa: E402


def _plummer(b=0.5):
    return PlummerPotential(amp=1e10, b=b, ro=8., vo=220.)


# --- input validation ---------------------------------------------------------

@pytest.mark.parametrize("bad", [
    pytest.param(object(), id='bare_object'),
    pytest.param('MWPotential2014', id='string'),
    pytest.param(42, id='int'),
    pytest.param(None, id='none'),
])
def test_a_non_potential_is_rejected(bad):
    with pytest.raises(TypeError, match="Can't use a .* as an external potential"):
        ExternalPotential(bad)


def test_a_potential_is_accepted_by_the_galpy_backend():
    # The control: the guard above must not be rejecting everything.
    f = ExternalPotential(_plummer())
    assert f.backend.name == 'galpy'


@pytest.mark.parametrize("container", [list, tuple])
def test_a_list_or_tuple_of_potentials_is_accepted_and_composed(container):
    f = ExternalPotential(container([_plummer(b=0.5), _plummer(b=0.9)]))
    pos = np.array([[8., 0., 0.]])
    expected = ExternalPotential(_plummer(b=0.5)).acc(pos, 0.0) + ExternalPotential(_plummer(b=0.9)).acc(pos, 0.0)
    np.testing.assert_allclose(f.acc(pos, 0.0), expected)


def test_a_list_containing_a_non_potential_names_the_bad_element():
    # Each element is checked before anything is combined. Combining first used
    # to recurse inside galpy's __add__ and surface as a RecursionError.
    with pytest.raises(TypeError, match=r"element 1 of the list.*got str"):
        ExternalPotential([_plummer(), 'not a potential'])


# --- evaluation ---------------------------------------------------------------

_ON_AXIS = np.array([[0., 0., 2.]])
_OFFSET = np.array([0.3, 0.4, 0.])      # kpc


def _nfw():
    return NFWPotential(a=2., normalize=0.35, ro=8., vo=220.)


def _spherical_case():
    # By symmetry, the force on the axis is the in-plane force at the same radius, turned onto z.
    a_r = ExternalPotential(_nfw()).acc(np.array([[2., 0., 0.]]), 0.0)[0, 0]
    return _nfw(), np.array([[0., 0., a_r]])


def _off_centre_case():
    # A halo shifted off the axis has a non-zero in-plane force on the axis, which goes
    # through both the R-force and the phi-torque. galpy evaluates the wrapped potential
    # at pos + offset, so the answer is the unshifted halo there, well away from the axis.
    shifted = RotateAndTiltWrapperPotential(pot=_nfw(), offset=_OFFSET * u.kpc, ro=8., vo=220.)
    return shifted, ExternalPotential(_nfw()).acc(_ON_AXIS + _OFFSET, 0.0)


def _bar_case():
    # The bar's potential goes as R^2 near the axis, so its force on the axis is exactly zero.
    return DehnenBarPotential(ro=8., vo=220.), np.zeros((1, 3))


@pytest.mark.parametrize("case", [_spherical_case, _off_centre_case, _bar_case],
                         ids=['spherical', 'off_centre', 'bar'])
def test_acceleration_on_the_z_axis_matches_the_exact_value(case):
    pot, expected = case()
    f = ExternalPotential(pot)
    on_axis = f.acc(_ON_AXIS, 0.0)
    scale = np.abs(f.acc(np.array([[2., 0., 0.]]), 0.0)).max()
    assert np.all(np.isfinite(on_axis))
    np.testing.assert_allclose(on_axis, expected, rtol=0, atol=1e-12 * scale)


def test_repr_names_the_backend_and_potential():
    assert repr(ExternalPotential(_plummer())) == "ExternalPotential(galpy: PlummerPotential)"


def test_the_sim_summary_labels_the_potential_by_its_components():
    sim = _one_particle_sim(ExternalPotential([_plummer(), _nfw()]))
    assert "ExternalPotential(PlummerPotential+NFWPotential)" in repr(sim)


# --- check_still_valid: called at the start of every run -----------------------

def _one_particle_sim(force):
    sim = Sim()
    sim.add_particles('p', pos=[[8., 0., 0.]], vel=[[0., 200., 0.]], mass=[1.])
    sim.add_external_force(force)
    return sim


class _UniformField(ExternalConservativeForce):
    """A user's own external force, with no check_still_valid method."""
    def acc(self, pos, t):
        return np.tile([0., 0., -1.], (len(pos), 1))

    def potential(self, pos, t):
        return pos[:, 2].copy()


def test_run_works_with_an_external_force_that_has_no_validity_check():
    # run() checks only the forces that have check_still_valid; calling it on every
    # external force would break forces users write themselves.
    sim = _one_particle_sim(_UniformField())
    sim.add_external_pot(_plummer())
    sim.run(t_end=0.01, dt=0.01, dt_out=0.01, method=None, progress=False, monitors=False)
    assert sim.times[-1] == pytest.approx(0.01)


def test_run_calls_check_still_valid_on_each_external_potential(monkeypatch):
    f = ExternalPotential(_plummer())

    def invalid():
        raise RuntimeError("potential invalidated")
    monkeypatch.setattr(f.backend, "check_still_valid", invalid)
    sim = _one_particle_sim(f)
    with pytest.raises(RuntimeError, match="potential invalidated"):
        sim.run(t_end=0.01, dt=0.01, dt_out=0.01, method=None, progress=False, monitors=False)


def test_a_valid_potential_runs():
    sim = _one_particle_sim(ExternalPotential(_plummer()))
    sim.run(t_end=0.01, dt=0.01, dt_out=0.01, method=None, progress=False, monitors=False)
    assert sim.times[-1] == pytest.approx(0.01)


# --- _dedup_key: the pickle path ----------------------------------------------

def test_equal_parameters_dedup_as_the_same_force():
    # The point of pickling rather than using identity: two separately
    # constructed but identical potentials ARE a duplicate registration.
    a, b = ExternalPotential(_plummer()), ExternalPotential(_plummer())
    assert a.backend.obj is not b.backend.obj        # genuinely distinct objects
    assert a._dedup_key() == b._dedup_key()


def test_different_parameters_are_not_duplicates():
    a = ExternalPotential(_plummer(b=0.5))
    b = ExternalPotential(_plummer(b=0.9))
    assert a._dedup_key() != b._dedup_key()


def test_different_potential_types_are_not_duplicates():
    a = ExternalPotential(_plummer())
    b = ExternalPotential(NFWPotential(amp=1e12, a=16., ro=8., vo=220.))
    assert a._dedup_key() != b._dedup_key()


def test_dedup_key_is_hashable_and_names_the_type_and_backend():
    # add_hook/add_external_force compare and store these, so they must be usable
    # as dict/set members, and must not collide across force classes or backends.
    key = ExternalPotential(_plummer())._dedup_key()
    hash(key)
    assert key[:2] == (ExternalPotential, 'galpy')


# --- _dedup_key: unchanged by a run --------------------------------------------

def _triaxial():
    return TriaxialNFWPotential(a=2., b=0.8, c=0.6, ro=8., vo=220.)


def test_the_triaxial_halo_really_caches_when_evaluated():
    # Guard for the test below: if galpy stopped caching, that test would pass
    # without exercising anything.
    p = _triaxial()
    before = pickle.dumps(p)
    ExternalPotential(p).acc(np.array([[8., 0., 1.]]), 0.0)
    assert pickle.dumps(p) != before


def test_a_copy_added_after_a_run_is_still_a_duplicate():
    # A notebook cell that adds a potential, re-run after sim.run(). The first halo
    # has cached results in itself during the run, so it no longer pickles like the
    # fresh copy; the key it took when its force was created still matches.
    sim = _one_particle_sim(ExternalPotential(_triaxial()))
    sim.run(t_end=0.01, dt=0.01, dt_out=0.01, method=None, progress=False, monitors=False)
    with pytest.raises(ValueError, match="equivalently-configured"):
        sim.add_external_pot(_triaxial())


def test_the_same_object_re_added_after_a_run_is_still_a_duplicate():
    # A notebook cell that adds a potential built in an earlier cell, re-run after
    # sim.run(). The halo has cached results in itself during the run; its key must not
    # change, or the field would be counted twice.
    halo = _triaxial()
    sim = _one_particle_sim(ExternalPotential(halo))
    sim.run(t_end=0.01, dt=0.01, dt_out=0.01, method=None, progress=False, monitors=False)
    with pytest.raises(ValueError, match="equivalently-configured"):
        sim.add_external_pot(halo)


# --- _dedup_key: the unpicklable fallback -------------------------------------

def _unpicklable_plummer():
    """A real galpy potential carrying an attribute pickle cannot serialise."""
    p = _plummer()
    p._gotcha = lambda: None            # lambdas are not picklable
    return p


def test_the_unpicklable_potential_is_genuinely_unpicklable():
    # Guard for the tests below: if galpy ever made this picklable, they would
    # silently start exercising the pickle path instead of the fallback.
    with pytest.raises(Exception):
        pickle.dumps(_unpicklable_plummer())


def test_an_unpicklable_potential_falls_back_to_object_identity():
    # Without the fallback, constructing the force would be fine but
    # *registering* it would raise PicklingError from inside add_external_force.
    p = _unpicklable_plummer()
    assert ExternalPotential(p)._dedup_key() == (ExternalPotential, 'galpy', (id(p),))


def test_the_fallback_still_dedups_the_same_object():
    # The fallback must remain useful for the case it can decide: the identical
    # potential object wrapped twice really is a duplicate.
    p = _unpicklable_plummer()
    assert ExternalPotential(p)._dedup_key() == ExternalPotential(p)._dedup_key()


def test_the_fallback_cannot_recognise_equal_but_distinct_potentials():
    # The documented cost of the fallback, pinned so it is a known trade rather
    # than a surprise: two equal-but-distinct unpicklable potentials do NOT
    # compare equal, so a duplicate registration slips through. Deliberate;
    # dedup errs toward accepting, never toward wrongly rejecting a force.
    a = ExternalPotential(_unpicklable_plummer())
    b = ExternalPotential(_unpicklable_plummer())
    assert a._dedup_key() != b._dedup_key()


def test_an_unpicklable_potential_is_still_a_working_force():
    # The fallback is about dedup identity only; the physics must be unaffected.
    f = ExternalPotential(_unpicklable_plummer())
    ref = ExternalPotential(_plummer())
    pos = np.array([[8., 0., 0.], [0., 8., 1.]])
    np.testing.assert_allclose(f.acc(pos, 0.0), ref.acc(pos, 0.0))
    np.testing.assert_allclose(f.potential(pos, 0.0), ref.potential(pos, 0.0))
