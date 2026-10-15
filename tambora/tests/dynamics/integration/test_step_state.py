"""
Tests for ``StepState``, the live, read-only view of the simulation passed
to hooks (``tambora/dynamics/integration/step_state.py``).


Testing Approach
----------------
- build a ``StepState`` in isolation over a hand-made ``StepResult`` and
``StepContext``

- Force objects are *stubs* that return fixed arrays and count calls, 
  so we can assert the per-step cache computes each expensive quantity 
  at most once. 

- Numeric agreement with a real ``Sim`` is left to the higher-level 
  integration tests; here we pin StepState's own logic:
  slicing, component views, caching, and reporting.
"""

import copy

import numpy as np
import pytest

from tambora.dynamics.integration.step_state import StepState, StepContext
from tambora.dynamics.integration.BaseIntegrator import StepResult


# --- fixed system: 5 particles, two components a=[0:3], b=[3:5] ---------------

POS = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [5, 1, 0]])
VEL = np.array([[1., 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 0], [0, 1, 1]])
MASS = np.array([1., 2, 3, 4, 5])
SLICES = {'a': slice(0, 3), 'b': slice(3, 5)}


# --- stubs: return fixed arrays, count calls ----------------------------------

class SpySolver:
    """Stand-in SelfGravityForce: fixed per-particle potential, counts calls."""
    def __init__(self, phi=-1.0):
        self.phi = phi
        self.calls = 0

    def acc_and_potential(self, pos, mass, eps=None):
        self.calls += 1
        return np.zeros_like(pos), np.full(len(mass), self.phi)


class SpyExternal:
    """Stand-in conservative external force: fixed potential, counts calls."""
    def __init__(self, phi=-2.0):
        self.phi = phi
        self.calls = 0

    def potential(self, pos, mass, t):
        self.calls += 1
        return np.full(len(mass), self.phi)


class SpyProgress:
    """Stand-in progress channel: records the fields reported to it."""
    def __init__(self):
        self.reported = {}

    def report(self, **fields):
        self.reported.update(fields)


def make_state(self_pot=None, self_acc=None, sg=None, ext=None, progress=None, t=1.0, step=3):
    """Build a StepState over the fixed system with the given stubs/overrides."""
    res = StepResult(POS.copy(), VEL.copy(), MASS.copy(), t,
                     self_acc=self_acc, self_pot=self_pot,
                     conserv_ext_acc=None, base_ext_acc=None, step=step)
    ctx = StepContext(dict(SLICES), sg, ext, None, progress)
    return StepState(res, ctx)


# --- kinematics & slicing -----------------------------------------------------

def test_full_and_component_kinematics():
    st = make_state()
    np.testing.assert_array_equal(st.pos(), POS)
    np.testing.assert_array_equal(st.a.pos(), POS[0:3])
    np.testing.assert_array_equal(st.a.vel(), VEL[0:3])
    np.testing.assert_array_equal(st.a.mass, MASS[0:3])
    np.testing.assert_array_equal(st.a.x(), POS[0:3, 0])
    np.testing.assert_array_equal(st.b.z(), POS[3:5, 2])
    np.testing.assert_allclose(st.a.r(), np.linalg.norm(POS[0:3], axis=-1))


# The six axis accessors are near-identical one-liners (self.pos()[..., 0] and
# friends), so the realistic bug is a wrong constant from copy-paste. 

@pytest.mark.parametrize("name, arr, col", [
    ('x', POS, 0), ('y', POS, 1), ('z', POS, 2),
    ('vx', VEL, 0), ('vy', VEL, 1), ('vz', VEL, 2),
])
def test_axis_accessors_select_the_right_column(name, arr, col):
    st = make_state()
    np.testing.assert_array_equal(getattr(st, name)(), arr[:, col])          # whole system
    np.testing.assert_array_equal(getattr(st.a, name)(), arr[0:3, col])      # component a
    np.testing.assert_array_equal(getattr(st.b, name)(), arr[3:5, col])      # component b


def test_pos_and_vel_are_component_local():
    st = make_state()
    np.testing.assert_array_equal(st.vel(), VEL)
    np.testing.assert_array_equal(st.b.pos(), POS[3:5])
    np.testing.assert_array_equal(st.b.vel(), VEL[3:5])


def test_r_is_the_norm_of_pos_at_every_scope():
    st = make_state()
    np.testing.assert_allclose(st.r(), np.linalg.norm(POS, axis=-1))
    np.testing.assert_allclose(st.b.r(), np.linalg.norm(POS[3:5], axis=-1))
    # r is a magnitude, so it must not simply track one axis.
    assert not np.allclose(st.r(), st.x())


def test_accessors_are_live_views_of_the_step_result():
    # pos()/vel() slice the StepResult rather than copying, so a hook that stashes
    # the array (instead of copying out of it) sees later steps' data. Pinning the
    # view semantics the hook docs warn about.
    st = make_state()
    seen = st.a.pos()
    st._result.pos[0, 0] = 99.0
    assert seen[0, 0] == 99.0
    assert st.a.x()[0] == 99.0


def test_KE_is_component_local():
    st = make_state()
    np.testing.assert_allclose(st.a.KE(), 0.5 * MASS[0:3] * np.sum(VEL[0:3] ** 2, axis=-1))
    np.testing.assert_allclose(st.KE(), 0.5 * MASS * np.sum(VEL ** 2, axis=-1))


def test_step_and_time_metadata():
    st = make_state(t=1.5, step=7)
    assert st.t == 1.5
    assert st.step == 7


def test_mass_is_a_property():
    # mass is time-independent, so it mirrors Sim/Component.mass as a @property.
    st = make_state()
    assert isinstance(st.mass, np.ndarray)
    with pytest.raises(TypeError):
        st.mass()                              # calling the array raises


# --- component views & attribute access ---------------------------------------

def test_attribute_access_matches_component_method():
    st = make_state()
    np.testing.assert_array_equal(st.a.pos(), st.component('a').pos())


def test_unknown_attribute_raises_attribute_error():
    st = make_state()
    with pytest.raises(AttributeError):
        st.nonexistent


def test_unknown_component_raises_key_error():
    st = make_state()
    with pytest.raises(KeyError):
        st.component('nonexistent')


# --- external_acc: sum of the two external force channels ---------------------
#
# Four cases (each channel present or not). The both-None case is the one worth
# pinning: it returns a scalar 0.0, not a zero array.

def test_external_acc_sums_both_channels():
    st = make_state()
    st._result.conserv_ext_acc = np.full((5, 3), 1.0)
    st._result.base_ext_acc = np.full((5, 3), 2.0)
    np.testing.assert_array_equal(st.external_acc(), np.full((5, 3), 3.0))


@pytest.mark.parametrize("channel", ['conserv_ext_acc', 'base_ext_acc'])
def test_external_acc_works_with_only_one_channel_set(channel):
    st = make_state()
    setattr(st._result, channel, np.full((5, 3), 7.0))
    np.testing.assert_array_equal(st.external_acc(), np.full((5, 3), 7.0))


def test_external_acc_is_component_local():
    st = make_state()
    st._result.conserv_ext_acc = np.arange(15.).reshape(5, 3)
    st._result.base_ext_acc = np.zeros((5, 3))
    np.testing.assert_array_equal(st.a.external_acc(), np.arange(15.).reshape(5, 3)[0:3])
    np.testing.assert_array_equal(st.b.external_acc(), np.arange(15.).reshape(5, 3)[3:5])


def test_external_acc_is_a_zero_vector_when_no_external_force_is_present():
    # The return type must not depend on the sim's configuration: a caller doing
    # external_acc()[..., 0] must get zeros, not a TypeError, when there is no
    # external force.
    st = make_state()
    out = st.external_acc()
    np.testing.assert_array_equal(out, np.zeros((5, 3)))
    assert out.shape == (5, 3)
    np.testing.assert_array_equal(out[..., 0], np.zeros(5))     # indexable


def test_external_acc_zero_vector_is_component_local():
    st = make_state()
    assert st.a.external_acc().shape == (3, 3)
    assert st.b.external_acc().shape == (2, 3)


def test_external_acc_shape_does_not_depend_on_which_channels_are_set():
    shapes = []
    for c, b in ((None, None), (np.ones((5, 3)), None),
                 (None, np.ones((5, 3))), (np.ones((5, 3)), np.ones((5, 3)))):
        st = make_state()
        st._result.conserv_ext_acc, st._result.base_ext_acc = c, b
        shapes.append(st.a.external_acc().shape)
    assert shapes == [(3, 3)] * 4


def test_external_acc_does_not_alias_or_mutate_the_step_result():
    # The `out = 0.0` seed matters: starting from `out = c[sl]` and using += would
    # write straight through the view into the StepResult, corrupting the
    # integrator's own array.
    st = make_state()
    C = np.full((5, 3), 1.0)
    B = np.full((5, 3), 2.0)
    st._result.conserv_ext_acc, st._result.base_ext_acc = C, B
    out = st.external_acc()
    assert not np.shares_memory(out, C)
    np.testing.assert_array_equal(C, 1.0)        # untouched
    np.testing.assert_array_equal(B, 2.0)


# --- __getattr__ guard --------------------------------------------------------

def test_private_attribute_lookup_does_not_recurse():
    # __getattr__ reads self._ctx, so it MUST refuse underscore names first: on an
    # object whose _ctx is not set (during __init__, or after __new__ from
    # copy/pickle), self._ctx would re-enter __getattr__ forever. This is a
    # RecursionError without the `name.startswith("_")` guard, not a nicety.
    bare = StepState.__new__(StepState)          # nothing initialised
    with pytest.raises(AttributeError):
        bare._ctx


def test_dunder_probes_raise_attribute_error_rather_than_recursing():
    # copy/pickle/IPython probe for *optional* dunders on arbitrary objects. Each
    # must get a clean AttributeError so the protocol falls back to its default.
    # Only names object() does not already define reach __getattr__ at all.
    # __getstate__/__reduce_ex__ are real methods on object since 3.11.
    bare = StepState.__new__(StepState)
    for probe in ('__deepcopy__', '__copy__', '_repr_html_'):
        assert not hasattr(object(), probe)      # else this proves nothing
        with pytest.raises(AttributeError):
            getattr(bare, probe)


def test_uninitialised_state_can_be_deepcopied():
    # The practical consequence of the guard.
    copy.deepcopy(StepState.__new__(StepState))


def test_unknown_public_attribute_names_the_object_and_the_attribute():
    st = make_state()
    with pytest.raises(AttributeError, match="no attribute or component named 'sat'"):
        st.sat


# --- self_potential -----------------------------------------------------------

def test_self_potential_fast_path_reads_precomputed_without_solver_call():
    sp = -np.arange(1., 6.)
    spy = SpySolver()
    st = make_state(self_pot=sp, sg=spy)
    # All-components + run solver reads the step's precomputed potential.
    np.testing.assert_allclose(st.self_potential(include_all_components=True), MASS * sp)
    np.testing.assert_allclose(st.a.self_potential(include_all_components=True), (MASS * sp)[0:3])
    assert spy.calls == 0                      # fast path never invokes the solver


def test_self_potential_custom_solver_computed_once_and_shared():
    run, alt = SpySolver(phi=-1.0), SpySolver(phi=-9.0)
    st = make_state(self_pot=-np.arange(1., 6.), sg=run)
    np.testing.assert_allclose(st.self_potential(include_all_components=True, solver=alt), MASS * -9.0)
    # A component view reuses the cached whole-system result (same solver id).
    np.testing.assert_allclose(st.a.self_potential(include_all_components=True, solver=alt), (MASS * -9.0)[0:3])
    assert alt.calls == 1                      # computed once, cached
    assert run.calls == 0                      # run solver untouched


def test_self_potential_isolated_uses_only_component_particles():
    from tambora.dynamics.forces.self_gravity import DirectSummationGravity
    solver = DirectSummationGravity(eps=0.1)
    _, full = DirectSummationGravity(eps=0.1).acc_and_potential(POS, MASS)
    st = make_state(self_pot=full, sg=solver)

    iso = st.a.self_potential(include_all_components=False)
    _, phi_a = DirectSummationGravity(eps=0.1).acc_and_potential(POS[0:3], MASS[0:3])
    np.testing.assert_allclose(iso, MASS[0:3] * phi_a)
    # Isolated (a's own gravity) differs from a's slice of the whole-system field.
    assert not np.allclose(iso, st.a.self_potential(include_all_components=True))


def test_self_potential_isolated_cached_per_component():
    spy = SpySolver()
    st = make_state(self_pot=-np.arange(1., 6.), sg=spy)
    st.a.self_potential(include_all_components=False)
    st.a.self_potential(include_all_components=False)
    assert spy.calls == 1                      # cached for component 'a'
    st.b.self_potential(include_all_components=False)
    assert spy.calls == 2                      # distinct slice -> recompute


def test_self_potential_family_requires_include_all_components():
    # No default: on a component view "self" is ambiguous, so the caller must
    # choose explicitly. Omitting it is a TypeError, not a silent wrong answer.
    st = make_state(self_pot=-np.arange(1., 6.), self_acc=np.zeros((5, 3)))
    for call in (lambda: st.self_potential(),
                 lambda: st.self_gravity(),
                 lambda: st.a.PE(),
                 lambda: st.a.energy()):
        with pytest.raises(TypeError):
            call()


# --- self_gravity (acceleration) ----------------------------------------------

def test_self_gravity_fast_path_reads_precomputed_without_solver_call():
    acc = np.arange(15.).reshape(5, 3)
    spy = SpySolver()
    st = make_state(self_acc=acc, sg=spy)
    np.testing.assert_array_equal(st.self_gravity(include_all_components=True), acc)
    np.testing.assert_array_equal(st.a.self_gravity(include_all_components=True), acc[0:3])
    assert spy.calls == 0                      # fast path never invokes the solver


def test_self_gravity_custom_solver_computed_once_and_shared():
    run, alt = SpySolver(), SpySolver()        # SpySolver returns a zero acceleration
    st = make_state(self_acc=np.ones((5, 3)), sg=run)
    np.testing.assert_allclose(st.self_gravity(include_all_components=True, solver=alt), np.zeros((5, 3)))
    np.testing.assert_allclose(st.a.self_gravity(include_all_components=True, solver=alt), np.zeros((3, 3)))
    assert alt.calls == 1                      # computed once, cached across views
    assert run.calls == 0                      # run solver untouched


def test_self_gravity_isolated_uses_only_component_particles():
    from tambora.dynamics.forces.self_gravity import DirectSummationGravity
    solver = DirectSummationGravity(eps=0.1)
    full_acc, full_pot = DirectSummationGravity(eps=0.1).acc_and_potential(POS, MASS)
    st = make_state(self_acc=full_acc, self_pot=full_pot, sg=solver)

    iso = st.a.self_gravity(include_all_components=False)
    acc_a, _ = DirectSummationGravity(eps=0.1).acc_and_potential(POS[0:3], MASS[0:3])
    np.testing.assert_allclose(iso, acc_a)
    # Isolated (a's own gravity) differs from a's slice of the whole-system field.
    assert not np.allclose(iso, st.a.self_gravity(include_all_components=True))


def test_self_gravity_and_potential_share_one_computation():
    # The whole point of the unified helper: acc + pot come from one call.
    alt = SpySolver()
    st = make_state(sg=SpySolver())
    st.self_gravity(include_all_components=True, solver=alt)
    st.self_potential(include_all_components=True, solver=alt)
    assert alt.calls == 1                      # one acc_and_potential serves both


# --- external_pot & system_energy ---------------------------------------------

def test_external_pot_computed_once_and_shared():
    ext = SpyExternal(phi=-2.0)
    st = make_state(ext=ext)
    np.testing.assert_allclose(st.external_pot(), MASS * -2.0)
    np.testing.assert_allclose(st.a.external_pot(), (MASS * -2.0)[0:3])
    assert ext.calls == 1                      # lazy compute-once, shared across views


def test_PE_and_energy_forward_self_potential_options():
    from tambora.dynamics.forces.self_gravity import DirectSummationGravity
    solver = DirectSummationGravity(eps=0.1)
    _, full = DirectSummationGravity(eps=0.1).acc_and_potential(POS, MASS)
    st = make_state(self_pot=full, sg=solver, ext=SpyExternal(phi=-2.0))
    a = st.a

    ext = MASS[0:3] * -2.0
    # PE forwards include_all_components -> isolated self-potential + external.
    # (Discriminating: if PE ignored the flag it would use self_potential(True),
    # which differs from self_potential(False).)
    np.testing.assert_allclose(
        a.PE(include_all_components=False),
        a.self_potential(include_all_components=False) + ext)
    # energy forwards too: KE + PE(isolated).
    np.testing.assert_allclose(
        a.energy(include_all_components=False),
        a.KE() + a.PE(include_all_components=False))


def test_system_energy_combines_ke_self_and_external():
    sp = -np.arange(1., 6.)
    st = make_state(self_pot=sp, sg=SpySolver(), ext=SpyExternal(phi=-2.0))
    expected = (np.sum(0.5 * MASS * np.sum(VEL ** 2, axis=-1))
                + 0.5 * np.sum(MASS * sp)
                + np.sum(MASS * -2.0))
    np.testing.assert_allclose(st.system_energy(), expected)


# --- bound_mask caching (spy on the diagnostics function) ---------------------

def _patch_bound_mask(monkeypatch):
    """Replace diagnostics.bound_mask with a call-counting stub; return counter."""
    calls = {'n': 0}
    def spy(pos, vel, mass, **kw):
        calls['n'] += 1
        return np.ones(len(mass), dtype=bool)
    monkeypatch.setattr('tambora.dynamics.diagnostics.bound_mask', spy)
    return calls


def test_bound_mask_computed_once_for_repeated_calls(monkeypatch):
    calls = _patch_bound_mask(monkeypatch)
    st = make_state()
    st.bound_mask('a', eps=0.1)
    st.bound_mask('a', eps=0.1)
    assert calls['n'] == 1                     # second call served from the step cache


def test_bound_mask_cache_keys_are_distinct(monkeypatch):
    calls = _patch_bound_mask(monkeypatch)
    st = make_state()
    st.bound_mask('a', eps=0.1)                     # 1
    st.bound_mask('a', eps=0.1)                     # cached
    st.bound_mask('b', eps=0.1)                     # 2: other component
    st.bound_mask('a', eps=0.2)                     # 3: other eps
    st.bound_mask('a', eps=0.1, method='direct')    # 4: other solver
    st.bound_mask('a', eps=0.1, max_iter=10)        # 5: other max_iter
    st.bound_mask(eps=0.1)                          # 6: whole system
    assert calls['n'] == 6


# --- reporting ----------------------------------------------------------------

def test_report_is_noop_without_progress():
    st = make_state(progress=None)
    st.report(anything=1)                      # must not raise


def test_report_forwards_fields_to_progress():
    prog = SpyProgress()
    st = make_state(progress=prog)
    st.report(nbound=5, frac=0.5)
    assert prog.reported == {'nbound': 5, 'frac': 0.5}


# --- _update: refresh data, drop cached results -------------------------------

def test_update_refreshes_data_and_clears_cache(monkeypatch):
    calls = _patch_bound_mask(monkeypatch)
    st = make_state(t=1.0, step=3)
    st.bound_mask('a', eps=0.1)
    st.bound_mask('a', eps=0.1)
    assert calls['n'] == 1                     # cached within the step

    new = StepResult(POS.copy(), VEL.copy(), MASS.copy(), 2.0,
                     self_acc=None, self_pot=None,
                     conserv_ext_acc=None, base_ext_acc=None, step=7)
    st._update(new)
    assert st.t == 2.0 and st.step == 7        # data refreshed
    st.bound_mask('a', eps=0.1)
    assert calls['n'] == 2                     # cache cleared -> recompute
