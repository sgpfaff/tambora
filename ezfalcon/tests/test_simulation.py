'''
Test the Sim class and its methods.
'''

import pytest 
from ezfalcon.simulation import Sim, Component
import numpy as np
from ezfalcon.util import G_INTERNAL
from ezfalcon.util.units import KMS_TO_KPCGYR
from ezfalcon.dynamics import DirectSummationGravity
import astropy.units as u


# --- Setup ----------------------------------------------------------------------------------- #

np.random.seed(42)

COMP1_NPTS = 50
COMP1_POS = np.random.rand(COMP1_NPTS, 3)
COMP1_VEL = np.random.rand(COMP1_NPTS, 3)
COMP1_MASS = np.random.rand(COMP1_NPTS)

COMP2_NPTS = 30
COMP2_POS = np.random.rand(COMP2_NPTS, 3)
COMP2_VEL = np.random.rand(COMP2_NPTS, 3)
COMP2_MASS = np.random.rand(COMP2_NPTS)


singlecomp = Sim()
singlecomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)
multicomp = Sim()
multicomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)
multicomp.add_particles('comp2',
                    pos=COMP2_POS, 
                    vel=COMP2_VEL, 
                    mass=COMP2_MASS)

# --- .add_particles() ------------------------------------------------------------------------ #

def test_add_particles_stores_correct_shape():
    sim = Sim()
    sim.add_particles('test', 
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    assert sim._init_pos.shape == (COMP1_NPTS, 3)
    assert sim._init_vel.shape == (COMP1_NPTS, 3)
    assert sim._mass.shape == (COMP1_NPTS,)

def test_add_particles_multiple_components():
    '''
    Test that adding multiple components stores the
    correct shapes.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    sim.add_particles('comp2',
                      pos=COMP2_POS, 
                      vel=COMP2_VEL, 
                      mass=COMP2_MASS)
    assert sim._init_pos.shape == (COMP1_NPTS + COMP2_NPTS, 3)
    assert sim._init_vel.shape == (COMP1_NPTS + COMP2_NPTS, 3)
    assert sim._mass.shape == (COMP1_NPTS + COMP2_NPTS,)

def test_add_component_with_same_name():
    '''
    Test that adding a component with the same 
    name raises an error.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    with pytest.raises(ValueError, match="Component 'comp1' already exists."):
        sim.add_particles('comp1',
                        pos=COMP2_POS, 
                        vel=COMP2_VEL, 
                        mass=COMP2_MASS)
    
def test_add_component_with_non_string_name():
    '''
    Test that adding a component with a non-string
    name raises an error.
    '''
    sim = Sim()
    with pytest.raises(TypeError, match="name must be a string."):
        sim.add_particles(123,
                        pos=COMP1_POS, 
                        vel=COMP1_VEL, 
                        mass=COMP1_MASS)

def test_add_component_after_run():
    '''
    Test that adding a component after run() raises an error.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    sim.run(t_end=1., dt=0.5, dt_out=0.5, method='direct', eps=0.0)
    with pytest.raises(RuntimeError, match="Cannot add components after run()"):
        sim.add_particles('comp2',
                        pos=COMP2_POS, 
                        vel=COMP2_VEL, 
                        mass=COMP2_MASS)

def test_add_particles_invalid_pos_shapes():
    '''
    Test that adding particles with invalid 
    position shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="pos must be shape \(N, 3\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=np.random.rand(COMP1_NPTS, 2),
                          vel=COMP1_VEL,
                          mass=COMP1_MASS)
    with pytest.raises(ValueError, match="pos must be shape \(N, 3\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=np.random.rand(COMP1_NPTS, 4),
                          vel=COMP1_VEL,
                          mass=COMP1_MASS)

def test_add_particles_invalid_vel_shapes():
    '''
    Test that adding particles with invalid
    velocity shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="vel must be shape \(N, 3\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=np.random.rand(COMP1_NPTS, 2),
                          mass=COMP1_MASS)
    with pytest.raises(ValueError, match="vel must be shape \(N, 3\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=np.random.rand(COMP1_NPTS, 4),
                          mass=COMP1_MASS)

def test_add_particles_invalid_mass_shapes():
    '''
    Test that adding particles with invalid
    mass shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="mass must be shape \(N,\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP1_VEL,
                          mass=np.random.rand(COMP1_NPTS, 2))
    with pytest.raises(ValueError, match="mass must be shape \(N,\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP1_VEL,
                          mass=np.random.rand(COMP1_NPTS, 4))

def test_add_particles_different_number_of_particles():
    '''
    Test that adding particles with different number
    of particles in pos, vel, and mass raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="pos, vel, mass must have same number of particles, received 50, 30, 50."):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP2_VEL,
                          mass=COMP1_MASS)

# --- Multi-component slicing ------------------------------------------------------------------------ #
#
# The slicing itself is tested in test_component.py,
# but here we test that the slices don't overlap and
# that the correct errors are raised when accessing non-existent components.


def test_component_slices_are_contiguous():
    '''
    Test that the component slices are contiguous and
    non-overlapping.
    '''
    comp1_slice = multicomp._slices['comp1']
    comp2_slice = multicomp._slices['comp2']
    assert comp1_slice.stop == comp2_slice.start
    assert comp1_slice == slice(0, COMP1_NPTS)
    assert comp2_slice == slice(COMP1_NPTS, COMP1_NPTS + COMP2_NPTS)

def test_non_existent_component_access():
    '''
    Test that accessing a non-existent component raises an error.
    '''
    with pytest.raises(AttributeError, match="\'Sim\' has no attribute or component named 'comp3'"):
        _ = multicomp.comp3


# --- accessors pre-run ------------------------------------------------------------------------ #

def test_accessors_are_correct_initially():
    '''
    Test that the pos, vel, mass, x, y, z, vx, vy, vz
    match the input values.
    '''
    np.testing.assert_array_equal(singlecomp.pos(0), COMP1_POS)
    np.testing.assert_allclose(singlecomp.vel(0), COMP1_VEL, rtol=1e-15)
    np.testing.assert_array_equal(singlecomp.mass, COMP1_MASS)
    np.testing.assert_array_equal(singlecomp.x(0), COMP1_POS[:, 0])
    np.testing.assert_array_equal(singlecomp.y(0), COMP1_POS[:, 1])
    np.testing.assert_array_equal(singlecomp.z(0), COMP1_POS[:, 2])
    np.testing.assert_allclose(singlecomp.vx(0), COMP1_VEL[:, 0], rtol=1e-15)
    np.testing.assert_allclose(singlecomp.vy(0), COMP1_VEL[:, 1], rtol=1e-15)
    np.testing.assert_allclose(singlecomp.vz(0), COMP1_VEL[:, 2], rtol=1e-15)

def test_accessors_match_init_values():
    '''
    Test that the pos, vel, mass, x, y, z, vx, vy, vz
    match the ._init_ values.
    '''
    np.testing.assert_array_equal(multicomp.pos(0), multicomp._init_pos)
    np.testing.assert_array_equal(multicomp.vel(0, return_internal=True), multicomp._init_vel)
    np.testing.assert_array_equal(multicomp.mass, multicomp._mass)
    np.testing.assert_array_equal(multicomp.x(0), multicomp._init_pos[:, 0])
    np.testing.assert_array_equal(multicomp.y(0), multicomp._init_pos[:, 1])
    np.testing.assert_array_equal(multicomp.z(0), multicomp._init_pos[:, 2])
    np.testing.assert_array_equal(multicomp.vx(0, return_internal=True), multicomp._init_vel[:, 0])
    np.testing.assert_array_equal(multicomp.vy(0, return_internal=True), multicomp._init_vel[:, 1])
    np.testing.assert_array_equal(multicomp.vz(0, return_internal=True), multicomp._init_vel[:, 2])


# --- POST-RUN, NO EXTERNAL POTENTIAL TESTS ------------------------------------------------------------------- #

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)
singlecomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)

# --- ._ti time indexing ------------------------------------------------------------------------ #

def test_ti_int_passthrough():
    '''
    Test that passing an integer to _ti returns the same integer.
    '''
    assert multicomp._ti(5) == 5

def test_ti_float_to_int():
    '''
    Test that passing a float returns that index of the nearest
    snapshot.
    '''
    assert multicomp._ti(0.2) == 2
    assert multicomp._ti(0.25) == 2
    assert multicomp._ti(0.26) == 3

def test_ti_positive_int_out_of_bounds():
    '''
    Test that passing a positive integer out of bounds raises an error.
    '''
    with pytest.raises(IndexError, match="Time index 100 is out of bounds for simulation with 11 snapshots. Please provide an index within \[-11, 10\]."):
        multicomp._ti(100)

def test_ti_negative_int_out_of_bounds():
    '''
    Test that passing a negative integer out of bounds raises an error.
    '''
    with pytest.raises(IndexError, match="Time index -100 is out of bounds for simulation with 11 snapshots. Please provide an index within \[-11, 10\]."):
        multicomp._ti(-100)

def test_ti_float_out_of_bounds():
    '''
    Test that passing a float out of bounds raises an error.
    '''
    with pytest.raises(ValueError, match="t=100.0 Gyr is out of bounds for simulation time range \[0.0, 1.0\] Gyr."):
        multicomp._ti(100.0)

def test_ti_fails_with_list():
    '''
    Test that passing a list raises an error.
    '''
    with pytest.raises(TypeError, match="t must be an int index, a float time, or ellipsis."):
        multicomp._ti([0, 5], vectorized=False)

def test_ti_vectorized_false_fails_with_ellipsis():
    '''
    Test that passing an ellipsis when vectorized=False raises an error.
    '''
    with pytest.raises(TypeError, match="This method is not vectorized, so t cannot be a list or ellipse. Please provide an integer index or a float time."):
        multicomp._ti(..., vectorized=False)

# --- output shapes ------------------------------------------------------------------------ #

def test_single_component_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for single component sims.
    '''
    assert singlecomp._positions.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._velocities.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._times.shape == (11,)

def test_multicomponent_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for multi-component sims.
    '''
    assert multicomp._positions.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._velocities.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._times.shape == (11,)


# --- self-gravity toggle ------------------------------------------------------------------------ #

def test_self_gravity_off_gives_zero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_off()
    sim.run(t_end=2, dt=1, dt_out=2)
    acc = sim.self_gravity()
    np.testing.assert_array_equal(acc, 0)

def test_self_gravity_on_gives_nonzero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_on()
    sim.run(t_end=2, dt=1, dt_out=2, method='direct', eps=0.0)
    acc = sim.self_gravity()
    assert not np.all(np.isclose(acc, 0, atol=1e-10))

# --- eps dict resolution -------------------------------------------------------------------- #

def test_resolve_eps_dict():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(10, 3)), np.zeros((10, 3)), np.ones(10) * 1e4)
    sim.add_particles('b', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5) * 1e4)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps={'a': 0.1, 'b': 0.05})
    assert sim._has_run

def test_resolve_eps_dict_missing_component():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5))
    sim.add_particles('b', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5))
    with pytest.raises(ValueError, match="eps dict is missing components"):
        sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps={'a': 0.1})

def test_resolve_eps_dict_extra_component():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5))
    with pytest.raises(ValueError, match="eps dict has unknown components"):
        sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps={'a': 0.1, 'z': 0.2})

def test_resolve_eps_dict_array_per_component():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5))
    sim.add_particles('b', np.random.normal(size=(3, 3)), np.zeros((3, 3)), np.ones(3))
    eps = {'a': np.full(5, 0.1), 'b': np.full(3, 0.05)}
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=eps)
    assert sim._has_run

def test_resolve_eps_invalid_type():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(5, 3)), np.zeros((5, 3)), np.ones(5))
    with pytest.raises(TypeError, match="eps must be a scalar or dict"):
        sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=[0.1, 0.2])

# --- .add_external_force() ---------------------------------------------------------------------- #
def test_add_external_force_rejects_non_force():
    class Test:
        ...
    testInstance = Test()
    sim = Sim()
    with pytest.raises(TypeError, match="Expected a ConservativeForce or BaseForce subclass"):
        sim.add_external_force(testInstance)

def test_add_external_force_rejects_self_gravity():
    sim = Sim()
    force = DirectSummationGravity(eps=0.1)
    with pytest.raises(TypeError, match="The provided force is a self-gravity force, not an external force."):
        sim.add_external_force(force)

# --- .add_external_pot() ------------------------------------------------------------------------ #

def test_add_external_pot_rejection():
    sim = Sim()
    with pytest.raises(TypeError, match="External potential must be a galpy Potential object."):
        sim.add_external_pot(lambda pos, t: pos)

# --- self-gravity acceleration accessors ------------------------------------------------------------------------ #

def test_acc_matches_direct():
    '''
    Test that the acceleration accessors match the directly
    obtained values.
    '''
    pos = np.array(np.random.normal(size=(10, 3)))
    vel = np.array(np.random.normal(size=(10, 3)))
    mass = np.array(np.random.normal(loc=1e9, scale=1e8, size=(10,)))
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    acc = sim.self_gravity(t=0, method='direct', eps=0.0, return_internal=True)
    acc_direct= DirectSummationGravity(eps=0.0).acc(pos, mass)
    np.testing.assert_allclose(acc, acc_direct, rtol=1e-10)

# --- external acceleration accessors -------------------------------------------------------- #

def test_external_acc_zero_without_pot():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.run(t_end=2, dt=1, dt_out=2, method='direct', eps=0.0)
    np.testing.assert_array_equal(sim.external_acc(-1), 0)

KEPLER_SIM = Sim()
from galpy.potential import KeplerPotential
kepler_pot = KeplerPotential(amp=1e9*u.Msun)
KEPLER_SIM.add_external_pot(kepler_pot)
KEPLER_SIM.add_particles('a', pos=np.array([[0.1, 0.2, 0.3]]), vel=np.array([[0, 0.0, 0]]), mass=np.array([1e8]))
KEPLER_ACC = -G_INTERNAL * 1e9 * np.array([[0.1, 0.2, 0.3]]) / (0.1**2 + 0.2**2 + 0.3**2)**(3/2)
KEPLER_POT = -G_INTERNAL * 1e8 * 1e9 / np.sqrt(0.1**2 + 0.2**2 + 0.3**2)

def test_external_acc_with_pot():
    acc = KEPLER_SIM.external_acc(0, return_internal=True)
    assert np.all(np.isclose(acc, KEPLER_ACC, rtol=1e-10))

def test_external_ax_with_pot():
    ax = KEPLER_SIM.external_ax(0, return_internal=True)
    assert np.all(np.isclose(ax, KEPLER_ACC[:, 0], rtol=1e-10))

def test_external_ay_with_pot():
    ay = KEPLER_SIM.external_ay(0, return_internal=True)
    assert np.all(np.isclose(ay, KEPLER_ACC[:, 1], rtol=1e-10))

def test_external_az_with_pot():
    az = KEPLER_SIM.external_az(0, return_internal=True)
    assert np.all(np.isclose(az, KEPLER_ACC[:, 2], rtol=1e-10))

# --- potential accessors ------------------------------------------------------------------------ #

def test_external_pot_against_direct():
    '''
    Aim: Verify compute_external_pot() returns m * phi_ext for a single particle
    in a Kepler potential, compared to the hand-computed value.

    If this fails: galpy bridge is returning wrong potential values, or
    compute_external_pot is not multiplying by mass.
    Relies on: galpy KeplerPotential being correct, _galpy_bridge conversion.
    '''
    pot = KEPLER_SIM.compute_external_pot(0, return_internal=True)
    assert np.all(np.isclose(pot, KEPLER_POT, rtol=1e-10))

def test_self_potential_against_direct():
    '''
    Aim: Verify self_potential() equals mass * DirectSummationGravity().potential()
    potential. This tests that the Sim accessor correctly multiplies by mass
    on top of the raw solver output. Uses non-unit masses [1e8, 1e10].

    If this fails: self_potential is not multiplying by mass, or is
    calling self_gravity incorrectly (wrong method, wrong kwargs).
    Relies on: DirectSummationGravity being correct (test_direct_summation.py).
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    pot_direct = DirectSummationGravity(eps=0.0).potential(pos, mass)
    pot_sim = sim.self_potential(t=0, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(pot_sim[0], mass[0] * pot_direct[0], rtol=1e-15)
    np.testing.assert_allclose(pot_sim[1], mass[1] * pot_direct[1], rtol=1e-15)

def test_PE_against_direct():
    '''
    Aim: Verify PE() = mass*phi_self + mass*phi_ext by computing both terms
    analytically and comparing to the Sim accessor. Uses non-unit masses
    and an external Kepler potential.

    If this fails: PE() is not correctly summing self + external, or one
    of the terms is missing its mass factor.
    Relies on: test_self_potential_against_direct, test_external_pot_against_direct.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    self_pot_direct = DirectSummationGravity(eps=0.0).potential(pos, mass)
    ext_pot_direct = ([-G_INTERNAL * mass[0] * 1e9 / np.linalg.norm(pos[0]), 
                       -G_INTERNAL * mass[1] * 1e9 / np.linalg.norm(pos[1])])
    pot_direct = mass * self_pot_direct + ext_pot_direct
    pot_sim = sim.PE(t=0, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(pot_sim[0], pot_direct[0], rtol=1e-10)
    np.testing.assert_allclose(pot_sim[1], pot_direct[1], rtol=1e-10)

def test_PE_is_sum_of_self_and_external():
    '''
    Aim: Verify PE() == self_potential() + compute_external_pot().
    This is a pure consistency check — it does not compare to analytical
    values, so it can pass even if both compute_ methods have the same bug.

    If this fails: PE() is doing something other than adding the two
    methods together (e.g. extra terms, wrong signs).
    Relies on: nothing external — only tests internal consistency of Sim.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    self_pot = sim.self_potential(t=0, method='direct', eps=0.0, return_internal=True)
    ext_pot = sim.compute_external_pot(t=0, return_internal=True)
    total_pot = sim.PE(t=0, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(total_pot, self_pot + ext_pot, rtol=1e-15)

# --- total energy accessors ------------------------------------------------------------------------ #

def test_energy_is_sum_of_KE_and_PE():
    '''
    Aim: Verify energy() == KE() + PE() at t=0 (before integration).
    Pure consistency check — if KE and PE are both wrong in the same
    way, this still passes. The individual KE/PE correctness is
    validated by the analytical spot-check tests above.

    If this fails: energy() has extra logic beyond summing KE + PE.
    Relies on: nothing external — only tests internal consistency of Sim.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    KE = sim.KE(t=0, return_internal=True)
    PE = sim.PE(t=0, method='direct', eps=0.0, return_internal=True)
    energy = sim.energy(t=0, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(energy, KE + PE, rtol=1e-15)

def test_energy_is_sum_of_KE_and_PE_after_run():
    '''
    Aim: Same as test_energy_is_sum_of_KE_and_PE, but checked at t=0.5
    after the simulation has evolved. Ensures that energy() remains
    consistent with KE() + PE() even after integration moves particles.

    If this fails: energy() uses stale/cached data instead of recomputing
    from the current snapshot positions and velocities.
    Relies on: run() completing without error, pos/vel accessors working
    at non-zero timesteps.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=0.0)
    np.testing.assert_allclose(sim.energy(t=0.5, method='direct', eps=0.0, return_internal=True), 
                               sim.KE(t=0.5, return_internal=True) + sim.PE(t=0.5, method='direct', eps=0.0, return_internal=True), rtol=1e-15)

def test_system_energy_is_sum_of_energies():
    '''
    Aim: Verify system_energy() == sum(KE) + 0.5*sum(m·phi_self) + sum(m·phi_ext).
    The 0.5 on the self-PE avoids double-counting pairwise interactions.
    This checks the scalar reduction logic, NOT whether individual
    terms have correct mass factors (that's the analytical tests' job).

    NOTE: the expected value here is built from Sim's own accessors
    (self_potential, compute_external_pot), so this is a
    consistency test. A shared mass-factor bug would escape.

    If this fails: system_energy() is combining terms incorrectly
    (e.g. missing the 0.5 on self-PE, or not summing over particles).
    Relies on: KE(), self_potential(), compute_external_pot()
    all returning per-particle arrays of the right shape.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    KE = np.sum(sim.KE(t=0, return_internal=True))
    PE = 0.5 * np.sum(sim.self_potential(t=0, method='direct', eps=0.0, return_internal=True)) + np.sum(sim.compute_external_pot(t=0, return_internal=True))
    system_energy = sim.system_energy(t=0, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(system_energy, KE + PE, rtol=1e-15)
   
def test_system_energy_is_sum_of_energies_after_run():
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=0.0)
    KE = np.sum(sim.KE(t=0.5, return_internal=True))
    PE = 0.5 *np.sum(sim.self_potential(t=0.5, method='direct', eps=0.0, return_internal=True)) + np.sum(sim.compute_external_pot(t=0.5, return_internal=True))
    system_energy = sim.system_energy(t=0.5, method='direct', eps=0.0, return_internal=True)
    np.testing.assert_allclose(system_energy, KE + PE, rtol=1e-15)


# --- Test .run method ------------------------------------------------------------------------ #

def test_negative_dt():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=-0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0)

def test_negative_dt_out():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=-0.1,
                  method='direct',
                  eps=0.0)

def test_negative_t_end():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=-1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0)

def test_dt_out_less_than_dt():
    with pytest.raises(ValueError, match="dt_out must be greater than or equal to dt."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.05,
                  method='direct',
                  eps=0.0)
def test_dt_out_not_a_multiple_of_dt():
    with pytest.raises(ValueError, match="dt_out must be a multiple of dt."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.03, 
                  dt_out=0.07,
                  method='direct',
                  eps=0.0)
def test_invalid_method():
    with pytest.raises(ValueError, match=r"Unknown method 'invalid_method' for self-gravity. Supported methods: \['direct', 'falcON'\]"):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='invalid_method',
                  eps=0.0) 

def test_invalid_kwargs():
    with pytest.raises(ValueError, match="{'invalid_kwarg'} is \(are\) invalid kwarg\(s\) for 'direct' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0,
                  invalid_kwarg=42)
    with pytest.raises(ValueError, match="{'invalid_kwarg'} is \(are\) invalid kwarg\(s\) for 'falcON' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='falcON',
                  eps=0.0,
                  theta=0.3,
                  invalid_kwarg=42)

# --- Energy tests with non-unit masses ------------------------------------------------------------------------ #

E_POS  = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])   # 2 kpc apart
E_VEL  = np.array([[0.5,  0.0, 0.0], [0.0, 0.0, 0.3]])
E_VEL_INTERNAL = E_VEL * KMS_TO_KPCGYR
E_MASS = np.array([3e8, 5e8])           # non-unit, non-equal, large enough that G*M*M/r >> 1e-8
E_SEP  = np.linalg.norm(E_POS[1] - E_POS[0])   # 2.0 kpc
E_EPS  = 0.0
E_KEPLER_MASS = 1e10   # Msun, for external Kepler potential


def _energy_sim(with_ext_pot=False):
    """Two massive particles, optionally in an external Kepler potential."""
    sim = Sim()
    sim.add_particles('pair', pos=E_POS.copy(), vel=E_VEL.copy(), mass=E_MASS.copy())
    if with_ext_pot:
        kep = KeplerPotential(amp=E_KEPLER_MASS * u.Msun)
        sim.add_external_pot(kep)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=E_EPS)
    return sim

# --- KE -----------------------------------------------------------------

def test_KE_nonunit_mass():
    '''
    Aim: Verify KE = ½ m |v|² with non-unit masses [3e8, 5e8] Msun.
    Also asserts the result differs from the unit-mass answer, so a
    missing mass factor would be caught.

    If this fails: KE() is not multiplying by particle mass.
    Relies on: vel accessor returning correct initial velocities.
    '''
    sim = _energy_sim()
    ke = sim.KE(t=0)
    expected = 0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1)
    np.testing.assert_allclose(ke, expected, rtol=1e-10)
    # Confirm it *differs* from the unit-mass answer
    assert not np.allclose(ke, 0.5 * np.sum(E_VEL ** 2, axis=-1))

# --- Self-gravitational PE -----------------------------------------------

def test_self_potential_is_mass_weighted():
    '''
    Aim: Verify compute_self_potential returns m_i * phi_i (not just phi_i)
    by comparing to the analytical two-body PE: m_i * (-G * m_j / r_ij).
    Uses non-unit masses so m*phi ≠ phi.

    If this fails: compute_self_potential is returning bare phi without
    the mass multiplication, or DirectSummationGravity potential is wrong.
    Relies on: DirectSummationGravity being correct (test_direct_summation.py).
    '''
    sim = _energy_sim()
    pe = sim.self_potential(t=0, method='direct',eps=E_EPS, return_internal=True)
    # Two-body: PE_i = m_i * (-G * m_j / r_ij)
    expected_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    expected_1 = -E_MASS[1] * G_INTERNAL * E_MASS[0] / E_SEP
    np.testing.assert_allclose(pe[0], expected_0, rtol=1e-2)
    np.testing.assert_allclose(pe[1], expected_1, rtol=1e-2)

def test_self_potential_differs_from_bare_phi():
    '''
    Aim: Negative sanity check — verify that the returned PE does NOT
    equal the bare (unweighted) potential phi_0 = -G*m1/r. If the
    mass factor were missing, they would match and this test would fail.

    If this fails: compute_self_potential is returning phi without
    multiplying by particle mass — the exact bug this was written to catch.
    Relies on: test_self_potential_is_mass_weighted (if that passes and
    this fails, there is a contradiction).
    '''
    sim = _energy_sim()
    pe = sim.self_potential(t=0, method='direct',eps=E_EPS, return_internal=True)
    bare_phi_0 = -G_INTERNAL * E_MASS[1] / E_SEP   # potential, not PE
    assert not np.isclose(pe[0], bare_phi_0, rtol=1e-2, atol=0)

# --- External potential (per unit mass) -----------------------------------

def test_external_pot_is_mass_weighted():
    '''
    Aim: Verify compute_external_pot returns m_i * phi_ext,i (not bare phi_ext)
    by comparing to the analytical Kepler potential m * (-G * M / r).
    Uses non-unit masses so m*phi ≠ phi.

    If this fails: compute_external_pot is not multiplying by mass, or
    the galpy bridge is returning wrong potential values.
    Relies on: galpy KeplerPotential, _galpy_bridge unit conversion.
    '''
    sim = _energy_sim(with_ext_pot=True)
    ext = sim.compute_external_pot(t=0, return_internal=True)
    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    expected_0 = -E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
    expected_1 = -E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1
    np.testing.assert_allclose(ext[0], expected_0, rtol=1e-2)
    np.testing.assert_allclose(ext[1], expected_1, rtol=1e-2)

# --- Total PE (self + external, both mass-weighted) -----------------------

def test_PE_includes_mass_on_external():
    '''
    Aim: Verify PE() = m*phi_self + m*phi_ext by computing both terms
    analytically with non-unit masses. This is the main end-to-end check
    that the total PE has mass factors on BOTH the self and external terms.

    If this fails: PE() is missing mass on either the self-gravity or
    external potential term, or the summation of the two is wrong.
    Relies on: test_self_potential_is_mass_weighted,
    test_external_pot_is_mass_weighted (individual terms correct).
    '''
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, method='direct', eps=E_EPS, return_internal=True)

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])

    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP   # same for both
    ext_pe_0 = -E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
    ext_pe_1 = -E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1

    expected = np.array([self_pe + ext_pe_0, self_pe + ext_pe_1])
    np.testing.assert_allclose(pe, expected, rtol=1e-2)

def test_PE_external_without_mass_is_wrong():
    '''
    Aim: Negative sanity check — verify PE does NOT match the value you
    would get if mass were missing from the external potential term.
    Computes the "wrong" answer (m*phi_self + phi_ext) and asserts PE differs.

    If this fails: PE() is not multiplying mass onto the external potential
    — the exact bug this was written to catch.
    Relies on: test_PE_includes_mass_on_external (if that passes and this
    fails, there is a contradiction).
    '''
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, method='direct', eps=E_EPS, return_internal=True)

    self_pe_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    bare_ext_0 = -G_INTERNAL * E_KEPLER_MASS / np.linalg.norm(E_POS[0])
    wrong_pe_0 = self_pe_0 + bare_ext_0   # missing mass on external
    assert not np.isclose(pe[0], wrong_pe_0, rtol=1e-2, atol=0)


# --- System energy --------------------------------------------------------

def test_system_energy_analytical():
    '''
    Aim: Verify system_energy() matches a fully hand-computed value:
    E = sum(0.5 m |v|^2) + (-G m0 m1 / r) + sum(-m_i G M_ext / r_i).
    The 0.5 on the self-PE double-counting cancels with the pairwise sum.
    All values are analytical — no Sim accessors in the expected value.

    If this fails: system_energy has a wrong coefficient (e.g. missing 0.5
    on self-PE), a missing mass factor on any term, or wrong signs.
    Relies on: DirectSummationGravity  being correct, galpy bridge being correct.
    This is the strongest energy test — independent of all other accessors.
    '''
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, method='direct', eps=E_EPS, use_cached=False, return_internal=True)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL_INTERNAL ** 2, axis=-1))

    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    ext_pe = (-E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
              - E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1)

    expected = ke + self_pe + ext_pe
    np.testing.assert_allclose(E, expected, rtol=1e-10)

def test_system_energy_mass_on_external_matters():
    '''
    Aim: Negative sanity check — verify system_energy does NOT match
    the value you get if mass is missing from the external potential sum.
    Computes the "wrong" answer with bare phi_ext and asserts it differs.

    If this fails: system_energy is not multiplying mass onto the
    external potential — the exact bug this was written to catch.
    Relies on: test_system_energy_analytical (if that passes and this
    fails, there is a contradiction).
    '''
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, method='direct', eps=E_EPS, return_internal=True)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL_INTERNAL ** 2, axis=-1))
    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP
    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    wrong_ext = (-G_INTERNAL * E_KEPLER_MASS / r0
                 - G_INTERNAL * E_KEPLER_MASS / r1)  # missing mass
    wrong_E = ke + self_pe + wrong_ext
    assert not np.isclose(E, wrong_E, rtol=1e-2, atol=0)


# --- self-gravity caching ------------------------------------------------------------------------ #

def _caching_test_sim():
    sim = Sim()
    sim.add_particles('test', pos=np.random.normal(size=(10, 3)), vel=np.random.normal(size=(10, 3)), mass=np.random.normal(loc=1e9, scale=1e8, size=(10,)))
    return sim
def _run_caching_test_sim(sim, cache_self_gravity, cache_self_potential):
    sim.run(t_end=1, dt=0.1, dt_out=0.1, 
            method='direct', eps=0.0, 
            cache_self_gravity_acc=cache_self_gravity, 
            cache_self_gravity_pot=cache_self_potential)
    
def test_no_self_gravity_pot_or_acc_caching_after_run():
    '''
    Test that self-gravity potential and acceleration are not cached during
    run if return_self_acceleration and return_potential are False. 
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    assert sim._cached_self_pot is None
    assert sim._cached_self_acc is None
    
def test_self_gravity_acc_caching_only_after_run():
    '''
    Test that self-gravity acceleration is cached and potential
    is not during run if cache_self_gravity is True and cache_self_potential is False.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    assert sim._cached_self_acc.shape == (len(sim._times), sim._positions.shape[1], 3)
    assert sim._cached_self_pot is None

def test_self_gravity_pot_caching_only_after_run():
    '''
    Test that self-gravity potential is cached and acceleration is not during run
    if cache_self_potential is True and cache_self_gravity is False.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    assert sim._cached_self_pot.shape == (len(sim._times), sim._positions.shape[1])
    assert sim._cached_self_acc is None

def test_self_gravity_acc_and_pot_caching_after_run():
    '''
    Test that self-gravity acceleration and potential are cached during run
    if cache_self_gravity and cache_self_potential are True.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    assert sim._cached_self_acc.shape == (len(sim._times), sim._positions.shape[1], 3)
    assert sim._cached_self_pot.shape == (len(sim._times), sim._positions.shape[1])

def test_self_gravity_acc_caching_matches_internal_array():
    '''
    Test that the self-gravity acceleration values when 
    using caching are the same as the internal arrays.
    
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    np.testing.assert_allclose(sim.self_gravity(return_internal=True), sim._cached_self_acc, rtol=1e-10)

def test_self_gravity_pot_caching_matches_internal_array():
    '''
    Test that the self-gravity potential values when 
    using caching are the same as the internal arrays.
    
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    np.testing.assert_allclose(sim.self_potential(return_internal=True), sim._mass * sim._cached_self_pot, rtol=1e-10)

def test_self_gravity_acc_cache_matches_direct_computation():
    '''
    Test that the self-gravity acceleration cached results match the internal array after the run.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    np.testing.assert_allclose(sim.self_gravity(t=0, method='direct', eps=0.0), sim.self_gravity(t=0, use_cached=True), rtol=1e-10)

def test_self_gravity_pot_cache_matches_direct_computation():
    '''
    Test that the self-gravity potential cached results match the internal array after the run.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    np.testing.assert_allclose(sim.self_potential(t=0, method='direct', eps=0.0), sim.self_potential(t=0, use_cached=True), rtol=1e-10)

def test_provides_method_but_use_cached_true_raises_error():
    '''
    Test that requesting a method-specific computation with use_cached=True
    raises an error, since the cache is not method-specific.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    with pytest.raises(ValueError, match="`method` should not be specified"):
        sim.self_gravity(t=0, method='direct', eps=0.0, use_cached=True)

def test_use_cache_true_before_run_raises_error():
    '''
    Test that trying to use the self-gravity cache before it has been populated by a run raises an error.
    '''
    sim = _caching_test_sim()
    with pytest.raises(ValueError, match="Cannot use cached results before run"):
        sim.self_gravity(t=0, method=None, eps=0.0, use_cached=True)
    with pytest.raises(ValueError, match="Cannot use cached results before run"):
        sim.self_gravity(t=0, method='direct', eps=0.0, use_cached=True)

def test_no_caching_without_method_raises_error():
    '''
    Test that setting use_cached=False without specifying a method raises an error.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    with pytest.raises(ValueError, match="`use_cached` is False but no `method` was provided"):
        sim.self_gravity(t=0, use_cached=False)

def test_caching_defaults_to_false_before_run():
    '''
    Test that calling an accessor before run() with no method raises
    an informative error about the simulation not having been run.
    '''
    sim = _caching_test_sim()
    with pytest.raises(ValueError, match="No cached results available .* the simulation has not been run yet"):
        sim.self_gravity(t=0)
    with pytest.raises(ValueError, match="No cached results available .* the simulation has not been run yet"):
        sim.self_potential(t=0)

def test_caching_fails_if_run_did_not_cache():
    '''
    Test that if you set cache_self_gravity=False but then try to use the cache after the run, it raises an error.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    with pytest.raises(ValueError, match="Cached self-potential is not available"):
        sim.self_potential(t=0, use_cached=True)
    with pytest.raises(ValueError, match="Cached self-gravity is not available"):
        sim.self_gravity(t=0, use_cached=True)

def test_vectorized_t_fails_without_cached_results():
    '''
    Test that passing an array of times to self_gravity without cached results raises an error, since the method-specific computation does not support vectorized t.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    with pytest.raises(ValueError, match="Cached self-gravity is not available"):
        sim.self_gravity(t=...)
    
def test_vectorized_t_works_with_cache():
    '''
    Test that passing an array of times to self_gravity with cached results returns an array of the correct shape.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    acc = sim.self_gravity(t=..., use_cached=True)
    assert acc.shape == (len(sim._times), sim._positions.shape[1], 3)

def test_vectorized_t_fails_if_method_provided():
    '''
    Test that passing an array of times to self_gravity with a method specified raises an error, 
    since method-specific computations do not support vectorized t.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    with pytest.raises((ValueError, TypeError), match="Cannot compute on-the-fly for all times."):
        sim.self_gravity(t=..., method='direct', eps=0.0)


# --- Output shape tests: single t vs. ellipsis -------------------------------------------- #

def _shape_test_sim():
    '''Create and run a sim for shape testing.'''
    sim = Sim()
    sim.add_particles('a',
                      pos=np.random.normal(size=(20, 3)),
                      vel=np.random.normal(size=(20, 3)),
                      mass=np.abs(np.random.normal(loc=1e9, scale=1e8, size=(20,))))
    sim.run(t_end=1, dt=0.1, dt_out=0.1, method='direct', eps=0.05)
    return sim

_SHAPE_SIM = _shape_test_sim()
_N_SNAP = len(_SHAPE_SIM._times)       # 11
_N_PART = _SHAPE_SIM._mass.shape[0]    # 20


# --- self_potential shapes ---

def test_self_potential_single_t_shape():
    result = _SHAPE_SIM.self_potential(t=0)
    assert result.shape == (_N_PART,)

def test_self_potential_ellipsis_shape():
    result = _SHAPE_SIM.self_potential(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

# --- self_gravity shapes ---

def test_self_gravity_single_t_shape():
    result = _SHAPE_SIM.self_gravity(t=0)
    assert result.shape == (_N_PART, 3)

def test_self_gravity_ellipsis_shape():
    result = _SHAPE_SIM.self_gravity(t=...)
    assert result.shape == (_N_SNAP, _N_PART, 3)

# --- KE shapes ---

def test_KE_single_t_shape():
    result = _SHAPE_SIM.KE(t=0)
    assert result.shape == (_N_PART,)

def test_KE_ellipsis_shape():
    result = _SHAPE_SIM.KE(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

# --- PE shapes ---

def test_PE_single_t_shape():
    result = _SHAPE_SIM.PE(t=0)
    assert result.shape == (_N_PART,)

def test_PE_ellipsis_shape():
    result = _SHAPE_SIM.PE(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

# --- energy (per-particle) shapes ---

def test_energy_single_t_shape():
    result = _SHAPE_SIM.energy(t=0)
    assert result.shape == (_N_PART,)

def test_energy_ellipsis_shape():
    result = _SHAPE_SIM.energy(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

# --- system_energy shapes ---

def test_system_energy_single_t_shape():
    result = _SHAPE_SIM.system_energy(t=0)
    assert np.isscalar(result) or result.shape == ()

def test_system_energy_ellipsis_shape():
    result = _SHAPE_SIM.system_energy(t=...)
    assert result.shape == (_N_SNAP,)

# --- dE shapes ---

def test_dE_shape():
    result = _SHAPE_SIM.dE()
    assert result.shape == (_N_SNAP,)

# --- compute on-the-fly: single t only ---

def test_self_potential_compute_single_t_shape():
    result = _SHAPE_SIM.self_potential(t=0, method='direct', eps=0.05)
    assert result.shape == (_N_PART,)

def test_self_gravity_compute_single_t_shape():
    result = _SHAPE_SIM.self_gravity(t=0, method='direct', eps=0.05)
    assert result.shape == (_N_PART, 3)

def test_system_energy_compute_single_t_shape():
    result = _SHAPE_SIM.system_energy(t=0, method='direct', eps=0.05)
    assert np.isscalar(result) or result.shape == ()


# --- Momentum Accessors ------------------------------------------------------------------------ #
# --- comparison with analytic calculation --- #

def test_momentum_analytic():
    '''
    Aim: Verify that momentum = m * velocity for each particle at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    momentum = sim.p(t=0, return_internal=True)
    expected = mass[:, None] * vel * KMS_TO_KPCGYR
    np.testing.assert_allclose(momentum, expected, rtol=1e-15)

def test_px_analytic():
    '''
    Aim: Verify that px accessor returns the x-component of momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    px = sim.px(t=0, return_internal=True)
    expected = mass * vel[:, 0] * KMS_TO_KPCGYR
    np.testing.assert_allclose(px, expected, rtol=1e-15)

def test_py_analytic():
    '''
    Aim: Verify that py accessor returns the y-component of momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    py = sim.py(t=0, return_internal=True)
    expected = mass * vel[:, 1] * KMS_TO_KPCGYR
    np.testing.assert_allclose(py, expected, rtol=1e-15)

def test_pz_analytic():
    '''
    Aim: Verify that pz accessor returns the z-component of momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    pz = sim.pz(t=0, return_internal=True)
    expected = mass * vel[:, 2] * KMS_TO_KPCGYR
    np.testing.assert_allclose(pz, expected, rtol=1e-15)

def test_L_analytic():
    '''
    Aim: Verify that L = r x p at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    L = sim.L(t=0, return_internal=True)
    momentum = mass[:, None] * vel * KMS_TO_KPCGYR
    expected = np.cross(pos, momentum)
    np.testing.assert_allclose(L, expected, rtol=1e-12)

def test_Lx_analytic():
    '''
    Aim: Verify that Lx accessor returns the x-component of angular momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    Lx = sim.Lx(t=0, return_internal=True)
    # Lx = y*pz - z*py
    expected = mass * (pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1]) * KMS_TO_KPCGYR
    np.testing.assert_allclose(Lx, expected, rtol=1e-14)

def test_Ly_analytic():
    '''
    Aim: Verify that Ly accessor returns the y-component of angular momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    Ly = sim.Ly(t=0, return_internal=True)
    # Ly = z*px - x*pz
    expected = mass * (pos[:, 2] * vel[:, 0] - pos[:, 0] * vel[:, 2]) * KMS_TO_KPCGYR
    np.testing.assert_allclose(Ly, expected, rtol=1e-14)

def test_Lz_analytic():
    '''
    Aim: Verify that Lz accessor returns the z-component of angular momentum at t=0.
    '''
    sim = Sim()
    pos = np.random.normal(size=(10, 3))
    vel = np.random.normal(size=(10, 3))
    mass = np.random.normal(loc=1e9, scale=1e8, size=(10,))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    Lz = sim.Lz(t=0, return_internal=True)
    # Lz = x*py - y*px
    expected = mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) * KMS_TO_KPCGYR
    np.testing.assert_allclose(Lz, expected, rtol=1e-14)

# -- momentum shapes --- #

def test_linear_momentum_single_t_shape():
    result = _SHAPE_SIM.p(t=0)
    assert result.shape == (_N_PART, 3)

def test_linear_momentum_ellipsis_shape():
    result = _SHAPE_SIM.p(t=...)
    assert result.shape == (_N_SNAP, _N_PART, 3)

def test_px_shape():
    result = _SHAPE_SIM.px(t=0)
    assert result.shape == (_N_PART,)

def test_px_ellipsis_shape():
    result = _SHAPE_SIM.px(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

def test_py_shape():
    result = _SHAPE_SIM.py(t=0)
    assert result.shape == (_N_PART,)

def test_py_ellipsis_shape():
    result = _SHAPE_SIM.py(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

def test_pz_shape():
    result = _SHAPE_SIM.pz(t=0)
    assert result.shape == (_N_PART,)

def test_pz_ellipsis_shape():
    result = _SHAPE_SIM.pz(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

def test_L_shape():
    result = _SHAPE_SIM.L(t=0)
    assert result.shape == (_N_PART, 3)

def test_L_ellipsis_shape():
    result = _SHAPE_SIM.L(t=...)
    assert result.shape == (_N_SNAP, _N_PART, 3)

def test_Lx_shape():
    result = _SHAPE_SIM.Lx(t=0)
    assert result.shape == (_N_PART,)

def test_Lx_ellipsis_shape():
    result = _SHAPE_SIM.Lx(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

def test_Ly_shape():
    result = _SHAPE_SIM.Ly(t=0)
    assert result.shape == (_N_PART,)

def test_Ly_ellipsis_shape():
    result = _SHAPE_SIM.Ly(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

def test_Lz_shape():
    result = _SHAPE_SIM.Lz(t=0)
    assert result.shape == (_N_PART,)

def test_Lz_ellipsis_shape():
    result = _SHAPE_SIM.Lz(t=...)
    assert result.shape == (_N_SNAP, _N_PART)

# --- consistency --- #

def test_px_consistent_with_momentum():
    result = _SHAPE_SIM.px(t=0)
    expected = _SHAPE_SIM.p(t=0)[:, 0]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_py_consistent_with_momentum():
    result = _SHAPE_SIM.py(t=0)
    expected = _SHAPE_SIM.p(t=0)[:, 1]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_pz_consistent_with_momentum():
    result = _SHAPE_SIM.pz(t=0)
    expected = _SHAPE_SIM.p(t=0)[:, 2]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_L_consistent_with_momentum_and_position():
    result = _SHAPE_SIM.L(t=0, return_internal=True)
    expected = np.cross(_SHAPE_SIM._positions[0], _SHAPE_SIM.p(t=0, return_internal=True))
    np.testing.assert_allclose(result, expected, rtol=1e-14)

def test_Lx_consistent_with_L():
    result = _SHAPE_SIM.Lx(t=0)
    expected = _SHAPE_SIM.L(t=0)[:, 0]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_Ly_consistent_with_L():
    result = _SHAPE_SIM.Ly(t=0)
    expected = _SHAPE_SIM.L(t=0)[:, 1]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_Lz_consistent_with_L():
    result = _SHAPE_SIM.Lz(t=0)
    expected = _SHAPE_SIM.L(t=0)[:, 2]
    np.testing.assert_allclose(result, expected, rtol=1e-15)

def test_L_w_center_pos():
    '''
    Test center_pos shift in calculation of L.
    '''
    sim = Sim()
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.array([[0, 0.1, 0], [0.1, 0.0, 0]])
    mass = np.array([1e8, 1e9])
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    L = sim.L(center_pos=np.array([0.5, 0.5, 0]), return_internal=True)
    expected = mass[:, None] * np.cross(pos - np.array([0.5, 0.5, 0]), vel * KMS_TO_KPCGYR)
    np.testing.assert_allclose(L[0], expected, rtol=1e-15)

def test_L_w_center_vel():
    '''
    Test center_vel shift in calculation of L.
    '''
    sim = Sim()
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.array([[0, 0.1, 0], [0.1, 0.0, 0]])
    mass = np.array([1e8, 1e9])
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    L = sim.L(center_vel=np.array([0.05, 0.05, 0]), return_internal=True)
    expected = mass[:, None] * np.cross(pos, vel * KMS_TO_KPCGYR - np.array([0.05, 0.05, 0]))
    np.testing.assert_allclose(L[0], expected, rtol=1e-15)

def test_L_w_center_pos_and_vel():
    '''
    Test center_pos and center_vel shift in calculation of L.
    '''
    sim = Sim()
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.array([[0, 0.1, 0], [0.1, 0.0, 0]])
    mass = np.array([1e8, 1e9])
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    L = sim.L(center_pos=np.array([0.5, 0.5, 0]), center_vel=np.array([0.05, 0.05, 0]), return_internal=True)
    expected = mass[:, None] * np.cross(pos - np.array([0.5, 0.5, 0]), vel * KMS_TO_KPCGYR - np.array([0.05, 0.05, 0]))
    np.testing.assert_allclose(L[0], expected, rtol=1e-15)

# --- coordinate transformations ----------------------------------------------------------------------- #

def _coord_test_sim():
    """Sim with random particles away from coordinate singularities."""
    rng = np.random.default_rng(42)
    sim = Sim()
    # Avoid z-axis (R=0) and origin (r=0) to prevent division-by-zero
    pos = rng.normal(loc=3.0, scale=1.0, size=(20, 3))
    vel = rng.normal(size=(20, 3))
    mass = np.abs(rng.normal(loc=1e9, scale=1e8, size=(20,)))
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    return sim

_COORD_SIM = _coord_test_sim()

def test_spherical_r_consistent_with_cartesian():
    '''
    Test that the spherical radius r is consistent with the Cartesian position.
    '''
    r_cartesian = np.linalg.norm(_COORD_SIM.pos(t=0), axis=-1)
    r_spherical = _COORD_SIM.r(t=0)
    np.testing.assert_allclose(r_cartesian, r_spherical, rtol=1e-15)

def test_spherical_r_consistent_with_cylindrical():
    '''
    Test that the spherical radius r is consistent with the cylindrical coordinates R and z.
    '''
    expected_r = np.sqrt(_COORD_SIM.cylR(t=0)**2 + _COORD_SIM.z(t=0)**2)
    np.testing.assert_allclose(_COORD_SIM.r(t=0), expected_r, rtol=1e-15)

def test_phi_consistent_with_cartesian():
    '''
    Test that the azimuthal angle phi is consistent with the Cartesian position.
    '''
    phi_cartesian = np.arctan2(_COORD_SIM.y(t=0), _COORD_SIM.x(t=0))
    phi_cylindrical = _COORD_SIM.phi(t=0)
    np.testing.assert_allclose(phi_cartesian, phi_cylindrical, rtol=1e-15)


# --- position coordinate identities (random particles) ------------------------------------------- #

def test_r_squared_equals_x2_y2_z2():
    """r^2 = x^2 + y^2 + z^2"""
    np.testing.assert_allclose(
        _COORD_SIM.r(t=0)**2,
        _COORD_SIM.x(t=0)**2 + _COORD_SIM.y(t=0)**2 + _COORD_SIM.z(t=0)**2,
        rtol=1e-14)

def test_cylR_squared_equals_x2_y2():
    """R^2 = x^2 + y^2"""
    np.testing.assert_allclose(
        _COORD_SIM.cylR(t=0)**2,
        _COORD_SIM.x(t=0)**2 + _COORD_SIM.y(t=0)**2,
        rtol=1e-14)

def test_cartesian_roundtrip_x():
    """x = r sin(theta) cos(phi)"""
    np.testing.assert_allclose(
        _COORD_SIM.x(t=0),
        _COORD_SIM.r(t=0) * np.sin(_COORD_SIM.theta(t=0)) * np.cos(_COORD_SIM.phi(t=0)),
        rtol=1e-14)

def test_cartesian_roundtrip_y():
    """y = r sin(theta) sin(phi)"""
    np.testing.assert_allclose(
        _COORD_SIM.y(t=0),
        _COORD_SIM.r(t=0) * np.sin(_COORD_SIM.theta(t=0)) * np.sin(_COORD_SIM.phi(t=0)),
        rtol=1e-14)

def test_cartesian_roundtrip_z():
    """z = r cos(theta)"""
    np.testing.assert_allclose(
        _COORD_SIM.z(t=0),
        _COORD_SIM.r(t=0) * np.cos(_COORD_SIM.theta(t=0)),
        rtol=1e-14)

def test_cylR_equals_r_sin_theta():
    """R = r sin(theta)"""
    np.testing.assert_allclose(
        _COORD_SIM.cylR(t=0),
        _COORD_SIM.r(t=0) * np.sin(_COORD_SIM.theta(t=0)),
        rtol=1e-14)

def test_theta_range():
    """theta is in  [0, pi]"""
    th = _COORD_SIM.theta(t=0)
    assert np.all(th >= 0) and np.all(th <= np.pi)

def test_phi_range():
    """phi is in [-pi, pi]"""
    ph = _COORD_SIM.phi(t=0)
    assert np.all(ph >= -np.pi) and np.all(ph <= np.pi)

# --- velocity decomposition identities (random particles) ---------------------------------------- #

def test_spherical_velocity_decomposition():
    """|v|^2 = vr^2 + vtheta^2 + (R*vphi)^2"""
    v_sq = np.sum(_COORD_SIM.vel(t=0, return_internal=True)**2, axis=-1)
    R = _COORD_SIM.cylR(t=0)
    recon = (_COORD_SIM.vr(t=0, return_internal=True)**2
             + _COORD_SIM.vtheta(t=0, return_internal=True)**2
             + (R * _COORD_SIM.vphi(t=0, return_internal=True))**2)
    np.testing.assert_allclose(recon, v_sq, rtol=1e-13)

def test_cylindrical_velocity_decomposition():
    """|v|^2 = vR^2 + (R*vphi)^2 + vz^2"""
    v_sq = np.sum(_COORD_SIM.vel(t=0, return_internal=True)**2, axis=-1)
    R = _COORD_SIM.cylR(t=0)
    recon = (_COORD_SIM.cylvR(t=0, return_internal=True)**2
             + (R * _COORD_SIM.vphi(t=0, return_internal=True))**2
             + _COORD_SIM.vz(t=0, return_internal=True)**2)
    np.testing.assert_allclose(recon, v_sq, rtol=1e-13)


# --- explicit velocity formulas (random particles) ----------------------------------------------- #

def test_vr_explicit_formula():
    """vr = (x*vx + y*vy + z*vz) / r"""
    pos = _COORD_SIM.pos(t=0)
    vel = _COORD_SIM.vel(t=0, return_internal=True)
    r = np.linalg.norm(pos, axis=-1)
    expected = (pos[:, 0]*vel[:, 0] + pos[:, 1]*vel[:, 1] + pos[:, 2]*vel[:, 2]) / r
    np.testing.assert_allclose(_COORD_SIM.vr(t=0, return_internal=True), expected, rtol=1e-15)

def test_cylvR_explicit_formula():
    """vR = (x*vx + y*vy) / R"""
    pos = _COORD_SIM.pos(t=0)
    vel = _COORD_SIM.vel(t=0, return_internal=True)
    R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    expected = (pos[:, 0]*vel[:, 0] + pos[:, 1]*vel[:, 1]) / R
    np.testing.assert_allclose(_COORD_SIM.cylvR(t=0, return_internal=True), expected, rtol=1e-15)

def test_vphi_explicit_formula():
    """vphi = (x*vy - y*vx) / R^2"""
    pos = _COORD_SIM.pos(t=0)
    vel = _COORD_SIM.vel(t=0, return_internal=True)
    R_sq = pos[:, 0]**2 + pos[:, 1]**2
    expected = (pos[:, 0]*vel[:, 1] - pos[:, 1]*vel[:, 0]) / R_sq
    np.testing.assert_allclose(_COORD_SIM.vphi(t=0, return_internal=True), expected, rtol=1e-15)

def test_vtheta_explicit_formula():
    """vtheta = [z(x*vx + y*vy) - R^2*vz] / (r*R)"""
    pos = _COORD_SIM.pos(t=0)
    vel = _COORD_SIM.vel(t=0, return_internal=True)
    R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    r = np.linalg.norm(pos, axis=-1)
    in_plane_dot = pos[:, 0]*vel[:, 0] + pos[:, 1]*vel[:, 1]
    expected = (pos[:, 2] * in_plane_dot - R**2 * vel[:, 2]) / (r * R)
    np.testing.assert_allclose(_COORD_SIM.vtheta(t=0, return_internal=True), expected, rtol=1e-15)


# --- known geometry: particle on x-axis ---------------------------------------------------------- #

def _axis_sim(pos, vel):
    sim = Sim()
    sim.add_particles('test', pos=np.atleast_2d(pos).astype(np.float64),
                      vel=np.atleast_2d(vel).astype(np.float64),
                      mass=np.array([1e9]))
    return sim

def test_x_axis_positions():
    """Particle at (5,0,0): r=5, φ=0, θ=π/2, R=5"""
    sim = _axis_sim([5, 0, 0], [0, 0, 0])
    assert sim.r(t=0)[0] == pytest.approx(5.0)
    assert sim.phi(t=0)[0] == pytest.approx(0.0)
    assert sim.theta(t=0)[0] == pytest.approx(np.pi / 2)
    assert sim.cylR(t=0)[0] == pytest.approx(5.0)

def test_x_axis_radial_motion():
    """Particle at (3,0,0) moving in +x: purely radial."""
    sim = _axis_sim([3, 0, 0], [2, 0, 0])
    assert sim.vr(t=0)[0] == pytest.approx(2.0)
    assert sim.cylvR(t=0)[0] == pytest.approx(2.0)
    assert sim.vphi(t=0)[0] == pytest.approx(0.0)
    assert sim.vtheta(t=0)[0] == pytest.approx(0.0, abs=1e-15)

def test_x_axis_tangential_motion():
    """Particle at (3,0,0) moving in +y: purely tangential."""
    sim = _axis_sim([3, 0, 0], [0, 5, 0])
    assert sim.vr(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.cylvR(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.vphi(t=0)[0] == pytest.approx(5.0 / 3.0)  # vphi = vT/R [km/s/kpc]
    assert sim.vtheta(t=0)[0] == pytest.approx(0.0, abs=1e-15)

def test_x_axis_polar_motion():
    """Particle at (3,0,0) moving in +z: purely polar (vtheta)."""
    sim = _axis_sim([3, 0, 0], [0, 0, 4])
    assert sim.vr(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.cylvR(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.vphi(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    # vtheta should capture the full z-velocity here (theta=pi/2 → ẑ is the -thetâ direction)
    assert sim.vtheta(t=0)[0] == pytest.approx(-4.0)


# --- known geometry: 45 deg in xz-plane ------------------------------------------------------------- #

def test_45deg_xz_positions():
    """Particle at (1,0,1): r=sqrt(2), phi=0, theta=pi/4, R=1"""
    sim = _axis_sim([1, 0, 1], [0, 0, 0])
    assert sim.r(t=0)[0] == pytest.approx(np.sqrt(2))
    assert sim.phi(t=0)[0] == pytest.approx(0.0)
    assert sim.theta(t=0)[0] == pytest.approx(np.pi / 4)
    assert sim.cylR(t=0)[0] == pytest.approx(1.0)

def test_45deg_xz_radial_motion():
    """Particle at (1,0,1) moving radially outward along (1,0,1)/sqrt(2)."""
    s = 1 / np.sqrt(2)
    sim = _axis_sim([1, 0, 1], [3*s, 0, 3*s])
    assert sim.vr(t=0)[0] == pytest.approx(3.0)
    assert sim.vtheta(t=0)[0] == pytest.approx(0.0, abs=1e-14)
    assert sim.vphi(t=0)[0] == pytest.approx(0.0, abs=1e-14)


# --- known geometry: xy-plane circle -------------------------------------------------------------- #

def test_circular_orbit_xy():
    """Particle at (R,0,0) with velocity (0,v,0): purely tangential."""
    R0, v0 = 4.0, 7.0
    sim = _axis_sim([R0, 0, 0], [0, v0, 0])
    assert sim.vr(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.cylvR(t=0)[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.vphi(t=0)[0] == pytest.approx(v0 / R0)
    # tangential linear velocity = R * φ̇ = v0
    assert _COORD_SIM.cylR(t=0)[0] * _COORD_SIM.vphi(t=0)[0] == pytest.approx(
        _COORD_SIM.cylR(t=0)[0] * _COORD_SIM.vphi(t=0)[0])  # tautology guard
    assert sim.cylR(t=0)[0] * sim.vphi(t=0)[0] == pytest.approx(v0)


# --- known geometry: particle at (1,1,0) --------------------------------------------------------- #

def test_xy_diagonal_positions():
    """Particle at (1,1,0): phi=pi/4, theta=pi/2, R=r=sqrt(2)"""
    sim = _axis_sim([1, 1, 0], [0, 0, 0])
    assert sim.phi(t=0)[0] == pytest.approx(np.pi / 4)
    assert sim.theta(t=0)[0] == pytest.approx(np.pi / 2)
    assert sim.r(t=0)[0] == pytest.approx(np.sqrt(2))
    assert sim.cylR(t=0)[0] == pytest.approx(np.sqrt(2))

def test_xy_diagonal_radial_motion():
    """Particle at (1,1,0) moving along (1,1,0): purely radial."""
    sim = _axis_sim([1, 1, 0], [1, 1, 0])
    assert sim.vr(t=0)[0] == pytest.approx(np.sqrt(2))
    assert sim.cylvR(t=0)[0] == pytest.approx(np.sqrt(2))
    assert sim.vphi(t=0)[0] == pytest.approx(0.0, abs=1e-15)

def test_xy_diagonal_tangential_motion():
    """Particle at (1,1,0) moving along (-1,1,0): purely tangential."""
    sim = _axis_sim([1, 1, 0], [-1, 1, 0])
    assert sim.vr(t=0)[0] == pytest.approx(0.0, abs=1e-14)
    assert sim.cylvR(t=0)[0] == pytest.approx(0.0, abs=1e-14)
    assert sim.vphi(t=0)[0] == pytest.approx(1.0)

# --- component-level accessors ----------------------------------------------------------------- #

@pytest.fixture
def two_component_sim():
    sim = Sim()
    sim.add_particles('stars', pos=np.random.normal(size=(10, 3)),
                      vel=np.random.normal(size=(10, 3)),
                      mass=np.abs(np.random.normal(loc=1e9, scale=1e8, size=10)))
    sim.add_particles('gas', pos=np.random.normal(size=(5, 3)),
                      vel=np.random.normal(size=(5, 3)),
                      mass=np.abs(np.random.normal(loc=1e8, scale=1e7, size=5)))
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.05)
    return sim

def test_component_self_ax(two_component_sim):
    ax = two_component_sim.stars.self_ax(t=0)
    assert ax.shape == (10,)

def test_component_self_ay(two_component_sim):
    ay = two_component_sim.stars.self_ay(t=0)
    assert ay.shape == (10,)

def test_component_self_az(two_component_sim):
    az = two_component_sim.stars.self_az(t=0)
    assert az.shape == (10,)

def test_component_self_potential_cached(two_component_sim):
    sp = two_component_sim.stars.self_potential(t=0)
    assert sp.shape == (10,)

def test_component_self_potential_cached_warns_include_all(two_component_sim):
    with pytest.warns(UserWarning, match="Using cached self-potential"):
        two_component_sim.stars.self_potential(t=0, include_all_components=False)

def test_component_energy(two_component_sim):
    E = two_component_sim.stars.energy(t=0)
    assert E.shape == (10,)

def test_component_PE(two_component_sim):
    PE = two_component_sim.stars.PE(t=0)
    assert PE.shape == (10,)

def test_component_has_run(two_component_sim):
    assert two_component_sim.stars._has_run

# --- component external potential / acceleration ----------------------------------------------- #

@pytest.fixture
def ext_pot_component_sim():
    sim = Sim()
    sim.add_particles('stars', pos=np.array([[1.0, 0.0, 0.0]]),
                      vel=np.zeros((1, 3)),
                      mass=np.array([1e8]))
    sim.add_particles('gas', pos=np.array([[0.0, 1.0, 0.0]]),
                      vel=np.zeros((1, 3)),
                      mass=np.array([1e7]))
    kepler = KeplerPotential(amp=1e9*u.Msun)
    sim.add_external_pot(kepler)
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.05)
    return sim

def test_component_compute_external_pot(ext_pot_component_sim):
    ext_pot = ext_pot_component_sim.stars.compute_external_pot(t=0)
    assert ext_pot.shape == (1,)
    assert ext_pot[0] != 0.0

def test_component_external_acc(ext_pot_component_sim):
    acc = ext_pot_component_sim.stars.external_acc(t=0)
    assert acc.shape == (1, 3)
    assert not np.all(acc == 0)

def test_component_external_ax(ext_pot_component_sim):
    ax = ext_pot_component_sim.stars.external_ax(t=0)
    assert ax.shape == (1,)

def test_component_external_ay(ext_pot_component_sim):
    ay = ext_pot_component_sim.stars.external_ay(t=0)
    assert ay.shape == (1,)

def test_component_external_az(ext_pot_component_sim):
    az = ext_pot_component_sim.stars.external_az(t=0)
    assert az.shape == (1,)

# --- uncached dE / decorator paths ------------------------------------------------------------- #

def test_dE_uncached():
    sim = Sim()
    sim.add_particles('a', pos=np.random.normal(size=(10, 3)),
                      vel=np.random.normal(size=(10, 3)) * 0.01,
                      mass=np.abs(np.random.normal(loc=1e9, scale=1e8, size=10)))
    sim.run(t_end=0.2, dt=0.1, dt_out=0.1, method='direct', eps=0.05)
    dE = sim.dE(use_cached=False, method='direct', eps=0.05)
    assert dE.shape == (len(sim.times),)
    assert dE[0] == 0.0

def test_dE_uncached_single_t():
    sim = Sim()
    sim.add_particles('a', pos=np.random.normal(size=(10, 3)),
                      vel=np.random.normal(size=(10, 3)) * 0.01,
                      mass=np.abs(np.random.normal(loc=1e9, scale=1e8, size=10)))
    sim.run(t_end=0.2, dt=0.1, dt_out=0.1, method='direct', eps=0.05)
    dE = sim.dE(t=-1, use_cached=False, method='direct', eps=0.05)
    assert isinstance(dE, (float, np.floating))

def test_self_ax_simulation(two_component_sim):
    ax = two_component_sim.self_ax(t=0)
    assert ax.shape == (15,)

def test_self_ay_simulation(two_component_sim):
    ay = two_component_sim.self_ay(t=0)
    assert ay.shape == (15,)

def test_self_az_simulation(two_component_sim):
    az = two_component_sim.self_az(t=0)
    assert az.shape == (15,)

# --- Time-parameter validation and integration edge cases ----------------------------------- #

import warnings as _warnings


def _simple_sim():
    """One-particle sim for fast validation checks (no self-gravity needed)."""
    sim = Sim()
    sim.add_particles('pt', pos=np.array([[1.0, 0, 0]]),
                      vel=np.array([[0.0, 0, 0]]),
                      mass=np.array([1.0]))
    return sim


# -- t_end not a multiple of dt -> warning (not error) --

def test_t_end_not_multiple_of_dt_warns():
    '''
    t_end=0.15 / dt=0.04 is not an integer ratio (3.75).
    Sim.run() should issue a warning reporting the actual end time.
    int(0.15/0.04) = 3, so actual_t_end = 3*0.04 = 0.12.
    '''
    sim = _simple_sim()
    with pytest.warns(UserWarning, match=r"Simulation will end at t=0\.12 Gyr instead"):
        sim.run(t_end=0.15, dt=0.04, dt_out=0.04, method='direct', eps=0.0)
    # Verify the simulation actually ends at the reported time (truncated, not rounded up)
    np.testing.assert_allclose(sim.times[-1], 0.12, atol=1e-14)


def test_t_end_not_multiple_of_dt_still_runs():
    '''
    Even when t_end is not a multiple of dt the simulation should
    complete and produce valid output (no raise).
    '''
    sim = _simple_sim()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        sim.run(t_end=0.15, dt=0.04, dt_out=0.04, method='direct', eps=0.0)
    assert sim._has_run
    assert len(sim.times) >= 2


# -- t_end not a multiple of dt_out -> warning --

def test_t_end_not_multiple_of_dt_out_warns():
    '''
    t_end=0.5 / dt_out=0.3 is not an integer ratio.
    n_steps=round(0.5/0.1)=5, steps_per_output=round(0.3/0.1)=3,
    nsnaps=5//3+1=2, last output at (2-1)*0.3 = 0.3.
    '''
    sim = _simple_sim()
    with pytest.warns(UserWarning, match=r"Last output will be at t=0\.3 Gyr instead of t=0\.5 Gyr"):
        sim.run(t_end=0.5, dt=0.1, dt_out=0.3, method='direct', eps=0.0)
    # Verify the simulation's last output is at the reported time
    np.testing.assert_allclose(sim.times[-1], 0.3, atol=1e-14)

def test_t_end_not_multiple_of_dt_out_nor_dt_warns():
    '''
    t_end=0.5 / dt_out=0.3 is not an integer ratio.
    n_steps=round(0.5/0.1)=5, steps_per_output=round(0.3/0.1)=3,
    nsnaps=5//3+1=2, last output at (2-1)*0.3 = 0.3.
    '''
    sim = _simple_sim()
    with pytest.warns(UserWarning, match=r"Last output will be at t=0\.3 Gyr instead of t=0\.5 Gyr"):
        sim.run(t_end=0.5, dt=0.3, dt_out=0.3, method='direct', eps=0.0)
    # Verify the simulation's last output is at the reported time
    np.testing.assert_allclose(sim.times[-1], 0.3, atol=1e-14)
    
# -- float-tolerant dt_out/dt check: accept valid inputs --

def test_float_tolerant_dt_out_dt_accepts_exact_multiples():
    '''
    dt_out=0.1, dt=0.001 -> ratio = 100 exactly.
    Should not raise despite floating-point representation.
    '''
    sim = _simple_sim()
    sim.run(t_end=0.1, dt=0.001, dt_out=0.1, method='direct', eps=0.0)
    assert sim._has_run


def test_float_tolerant_dt_out_dt_accepts_tricky_floats():
    '''
    dt_out=0.075, dt=0.025 -> 3.0 in exact arithmetic, but
    0.075 and 0.025 are not exact in binary. The tolerant check
    should accept this.
    '''
    sim = _simple_sim()
    sim.run(t_end=0.075, dt=0.025, dt_out=0.075, method='direct', eps=0.0)
    assert sim._has_run


def test_float_tolerant_dt_out_dt_rejects_genuine_nonmultiple():
    '''
    dt_out=0.07, dt=0.03 -> ratio approx 2.333, genuinely not an integer.
    Should raise ValueError.
    '''
    sim = _simple_sim()
    with pytest.raises(ValueError, match="dt_out must be a multiple of dt."):
        sim.run(t_end=1.0, dt=0.03, dt_out=0.07, method='direct', eps=0.0)


# -- exact multiples produce no warnings --

def test_exact_multiples_no_warnings():
    '''
    dt_out and t_end are exact multiples of dt.
    No UserWarning should be raised.
    '''
    sim = _simple_sim()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")   # turn warnings into errors
        sim.run(t_end=0.5, dt=0.1, dt_out=0.1, method='direct', eps=0.0)
    assert sim._has_run


# -- no zero-position rows (the arange overshoot bug) --

def test_no_zero_position_rows_exact():
    '''
    With non-exact multiples the last snapshot must NOT be all-zeros
    (regression: np.arange overshoot allocated an extra row).
    '''
    sim = _simple_sim()
    sim.run(t_end=0.5, dt=0.1, dt_out=0.2, method='direct', eps=0.0)
    last_pos = sim.pos(-1, return_internal=True)
    assert not np.allclose(last_pos, 0.0), "Last snapshot is all zeros — arange overshoot bug"


def test_no_zero_position_rows_tricky():
    '''
    Use dt_out/dt that are exact in principle but tricky as floats
    (0.002 / 0.0001 = 20, but 0.002 % 0.0001 != 0 in binary).
    '''
    sim = _simple_sim()
    sim.run(t_end=0.002, dt=0.0001, dt_out=0.002, method='direct', eps=0.0)
    last_pos = sim.pos(-1, return_internal=True)
    assert not np.allclose(last_pos, 0.0), "Last snapshot is all zeros — arange overshoot bug"


def test_nsnaps_matches_times_length():
    '''
    The number of stored snapshots must equal len(sim.times).
    No extra unwritten rows should exist.
    '''
    sim = _simple_sim()
    sim.run(t_end=1.0, dt=0.1, dt_out=0.5, method='direct', eps=0.0)
    assert sim._positions.shape[0] == len(sim.times)
    assert sim._velocities.shape[0] == len(sim.times)


def test_times_array_correct():
    '''
    sim.times should be [0, dt_out, 2*dt_out, ...] with the right count,
    constructed via integer arithmetic not np.arange.
    '''
    sim = _simple_sim()
    sim.run(t_end=1.0, dt=0.1, dt_out=0.5, method='direct', eps=0.0)
    expected_times = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(sim.times, expected_times, atol=1e-14)


# --- Time-dependent external potential bug regression -------------------------------------- #

def test_compute_external_pot_uses_physical_time():
    '''
    Regression: compute_external_pot must pass the *physical* time (Gyr)
    to the external potential function, not the integer snapshot index.

    A DehnenSmoothWrapperPotential that is zero at t=0 and fully grown
    by t=tform+tsteady should give nonzero potential at later snapshots.
    If the bug is present (index passed instead of time), the potential
    would be evaluated at the wrong time.
    '''
    from galpy.potential import (LogarithmicHaloPotential,
                                 DehnenSmoothWrapperPotential)
    static_pot = LogarithmicHaloPotential(normalize=1., q=0.9)
    smooth_pot = DehnenSmoothWrapperPotential(
        pot=static_pot, tform=0., tsteady=1.0 * u.Gyr
    )

    sim = Sim()
    sim.add_external_pot(smooth_pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    vel = np.array([[0.0, 220.0 * KMS_TO_KPCGYR, 0.0]])
    mass = np.array([1e5])
    sim.add_particles('star', pos=pos, vel=vel, mass=mass)

    # Run so we get multiple snapshots at different physical times
    sim.run(t_end=1.0, dt=0.1, dt_out=0.5, method='direct', eps=0.0)
    # times = [0.0, 0.5, 1.0] -> indices [0, 1, 2]

    # At index 2, the physical time is 1.0 Gyr.
    # Get potential via integer index
    pot_idx = sim.compute_external_pot(2, return_internal=True)
    # Get potential via float time (unambiguously physical time)
    pot_time = sim.compute_external_pot(1.0, return_internal=True)

    # These must agree — if the bug were present, pot_idx would use t=2
    # (the index) instead of t=1.0 Gyr, giving different results.
    np.testing.assert_allclose(pot_idx, pot_time, rtol=1e-14)


def test_external_acc_uses_physical_time():
    '''
    Regression: external_acc must pass the *physical* time (Gyr)
    to the external acceleration function, not the raw integer index.
    '''
    from galpy.potential import (LogarithmicHaloPotential,
                                 DehnenSmoothWrapperPotential)
    static_pot = LogarithmicHaloPotential(normalize=1., q=0.9)
    smooth_pot = DehnenSmoothWrapperPotential(
        pot=static_pot, tform=0., tsteady=2.0 * u.Gyr
    )

    sim = Sim()
    sim.add_external_pot(smooth_pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    vel = np.array([[0.0, 220.0 * KMS_TO_KPCGYR, 0.0]])
    mass = np.array([1e5])
    sim.add_particles('star', pos=pos, vel=vel, mass=mass)

    sim.run(t_end=1.0, dt=0.1, dt_out=0.5, method='direct', eps=0.0)

    # At index 2, the physical time is 1.0 Gyr.
    acc_idx = sim.external_acc(2, return_internal=True)
    acc_time = sim.external_acc(1.0, return_internal=True)

    np.testing.assert_allclose(acc_idx, acc_time, rtol=1e-14)
