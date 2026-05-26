import pytest
import numpy as np
from tambora.simulation import Sim, Component
from tambora.util import G_INTERNAL
from tambora.dynamics import DirectSummationGravity

np.random.seed(42)
direct_gravity = DirectSummationGravity(eps=0.0)

COMP1_NPTS = 50
COMP1_POS = np.random.rand(COMP1_NPTS, 3)
COMP1_VEL = np.random.rand(COMP1_NPTS, 3)
COMP1_MASS = np.random.rand(COMP1_NPTS)

COMP2_NPTS = 30
COMP2_POS = np.random.rand(COMP2_NPTS, 3)
COMP2_VEL = np.random.rand(COMP2_NPTS, 3)
COMP2_MASS = np.random.rand(COMP2_NPTS)

multicomp = Sim()
multicomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)
multicomp.add_particles('comp2',
                    pos=COMP2_POS, 
                    vel=COMP2_VEL, 
                    mass=COMP2_MASS)

# --- Basic attribute access ------------------------------------------------- #

def test_component_attribute_access_pre_run():
    '''
    Aim: Verify that accessing a named component via sim.comp1 returns a
    Component that slices into the correct portion of the Sim arrays.

    If this fails: __getattr__ is not creating Component with the right slice.
    Relies on: add_particles correctly building _slices.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1, Component)
    assert np.all(comp1.pos(t=0) == multicomp._init_pos[multicomp._slices['comp1']])
    assert np.all(comp1.vel(t=0, return_internal=True) == multicomp._init_vel[multicomp._slices['comp1']])
    assert np.all(comp1.mass == multicomp._mass[multicomp._slices['comp1']])

def test_component_returns_view_not_copy():
    '''
    Aim: Verify that component.mass is a view into (not a copy of)
    the parent Sim's mass array, so they share memory.

    If this fails: Component is copying data instead of slicing.
    Relies on: add_particles, __getattr__.
    '''
    comp1 = multicomp.comp1
    assert np.shares_memory(comp1.mass, multicomp._mass)

def test_both_components_partition_full_arrays():
    '''
    Aim: Verify comp1 and comp2 together cover all particles exactly once.
    Concatenating their masses should reproduce the full Sim mass array.

    If this fails: Slices are overlapping, have gaps, or are wrong size.
    Relies on: add_particles building contiguous non-overlapping slices.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    combined_mass = np.concatenate([comp1.mass, comp2.mass])
    np.testing.assert_array_equal(combined_mass, multicomp._mass)

def test_component_mass_is_read_only_property():
    '''
    Aim: Verify mass is a property, not a callable — calling it should
    raise TypeError, indexing should work.

    If this fails: mass was accidentally defined as a method instead of property.
    Relies on: Component.mass @property.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1.mass, np.ndarray)
    with pytest.raises(TypeError):
        comp1.mass()

# --- Shapes after run ------------------------------------------------------- #

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)
_N_SNAP_MC = 11

def test_component_accessor_shapes_after_run():
    '''
    Aim: Verify that all position/velocity accessors return arrays with
    the correct component-specific shapes after run.

    If this fails: Slicing is grabbing too many or too few particles.
    Relies on: run() populating _positions/_velocities correctly.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    assert comp1.pos().shape == (11, COMP1_NPTS, 3)
    assert comp1.vel().shape == (11, COMP1_NPTS, 3)
    assert comp1.mass.shape == (COMP1_NPTS,)
    assert comp2.pos().shape == (11, COMP2_NPTS, 3)
    assert comp2.vel().shape == (11, COMP2_NPTS, 3)
    assert comp2.mass.shape == (COMP2_NPTS,)

# --- Scalar accessors (x, y, z, vx, vy, vz) -------------------------------- #

def test_component_xyz_match_pos_columns():
    '''
    Aim: Verify x(), y(), z() return the correct columns of pos().

    If this fails: Column indices are swapped in the axis slicing.
    Relies on: pos() being correct.
    '''
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.x(0), comp1.pos(0)[:, 0])
    np.testing.assert_array_equal(comp1.y(0), comp1.pos(0)[:, 1])
    np.testing.assert_array_equal(comp1.z(0), comp1.pos(0)[:, 2])

def test_component_vxvyvz_match_vel_columns():
    '''
    Aim: Verify vx(), vy(), vz() return the correct columns of vel().

    If this fails: Column indices are swapped in the axis slicing.
    Relies on: vel() being correct.
    '''
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.vx(0), comp1.vel(0)[:, 0])
    np.testing.assert_array_equal(comp1.vy(0), comp1.vel(0)[:, 1])
    np.testing.assert_array_equal(comp1.vz(0), comp1.vel(0)[:, 2])

# --- Slicing correctness: component data != other component's data ---------- #

def test_comp1_data_differs_from_comp2():
    '''
    Aim: Verify that comp1 and comp2 return different data.

    If this fails: Both components reference the same slice.
    Relies on: Different random ICs for COMP1 and COMP2.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    assert not np.array_equal(comp1.pos(0), comp2.pos(0)[:COMP1_NPTS])
    assert not np.array_equal(comp1.mass, comp2.mass[:COMP1_NPTS])

# --- Time indexing through Component ---------------------------------------- #

def test_component_int_time_index():
    comp1 = multicomp.comp1
    assert comp1.pos(0).shape == (COMP1_NPTS, 3)
    assert comp1.pos(-1).shape == (COMP1_NPTS, 3)

def test_component_float_time_index():
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.pos(0.0), comp1.pos(0))
    np.testing.assert_array_equal(comp1.vel(0.0), comp1.vel(0))

def test_component_ellipsis_returns_all_snapshots():
    comp1 = multicomp.comp1
    all_pos = comp1.pos()
    assert all_pos.shape == (11, COMP1_NPTS, 3)
    np.testing.assert_array_equal(all_pos[0], comp1.pos(0))
    np.testing.assert_array_equal(all_pos[-1], comp1.pos(-1))

# --- Component data matches parent Sim sliced data ------------------------- #

def test_component_pos_matches_sim_sliced():
    comp1 = multicomp.comp1
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(comp1.pos(t), multicomp.pos(t)[sl])
        np.testing.assert_array_equal(comp1.vel(t), multicomp.vel(t)[sl])

def test_component_vel_matches_sim_sliced():
    comp2 = multicomp.comp2
    sl = multicomp._slices['comp2']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(comp2.vel(t), multicomp.vel(t)[sl])

# --- KE -------------------------------------------------------------------- #

def test_component_KE_matches_manual():
    comp1 = multicomp.comp1
    ke = comp1.KE(t=0)
    expected = 0.5 * comp1.mass * np.sum(comp1.vel(0)**2, axis=-1)
    np.testing.assert_allclose(ke, expected, rtol=1e-15)

def test_component_KE_differs_between_components():
    assert not np.array_equal(multicomp.comp1.KE(0), multicomp.comp2.KE(0)[:COMP1_NPTS])

def test_component_KE_sums_to_sim_KE():
    sim_ke = multicomp.KE(t=0)
    comp1_ke = multicomp.comp1.KE(t=0)
    comp2_ke = multicomp.comp2.KE(t=0)
    combined = np.concatenate([comp1_ke, comp2_ke])
    np.testing.assert_allclose(combined, sim_ke, rtol=1e-15)

# --- Component-specific shape tests for all accessors ---------------------------------------- #

# --- Coordinate position shapes --- #

def test_comp_r_shape_single_t():
    assert multicomp.comp1.r(t=0).shape == (COMP1_NPTS,)

def test_comp_r_shape_ellipsis():
    assert multicomp.comp1.r().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_phi_shape_single_t():
    assert multicomp.comp1.phi(t=0).shape == (COMP1_NPTS,)

def test_comp_phi_shape_ellipsis():
    assert multicomp.comp1.phi().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_theta_shape_single_t():
    assert multicomp.comp1.theta(t=0).shape == (COMP1_NPTS,)

def test_comp_theta_shape_ellipsis():
    assert multicomp.comp1.theta().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_cylR_shape_single_t():
    assert multicomp.comp1.cylR(t=0).shape == (COMP1_NPTS,)

def test_comp_cylR_shape_ellipsis():
    assert multicomp.comp1.cylR().shape == (_N_SNAP_MC, COMP1_NPTS)

# --- Coordinate velocity shapes --- #

def test_comp_vr_shape_single_t():
    assert multicomp.comp1.vr(t=0).shape == (COMP1_NPTS,)

def test_comp_vr_shape_ellipsis():
    assert multicomp.comp1.vr().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_vphi_shape_single_t():
    assert multicomp.comp1.vphi(t=0).shape == (COMP1_NPTS,)

def test_comp_vphi_shape_ellipsis():
    assert multicomp.comp1.vphi().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_vtheta_shape_single_t():
    assert multicomp.comp1.vtheta(t=0).shape == (COMP1_NPTS,)

def test_comp_vtheta_shape_ellipsis():
    assert multicomp.comp1.vtheta().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_cylvR_shape_single_t():
    assert multicomp.comp1.cylvR(t=0).shape == (COMP1_NPTS,)

def test_comp_cylvR_shape_ellipsis():
    assert multicomp.comp1.cylvR().shape == (_N_SNAP_MC, COMP1_NPTS)

# --- Momentum shapes --- #

def test_comp_p_shape_single_t():
    assert multicomp.comp1.p(t=0).shape == (COMP1_NPTS, 3)

def test_comp_p_shape_ellipsis():
    assert multicomp.comp1.p().shape == (_N_SNAP_MC, COMP1_NPTS, 3)

def test_comp_px_shape_single_t():
    assert multicomp.comp1.px(t=0).shape == (COMP1_NPTS,)

def test_comp_px_shape_ellipsis():
    assert multicomp.comp1.px().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_py_shape_single_t():
    assert multicomp.comp1.py(t=0).shape == (COMP1_NPTS,)

def test_comp_py_shape_ellipsis():
    assert multicomp.comp1.py().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_pz_shape_single_t():
    assert multicomp.comp1.pz(t=0).shape == (COMP1_NPTS,)

def test_comp_pz_shape_ellipsis():
    assert multicomp.comp1.pz().shape == (_N_SNAP_MC, COMP1_NPTS)

# --- Angular momentum shapes --- #

def test_comp_L_shape_single_t():
    assert multicomp.comp1.L(t=0).shape == (COMP1_NPTS, 3)

def test_comp_L_shape_ellipsis():
    assert multicomp.comp1.L().shape == (_N_SNAP_MC, COMP1_NPTS, 3)

def test_comp_Lx_shape_single_t():
    assert multicomp.comp1.Lx(t=0).shape == (COMP1_NPTS,)

def test_comp_Lx_shape_ellipsis():
    assert multicomp.comp1.Lx().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_Ly_shape_single_t():
    assert multicomp.comp1.Ly(t=0).shape == (COMP1_NPTS,)

def test_comp_Ly_shape_ellipsis():
    assert multicomp.comp1.Ly().shape == (_N_SNAP_MC, COMP1_NPTS)

def test_comp_Lz_shape_single_t():
    assert multicomp.comp1.Lz(t=0).shape == (COMP1_NPTS,)

def test_comp_Lz_shape_ellipsis():
    assert multicomp.comp1.Lz().shape == (_N_SNAP_MC, COMP1_NPTS)

# --- KE shapes --- #

def test_comp_KE_shape_single_t():
    assert multicomp.comp1.KE(t=0).shape == (COMP1_NPTS,)

def test_comp_KE_shape_ellipsis():
    assert multicomp.comp1.KE().shape == (_N_SNAP_MC, COMP1_NPTS)

# --- Component accessors match Sim sliced accessors -------------------------------------------------------------- #

def test_comp_r_matches_sim_sliced():
    '''comp.r(t) == sim.r(t)[slice]'''
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.r(t), multicomp.r(t)[sl])

def test_comp_phi_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.phi(t), multicomp.phi(t)[sl])

def test_comp_theta_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.theta(t), multicomp.theta(t)[sl])

def test_comp_cylR_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.cylR(t), multicomp.cylR(t)[sl])

def test_comp_vr_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.vr(t), multicomp.vr(t)[sl])

def test_comp_vphi_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.vphi(t), multicomp.vphi(t)[sl])

def test_comp_vtheta_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.vtheta(t), multicomp.vtheta(t)[sl])

def test_comp_cylvR_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.cylvR(t), multicomp.cylvR(t)[sl])

def test_comp_p_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.p(t), multicomp.p(t)[sl])

def test_comp_px_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.px(t), multicomp.px(t)[sl])

def test_comp_py_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.py(t), multicomp.py(t)[sl])

def test_comp_pz_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.pz(t), multicomp.pz(t)[sl])

def test_comp_L_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.L(t), multicomp.L(t)[sl])

def test_comp_Lx_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.Lx(t), multicomp.Lx(t)[sl])

def test_comp_Ly_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.Ly(t), multicomp.Ly(t)[sl])

def test_comp_Lz_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.Lz(t), multicomp.Lz(t)[sl])

def test_comp_KE_matches_sim_sliced():
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(multicomp.comp1.KE(t), multicomp.KE(t)[sl])


# --- Verify comp2 accessors (second slice offset) ------------------------------- #

def test_comp2_r_matches_sim_sliced():
    sl = multicomp._slices['comp2']
    np.testing.assert_array_equal(multicomp.comp2.r(0), multicomp.r(0)[sl])

def test_comp2_p_matches_sim_sliced():
    sl = multicomp._slices['comp2']
    np.testing.assert_array_equal(multicomp.comp2.p(0), multicomp.p(0)[sl])

def test_comp2_L_matches_sim_sliced():
    sl = multicomp._slices['comp2']
    np.testing.assert_array_equal(multicomp.comp2.L(0), multicomp.L(0)[sl])

# --- Component coordinate identities (computed from component data) ---------------------------------------- #

def test_comp_r_squared_identity():
    """r^2 = x^2 + y^2 + z^2 via component accessors."""
    c = multicomp.comp1
    np.testing.assert_allclose(
        c.r(t=0)**2,
        c.x(t=0)**2 + c.y(t=0)**2 + c.z(t=0)**2,
        rtol=1e-14)

def test_comp_cylR_squared_identity():
    """R^2 = x^2 + y^2 via component accessors."""
    c = multicomp.comp1
    np.testing.assert_allclose(
        c.cylR(t=0)**2,
        c.x(t=0)**2 + c.y(t=0)**2,
        rtol=1e-14)

def test_comp_cartesian_roundtrip():
    """x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta)"""
    c = multicomp.comp1
    r, th, ph = c.r(t=0), c.theta(t=0), c.phi(t=0)
    np.testing.assert_allclose(c.x(t=0), r * np.sin(th) * np.cos(ph), rtol=1e-13)
    np.testing.assert_allclose(c.y(t=0), r * np.sin(th) * np.sin(ph), rtol=1e-13)
    np.testing.assert_allclose(c.z(t=0), r * np.cos(th), rtol=1e-13)

def test_comp_spherical_velocity_decomposition():
    """|v|^2 = vr^2 + vtheta^2 + (R*vphi)^2 via component accessors."""
    c = multicomp.comp1
    v_sq = np.sum(c.vel(t=0, return_internal=True)**2, axis=-1)
    R = c.cylR(t=0)
    recon = c.vr(t=0, return_internal=True)**2 + c.vtheta(t=0, return_internal=True)**2 + (R * c.vphi(t=0, return_internal=True))**2
    np.testing.assert_allclose(recon, v_sq, rtol=1e-13)

def test_comp_cylindrical_velocity_decomposition():
    """|v|^2 = cylvR^2 + (R*vphi)^2 + vz^2 via component accessors."""
    c = multicomp.comp1
    v_sq = np.sum(c.vel(t=0, return_internal=True)**2, axis=-1)
    R = c.cylR(t=0)
    recon = c.cylvR(t=0, return_internal=True)**2 + (R * c.vphi(t=0, return_internal=True))**2 + c.vz(t=0, return_internal=True)**2
    np.testing.assert_allclose(recon, v_sq, rtol=1e-13)

# --- Component momentum analytical tests ----------------------------------------------- #

def test_comp_momentum_analytic():
    '''comp.p(t=0) == comp.mass[:, None] * comp.vel(t=0)'''
    c = multicomp.comp1
    np.testing.assert_allclose(c.p(t=0), c.mass[:, None] * c.vel(t=0), rtol=1e-15)

def test_comp_px_analytic():
    c = multicomp.comp1
    np.testing.assert_allclose(c.px(t=0), c.mass * c.vx(t=0), rtol=1e-15)

def test_comp_py_analytic():
    c = multicomp.comp1
    np.testing.assert_allclose(c.py(t=0), c.mass * c.vy(t=0), rtol=1e-15)

def test_comp_pz_analytic():
    c = multicomp.comp1
    np.testing.assert_allclose(c.pz(t=0), c.mass * c.vz(t=0), rtol=1e-15)

def test_comp_L_analytic():
    '''comp.L(t=0) == mass[:,None] * cross(pos, vel)'''
    c = multicomp.comp1
    expected = c.mass[:, None] * np.cross(c.pos(t=0, return_internal=True), c.vel(t=0, return_internal=True))
    np.testing.assert_allclose(c.L(t=0, return_internal=True), expected, rtol=1e-15)

def test_comp_Lx_explicit():
    '''Lx = m * (y*vz - z*vy)'''
    c = multicomp.comp1
    expected = c.mass * (c.y(t=0, return_internal=True) * c.vz(t=0, return_internal=True) - c.z(t=0, return_internal=True) * c.vy(t=0, return_internal=True))
    np.testing.assert_allclose(c.Lx(t=0, return_internal=True), expected, rtol=1e-15)

def test_comp_Ly_explicit():
    '''Ly = m * (z*vx - x*vz)'''
    c = multicomp.comp1
    expected = c.mass * (c.z(t=0, return_internal=True) * c.vx(t=0, return_internal=True) - c.x(t=0, return_internal=True) * c.vz(t=0, return_internal=True))
    np.testing.assert_allclose(c.Ly(t=0, return_internal=True), expected, rtol=1e-15)

def test_comp_Lz_explicit():
    '''Lz = m * (x*vy - y*vx)'''
    c = multicomp.comp1
    expected = c.mass * (c.x(t=0, return_internal=True) * c.vy(t=0, return_internal=True) - c.y(t=0, return_internal=True) * c.vx(t=0, return_internal=True))
    np.testing.assert_allclose(c.Lz(t=0, return_internal=True), expected, rtol=1e-15)


# --- Component momentum consistency ------------------------------------------------------ #

def test_comp_px_consistent_with_p():
    np.testing.assert_allclose(multicomp.comp1.px(0), multicomp.comp1.p(0)[:, 0], rtol=1e-15)

def test_comp_py_consistent_with_p():
    np.testing.assert_allclose(multicomp.comp1.py(0), multicomp.comp1.p(0)[:, 1], rtol=1e-15)

def test_comp_pz_consistent_with_p():
    np.testing.assert_allclose(multicomp.comp1.pz(0), multicomp.comp1.p(0)[:, 2], rtol=1e-15)

def test_comp_Lx_consistent_with_L():
    np.testing.assert_allclose(multicomp.comp1.Lx(0), multicomp.comp1.L(0)[:, 0], rtol=1e-15)

def test_comp_Ly_consistent_with_L():
    np.testing.assert_allclose(multicomp.comp1.Ly(0), multicomp.comp1.L(0)[:, 1], rtol=1e-15)

def test_comp_Lz_consistent_with_L():
    np.testing.assert_allclose(multicomp.comp1.Lz(0), multicomp.comp1.L(0)[:, 2], rtol=1e-15)


# --- Component L with center_pos / center_vel ------------------------------------------------------ # 

def test_comp_L_center_pos():
    c = multicomp.comp1
    center = np.array([0.5, 0.5, 0.0])
    L = c.L(t=0, center_pos=center, return_internal=True)
    expected = c.mass[:, None] * np.cross(c.pos(t=0, return_internal=True) - center, c.vel(t=0, return_internal=True))
    np.testing.assert_allclose(L, expected, rtol=1e-15)

def test_comp_L_center_vel():
    c = multicomp.comp1
    cv = np.array([0.05, 0.05, 0.0])
    L = c.L(t=0, center_vel=cv, return_internal=True)
    expected = c.mass[:, None] * np.cross(c.pos(t=0, return_internal=True), c.vel(t=0, return_internal=True) - cv)
    np.testing.assert_allclose(L, expected, rtol=1e-15)

def test_comp_L_center_pos_and_vel():
    c = multicomp.comp1
    cp = np.array([0.5, 0.5, 0.0])
    cv = np.array([0.05, 0.05, 0.0])
    L = c.L(t=0, center_pos=cp, center_vel=cv, return_internal=True)
    expected = c.mass[:, None] * np.cross(c.pos(t=0, return_internal=True) - cp, c.vel(t=0, return_internal=True) - cv)
    np.testing.assert_allclose(L, expected, rtol=1e-15)

# --- Component momentum / angular momentum partitioning ---------------------------------------- #

def test_comp_p_partitions_sim_p():
    '''Concatenation of component momenta equals full sim momentum.'''
    sim_p = multicomp.p(t=0)
    combined = np.concatenate([multicomp.comp1.p(t=0), multicomp.comp2.p(t=0)])
    np.testing.assert_allclose(combined, sim_p, rtol=1e-15)

def test_comp_L_partitions_sim_L():
    '''Concatenation of component angular momenta equals full sim angular momentum.'''
    sim_L = multicomp.L(t=0)
    combined = np.concatenate([multicomp.comp1.L(t=0), multicomp.comp2.L(t=0)])
    np.testing.assert_allclose(combined, sim_L, rtol=1e-15)


# --- Component velocity explicit formulas ---------------------------------------- #

def test_comp_vr_explicit():
    """vr = (x*vx + y*vy + z*vz) / r via component accessors."""
    c = multicomp.comp1
    expected = (c.x(0)*c.vx(0) + c.y(0)*c.vy(0) + c.z(0)*c.vz(0)) / c.r(0)
    np.testing.assert_allclose(c.vr(0), expected, rtol=1e-15)

def test_comp_cylvR_explicit():
    """cylvR = (x*vx + y*vy) / R via component accessors."""
    c = multicomp.comp1
    expected = (c.x(0)*c.vx(0) + c.y(0)*c.vy(0)) / c.cylR(0)
    np.testing.assert_allclose(c.cylvR(0), expected, rtol=1e-15)

def test_comp_vphi_explicit():
    """vphi = (x*vy - y*vx) / R^2 via component accessors."""
    c = multicomp.comp1
    expected = (c.x(0)*c.vy(0, return_internal=True) - c.y(0)*c.vx(0, return_internal=True)) / c.cylR(0)**2
    np.testing.assert_allclose(c.vphi(0, return_internal=True), expected, rtol=1e-15)

def test_comp_vtheta_explicit():
    """vtheta = [z(x*vx + y*vy) - R^2*vz] / (r*R) via component accessors."""
    c = multicomp.comp1
    R = c.cylR(0)
    r = c.r(0)
    in_plane = c.x(0, return_internal=True)*c.vx(0, return_internal=True) + c.y(0, return_internal=True)*c.vy(0, return_internal=True)
    expected = (c.z(0, return_internal=True) * in_plane - R**2 * c.vz(0, return_internal=True)) / (r * R)
    np.testing.assert_allclose(c.vtheta(0, return_internal=True), expected, rtol=1e-13)

# --- Ellipsis accessor matches sim sliced (all snapshots) ---------------------------------------- #

def test_comp_r_ellipsis_matches_sim():
    sl = multicomp._slices['comp1']
    np.testing.assert_array_equal(multicomp.comp1.r(), multicomp.r()[:, sl])

def test_comp_vr_ellipsis_matches_sim():
    sl = multicomp._slices['comp1']
    np.testing.assert_allclose(multicomp.comp1.vr(), multicomp.vr()[:, sl], rtol=1e-15)

def test_comp_p_ellipsis_matches_sim():
    sl = multicomp._slices['comp1']
    np.testing.assert_allclose(multicomp.comp1.p(), multicomp.p()[:, sl], rtol=1e-15)

def test_comp_L_ellipsis_matches_sim():
    sl = multicomp._slices['comp1']
    np.testing.assert_allclose(multicomp.comp1.L(), multicomp.L()[:, sl], rtol=1e-15)

def test_comp_KE_ellipsis_matches_sim():
    sl = multicomp._slices['comp1']
    np.testing.assert_allclose(multicomp.comp1.KE(), multicomp.KE()[:, sl], rtol=1e-15)


# --- include_all_components tests for self-gravity -------------------------------------------------- #

# Two-component sim with direct summation, eps=0, for analytical verification.
# comp1 ("disk"): 2 particles at x=1 and x=2
# comp2 ("sat"):  1 particle  at x=-5

_IAC_COMP1_POS = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_IAC_COMP1_VEL = np.zeros((2, 3))
_IAC_COMP1_MASS = np.array([1e9, 1e9])

_IAC_COMP2_POS = np.array([[-3.0, 0.0, 0.0]])
_IAC_COMP2_VEL = np.zeros((1, 3))
_IAC_COMP2_MASS = np.array([2e9])

def _iac_sim():
    """Two-component sim for include_all_components testing."""
    sim = Sim()
    sim.add_particles('disk', pos=_IAC_COMP1_POS.copy(),
                      vel=_IAC_COMP1_VEL.copy(), mass=_IAC_COMP1_MASS.copy())
    sim.add_particles('sat', pos=_IAC_COMP2_POS.copy(),
                      vel=_IAC_COMP2_VEL.copy(), mass=_IAC_COMP2_MASS.copy())
    sim.run(t_end=0.1, dt=0.1, dt_out=0.1, method='direct', eps=0.0,
            cache_self_gravity_acc=True, cache_self_gravity_pot=True)
    return sim

_IAC_SIM = _iac_sim()


def test_iac_self_gravity_all_components():
    '''
    Aim: With include_all_components=True, self-gravity on disk
    particles includes force from sat. Verify vs analytical direct sum.

    If this fails: include_all_components is not using all particles.
    '''
    disk = _IAC_SIM.disk
    all_pos = np.concatenate([_IAC_COMP1_POS, _IAC_COMP2_POS])
    all_mass = np.concatenate([_IAC_COMP1_MASS, _IAC_COMP2_MASS])
    
    acc_all = direct_gravity.acc(all_pos, all_mass)
    acc_disk_expected = acc_all[:2]

    acc_disk = disk.self_gravity(t=0, method='direct', eps=0.0, include_all_components=True, return_internal=True)
    np.testing.assert_allclose(acc_disk, acc_disk_expected, rtol=1e-15)


def test_iac_self_gravity_own_component_only():
    '''
    Aim: With include_all_components=False, self-gravity on disk
    only includes disk-disk forces, ignoring sat.

    If this fails: include_all_components=False still includes other components.
    '''
    disk = _IAC_SIM.disk
    acc_disk_only= direct_gravity.acc(_IAC_COMP1_POS, _IAC_COMP1_MASS)

    acc_disk = disk.self_gravity(t=0, method='direct', eps=0.0, include_all_components=False, return_internal=True)
    np.testing.assert_allclose(acc_disk, acc_disk_only, rtol=1e-15)


def test_iac_self_gravity_all_differs_from_own():
    '''
    Aim: Verify include_all_components=True and False give DIFFERENT results.

    If this fails: include_all_components flag has no effect.
    '''
    disk = _IAC_SIM.disk
    acc_all = disk.self_gravity(t=0, method='direct', eps=0.0, include_all_components=True)
    acc_own = disk.self_gravity(t=0, method='direct', eps=0.0, include_all_components=False)
    assert not np.allclose(acc_all, acc_own, rtol=1e-5)


def test_iac_self_potential_all_components():
    '''
    Aim: With include_all_components=True, self-potential on disk
    includes contributions from sat. Verify analytically.

    If this fails: include_all_components is not using all particles for potential.
    '''
    disk = _IAC_SIM.disk
    all_pos = np.concatenate([_IAC_COMP1_POS, _IAC_COMP2_POS])
    all_mass = np.concatenate([_IAC_COMP1_MASS, _IAC_COMP2_MASS])
    pot_all = direct_gravity.potential(all_pos, all_mass)
    expected = _IAC_COMP1_MASS * pot_all[:2]

    result = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=True, return_internal=True)
    np.testing.assert_allclose(result, expected, rtol=1e-15)


def test_iac_self_potential_own_component_only():
    '''
    Aim: With include_all_components=False, only disk-disk interactions.

    If this fails: include_all_components=False still includes other particles.
    '''
    disk = _IAC_SIM.disk
    pot_disk_only = direct_gravity.potential(_IAC_COMP1_POS, _IAC_COMP1_MASS)
    expected = _IAC_COMP1_MASS * pot_disk_only

    result = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=False, return_internal=True)
    np.testing.assert_allclose(result, expected, rtol=1e-15)


def test_iac_self_potential_all_differs_from_own():
    '''Verify include_all_components=True and False differ for potential.'''
    disk = _IAC_SIM.disk
    pot_all = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=True)
    pot_own = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=False)
    assert not np.allclose(pot_all, pot_own, rtol=1e-1)


def test_iac_sat_self_gravity_own_is_zero():
    '''
    Aim: Sat has only 1 particle. With include_all_components=False,
    self-gravity should be zero (no other particle to interact with).

    If this fails: Single-particle self-gravity is non-zero.
    '''
    sat = _IAC_SIM.sat
    acc = sat.self_gravity(t=0, method='direct', eps=0.0, include_all_components=False)
    np.testing.assert_allclose(acc, 0.0, atol=1e-30)


def test_iac_sat_self_gravity_all_is_nonzero():
    '''
    Aim: Sat has 1 particle. With include_all_components=True,
    it should feel force from the 2 disk particles.

    If this fails: include_all_components=True is not including disk particles.
    '''
    sat = _IAC_SIM.sat
    acc = sat.self_gravity(t=0, method='direct', eps=0.0, include_all_components=True)
    assert np.any(np.abs(acc) > 1e-30)


def test_iac_sat_self_gravity_all_analytical():
    '''
    Aim: Verify the exact acceleration on sat from all 3 particles.
    Sat is at (-3,0,0), disk particles at (1,0,0) and (2,0,0).

    If this fails: The all-components gravity calculation is wrong.
    '''
    sat = _IAC_SIM.sat
    all_pos = np.concatenate([_IAC_COMP1_POS, _IAC_COMP2_POS])
    all_mass = np.concatenate([_IAC_COMP1_MASS, _IAC_COMP2_MASS])
    acc_all = direct_gravity.acc(all_pos, all_mass)
    acc_sat_expected = acc_all[2:]  # sat is particle index 2

    acc_sat = sat.self_gravity(t=0, method='direct', eps=0.0, include_all_components=True, return_internal=True)
    np.testing.assert_allclose(acc_sat, acc_sat_expected, rtol=1e-15)


def test_iac_self_potential_analytical_two_body():
    '''
    Aim: Verify self_potential for disk particles analytically.
    Particle 0 at (1,0,0), mass 1e9. Particle 1 at (2,0,0), mass 1e9.
    Sat at (-3,0,0), mass 2e9.

    include_all_components=True:
      phi_0 = -G*m1/|r0-r1| - G*m_sat/|r0-r_sat| 
      PE_0 = m0 * phi_0

    include_all_components=False:
      phi_0 = -G*m1/|r0-r1|
      PE_0 = m0 * phi_0
    '''
    disk = _IAC_SIM.disk
    m0, m1, m_sat = 1e9, 1e9, 2e9
    r01 = 1.0   # |r0 - r1| = |1 - 2| = 1
    r0_sat = 4.0  # |1 - (-3)| = 4
    r1_sat = 5.0  # |2 - (-3)| = 5

    # All components
    phi_0_all = -G_INTERNAL * m1 / r01 - G_INTERNAL * m_sat / r0_sat
    phi_1_all = -G_INTERNAL * m0 / r01 - G_INTERNAL * m_sat / r1_sat
    expected_all = np.array([m0 * phi_0_all, m1 * phi_1_all])
    result_all = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=True, return_internal=True)
    np.testing.assert_allclose(result_all, expected_all, rtol=1e-15)

    # Own component only
    phi_0_own = -G_INTERNAL * m1 / r01
    phi_1_own = -G_INTERNAL * m0 / r01
    expected_own = np.array([m0 * phi_0_own, m1 * phi_1_own])
    result_own = disk.self_potential(t=0, method='direct', eps=0.0, include_all_components=False, return_internal=True)
    np.testing.assert_allclose(result_own, expected_own, rtol=1e-15)
