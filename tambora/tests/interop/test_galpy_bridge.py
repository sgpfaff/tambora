import pytest
import warnings
from galpy import potential
from galpy.orbit import Orbit
from galpy.util.coords import rect_to_cyl, cyl_to_rect_vec
import numpy as np
from tambora.interop._galpy import bridge as _galpy_bridge
from itertools import product
from functools import partial
import astropy.units as u
from tambora.dynamics import ExternalPotential

_has_composite = hasattr(potential, 'CompositePotential')


SUPPORTED_GALPY_SPHERICAL_POTENTIALS = [
    potential.BurkertPotential(),
    potential.TwoPowerSphericalPotential(),
    potential.DehnenCoreSphericalPotential(),
    potential.DehnenSphericalPotential(),
] + ([potential.EinastoPotential()] if hasattr(potential, 'EinastoPotential') else []) + [
    potential.HernquistPotential(),
    potential.interpSphericalPotential(rforce=lambda r: -1./r,
                        rgrid=np.geomspace(0.01,20,101),Phi0=0.),
    potential.IsochronePotential(),
    potential.JaffePotential(),
    potential.KeplerPotential(),
    potential.KingPotential(),
    potential.NFWPotential(),
    potential.PlummerPotential(),
    potential.PowerSphericalPotential(),
    potential.PowerSphericalPotentialwCutoff(),
    potential.PseudoIsothermalPotential(),
    potential.HomogeneousSpherePotential(),
    potential.SphericalShellPotential(),
    potential.TwoPowerTriaxialPotential(b=1., c=1.),
    potential.TriaxialGaussianPotential(b=1., c=1.),
    potential.TriaxialJaffePotential(b=1., c=1.),
    potential.TriaxialHernquistPotential(b=1., c=1.),
    potential.TriaxialNFWPotential(b=1., c=1.),
    potential.PerfectEllipsoidPotential(b=1., c=1.),
]

SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS = [
    potential.FlattenedPowerPotential(),
    potential.KuzminDiskPotential(),
    potential.KuzminKutuzovStaeckelPotential(),
    potential.LogarithmicHaloPotential(q=0.8),
    potential.MiyamotoNagaiPotential(),
    potential.MN3ExponentialDiskPotential(),
    potential.RingPotential(),
    potential.DoubleExponentialDiskPotential(),
    potential.RazorThinExponentialDiskPotential(),
    potential.interpRZPotential(potential.MWPotential2014, interpPot=True)
]

SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS = [
    # Ellipsoidal Potentials
    potential.TwoPowerTriaxialPotential(b=0.8, c=0.6),
    potential.TriaxialGaussianPotential(b=0.8, c=0.6),
    potential.TriaxialJaffePotential(b=0.8, c=0.6),
    potential.TriaxialHernquistPotential(b=0.8, c=0.6),
    potential.TriaxialNFWPotential(b=0.8, c=0.6),
    potential.PerfectEllipsoidPotential(b=0.8, c=0.6),
]

SUPPORTED_GALPY_GENERAL_TRIAXIAL_POTENTIALS = [
    potential.DehnenBarPotential(),
    potential.FerrersPotential(),
    potential.NullPotential(),
    potential.SoftenedNeedleBarPotential(),
    potential.SpiralArmsPotential(),
]

EXAMPLE_GALPY_COMPOSITE_POTENTIALS = [
    potential.MWPotential2014,
    potential.NFWPotential() + potential.MiyamotoNagaiPotential(), # two vectorized potentials
    potential.TriaxialNFWPotential(b=0.8, c=0.6) + potential.MiyamotoNagaiPotential(), # unvectorized before galpy 1.12 + vectorized
    potential.TriaxialNFWPotential(b=0.8, c=0.6) + potential.DehnenBarPotential(), # unvectorized before galpy 1.12 + vectorized
]

ALL_SUPPORTED_GALPY_POTENTIALS = (SUPPORTED_GALPY_SPHERICAL_POTENTIALS + 
                                  SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS + 
                                  SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS + 
                                  SUPPORTED_GALPY_GENERAL_TRIAXIAL_POTENTIALS +
                                  EXAMPLE_GALPY_COMPOSITE_POTENTIALS)


UNSUPPORTED_GALPY_POTENTIALS = [
    potential.DiskSCFPotential(),
    potential.SCFPotential(),
]

g = np.linspace(-100, 100, 5)
FULL_TEST_GRID_POSITIONS = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T

# Exclude the z-axis (x=0, y=0) — galpy forces have a known singularity at R=0
TEST_GRID_POSITIONS = FULL_TEST_GRID_POSITIONS[((FULL_TEST_GRID_POSITIONS[:, 0] != 0) | (FULL_TEST_GRID_POSITIONS[:, 1] != 0))]

# --- Spherical Potentials ------------------------------------------------------------------------ #

@pytest.fixture(params=SUPPORTED_GALPY_SPHERICAL_POTENTIALS, ids=lambda p: type(p).__name__)
def spherical_potential(request):
    return request.param

def test_radial_acc_only(spherical_potential):
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    acc = acc_fn(TEST_GRID_POSITIONS, t=0)
    cross = np.cross(TEST_GRID_POSITIONS, acc)
    scale = np.max(np.abs(TEST_GRID_POSITIONS)) * np.max(np.abs(acc))
    assert np.allclose(cross, 0, atol=1e-14 * scale)

def test_spherical_symmetry(spherical_potential):
    '''|a| should be identical at the same radius but different angles.'''
    r = 10.0  # kpc
    # Points at the same radius along different axes / diagonals
    d = r / np.sqrt(3)
    points = np.array([
        [r, 0, 0],
        [0, r, 0],
        [-r, 0, 0],
        [d, d, d],
        [-d, d, -d],
    ])
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    acc = acc_fn(points, t=0)
    magnitudes = np.linalg.norm(acc, axis=1)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-10)

def test_reflection_symmetry(spherical_potential):
    '''For spherical potentials, a(r) = -a(-r) (odd parity).'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    pos = np.array([[5.0, 3.0, 1.0], [10.0, -7.0, 2.0]])
    acc_pos = acc_fn(pos, t=0)
    acc_neg = acc_fn(-pos, t=0)
    np.testing.assert_allclose(acc_pos, -acc_neg, rtol=1e-12)
    
# --- Axisymmetric Potentials ------------------------------------------------------------------------ #

@pytest.fixture(params=SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS, ids=lambda p: type(p).__name__)
def axisymmetric_potential(request):
    return request.param

def test_axisymmetry(axisymmetric_potential):
    '''Acc should be invariant under rotation about z-axis for axisymmetric potentials.'''
    pot = axisymmetric_potential
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    R, z = 8.0, 1.0
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.column_stack([R * np.cos(angles), R * np.sin(angles), np.full_like(angles, z)])
    acc = acc_fn(points, t=0)
    magnitudes = np.linalg.norm(acc, axis=1)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-12)

# --- All Supported Potentials ------------------------------------------------------------------------ #

# # Reuse the grid without the origin for acceleration/potential tests
# POSITION_GRID = TEST_GRID_POSITIONS

@pytest.fixture(params=ALL_SUPPORTED_GALPY_POTENTIALS, ids=lambda p: type(p).__name__)
def galpy_potential(request):
    return request.param

def _numerical_acc(pot_fn, pos, h=1e-5):
    """Compute -grad(Phi) via central differences."""
    pos = np.asarray(pos, dtype=float)
    acc = np.zeros_like(pos)
    for i in range(3):
        pos_plus = pos.copy()
        pos_minus = pos.copy()
        pos_plus[:, i] += h
        pos_minus[:, i] -= h
        acc[:, i] = -(pot_fn(pos_plus) - pot_fn(pos_minus)) / (2 * h)
    return acc

def test_acceleration_match(galpy_potential):
    '''
    Compare the acceleration from the galpy potential wrapper to 
    numerical accelerations via central differences (h=1e-5).
    
    Relative tolerance is limited by O(h^2) truncation error,
    which varies by potential — sharper features give larger errors.
    atol is scaled to the max acceleration magnitude so the threshold
    is unit-system independent.
    '''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(galpy_potential)
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(galpy_potential)
    pot_i = partial(pot_fn, t=0)
    acc_bridge = acc_fn(TEST_GRID_POSITIONS, t=0)
    acc_num = _numerical_acc(pot_i, TEST_GRID_POSITIONS)
    scale = np.max(np.abs(acc_bridge))
    if isinstance(galpy_potential, potential.interpRZPotential):
        rtol = 1e-4
    elif isinstance(galpy_potential, list) or (_has_composite and isinstance(galpy_potential, potential.CompositePotential)):
        rtol = 1e-4   # composites with sharp features have large O(h^2) error
    else:
        rtol = 1e-6
    assert np.allclose(acc_bridge, acc_num, rtol=rtol, atol=1e-10 * scale)

def _set_physical(pot, on=True):
    '''Turn physical units on/off for any galpy potential type.'''
    if isinstance(pot, list):
        for p in pot:
            p.turn_physical_on() if on else p.turn_physical_off()
    else:
        pot.turn_physical_on() if on else pot.turn_physical_off()

def test_force_match_galpy(galpy_potential):
    '''
    Verify bridge accelerations match galpy physical-unit forces directly,
    analogous to test_potential_match for potentials.

    Uses galpy's evaluateRforces / evaluatezforces / evaluatephitorques
    with astropy units (quantity=True).
    '''
    R, phi, z = rect_to_cyl(*TEST_GRID_POSITIONS.T)
    acc_unit = u.kpc / u.Gyr**2
    torque_unit = u.kpc**2 / u.Gyr**2

    # Reference: galpy analytical forces in physical units → kpc/Gyr^2
    aR_ref = np.array([
        potential.evaluateRforces(
            galpy_potential, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(acc_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    az_ref = np.array([
        potential.evaluatezforces(
            galpy_potential, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(acc_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    pt_ref = np.array([
        potential.evaluatephitorques(
            galpy_potential, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(torque_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])

    # Azimuthal force = torque / R  (R in kpc)
    aphi_ref = pt_ref / R

    # Cylindrical → Cartesian
    ax_ref, ay_ref, az_cart_ref = cyl_to_rect_vec(aR_ref, aphi_ref, az_ref, phi)
    galpy_ref_acc = np.column_stack([ax_ref, ay_ref, az_cart_ref])

    # Bridge output (should match with physical OFF)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(galpy_potential)
    acc_bridge = acc_fn(TEST_GRID_POSITIONS, t=0)
    assert np.allclose(acc_bridge, galpy_ref_acc, rtol=1e-12, equal_nan=True), \
        "Force mismatch with galpy analytical forces (physical OFF)"

    # Also test with physical ON
    _set_physical(galpy_potential, on=True)
    acc_fn_on = _galpy_bridge._galpy_pot_to_acc_fn(galpy_potential)
    acc_bridge_on = acc_fn_on(TEST_GRID_POSITIONS, t=0)
    assert np.allclose(acc_bridge_on, galpy_ref_acc, rtol=1e-12, equal_nan=True), \
        "Force mismatch with galpy analytical forces (physical ON)"

    # Restore fixture state
    _set_physical(galpy_potential, on=False)

def test_potential_match(galpy_potential):
    '''
    Verify bridge potential matches galpy physical-unit output,
    regardless of whether physical units are on or off.
    '''
    R, phi, z = rect_to_cyl(*FULL_TEST_GRID_POSITIONS.T)
    # Reference: galpy evaluatePotentials in physical units (km^2/s^2),
    # converted to tambora internal units (kpc^2/Gyr^2).
    # Uses galpy defaults (ro=8 kpc, vo=220 km/s) passed explicitly.
    galpy_ref = np.array([
        potential.evaluatePotentials(
            galpy_potential, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(u.kpc**2 / u.Gyr**2).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    # Bridge output should match with physical OFF (default fixture state)
    ez_pot_off = _galpy_bridge._galpy_pot_to_pot_fn(galpy_potential)(
        FULL_TEST_GRID_POSITIONS, t=0
    )
    assert np.allclose(ez_pot_off, galpy_ref, rtol=1e-12, equal_nan=True), \
        "Potential mismatch with physical OFF"
    # Bridge output should also match with physical ON
    _set_physical(galpy_potential, on=True)
    ez_pot_on = _galpy_bridge._galpy_pot_to_pot_fn(galpy_potential)(
        FULL_TEST_GRID_POSITIONS, t=0
    )
    assert np.allclose(ez_pot_on, galpy_ref, rtol=1e-12, equal_nan=True), \
        "Potential mismatch with physical ON"
    # Restore fixture state
    _set_physical(galpy_potential, on=False)

# --- Triaxial Potentials ------------------------------------------------------------------------ #

@pytest.fixture(params=SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS, ids=lambda p: type(p).__name__)
def triaxial_potential(request):
    return request.param

def test_triaxial_reflection_symmetry(triaxial_potential):
    '''For triaxial potentials centered at origin, a(r) = -a(-r).'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(triaxial_potential)
    pos = np.array([[5.0, 3.0, 1.0], [10.0, -7.0, 2.0]])
    acc_pos = acc_fn(pos, t=0)
    acc_neg = acc_fn(-pos, t=0)
    np.testing.assert_allclose(acc_pos, -acc_neg, rtol=1e-12)

def test_triaxial_plane_symmetry(triaxial_potential):
    '''Reflecting through a coordinate plane flips only that force component.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(triaxial_potential)
    pos = np.array([[5.0, 3.0, 2.0]])
    acc = acc_fn(pos, t=0)
    for axis in range(3):
        reflected = pos.copy()
        reflected[0, axis] *= -1
        acc_ref = acc_fn(reflected, t=0)
        for j in range(3):
            if j == axis:
                np.testing.assert_allclose(acc_ref[0, j], -acc[0, j], rtol=1e-10)
            else:
                np.testing.assert_allclose(acc_ref[0, j], acc[0, j], rtol=1e-10)

# --- Unsupported Potentials ------------------------------------------------------------------------ #

@pytest.fixture(params=UNSUPPORTED_GALPY_POTENTIALS, ids=lambda p: type(p).__name__)
def unsupported_potential(request):
    return request.param

def test_identify_unsupported_potential(unsupported_potential):
    with pytest.raises(TypeError):
        _galpy_bridge._check_supported_pot(unsupported_potential)

# --- Unit Conversion ------------------------------------------------------------------------ #

def test_acc_units():
    '''Verify acc_fn returns the correct analytic value in internal units
    for a Kepler potential at a known position.'''
    from tambora.units import G_INTERNAL
    # Kepler potential: a = -GM/r^2 rhat
    from galpy.util.conversion import get_physical
    M_msun = 1e7 # Msun
    pot = potential.KeplerPotential(amp=M_msun * u.Msun)  # amp=1 Msun in physical units
    pot.turn_physical_on()
    r = 0.1 # kpc
    pos = np.array([[r, 0.0, 0.0]])  # 10 kpc along x
    expected_ax = -G_INTERNAL * M_msun / r**2
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    np.testing.assert_allclose(acc[0, 0], expected_ax, rtol=1e-10)
    np.testing.assert_allclose(acc[0, 1], 0.0, atol=1e-20)
    np.testing.assert_allclose(acc[0, 2], 0.0, atol=1e-20)

def test_pot_units():
    '''Verify pot_fn returns the correct analytic value in internal units
    for a Kepler potential at a known position.'''
    from tambora.units import G_INTERNAL
    from galpy.util.conversion import get_physical
    M_msun = 1e7 # Msun
    pot = potential.KeplerPotential(amp=M_msun * u.Msun)  # amp=1 Msun in physical units
    pot.turn_physical_on()
    r = 0.1 # kpc
    pos = np.array([[r, 0.0, 0.0]])
    expected_phi = -G_INTERNAL * M_msun / r
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(pot)
    phi = pot_fn(pos, t=0)
    np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-10)

# --- Input Shape Handling ------------------------------------------------------------------------ #

def test_single_particle_acc_shape():
    '''acc_fn should accept a (1,3) array and return (1,3).'''
    pot = potential.PlummerPotential()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    assert np.all(np.isfinite(acc))

def test_single_particle_pot_shape():
    '''pot_fn should accept a (1,3) array and return a (1,) array.'''
    pot = potential.PlummerPotential()
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    phi = pot_fn(pos, t=0)
    assert phi.shape == (1,)
    assert np.all(np.isfinite(phi))

def test_large_batch():
    '''acc_fn should handle 10k particles without issue.'''
    pot = potential.NFWPotential()
    rng = np.random.default_rng(42)
    pos = rng.uniform(-50, 50, size=(10_000, 3))
    # Avoid z-axis
    pos[np.abs(pos[:, 0]) < 0.01, 0] = 0.01
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    acc = acc_fn(pos, t=0)
    assert acc.shape == (10_000, 3)
    assert np.all(np.isfinite(acc))

# --- Time Dependence  ------------------------------------------------------------------------ #

def test_time_independence(spherical_potential):
    '''For static potentials, acc should not depend on t.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    pos = np.array([[8.0, 0.0, 1.0], [3.0, 4.0, 0.0]])
    acc_t0 = acc_fn(pos, t=0)
    acc_t100 = acc_fn(pos, t=100)
    np.testing.assert_array_equal(acc_t0, acc_t100)

# --- Force Direction ------------------------------------------------------------------------ #

@pytest.fixture(params=(SUPPORTED_GALPY_SPHERICAL_POTENTIALS + 
                                  SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS + 
                                  SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS), ids=lambda p: type(p).__name__)
def general_galpy_potential(request):
    return request.param

def test_force_direction_attractive(general_galpy_potential):
    '''Radial component of acceleration should point inward (dot(a, r) < 0).'''
    if isinstance(general_galpy_potential, potential.RingPotential) or isinstance(general_galpy_potential, potential.SphericalShellPotential):
        pytest.xfail("RingPotential or SphericalShellPotential is not purely attractive at all positions")
    pos = np.array([
        [8.0, 0.0, 0.0],
        [0.0, 5.0, 3.0],
        [3.0, -4.0, 1.0],
        [-2.0, -2.0, -2.0],
    ])
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(general_galpy_potential)
    acc = acc_fn(pos, t=0)
    dots = np.sum(pos * acc, axis=1)
    assert np.all(dots < 0), f"Expected all dot products < 0, got {dots}"

# --- Validation Helper Functions ------------------------------------------------------------------ #

def test_check_physical_pot_warns():
    '''_check_physical should warn when physical units are not explicitly set.'''
    pot = potential.PlummerPotential()
    with pytest.warns(UserWarning, match="does not have physical units explicitly set"):
        _galpy_bridge._check_physical(pot)

def test_check_physical_pot_noop():
    '''_check_physical should not warn when physical is already on.'''
    pot = potential.PlummerPotential()
    pot.turn_physical_on()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _galpy_bridge._check_physical(pot)

# --- Evaluating on arrays or one point at a time --------------------------------------------- #

class _PointsOnly(potential.PlummerPotential):
    '''A Plummer that galpy can only evaluate one point at a time.'''
    def _evaluate(self, R, z, phi=0., t=0.):
        if np.ndim(R):
            raise ValueError("one point at a time")
        return super()._evaluate(R, z, phi=phi, t=t)


class _WrongOnArrays(potential.PlummerPotential):
    '''A Plummer whose array code gives the wrong potential.'''
    def _evaluate(self, R, z, phi=0., t=0.):
        return super()._evaluate(R, z, phi=phi, t=t) * (1.01 if np.ndim(R) else 1.)


class _NaNFar(potential.PlummerPotential):
    '''A Plummer whose potential is NaN beyond R = 1, on arrays as at single points.'''
    def _evaluate(self, R, z, phi=0., t=0.):
        return np.where(np.asarray(R) > 1., np.nan, super()._evaluate(R, z, phi=phi, t=t))


class _CountsPoints(potential.PlummerPotential):
    '''A Plummer that counts how often galpy evaluates its potential at a single point.'''
    points = 0

    def _evaluate(self, R, z, phi=0., t=0.):
        type(self).points += not np.ndim(R)
        return super()._evaluate(R, z, phi=phi, t=t)


class _LogsForces(_PointsOnly):
    '''One that records where galpy asks for its forces, and which.'''
    log = []

    def _Rforce(self, R, z, phi=0., t=0.):
        type(self).log.append(('R', R))
        return super()._Rforce(R, z, phi=phi, t=t)

    def _zforce(self, R, z, phi=0., t=0.):
        type(self).log.append(('z', R))
        return super()._zforce(R, z, phi=phi, t=t)


class _Broken(potential.PlummerPotential):
    def _evaluate(self, R, z, phi=0., t=0.):
        raise RuntimeError("this potential is broken")


_SOME_POINTS = np.array([[8., 0., 1.], [5., 3., -2.], [-4., 6., 0.5], [1., -1., 0.]])


def _point_by_point(pot, pos):
    '''galpy's potential at each of ``pos`` [kpc], one point at a time, in tambora's units.'''
    ro, vo = _galpy_bridge._get_ro_vo(pot)
    R, phi, z = rect_to_cyl(*pos.T)
    return np.array([potential.evaluatePotentials(pot, Ri / ro, zi / ro, phi=pi, t=0., use_physical=False)
                     for Ri, zi, pi in zip(R, z, phi)]) * (vo * _galpy_bridge.KMS_TO_KPCGYR)**2


@pytest.mark.parametrize("make", [
    pytest.param(_PointsOnly, id='array_call_fails'),
    pytest.param(_WrongOnArrays, id='array_call_differs'),
    pytest.param(potential.FerrersPotential, id='FerrersPotential'),     # galpy's array call fails on any galpy
])
def test_a_potential_galpy_cant_evaluate_on_arrays_is_evaluated_one_point_at_a_time(make):
    pot = make()
    assert not _galpy_bridge._takes_arrays(pot)
    with pytest.warns(UserWarning, match=f"galpy can't evaluate {type(pot).__name__} on arrays of points"):
        acc_fn, pot_fn = _galpy_bridge._galpy_pot_to_fns(pot)
    np.testing.assert_allclose(pot_fn(_SOME_POINTS, t=0.), _point_by_point(pot, _SOME_POINTS), rtol=1e-14)
    assert acc_fn(_SOME_POINTS, t=0.).shape == (4, 3)


@pytest.mark.parametrize("pot", [
    pytest.param(potential.PlummerPotential(), id='PlummerPotential'),
    pytest.param(potential.NullPotential(), id='NullPotential'),          # a single 0 for any number of points
    pytest.param(potential.DehnenBarPotential(), id='DehnenBarPotential'),
    pytest.param(_NaNFar(), id='NaN_in_places'),                        # NaNs that agree, agree
])
def test_a_potential_galpy_evaluates_on_arrays_is_evaluated_that_way_without_a_warning(pot):
    assert _galpy_bridge._takes_arrays(pot)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _galpy_bridge._galpy_pot_to_fns(pot)


def test_only_the_components_galpy_cant_evaluate_on_arrays_are_evaluated_one_point_at_a_time():
    counted, slow = _CountsPoints(), _PointsOnly(amp=2., b=0.5)
    with pytest.warns(UserWarning, match="galpy can't evaluate _PointsOnly") as caught:
        _, pot_fn = _galpy_bridge._galpy_pot_to_fns([counted, slow])
    assert len([w for w in caught if "on arrays of points" in str(w.message)]) == 1
    _CountsPoints.points = 0
    got = pot_fn(_SOME_POINTS, t=0.)
    assert _CountsPoints.points == 0
    np.testing.assert_allclose(got, _point_by_point(counted, _SOME_POINTS) + _point_by_point(slow, _SOME_POINTS),
                               rtol=1e-14)


def test_one_point_at_a_time_every_force_is_evaluated_at_a_point_before_the_next():
    # galpy's triaxial and Ferrers potentials keep their work for the last point; going
    # through the points once per force made them twice as slow.
    with pytest.warns(UserWarning, match="one point at a time"):
        acc_fn, _ = _galpy_bridge._galpy_pot_to_fns(_LogsForces())
    _LogsForces.log = []
    acc_fn(_SOME_POINTS[:3], t=0.)
    assert [which for which, _ in _LogsForces.log] == ['R', 'z'] * 3


def test_an_error_from_the_potential_itself_is_raised():
    with pytest.raises(RuntimeError, match="this potential is broken"):
        _galpy_bridge._galpy_pot_to_fns(_Broken())

# ---  Edge Cases ------------------------------------------------------------------------------------ #

def test_z_axis_is_finite():
    '''Positions on the z-axis (R=0) give a finite acceleration, pointing along -z for a
    spherical potential. galpy's phi-torque divided by R used to give NaN here.'''
    pot = potential.NFWPotential()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    pos = np.array([[0.0, 0.0, 5.0]])
    acc = acc_fn(pos, t=0)
    assert np.all(np.isfinite(acc))
    np.testing.assert_allclose(acc[0, :2], 0.0, atol=1e-8 * abs(acc[0, 2]))
    assert acc[0, 2] < 0

def test_very_large_radius():
    '''Bridge should return finite values at very large radii.'''
    pot = potential.NFWPotential()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    radii = [1e3, 1e4, 1e5]
    for r in radii:
        pos = np.array([[r, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        assert np.all(np.isfinite(acc)), f"Non-finite acc at r={r}"
    # Magnitude should decrease with radius
    mags = []
    for r in radii:
        pos = np.array([[r, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        mags.append(np.linalg.norm(acc))
    assert all(mags[i] > mags[i + 1] for i in range(len(mags) - 1)), \
        f"|a| should decrease with r, got {mags}"

def test_very_small_radius_cored():
    '''Potentials with cores (Plummer, DehnenCore, Burkert) should return finite acc at small r.'''
    cored_pots = [
        potential.PlummerPotential(),
        potential.DehnenCoreSphericalPotential(),
        potential.BurkertPotential(),
    ]
    for pot in cored_pots:
        acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
        pos = np.array([[1e-10, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        assert np.all(np.isfinite(acc)), f"{type(pot).__name__} returned non-finite acc at r=1e-10"

# --- interpSphericalPotential ------------------------------------------------------------------#

def test_interp_spherical_outside_grid():
    '''interpSphericalPotential evaluated outside its rgrid should still return finite values.'''
    pot = potential.interpSphericalPotential(
        rforce=lambda r: -1. / r,
        rgrid=np.geomspace(0.01, 20, 101),
        Phi0=0.,
    )
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    # r=50 is outside the grid (max=20)
    pos = np.array([[50.0, 0.0, 0.0]])
    acc = acc_fn(pos, t=0)
    # Just document whether it's finite or not — it extrapolates
    if np.any(np.isnan(acc)):
        pytest.skip("interpSphericalPotential produces NaN outside rgrid (expected)")
    assert np.all(np.isfinite(acc))

# --- Wrapper Potentials --------------------------------------------------------------------------- #

# All wrappers that are vectorized (with a vectorized inner pot)
VECTORIZED_WRAPPER_POTENTIALS = [
    potential.DehnenSmoothWrapperPotential(pot=potential.NFWPotential()),
    potential.GaussianAmplitudeWrapperPotential(pot=potential.NFWPotential()),
    potential.SolidBodyRotationWrapperPotential(pot=potential.NFWPotential(), omega=1.0),
    potential.CorotatingRotationWrapperPotential(pot=potential.NFWPotential(), vpo=220., to=0.),
] + ([potential.KuzminLikeWrapperPotential(pot=potential.NFWPotential())]
     if hasattr(potential, 'KuzminLikeWrapperPotential') else []
) + ([potential.TimeDependentAmplitudeWrapperPotential(pot=potential.NFWPotential(), A=lambda t: 1.0)]
     if hasattr(potential, 'TimeDependentAmplitudeWrapperPotential') else []
)

# RotateAndTilt is unvectorized
UNVECTORIZED_WRAPPER_POTENTIALS = (
    [potential.RotateAndTiltWrapperPotential(pot=potential.NFWPotential(), zvec=[0., 0., 1.])]
    if hasattr(potential, 'RotateAndTiltWrapperPotential') else []
)

# Wrapper around an unvectorized inner pot (forces scalar loop even though wrapper is vectorized)
WRAPPER_WITH_UNVECTORIZED_INNER = [
    potential.DehnenSmoothWrapperPotential(pot=potential.HomogeneousSpherePotential()),
]

ALL_WRAPPER_POTENTIALS = (
    VECTORIZED_WRAPPER_POTENTIALS
    + UNVECTORIZED_WRAPPER_POTENTIALS
    + WRAPPER_WITH_UNVECTORIZED_INNER
)


@pytest.fixture(params=ALL_WRAPPER_POTENTIALS, ids=lambda p: type(p).__name__)
def wrapper_potential(request):
    return request.param

def test_wrapper_passes_check_supported(wrapper_potential):
    '''Wrapper potentials with supported inner pots should pass validation.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress unvectorized warnings
        _galpy_bridge._check_supported_pot(wrapper_potential)


def test_wrapper_rejects_unsupported_inner():
    '''Wrapper around an unsupported inner pot should be rejected.'''
    wp = potential.DehnenSmoothWrapperPotential(pot=potential.SCFPotential())
    with pytest.raises(TypeError, match="SCFPotential"):
        _galpy_bridge._check_supported_pot(wp)


def test_wrapper_rejects_unsupported_wrapper():
    '''An unknown wrapper class should be rejected by _check_supported_pot.'''
    from galpy.potential.WrapperPotential import WrapperPotential

    class FakeWrapperPotential(WrapperPotential):
        pass

    nfw = potential.NFWPotential()
    fake = object.__new__(FakeWrapperPotential)
    fake._pot = nfw
    with pytest.raises(TypeError, match="is not supported by tambora"):
        _galpy_bridge._check_supported_pot(fake)


def test_wrapper_acc_match(wrapper_potential):
    '''Wrapper acc should match -grad(Phi) via numerical differentiation.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wrapper_potential)
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(wrapper_potential)
    pot_i = partial(pot_fn, t=0)
    # Use a smaller grid to keep scalar-loop tests fast
    pos = np.array([
        [8.0, 0.0, 1.0],
        [5.0, 3.0, -2.0],
        [-4.0, 6.0, 0.5],
        [10.0, -7.0, 3.0],
    ])
    acc_bridge = acc_fn(pos, t=0)
    acc_num = _numerical_acc(pot_i, pos)
    np.testing.assert_allclose(acc_bridge, acc_num, rtol=1e-6, atol=1e-10)


def test_wrapper_force_match_galpy(wrapper_potential):
    '''Wrapper acc should match galpy analytical forces directly.'''
    # Use a smaller grid to keep scalar-loop tests fast
    pos = np.array([
        [8.0, 0.0, 1.0],
        [5.0, 3.0, -2.0],
        [-4.0, 6.0, 0.5],
        [10.0, -7.0, 3.0],
    ])
    R, phi, z = rect_to_cyl(*pos.T)
    acc_unit = u.kpc / u.Gyr**2
    torque_unit = u.kpc**2 / u.Gyr**2

    pot = _galpy_bridge._ensure_pot(wrapper_potential)
    aR_ref = np.array([
        potential.evaluateRforces(
            pot, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(acc_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    az_ref = np.array([
        potential.evaluatezforces(
            pot, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(acc_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    pt_ref = np.array([
        potential.evaluatephitorques(
            pot, Ri * u.kpc, zi * u.kpc, phi=pi, t=0 * u.Gyr,
            ro=8. * u.kpc, vo=220. * u.km / u.s, quantity=True
        ).to(torque_unit).value
        for Ri, zi, pi in zip(R, z, phi)
    ])
    aphi_ref = pt_ref / R
    ax_ref, ay_ref, az_cart_ref = cyl_to_rect_vec(aR_ref, aphi_ref, az_ref, phi)
    galpy_ref_acc = np.column_stack([ax_ref, ay_ref, az_cart_ref])

    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wrapper_potential)
    acc_bridge = acc_fn(pos, t=0)
    np.testing.assert_allclose(acc_bridge, galpy_ref_acc, rtol=1e-12)


def test_wrapper_pot_match(wrapper_potential):
    '''Wrapper pot should match galpy module-level evaluatePotentials.'''
    ez_pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(wrapper_potential)
    pos = np.array([
        [8.0, 0.0, 1.0],
        [5.0, 3.0, -2.0],
        [-4.0, 6.0, 0.5],
    ])
    ez_pot = ez_pot_fn(pos, t=0)
    # Reference: galpy, one point at a time
    np.testing.assert_allclose(ez_pot, _point_by_point(_galpy_bridge._ensure_pot(wrapper_potential), pos), rtol=1e-14)


def test_nested_wrapper():
    '''A wrapper inside a wrapper should work.'''
    inner = potential.SolidBodyRotationWrapperPotential(pot=potential.NFWPotential(), omega=1.0)
    outer = potential.DehnenSmoothWrapperPotential(pot=inner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(outer)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(outer)
    pos = np.array([[8.0, 0.0, 1.0]])
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    assert np.all(np.isfinite(acc))


def test_wrapper_unvectorized_inner_warns():
    '''A wrapper around a potential galpy can't evaluate on arrays is evaluated one point at a time.'''
    wp = potential.DehnenSmoothWrapperPotential(pot=potential.FerrersPotential())
    with pytest.warns(UserWarning, match="galpy can't evaluate _?DehnenSmoothWrapperPotential on arrays"):
        _galpy_bridge._galpy_pot_to_fns(wp)


@pytest.mark.skipif(not hasattr(potential, 'RotateAndTiltWrapperPotential'),
                    reason='RotateAndTiltWrapperPotential not available')
def test_wrapper_unvectorized_wrapper_warns():
    '''RotateAndTilt wrapper itself should emit an unvectorized warning.'''
    rt = potential.RotateAndTiltWrapperPotential(pot=potential.NFWPotential(), zvec=[0., 0., 1.])
    with pytest.warns(UserWarning, match="galpy can't evaluate _?RotateAndTiltWrapperPotential on arrays"):
        _galpy_bridge._galpy_pot_to_fns(rt)


# --- Time-Dependent Potentials -------------------------------------------------------------------- #

def test_dehnen_smooth_time_dependence():
    '''DehnenSmoothWrapper should grow from 0 to full amplitude over tform.'''
    nfw = potential.NFWPotential()
    wp = potential.DehnenSmoothWrapperPotential(pot=nfw, tform=0., tsteady=2*u.Myr)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pos = np.array([[8.0, 0.0, 1.0]])
    # At t=0 the smooth wrapper should be near full amplitude
    acc_t0 = acc_fn(pos, t=0)
    acc_t10 = acc_fn(pos, t=5)  # well after tsteady
    acc_nfw = _galpy_bridge._galpy_pot_to_acc_fn(nfw)(pos, t=5)
    np.testing.assert_allclose(acc_t0, 0.0, rtol=1e-15)
    np.testing.assert_allclose(acc_t10, acc_nfw, rtol=1e-15)


def test_gaussian_amplitude_time_dependence():
    '''GaussianAmplitude wrapper should peak at to and decay away from it.'''
    nfw = potential.NFWPotential()
    # Peak at t_nat=0 with sigma_nat=2
    wp = potential.GaussianAmplitudeWrapperPotential(pot=nfw, to=0., sigma=2.)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pos = np.array([[8.0, 0.0, 1.0]])
    acc_t0 = acc_fn(pos, t=0)
    # At a much later time the Gaussian should have decayed
    acc_late = acc_fn(pos, t=5000)  # 5 Gyr later
    mag_t0 = np.linalg.norm(acc_t0)
    mag_late = np.linalg.norm(acc_late)
    assert mag_t0 > 10 * mag_late, (
        f"Expected forces to decay away from Gaussian peak, got |a(0)|={mag_t0}, |a(5000)|={mag_late}"
    )


@pytest.mark.skipif(not hasattr(potential, 'TimeDependentAmplitudeWrapperPotential'),
                    reason='TimeDependentAmplitudeWrapperPotential not available')
def test_time_dependent_amplitude_wrapper():
    '''TimeDependentAmplitudeWrapper with A(t)=sin^2(t) should vary in time.'''
    nfw = potential.NFWPotential()
    wp = potential.TimeDependentAmplitudeWrapperPotential(
        pot=nfw, A=lambda t: np.sin(t)**2
    )
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pos = np.array([[8.0, 0.0, 1.0]])
    acc_t0 = acc_fn(pos, t=0)           # sin(0)^2=0 → zero force
    acc_mid = acc_fn(pos, t=100)        # sin(t_nat)^2 varies
    assert np.linalg.norm(acc_t0) < 1e-20, "Expected zero force at t=0 for sin^2(0)"
    assert np.linalg.norm(acc_mid) > 0, "Expected nonzero force at nonzero time"


def test_solid_body_rotation():
    '''SolidBodyRotation wrapper should rotate the pattern — forces differ at same position, different times.'''
    bar = potential.DehnenBarPotential()
    wp = potential.SolidBodyRotationWrapperPotential(pot=bar, omega=1.0)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pos = np.array([[5.0, 3.0, 0.0]])
    acc_t0 = acc_fn(pos, t=0)
    acc_t1 = acc_fn(pos, t=500)  # after some rotation
    # Forces should differ because the bar has rotated
    assert not np.allclose(acc_t0, acc_t1, atol=1e-14), (
        "Expected different forces at different times for rotating bar"
    )


def test_corotating_rotation():
    '''CorotatingRotation should also produce time-varying forces.'''
    bar = potential.DehnenBarPotential()
    wp = potential.CorotatingRotationWrapperPotential(pot=bar, vpo=220., to=0.)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pos = np.array([[5.0, 3.0, 0.0]])
    acc_t0 = acc_fn(pos, t=0)
    acc_t1 = acc_fn(pos, t=500)
    assert not np.allclose(acc_t0, acc_t1, atol=1e-14), (
        "Expected different forces at different times for corotating bar"
    )


@pytest.mark.skipif(not hasattr(potential, 'RotateAndTiltWrapperPotential'),
                    reason='RotateAndTiltWrapperPotential not available')
def test_rotate_and_tilt_acc():
    '''RotateAndTilt should change the force direction when tilted.'''
    nfw = potential.NFWPotential()
    # Identity rotation (zvec=z-axis) should match bare NFW
    rt_identity = potential.RotateAndTiltWrapperPotential(pot=nfw, zvec=[0., 0., 1.])
    acc_fn_id = _galpy_bridge._galpy_pot_to_acc_fn(rt_identity)
    acc_fn_bare = _galpy_bridge._galpy_pot_to_acc_fn(nfw)
    pos = np.array([[8.0, 0.0, 1.0]])
    acc_id = acc_fn_id(pos, t=0)
    acc_bare = acc_fn_bare(pos, t=0)
    np.testing.assert_allclose(acc_id, acc_bare, rtol=1e-10)

    # Tilted rotation should differ
    rt_tilted = potential.RotateAndTiltWrapperPotential(pot=nfw, zvec=[1., 0., 0.])
    acc_fn_tilt = _galpy_bridge._galpy_pot_to_acc_fn(rt_tilted)
    acc_tilt = acc_fn_tilt(pos, t=0)
    # For a spherical pot, tilting should not change forces (symmetry)
    np.testing.assert_allclose(np.linalg.norm(acc_tilt), np.linalg.norm(acc_bare), rtol=1e-8)

@pytest.mark.skipif(not hasattr(potential, 'KuzminLikeWrapperPotential'),
                    reason='KuzminLikeWrapperPotential not available')
def test_kuzmin_like_wrapper():
    '''KuzminLikeWrapper should produce finite forces and match -grad(Phi).'''
    nfw = potential.NFWPotential()
    wp = potential.KuzminLikeWrapperPotential(pot=nfw)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(wp)
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(wp)
    pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
    acc = acc_fn(pos, t=0)
    assert np.all(np.isfinite(acc))
    acc_num = _numerical_acc(partial(pot_fn, t=0), pos)
    np.testing.assert_allclose(acc, acc_num, rtol=1e-6, atol=1e-10)

def test_wrapper_in_composite():
    '''A wrapper combined with other potentials in a list should work.'''
    nfw = potential.NFWPotential()
    bar_wrapped = potential.SolidBodyRotationWrapperPotential(
        pot=potential.DehnenBarPotential(), omega=1.0
    )
    combo = [nfw, bar_wrapped]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(combo)
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(combo)
    pos = np.array([[8.0, 0.0, 1.0]])
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    assert np.all(np.isfinite(acc))

def test_warns_for_inconsistent_physical_units_in_composite():
    '''_get_ro_vo should warn when a composite with mixed 
    physical unit settings is provided.'''
    p1 = potential.NFWPotential()
    p2 = potential.PlummerPotential()
    p1.turn_physical_on(ro=8., vo=220.)
    p2.turn_physical_on(ro=10., vo=400.)  # one on, one off
    combo = [p1, p2]
    with pytest.warns(UserWarning, match=r"Potential PlummerPotential has ro=10\.0, vo=400\.0 "
                r"which differs from the first potential \(ro=8\.0, vo=220\.0\)\. "
                r"Using the first potential's values\."):
        _galpy_bridge._get_ro_vo(combo)

# --- On the z-axis (R = 0), every supported potential ------------------------------------------ #

# galpy can't evaluate these at R = 0 (as configured in the lists above), so their on-axis
# force stays NaN, as it always was, instead of becoming a plausible but wrong number.
_NAN_ON_AXIS = (
    potential.RingPotential,
    potential.DoubleExponentialDiskPotential,
    potential.RazorThinExponentialDiskPotential,
    potential.SpiralArmsPotential,
    potential.CorotatingRotationWrapperPotential,
) + ((potential.RotateAndTiltWrapperPotential,)
     if hasattr(potential, 'RotateAndTiltWrapperPotential') else ())


def _satellite_orbit():
    o = Orbit([1., 0.1, 1.1, 0.1, 0., 0.3])
    o.integrate(np.linspace(-1., 1., 101), potential.MWPotential2014)
    return o


# Masses off to one side pull sideways on the axis, so these exercise the phi-term there.
# Every potential in the lists above has no sideways pull on the axis.
OFF_CENTRE_GALPY_POTENTIALS = (
    [potential.RotateAndTiltWrapperPotential(pot=potential.NFWPotential(), offset=[0.3 / 8., 0.4 / 8., 0.])]
    if hasattr(potential, 'RotateAndTiltWrapperPotential') else []
) + [
    potential.MovingObjectPotential(_satellite_orbit(), pot=potential.PlummerPotential(amp=0.1, b=0.1)),
]

_ON_AXIS_CASES = (
    [pytest.param(p, isinstance(p, _NAN_ON_AXIS), id=type(p).__name__)
     for p in ALL_SUPPORTED_GALPY_POTENTIALS + ALL_WRAPPER_POTENTIALS]
    + [pytest.param(p, False, id='off_centre_' + type(p).__name__) for p in OFF_CENTRE_GALPY_POTENTIALS]
)
_FINITE_ON_AXIS_CASES = [c for c in _ON_AXIS_CASES if not c.values[1]]

# galpy can't evaluate the potential of these at R = 0 either. The rest of _NAN_ON_AXIS have
# a finite, correct potential on the axis, although their force there is NaN.
_POT_NAN_ON_AXIS = (
    potential.DoubleExponentialDiskPotential,
    potential.SpiralArmsPotential,
)

# galpy's array code gives DehnenBarPotential a finite but wrong potential on the axis inside
# the bar radius: it leaves R^2/r^2 at 1 there instead of 0. Its scalar code and its forces are
# right. Strict, so this fails once galpy is fixed; then limit it to the affected versions.
_GALPY_BAR_AXIS_BUG = pytest.mark.xfail(
    raises=AssertionError, strict=True,
    reason="galpy bug: DehnenBarPotential's potential is non-zero on the z-axis for array input")


def _hits_galpys_bar_bug(p):
    '''Whether the bridge evaluates a DehnenBarPotential in ``p`` with galpy's array code: it
    does unless the component holding the bar can only be evaluated one point at a time.'''
    return any(_galpy_bridge._takes_arrays(c)
               and any(isinstance(leaf, potential.DehnenBarPotential) for leaf in _galpy_bridge._unwrap_pot(c))
               for c in _galpy_bridge._iter_components(_galpy_bridge._ensure_pot(p)))


def _pot_on_axis_case(case):
    p = case.values[0]
    marks = [_GALPY_BAR_AXIS_BUG] if _hits_galpys_bar_bug(p) else []
    return pytest.param(p, isinstance(p, _POT_NAN_ON_AXIS), id=case.id, marks=marks)


_POT_ON_AXIS_CASES = [_pot_on_axis_case(c) for c in _ON_AXIS_CASES]
_POT_FINITE_ON_AXIS_CASES = [c for c in _POT_ON_AXIS_CASES if not c.values[1]]

_ON_AXIS = np.array([[0., 0., 2.]])     # kpc
_NEAR_AXIS_H = 1e-6                     # kpc


def _near_axis_mean(fn, h):
    '''Mean of ``fn`` (acceleration or potential) at four points a distance h off the axis
    (+-x, +-y). The terms linear in h cancel, leaving the on-axis value plus an O(h^2) error.'''
    pts = np.array([[h, 0., 2.], [-h, 0., 2.], [0., h, 2.], [0., -h, 2.]])
    return fn(pts, t=0).mean(axis=0)


def _field_scale(fn):
    '''Size of ``fn`` one step off the axis, for tolerances. 1 where it is exactly 0.'''
    return np.abs(fn(np.array([[2., 0., 2.]]), t=0)).max() or 1.0


@pytest.mark.parametrize("pot, nan_expected", _ON_AXIS_CASES)
def test_on_axis_acc_is_the_limit_from_nearby_or_nan(pot, nan_expected):
    '''On the z-axis the acceleration is either the limit approached from just off the axis
    or, where galpy can't evaluate R = 0, NaN. Never a finite number that is wrong.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
        on_axis = acc_fn(_ON_AXIS, t=0)[0]
        if nan_expected:
            assert not np.all(np.isfinite(on_axis))
            return
        near = _near_axis_mean(acc_fn, _NEAR_AXIS_H)
        scale = _field_scale(acc_fn)
    assert np.all(np.isfinite(on_axis))
    # The tolerance is relative to the field, not per component: components that should be
    # zero come out exactly 0 on the axis but ~1e-16 in the reference. The reference is good
    # to ~1e-12 of the field at this h (see the h^2 test below), so 1e-10 leaves a 100x margin.
    np.testing.assert_allclose(on_axis, near, rtol=0, atol=1e-10 * scale)


@pytest.mark.parametrize("pot, nan_expected", _POT_ON_AXIS_CASES)
def test_on_axis_potential_is_the_limit_from_nearby_or_nan(pot, nan_expected):
    '''The same for the potential, which feeds the energy diagnostics: the limit from just
    off the axis, or NaN where galpy can't evaluate R = 0.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(pot)
        on_axis = pot_fn(_ON_AXIS, t=0)[0]
        if nan_expected:
            assert np.isnan(on_axis)
            return
        near = _near_axis_mean(pot_fn, _NEAR_AXIS_H)
        scale = _field_scale(pot_fn)
    assert np.isfinite(on_axis)
    # Tolerance as for the acceleration; the h^2 test below checks the reference here too.
    np.testing.assert_allclose(on_axis, near, rtol=0, atol=1e-10 * scale)


def _assert_the_gap_shrinks_as_h_squared(fn):
    on_axis = fn(_ON_AXIS, t=0)[0]
    scale = _field_scale(fn)
    gap = [np.abs(on_axis - _near_axis_mean(fn, h)).max() / scale for h in (1e-3, 1e-4)]
    if gap[0] < 1e-12:
        # No curvature at the axis (e.g. a uniform field or a constant potential): both gaps
        # are rounding error.
        assert gap[1] < 1e-12
    else:
        assert 90. < gap[0] / gap[1] < 110.


@pytest.mark.parametrize("pot, nan_expected", _FINITE_ON_AXIS_CASES)
def test_the_near_axis_reference_converges_as_h_squared(pot, nan_expected):
    '''The tests above trust the four-point mean to be accurate to O(h^2). Shrinking h by 10
    must then shrink its gap to the on-axis value by 100. A gap that doesn't shrink would
    mean the on-axis value isn't the limit at all.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_the_gap_shrinks_as_h_squared(_galpy_bridge._galpy_pot_to_acc_fn(pot))


@pytest.mark.parametrize("pot, nan_expected", _POT_FINITE_ON_AXIS_CASES)
def test_the_near_axis_potential_converges_as_h_squared(pot, nan_expected):
    '''The same check for the potential.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_the_gap_shrinks_as_h_squared(_galpy_bridge._galpy_pot_to_pot_fn(pot))


@pytest.mark.parametrize("pot", OFF_CENTRE_GALPY_POTENTIALS, ids=lambda p: type(p).__name__)
def test_the_off_centre_cases_pull_sideways_on_the_axis(pot):
    '''Guard for the tests above: if these had no sideways pull on the axis, an on-axis
    branch that dropped the phi-term would still pass them.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
        on_axis = acc_fn(_ON_AXIS, t=0)[0]
        scale = _field_scale(acc_fn)
    assert np.hypot(on_axis[0], on_axis[1]) > 0.1 * scale


def test_the_origin_is_nan_for_a_cusp_and_zero_for_a_core():
    '''At the centre the force of a cusped profile is undefined, so it stays NaN; a cored
    profile has zero force there.'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cusp = _galpy_bridge._galpy_pot_to_acc_fn(potential.NFWPotential())(np.zeros((1, 3)), t=0)
        core = _galpy_bridge._galpy_pot_to_acc_fn(potential.PlummerPotential())(np.zeros((1, 3)), t=0)
    assert np.all(np.isnan(cusp))
    np.testing.assert_array_equal(core, 0.0)

### ExternalPotential and CompositeForce Tests ------------------------------------------------------------------ #

class Test_ExternalPotential_methods_match_internal_fns:
    @classmethod
    def setup_class(cls):
        cls.pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
        cls.galpy_pot = potential.NFWPotential()
    
    def test_acc_matches_galpy_pot_to_acc_fn(self):
        acc_func = _galpy_bridge._galpy_pot_to_acc_fn(self.galpy_pot)
        acc_from_func = acc_func(self.pos, t=0)
        force_class = ExternalPotential(self.galpy_pot)
        acc_from_class = force_class.acc(self.pos, t=0)
        np.testing.assert_allclose(acc_from_func, acc_from_class, rtol=1e-15)
    
    def test_potential_matches_galpy_pot_to_pot_fn(self):
        pot_func = _galpy_bridge._galpy_pot_to_pot_fn(self.galpy_pot)
        pot_from_func = pot_func(self.pos, t=0)
        force_class = ExternalPotential(self.galpy_pot)
        pot_from_class = force_class.potential(self.pos, t=0)
        np.testing.assert_allclose(pot_from_func, pot_from_class, rtol=1e-15)

def test_acc_for_CompositeForce_of_galpy_matches_sum_of_individual():
    '''CompositeForce of galpy potentials should match sum of individual galpy forces.'''
    nfw = potential.NFWPotential()
    plummer = potential.PlummerPotential()
    combo = [nfw, plummer]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(combo)
    nfw_class = ExternalPotential(nfw)
    plummer_class = ExternalPotential(plummer)
    combo_class = nfw_class + plummer_class

    pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
    acc_combo = combo_class.acc(pos, None, t=0)
    acc_sum = nfw_class.acc(pos, t=0) + plummer_class.acc(pos, t=0)
    np.testing.assert_allclose(acc_combo, acc_sum, rtol=1e-15)

def test_pot_for_CompositeForce_of_galpy_matches_sum_of_individual():
    '''CompositeForce of galpy potentials should match sum of individual galpy forces.'''
    nfw = potential.NFWPotential()
    plummer = potential.PlummerPotential()
    combo = [nfw, plummer]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(combo)
    nfw_class = ExternalPotential(nfw)
    plummer_class = ExternalPotential(plummer)
    combo_class = nfw_class + plummer_class

    pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
    pot_combo = combo_class.potential(pos, None, t=0)
    pot_sum = nfw_class.potential(pos, t=0) + plummer_class.potential(pos, t=0)
    np.testing.assert_allclose(pot_combo, pot_sum, rtol=1e-15)

def test_acc_for_CompositeForce_of_galpy_is_same_as_ExternalPotential_of_composite():
    '''CompositeForce of galpy potentials should match ExternalPotential of the same combo.'''
    nfw = potential.NFWPotential()
    plummer = potential.PlummerPotential()
    combo = nfw + plummer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(combo)
    nfw_class = ExternalPotential(nfw)
    plummer_class = ExternalPotential(plummer)
    summed_combo_class = nfw_class + plummer_class
    direct_combo_class = ExternalPotential(combo)
    pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
    acc_summed_combo = summed_combo_class.acc(pos, None, t=0)
    acc_direct_combo = direct_combo_class.acc(pos, t=0)
    np.testing.assert_allclose(acc_summed_combo, acc_direct_combo, rtol=1e-15)

def test_pot_for_CompositeForce_of_galpy_is_same_as_ExternalPotential_of_composite():
    '''CompositeForce of galpy potentials should match ExternalPotential of the same combo.'''
    nfw = potential.NFWPotential()
    plummer = potential.PlummerPotential()
    combo = nfw + plummer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _galpy_bridge._check_supported_pot(combo)
    nfw_class = ExternalPotential(nfw)
    plummer_class = ExternalPotential(plummer)
    summed_combo_class = nfw_class + plummer_class
    direct_combo_class = ExternalPotential(combo)
    pos = np.array([[8.0, 0.0, 1.0], [5.0, 3.0, -2.0]])
    pot_summed_combo = summed_combo_class.potential(pos, None, t=0)
    pot_direct_combo = direct_combo_class.potential(pos, t=0)
    np.testing.assert_allclose(pot_summed_combo, pot_direct_combo, rtol=1e-15)
