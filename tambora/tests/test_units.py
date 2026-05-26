"""
Tests for unit conversion system.

Verifies:
- Conversion constants match astropy
- unit_handler decorator applies correct factors
- return_internal=True bypasses conversion
- add_particles converts velocity input correctly
- All accessor categories return correct user units
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ezfalcon.simulation import Sim
from ezfalcon.util.units import (
    G_INTERNAL,
    G_KPC_KMS,
    KMS_TO_KPCGYR,
    KPCGYR_TO_KMS,
    KMS_TO_KPCMYR,
    KPCMYR_TO_KMS,
    KM_TO_KPC,
    GYR_TO_MYR,
    INTERNAL_TO_USER_UNITS,
    unit_handler,
)

astropy = pytest.importorskip("astropy")
import astropy.units as u
import astropy.constants as const


# --------------------------------------------------------------------------- #
#  Conversion-constant validation against astropy
# --------------------------------------------------------------------------- #

class TestConversionConstantsAgainstAstropy:
    """Validate hardcoded conversion factors against astropy."""

    def test_kms_to_kpcmyr(self):
        one_km_s = (1.0 * u.km / u.s).to(u.kpc / u.Myr).value
        assert_allclose(KMS_TO_KPCMYR, one_km_s, rtol=1e-10)

    def test_kpcmyr_to_kms(self):
        one_kpc_myr = (1.0 * u.kpc / u.Myr).to(u.km / u.s).value
        assert_allclose(KPCMYR_TO_KMS, one_kpc_myr, rtol=1e-10)

    def test_kms_kpcmyr_inverse(self):
        assert_allclose(KMS_TO_KPCMYR * KPCMYR_TO_KMS, 1.0, atol=1e-15)

    def test_km_to_kpc(self):
        one_km_in_kpc = (1.0 * u.km).to(u.kpc).value
        assert_allclose(KM_TO_KPC, one_km_in_kpc, rtol=1e-10)

    def test_gyr_to_myr(self):
        assert GYR_TO_MYR == 1000.0

    def test_kms_to_kpcgyr(self):
        one_km_s = (1.0 * u.km / u.s).to(u.kpc / u.Gyr).value
        assert_allclose(KMS_TO_KPCGYR, one_km_s, rtol=1e-10)

    def test_kpcgyr_to_kms(self):
        one_kpc_gyr = (1.0 * u.kpc / u.Gyr).to(u.km / u.s).value
        assert_allclose(KPCGYR_TO_KMS, one_kpc_gyr, rtol=1e-10)

    def test_kms_kpcgyr_inverse(self):
        assert_allclose(KMS_TO_KPCGYR * KPCGYR_TO_KMS, 1.0, atol=1e-15)

    def test_G_internal(self):
        """G in kpc^3 Msun^-1 Gyr^-2."""
        G_astropy = const.G.to(u.kpc**3 / u.Msun / u.Gyr**2).value
        assert_allclose(G_INTERNAL, G_astropy, rtol=1e-15)

    def test_G_kpc_kms(self):
        """G in kpc (km/s)^2 Msun^-1."""
        G_astropy = const.G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
        assert_allclose(G_KPC_KMS, G_astropy, rtol=1e-15)

    def test_G_internal_vs_G_kpc_kms(self):
        """G_INTERNAL = G_KPC_KMS * KMS_TO_KPCGYR^2."""
        assert_allclose(G_INTERNAL, G_KPC_KMS * KMS_TO_KPCGYR**2, rtol=1e-15)


# --------------------------------------------------------------------------- #
#  INTERNAL_TO_USER_UNITS factor consistency
# --------------------------------------------------------------------------- #

class TestInternalToUserFactors:
    """Check that INTERNAL_TO_USER_UNITS factors are self-consistent."""

    def test_length_identity(self):
        assert INTERNAL_TO_USER_UNITS['length'] == 1.0

    def test_mass_identity(self):
        assert INTERNAL_TO_USER_UNITS['mass'] == 1.0

    def test_time_identity(self):
        assert INTERNAL_TO_USER_UNITS['time'] == 1.0

    def test_velocity_factor(self):
        assert_allclose(
            INTERNAL_TO_USER_UNITS['velocity'], KPCGYR_TO_KMS, rtol=1e-15
        )

    def test_velocity_factor_astropy(self):
        expected = (1.0 * u.kpc / u.Gyr).to(u.km / u.s).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['velocity'], expected, rtol=1e-10
        )

    def test_momentum_factor(self):
        """Msun kpc/Gyr -> Msun km/s."""
        assert_allclose(
            INTERNAL_TO_USER_UNITS['momentum'], KPCGYR_TO_KMS, rtol=1e-15
        )

    def test_momentum_factor_astropy(self):
        """Msun kpc/Gyr -> Msun km/s."""
        expected = (1.0 * u.Msun * u.kpc / u.Gyr).to(u.Msun * u.km / u.s).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['momentum'], expected, rtol=1e-10
        )

    def test_energy_factor(self):
        """Msun (kpc/Gyr)^2 -> Msun (km/s)^2."""
        assert_allclose(
            INTERNAL_TO_USER_UNITS['energy'], KPCGYR_TO_KMS**2, rtol=1e-15
        )

    def test_energy_factor_astropy(self):
        """Msun (kpc/Gyr)^2 -> Msun (km/s)^2."""
        expected = (1.0 * u.Msun * (u.kpc / u.Gyr)**2).to(u.Msun * (u.km / u.s)**2).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['energy'], expected, rtol=1e-10
        )

    def test_acceleration_factor(self):
        """kpc/Gyr^2 -> km/s/Gyr  (factor = KPCGYR_TO_KMS)."""
        assert_allclose(
            INTERNAL_TO_USER_UNITS['acceleration'],
            KPCGYR_TO_KMS,
            rtol=1e-15,
        )

    def test_acceleration_factor_astropy(self):
        """kpc/Gyr^2 -> km/s/Gyr  (factor = KPCGYR_TO_KMS)."""
        expected = (1.0 * u.kpc / u.Gyr**2).to(u.km / u.s / u.Gyr).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['acceleration'], expected, rtol=1e-10
        )


    def test_angular_momentum_factor(self):
        """Msun kpc^2/Gyr -> Msun kpc km/s  (factor = KPCGYR_TO_KMS)."""
        assert_allclose(
            INTERNAL_TO_USER_UNITS['angular_momentum'],
            KPCGYR_TO_KMS,
            rtol=1e-15,
        )

    def test_angular_momentum_factor_astropy(self):
        """Msun kpc^2/Gyr -> Msun kpc km/s  (factor = KPCGYR_TO_KMS)."""
        expected = (1.0 * u.Msun * u.kpc**2 / u.Gyr).to(u.Msun * u.kpc * u.km / u.s).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['angular_momentum'], expected, rtol=1e-10
        )

    def test_angular_velocity_factor(self):
        """1/Gyr -> km/s/kpc  (factor = KPCGYR_TO_KMS)."""
        assert_allclose(
            INTERNAL_TO_USER_UNITS['angular_velocity'], KPCGYR_TO_KMS, rtol=1e-15
        )

    def test_angular_velocity_factor_astropy(self):
        """1/Gyr -> km/s/kpc  (factor = KPCGYR_TO_KMS)."""
        expected = (1.0 / u.Gyr).to(u.km / u.s / u.kpc).value
        assert_allclose(
            INTERNAL_TO_USER_UNITS['angular_velocity'], expected, rtol=1e-10
        )

    def test_angle_identity(self):
        assert INTERNAL_TO_USER_UNITS['angle'] == 1.0


# --------------------------------------------------------------------------- #
#  unit_handler decorator behaviour
# --------------------------------------------------------------------------- #

class TestUnitHandlerDecorator:
    """Test the unit_handler decorator mechanics."""

    def test_decorator_applies_factor(self):
        @unit_handler('velocity')
        def dummy():
            return np.array([1.0, 2.0, 3.0])

        result = dummy()
        assert_allclose(result, np.array([1.0, 2.0, 3.0]) * KPCGYR_TO_KMS)

    def test_return_internal_bypasses(self):
        @unit_handler('velocity')
        def dummy():
            return np.array([1.0, 2.0, 3.0])

        result = dummy(return_internal=True)
        assert_allclose(result, np.array([1.0, 2.0, 3.0]))

    def test_identity_factor_unchanged(self):
        @unit_handler('length')
        def dummy():
            return np.array([5.0])

        assert_allclose(dummy(), np.array([5.0]))
        assert_allclose(dummy(return_internal=True), np.array([5.0]))

    def test_scalar_input(self):
        @unit_handler('velocity')
        def dummy():
            return 1.0

        assert_allclose(dummy(), KPCGYR_TO_KMS)

    def test_preserves_function_name(self):
        @unit_handler('velocity')
        def my_func():
            return 1.0

        assert my_func.__name__ == 'my_func'

    def test_passes_through_kwargs(self):
        @unit_handler('velocity')
        def dummy(a, b=10):
            return a + b

        assert_allclose(dummy(1.0, b=2.0), 3.0 * KPCGYR_TO_KMS)

    def test_invalid_unit_key_raises(self):
        with pytest.raises(KeyError):
            @unit_handler('invalid_key')
            def dummy():
                return 1.0


# --------------------------------------------------------------------------- #
#  Sim accessor conversion round-trips
# --------------------------------------------------------------------------- #

@pytest.fixture
def sim():
    np.random.seed(99)
    s = Sim()
    s.add_particles(
        'stars',
        pos=np.random.rand(20, 3),
        vel=np.random.rand(20, 3) * 100,  # 0-100 km/s
        mass=np.random.rand(20) * 1e6,
    )
    return s


class TestSimAccessorConversion:
    """Verify that every accessor category obeys:
       user_value == internal_value * factor
    """

    # --- length (factor = 1, trivial but worth checking) --------

    @pytest.mark.parametrize("method", ["pos", "x", "y", "z", "r", "cylR"])
    def test_length_accessors(self, sim, method):
        user = getattr(sim, method)(t=0)
        internal = getattr(sim, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['length'])

    # --- velocity -----------------------------------------------

    @pytest.mark.parametrize(
        "method", ["vel", "vx", "vy", "vz", "vr", "vtheta", "cylvR"]
    )
    def test_velocity_accessors(self, sim, method):
        user = getattr(sim, method)(t=0)
        internal = getattr(sim, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['velocity'],
                        rtol=1e-14)

    # --- momentum -----------------------------------------------

    @pytest.mark.parametrize("method", ["p", "px", "py", "pz"])
    def test_momentum_accessors(self, sim, method):
        user = getattr(sim, method)(t=0)
        internal = getattr(sim, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['momentum'],
                        rtol=1e-14)

    # --- angular momentum ---------------------------------------

    @pytest.mark.parametrize("method", ["L", "Lx", "Ly", "Lz"])
    def test_angular_momentum_accessors(self, sim, method):
        user = getattr(sim, method)(t=0)
        internal = getattr(sim, method)(t=0, return_internal=True)
        assert_allclose(
            user, internal * INTERNAL_TO_USER_UNITS['angular_momentum'],
            rtol=1e-14,
        )

    # --- angular velocity (vphi) --------------------------------

    def test_vphi_accessor(self, sim):
        user = sim.vphi(t=0)
        internal = sim.vphi(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['angular_velocity'],
                        rtol=1e-14)

    # --- energy -------------------------------------------------

    def test_KE(self, sim):
        user = sim.KE(t=0)
        internal = sim.KE(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- acceleration (self-gravity) ----------------------------

    def test_self_gravity(self, sim):
        user = sim.self_gravity(t=0, method='direct', eps=0.01)
        internal = sim.self_gravity(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(
            user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
            rtol=1e-14,
        )

    @pytest.mark.parametrize("method", ["self_ax", "self_ay", "self_az"])
    def test_self_acc_components(self, sim, method):
        user = getattr(sim, method)(t=0, method='direct', eps=0.01)
        internal = getattr(sim, method)(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(
            user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
            rtol=1e-14,
        )

    # --- self potential (energy) --------------------------------

    def test_self_potential(self, sim):
        user = sim.self_potential(t=0, method='direct', eps=0.01)
        internal = sim.self_potential(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- PE (total potential energy) ----------------------------

    def test_PE(self, sim):
        user = sim.PE(t=0, method='direct', eps=0.01)
        internal = sim.PE(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- energy (total = KE + PE) -------------------------------

    def test_energy(self, sim):
        user = sim.energy(t=0, method='direct', eps=0.01)
        internal = sim.energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- system_energy (scalar sum) -----------------------------

    def test_system_energy(self, sim):
        user = sim.system_energy(t=0, method='direct', eps=0.01)
        internal = sim.system_energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)


# --------------------------------------------------------------------------- #
#  External-potential accessor conversion round-trips
# --------------------------------------------------------------------------- #

@pytest.fixture
def sim_ext():
    """Sim with an external galpy potential attached."""
    from galpy import potential as gp
    np.random.seed(99)
    s = Sim()
    s.add_particles(
        'stars',
        pos=np.random.rand(10, 3) + 1.0,   # offset from origin to avoid R=0
        vel=np.random.rand(10, 3) * 100,
        mass=np.random.rand(10) * 1e6,
    )
    s.add_external_pot(gp.PlummerPotential())
    return s


class TestExternalAccessorConversion:
    """Verify external potential / acceleration accessors obey
       user_value == internal_value * factor."""

    def test_compute_external_pot(self, sim_ext):
        user = sim_ext.compute_external_pot(t=0)
        internal = sim_ext.compute_external_pot(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_external_acc(self, sim_ext):
        user = sim_ext.external_acc(t=0)
        internal = sim_ext.external_acc(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    @pytest.mark.parametrize("method", ["external_ax", "external_ay", "external_az"])
    def test_external_acc_components(self, sim_ext, method):
        user = getattr(sim_ext, method)(t=0)
        internal = getattr(sim_ext, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    def test_PE_with_external(self, sim_ext):
        user = sim_ext.PE(t=0, method='direct', eps=0.01)
        internal = sim_ext.PE(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_energy_with_external(self, sim_ext):
        user = sim_ext.energy(t=0, method='direct', eps=0.01)
        internal = sim_ext.energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_system_energy_with_external(self, sim_ext):
        user = sim_ext.system_energy(t=0, method='direct', eps=0.01)
        internal = sim_ext.system_energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)


# --------------------------------------------------------------------------- #
#  Component accessor conversion round-trips
# --------------------------------------------------------------------------- #

class TestComponentAccessorConversion:
    """Same checks on Component objects."""

    @pytest.mark.parametrize("method", ["pos", "x", "y", "z", "r", "cylR"])
    def test_length_accessors(self, sim, method):
        comp = sim.stars
        user = getattr(comp, method)(t=0)
        internal = getattr(comp, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['length'])

    @pytest.mark.parametrize(
        "method", ["vel", "vx", "vy", "vz", "vr", "vtheta", "cylvR"]
    )
    def test_velocity_accessors(self, sim, method):
        comp = sim.stars
        user = getattr(comp, method)(t=0)
        internal = getattr(comp, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['velocity'],
                        rtol=1e-14)

    @pytest.mark.parametrize("method", ["p", "px", "py", "pz"])
    def test_momentum_accessors(self, sim, method):
        comp = sim.stars
        user = getattr(comp, method)(t=0)
        internal = getattr(comp, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['momentum'],
                        rtol=1e-14)

    @pytest.mark.parametrize("method", ["L", "Lx", "Ly", "Lz"])
    def test_angular_momentum_accessors(self, sim, method):
        comp = sim.stars
        user = getattr(comp, method)(t=0)
        internal = getattr(comp, method)(t=0, return_internal=True)
        assert_allclose(
            user, internal * INTERNAL_TO_USER_UNITS['angular_momentum'],
            rtol=1e-14,
        )

    def test_vphi_accessor(self, sim):
        comp = sim.stars
        user = comp.vphi(t=0)
        internal = comp.vphi(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['angular_velocity'],
                        rtol=1e-14)

    def test_KE(self, sim):
        comp = sim.stars
        user = comp.KE(t=0)
        internal = comp.KE(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- self potential (energy) --------------------------------

    def test_self_potential(self, sim):
        comp = sim.stars
        user = comp.self_potential(t=0, method='direct', eps=0.01)
        internal = comp.self_potential(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- PE (total potential energy) ----------------------------

    def test_PE(self, sim):
        comp = sim.stars
        user = comp.PE(t=0, method='direct', eps=0.01)
        internal = comp.PE(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- energy (total = KE + PE) -------------------------------

    def test_energy(self, sim):
        comp = sim.stars
        user = comp.energy(t=0, method='direct', eps=0.01)
        internal = comp.energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    # --- acceleration (self-gravity) ----------------------------

    def test_self_gravity(self, sim):
        comp = sim.stars
        user = comp.self_gravity(t=0, method='direct', eps=0.01)
        internal = comp.self_gravity(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    @pytest.mark.parametrize("method", ["self_ax", "self_ay", "self_az"])
    def test_self_acc_components(self, sim, method):
        comp = sim.stars
        user = getattr(comp, method)(t=0, method='direct', eps=0.01)
        internal = getattr(comp, method)(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)


class TestComponentExternalAccessorConversion:
    """Component accessor checks with an external potential."""

    def test_compute_external_pot(self, sim_ext):
        comp = sim_ext.stars
        user = comp.compute_external_pot(t=0)
        internal = comp.compute_external_pot(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_external_acc(self, sim_ext):
        comp = sim_ext.stars
        user = comp.external_acc(t=0)
        internal = comp.external_acc(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    @pytest.mark.parametrize("method", ["external_ax", "external_ay", "external_az"])
    def test_external_acc_components(self, sim_ext, method):
        comp = sim_ext.stars
        user = getattr(comp, method)(t=0)
        internal = getattr(comp, method)(t=0, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    def test_PE_with_external(self, sim_ext):
        comp = sim_ext.stars
        user = comp.PE(t=0, method='direct', eps=0.01)
        internal = comp.PE(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_energy_with_external(self, sim_ext):
        comp = sim_ext.stars
        user = comp.energy(t=0, method='direct', eps=0.01)
        internal = comp.energy(t=0, method='direct', eps=0.01, return_internal=True)
        assert_allclose(user, internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

# --------------------------------------------------------------------------- #
#  add_particles velocity input conversion
# --------------------------------------------------------------------------- #

class TestAddParticlesVelocityConversion:
    """Verify that user velocity input (km/s) is stored internally as kpc/Gyr."""

    def test_stored_velocity_is_internal(self):
        vel_kms = np.array([[100.0, 200.0, 300.0]])
        s = Sim()
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms,
                        mass=np.ones(1))
        # Internal storage should be kpc/Gyr
        assert_allclose(s._init_vel, vel_kms * KMS_TO_KPCGYR, rtol=1e-15)

    def test_accessor_returns_user_units(self):
        vel_kms = np.array([[100.0, 200.0, 300.0]])
        s = Sim()
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms,
                        mass=np.ones(1))
        # Default accessor should return km/s (round-trip)
        assert_allclose(s.vel(t=0), vel_kms, rtol=1e-14)

    def test_return_internal_gives_kpc_gyr(self):
        vel_kms = np.array([[100.0, 200.0, 300.0]])
        s = Sim()
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms,
                        mass=np.ones(1))
        assert_allclose(s.vel(t=0, return_internal=True),
                        vel_kms * KMS_TO_KPCGYR, rtol=1e-15)


# --------------------------------------------------------------------------- #
#  Astropy cross-validation of physical quantities
# --------------------------------------------------------------------------- #

class TestAstropyPhysicalConsistency:
    """Use astropy to independently convert known quantities and
    compare against the unit_handler output."""

    def test_velocity_km_s(self):
        """A particle with internal vel 1 kpc/Gyr should show ~0.978 km/s."""
        s = Sim()
        # Input 1 kpc/Gyr expressed in km/s
        vel_kms_input = np.array([[(1.0 * u.kpc / u.Gyr).to(u.km / u.s).value,
                                    0.0, 0.0]])
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms_input,
                        mass=np.ones(1))
        # Internal should be exactly 1 kpc/Gyr
        assert_allclose(s.vel(t=0, return_internal=True)[0, 0], 1.0, rtol=1e-10)
        # User output should match astropy conversion
        expected_kms = (1.0 * u.kpc / u.Gyr).to(u.km / u.s).value
        assert_allclose(s.vel(t=0)[0, 0], expected_kms, rtol=1e-10)

    def test_KE_units(self):
        """KE = 0.5 * m * v^2. Check user-unit KE against astropy."""
        mass_msun = 1e6
        vel_kms = np.array([[100.0, 0.0, 0.0]])
        s = Sim()
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms,
                        mass=np.array([mass_msun]))

        KE_astropy = 0.5 * mass_msun * 100.0**2  # Msun (km/s)^2
        assert_allclose(s.KE(t=0), KE_astropy, rtol=1e-10)

    def test_momentum_units(self):
        """p = m * v. Check user-unit p against astropy."""
        mass_msun = 1e6
        vel_kms = np.array([[100.0, 200.0, 300.0]])
        s = Sim()
        s.add_particles('test', pos=np.zeros((1, 3)), vel=vel_kms,
                        mass=np.array([mass_msun]))

        # Expected: Msun * km/s
        expected_p = mass_msun * vel_kms  # Msun km/s
        assert_allclose(s.p(t=0), expected_p, rtol=1e-10)

    def test_self_gravity_dimensional_consistency(self):
        """G * M / r^2 should have acceleration dimensions.
        Verify the self-gravity output for a two-body system."""
        m1, m2 = 1e10, 1e10  # Msun
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 1 kpc apart
        vel = np.zeros((2, 3))  # stationary in km/s
        s = Sim()
        s.add_particles('test', pos=pos, vel=vel, mass=np.array([m1, m2]))

        # Compute in internal units
        acc_internal = s.self_gravity(t=0, method='direct', eps=0.0,
                                     return_internal=True)  # kpc/Gyr^2
        # Particle 0 should be attracted toward particle 1 (+x direction)
        # |a| = G * m2 / r^2 = G_INTERNAL * 1e10 / 1.0^2
        expected_acc_x = G_INTERNAL * m2 / 1.0**2
        assert_allclose(abs(acc_internal[0, 0]), expected_acc_x, rtol=1e-10)

        # Verify user-unit output matches internal * factor
        acc_user = s.self_gravity(t=0, method='direct', eps=0.0)
        assert_allclose(acc_user, acc_internal * INTERNAL_TO_USER_UNITS['acceleration'],
                        rtol=1e-14)

    def test_self_potential_dimensional_consistency(self):
        """phi = -G * m / r. self_potential returns m_i * phi_i (energy)."""
        m1, m2 = 1e8, 1e8
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # 2 kpc apart
        vel = np.zeros((2, 3))
        s = Sim()
        s.add_particles('test', pos=pos, vel=vel, mass=np.array([m1, m2]))

        pot_internal = s.self_potential(t=0, method='direct', eps=0.0,
                                       return_internal=True)
        # self_potential returns m_i * phi_i = m1 * (-G * m2 / r)
        expected_pot0 = m1 * (-G_INTERNAL * m2 / 2.0)
        assert_allclose(pot_internal[0], expected_pot0, rtol=1e-10)

        # User-unit consistency
        pot_user = s.self_potential(t=0, method='direct', eps=0.0)
        assert_allclose(pot_user, pot_internal * INTERNAL_TO_USER_UNITS['energy'],
                        rtol=1e-14)

    def test_angular_momentum_cross_product(self):
        """L = r x (m*v). Verify user output against manual astropy calc."""
        pos = np.array([[1.0, 0.0, 0.0]])   # kpc
        vel_kms = np.array([[0.0, 100.0, 0.0]])  # km/s
        mass = np.array([1e6])
        s = Sim()
        s.add_particles('test', pos=pos, vel=vel_kms, mass=mass)

        # Internal: L = r x (m * v_internal)
        # L_z = x * m * vy_internal = 1.0 * 1e6 * (100 * KMS_TO_KPCGYR)
        L_internal = s.L(t=0, return_internal=True)
        expected_Lz_internal = 1.0 * 1e6 * 100.0 * KMS_TO_KPCGYR
        assert_allclose(L_internal[0, 2], expected_Lz_internal, rtol=1e-12)

        # User output = internal * factor
        L_user = s.L(t=0)
        assert_allclose(L_user, L_internal * INTERNAL_TO_USER_UNITS['angular_momentum'],
                        rtol=1e-14)
