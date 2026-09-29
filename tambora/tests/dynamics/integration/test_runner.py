import pytest
import numpy as np

from tambora.dynamics.integration import _runner, StepState
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from tambora.interop._galpy.bridge import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn
from tambora.simulation import Sim
from tambora.dynamics import ExternalPotential, DirectSummationGravity, NullForce, NullSelfGravity
from tambora.dynamics.forces.CompositeForce import _CompositeConservative
from tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u

class TestOrbitIntegrationOutputShapes:
    @classmethod
    def setup_class(cls):
        from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec
        t_end = 0.01 * u.Gyr
        dt = 5e-6 * u.Gyr
        integrator = LeapfrogIntegrator()
        cls.t_end = t_end
        cls.ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
        pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
        pot.turn_physical_on()
        ext_force = _CompositeConservative([]) + ExternalPotential(pot)

        pos = np.array([[8., 0., 0.], [8., 0., 0.]])
        vel = np.array([[0.1, 220.0, 0.5], [0.2, 220.0, 0.5]])
        vel_internal = (vel * u.km/u.s).to(u.kpc/u.Gyr).value
        mass = np.array([1., 1.])
        (cls.pos_out, cls.vel_out, cls.ts_out,
        cls.self_acc_out, cls.self_pot_out)  = _runner(pos, vel_internal, mass, integrator, NullSelfGravity(), ext_force, NullForce(),
                                            0.0, t_end.value, dt.value, dt.value,
                                            return_self_gravity_pot=True, return_self_gravity_acc=True)

    def test_integrate_multiple_orbits_output_pos_shape(self):
        '''
        Test that the output positions have the correct shape when integrating multiple orbits.
        '''
        assert self.pos_out.shape == (len(self.ts_out), 2, 3)

    def test_integrate_multiple_orbits_output_vel_shape(self):
        '''
        Test that the output velocities have the correct shape when integrating multiple orbits.
        '''
        assert self.vel_out.shape == (len(self.ts_out), 2, 3)

    def test_integrate_multiple_orbits_output_self_acc_shape(self):
        '''
        Test that the output self-accelerations have the correct shape when integrating multiple orbits.
        '''
        assert self.self_acc_out.shape == (len(self.ts_out), 2, 3)

    def test_integrate_multiple_orbits_output_self_pot_shape(self):
        '''
        Test that the output self-potentials have the correct shape when integrating multiple orbits.
        '''
        assert self.self_pot_out.shape == (len(self.ts_out), 2)

# --- return option --------------------------------------------------------------------------------
class TestReturnOptions:
    @classmethod
    def setup_class(cls):
        t_end = 0.01 * u.Gyr
        dt = 5e-6 * u.Gyr
        integrator = LeapfrogIntegrator()
        cls.t_end = t_end
        cls.dt = dt
        cls.ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
        pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
        pot.turn_physical_on()
        cls.integrator = integrator
        cls.self_gravity = DirectSummationGravity(eps=0.0)
        cls.ext_force = _CompositeConservative([]) + ExternalPotential(pot)
        cls.pos = np.array([[8., 0., 0.], [8., 0., 0.]])
        vel = np.array([[0.1, 220.0, 0.5], [0.2, 220.0, 0.5]])
        cls.vel_internal = (vel * u.km/u.s).to(u.kpc/u.Gyr).value
        cls.mass = np.array([1., 1.])
    
    def test_return_self_potential_only(self):
        '''
        Test that we can return just the self-potential without the self-acceleration.
        '''
        out = _runner(self.pos, self.vel_internal, self.mass, self.integrator, self.self_gravity, self.ext_force, NullForce(), 
                      0.0, self.t_end.value, self.dt.value*10, self.dt.value*10,
                      return_self_gravity_pot=True, return_self_gravity_acc=False)
        assert out[-1].shape == (len(out[2]), self.mass.shape[0])
        assert out[-2] is None

    def test_return_self_acceleration_only(self):
        '''
        Test that we can return just the self-acceleration without the self-potential.
        '''

        out = _runner(self.pos, self.vel_internal, self.mass, self.integrator, self.self_gravity, self.ext_force, NullForce(), 
                      0.0, self.t_end.value, self.dt.value*10, self.dt.value*10,
                                        return_self_gravity_pot=False, return_self_gravity_acc=True)
        assert out[-2].shape == (len(out[2]), self.mass.shape[0], 3)
        assert out[-1] is None

    def test_return_self_acceleration_and_potential(self):
        '''
        Test that we can return both the self-acceleration and self-potential.
        '''
        out = _runner(self.pos, self.vel_internal, self.mass, self.integrator, self.self_gravity, self.ext_force, NullForce(), 
                      0.0, self.t_end.value, self.dt.value*10, self.dt.value*10,
                      return_self_gravity_pot=True, return_self_gravity_acc=True)
        assert out[-2].shape == (len(out[2]), self.mass.shape[0], 3)
        assert out[-1].shape == (len(out[2]), self.mass.shape[0])

    def test_return_neither_self_acceleration_nor_potential(self):
        '''
        Test that we can return neither the self-acceleration nor self-potential.
        '''
        out = _runner(self.pos, self.vel_internal, self.mass, self.integrator, self.self_gravity, self.ext_force, NullForce(), 
                      0.0, self.t_end.value, self.dt.value*10, self.dt.value*10,
                      return_self_gravity_pot=False, return_self_gravity_acc=False)
        assert out[-1] is None
        assert out[-2] is None

class TestTimeInputs:
    def test_forward_with_negative_dt_raises(self):
        # Forward span (t0 < t_end) requires a positive dt.
        with pytest.raises(ValueError, match="dt must be positive for forwards integration"):
            _runner(*[None]*7, t0=0.0,
                    t_end=1.0,
                    dt=-0.1,
                    dt_out=-0.1)

    def test_backward_with_positive_dt_raises(self):
        # Backward span (t0 > t_end) requires a negative dt.
        with pytest.raises(ValueError, match="dt must be negative for backwards integration"):
            _runner(*[None]*7, t0=1.0,
                    t_end=0.0,
                    dt=0.1,
                    dt_out=0.1)

    def test_dt_out_opposite_sign_to_dt_raises(self):
        # dt_out must share dt's sign (valid forward dt here, but a negative dt_out).
        with pytest.raises(ValueError, match="dt_out must have the same sign as dt"):
            _runner(*[None]*7, t0=0.0,
                    t_end=1.0,
                    dt=0.1,
                    dt_out=-0.1)

    def test_backward_integration_outputs_descending_times(self):
        # A consistent backward run (negative dt and dt_out) is accepted and
        # produces snapshot times running from t0 down to t_end.
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _, _, ts_out, _, _ = _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                _CompositeConservative([]), NullForce(), t0=0.0,
                t_end=-1.0,
                dt=-0.1,
                dt_out=-0.1)
        np.testing.assert_allclose(ts_out[0], 0.0)
        np.testing.assert_allclose(ts_out[-1], -1.0)
        assert ts_out.shape == (11,)

    def test_nonzero_t0_offsets_output_times(self):
        # t0 shifts the snapshot times: a run over [t0, t_end] starts at t0, ends at t_end.
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _, _, ts_out, _, _ = _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                _CompositeConservative([]), NullForce(), t0=1.0,
                t_end=2.0,
                dt=0.1,
                dt_out=0.1)
        np.testing.assert_allclose(ts_out[0], 1.0)
        np.testing.assert_allclose(ts_out[-1], 2.0)
        assert ts_out.shape == (11,)

    def test_dt_out_less_than_dt(self):
        with pytest.raises(ValueError, match="absolute value of dt_out must be greater than or equal to dt"):
            _runner(*[None]*7, t0=0.0,
                    t_end=1.0, 
                    dt=0.1, 
                    dt_out=0.05)
            
    def test_dt_out_not_a_multiple_of_dt_fails(self):
        with pytest.raises(ValueError, match="dt_out must be a multiple of dt."):
            _runner(*[None]*7, t0=0.0,
                    t_end=1.0, 
                    dt=0.1, 
                    dt_out=0.13)
            
    def test_dt_not_multiple_of_t_end_warns(self):
        t_end=1.0
        dt=0.13
        dt_out=0.13
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        with pytest.warns(UserWarning, match="is not an exact multiple of dt=0.13 Gyr"):
            _, _, ts_out, _, _ = _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)
        np.testing.assert_equal(ts_out[-1], 0.91)

    def test_dt_out_not_multiple_of_t_end_warns(self):
        t_end=1.0
        dt=0.2
        dt_out=0.4
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        with pytest.warns(UserWarning, match="Last output will be at t=0.8 Gyr instead of t=1.0 Gyr"):
            _, _, ts_out, _, _ = _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)
        np.testing.assert_equal(ts_out[-1], 0.8)
    
    def test_accepts_dt_out_equal_to_t_end(self):
        '''
        dt_out=0.075, dt=0.025 -> 3.0 in exact arithmetic, but
        0.075 and 0.025 are not exact in binary. The tolerant check
        should accept this.
        '''
        t_end=0.1
        dt=0.005
        dt_out=0.1
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)
        
    def test_accepts_dt_equal_to_t_end(self):
        '''
        dt_out=0.075, dt=0.025 -> 3.0 in exact arithmetic, but
        0.075 and 0.025 are not exact in binary. The tolerant check
        should accept this.
        '''
        t_end=0.1
        dt=0.1
        dt_out=0.1
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)
    
    def test_float_tolerant_dt_out_dt_accepts_exact_multiples(self):
        '''
        dt_out=0.1, dt=0.001 -> ratio = 100 exactly.
        Should not raise despite floating-point representation.
        '''
        t_end=1.0
        dt=0.001
        dt_out=0.01
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)


    def test_float_tolerant_dt_out_dt_accepts_tricky_floats(self):
        '''
        dt_out=0.075, dt=0.025 -> 3.0 in exact arithmetic, but
        0.075 and 0.025 are not exact in binary. The tolerant check
        should accept this.
        '''
        t_end=0.1
        dt=0.025
        dt_out=0.075
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)

    def test_float_tolerant_dt_out_dt_rejects_genuine_nonmultiple(self):
        '''
        dt_out=0.07, dt=0.03 -> ratio approx 2.333, genuinely not an integer.
        Should raise ValueError.
        '''
        t_end=1.0
        dt=0.03
        dt_out=0.07
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        with pytest.raises(ValueError, match="dt_out must be a multiple of dt."):
            _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                        _CompositeConservative([]), NullForce(), t0=0.0,
                        t_end=t_end, 
                        dt=dt, 
                        dt_out=dt_out)

    def test_exact_multiples_no_warnings(self):
        '''
        dt_out and t_end are exact multiples of dt.
        No UserWarning should be raised.
        '''
        t_end=1.0
        dt=0.1
        dt_out=0.1
        pos = np.zeros((1, 3))
        vel = np.zeros((1, 3))
        mass = np.ones(1)
        _runner(pos, vel, mass, LeapfrogIntegrator(), NullSelfGravity(),
                    _CompositeConservative([]), NullForce(), t0=0.0,
                    t_end=t_end, 
                    dt=dt, 
                    dt_out=dt_out)

class TestTimeStepInputs:
    @classmethod
    def setup_class(cls):
        t_end = 0.01 * u.Gyr
        dt = 5e-6 * u.Gyr
        cls.integrator = LeapfrogIntegrator()
        cls.t_end = t_end
        cls.ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
        pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
        pot.turn_physical_on()
        R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
        cls.pos = np.array([cyl_to_rect(R, phi, z)])
        cls.vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Gyr).value])
        cls.ext_force = _CompositeConservative([]) + ExternalPotential(pot)
        
    def test_output_shape_for_dt_out_multiple_of_dt(self, t_end=0.1, dt=0.01, dt_out=0.02):
        '''
        Test that the output arrays have the correct shape when dt_out is a multiple of dt.
        '''
        pos_out, vel_out, ts_out, _, _ = _runner(self.pos, self.vel, np.array([1.]), self.integrator, NullSelfGravity(), self.ext_force, NullForce(),
                                            0.0, t_end, dt, dt_out,
                                            return_self_gravity_pot=False, return_self_gravity_acc=False)
        expected_num_outputs = int(t_end / dt_out) + 1
        assert ts_out.shape == (expected_num_outputs,)
        assert pos_out.shape == (expected_num_outputs, 1, 3)
        assert vel_out.shape == (expected_num_outputs, 1, 3)
    
    def test_output_shape_for_dt_not_multiple_of_t_end(self, t_end=0.1, dt=0.011, dt_out=0.022):
        '''
        Test that the output arrays have the correct shape when dt_out is not a multiple of dt.
        '''
        pos_out, vel_out, ts_out, _, _ = _runner(self.pos, self.vel, np.array([1.]), self.integrator, NullSelfGravity(), self.ext_force, NullForce(),
                                            0.0, t_end, dt, dt_out,
                                            return_self_gravity_pot=False, return_self_gravity_acc=False)
        expected_num_outputs = int(t_end / dt_out) + 1
        assert ts_out.shape == (expected_num_outputs,)
        assert pos_out.shape == (expected_num_outputs, 1, 3)
        assert vel_out.shape == (expected_num_outputs, 1, 3)


    def test_output_shape_for_dt_out_not_multiple_of_t_end(self, t_end=0.1, dt=0.01, dt_out=0.03):
        '''
        Test that the output arrays have the correct shape when dt_out is not a multiple of dt.
        '''
        pos_out, vel_out, ts_out, _, _ = _runner(self.pos, self.vel, np.array([1.]), self.integrator, NullSelfGravity(), self.ext_force, NullForce(),
                                            0.0, t_end, dt, dt_out,
                                            return_self_gravity_pot=False, return_self_gravity_acc=False)
        expected_num_outputs = int(t_end / dt_out) + 1
        assert ts_out.shape == (expected_num_outputs,)
        assert pos_out.shape == (expected_num_outputs, 1, 3)
        assert vel_out.shape == (expected_num_outputs, 1, 3)

# --- time-dependent external potential --------------------------------------------------------- #

from galpy.potential import DehnenSmoothWrapperPotential
from tambora.units import KMS_TO_KPCGYR

def test_time_dependent_potential_matches_galpy():
    '''
    Integrate an orbit in a DehnenSmoothWrapperPotential (NFW that grows from
    zero to full strength) and compare the result against galpy's own orbit
    integration.  This verifies that the integration time is correctly
    forwarded to external force functions.
    '''
    # Static NFW wrapped in a smooth growth envelope.
    nfw_td = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    smooth_pot = DehnenSmoothWrapperPotential(pot=nfw_td, tform=0., tsteady=0.02*u.Gyr)
    smooth_pot.turn_physical_on()

    # Integration parameters
    td_t_end = 0.01 * u.Gyr
    td_dt = 5e-6 * u.Gyr
    td_ts = np.arange(0, td_t_end.value + td_dt.value, td_dt.value) * u.Gyr

    # Initial conditions: particle at R=8 kpc with some velocity
    td_R, td_vR, td_vT, td_z, td_vz, td_phi = 8., 0.1, 220.0, 0., 0.5, 0.
    td_pos = np.array([cyl_to_rect(td_R, td_phi, td_z)])
    td_vel = np.array([(cyl_to_rect_vec(td_vR, td_vT, td_vz, td_phi) * u.km / u.s).to(u.kpc / u.Gyr).value])

    integrator = LeapfrogIntegrator()
    # tambora integration
    td_force = _CompositeConservative([]) + ExternalPotential(smooth_pot)
    td_pos_out, td_vel_out, td_ts_out, _, _ = _runner(
        td_pos, td_vel, np.array([1.0]),
        integrator, NullSelfGravity(), td_force, NullForce(),
        0.0, td_t_end.value, td_dt.value, td_dt.value, 
        return_self_gravity_pot=False, return_self_gravity_acc=False,
    )

    # galpy reference integration
    td_o = Orbit([td_R * u.kpc, td_vR * u.km / u.s, td_vT * u.km / u.s,
                  td_z * u.kpc, td_vz * u.km / u.s, td_phi * u.rad])
    td_o.integrate(td_ts, smooth_pot, method='leapfrog_c', dt=td_dt)
    td_o.turn_physical_on()

    np.testing.assert_allclose(
        td_pos_out[..., 0].squeeze(), td_o.x(td_ts, use_physical=True, quantity=False).T.squeeze(),
        rtol=1e-8,
        err_msg="x position does not match galpy for time-dependent potential.",
    )
    np.testing.assert_allclose(
        td_pos_out[..., 1].squeeze(), td_o.y(td_ts, use_physical=True, quantity=False).T.squeeze(),
        rtol=1e-6,
        err_msg="y position does not match galpy for time-dependent potential.",
    )
    np.testing.assert_allclose(
        td_pos_out[..., 2].squeeze(), td_o.z(td_ts, use_physical=True, quantity=False).T.squeeze(),
        rtol=1e-6,
        err_msg="z position does not match galpy for time-dependent potential.",
    )

def test_time_dependent_potential_differs_from_static():
    '''
    Verify that a time-dependent potential actually produces different
    trajectories than a static one.  This catches the case where t=0 is
    silently passed — the growing potential at t=0 has zero force, so the
    particle would drift in a straight line.
    '''
    nfw_static = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    nfw_growing = DehnenSmoothWrapperPotential(pot=nfw_static, tform=0., tsteady=0.02*u.Gyr)
    nfw_growing.turn_physical_on()
    nfw_growing_force = _CompositeConservative([]) + ExternalPotential(nfw_growing)

    td2_t_end = 0.01 * u.Gyr
    td2_dt = 5e-6 * u.Gyr

    td2_R, td2_phi, td2_z = 8., 0., 0.
    td2_vR, td2_vT, td2_vz = 0.1, 220.0, 0.5
    td2_pos = np.array([cyl_to_rect(td2_R, td2_phi, td2_z)])
    td2_vel = np.array([(cyl_to_rect_vec(td2_vR, td2_vT, td2_vz, td2_phi) * u.km / u.s).to(u.kpc / u.Gyr).value])

    integrator = LeapfrogIntegrator()

    # Integrate in growing potential
    #acc_growing = _galpy_pot_to_acc_fn(nfw_growing)
    pos_growing, _, _, _, _ = _runner(
        td2_pos.copy(), td2_vel.copy(), np.array([1.0]),
        integrator, NullSelfGravity(), nfw_growing_force, NullForce(),
        0.0, td2_t_end.value, td2_dt.value, td2_dt.value,
        return_self_gravity_pot=False, return_self_gravity_acc=False,
    )

    # Integrate in static potential
    nfw_static.turn_physical_on()
    # acc_static = _galpy_pot_to_acc_fn(nfw_static)
    static_ext_force = _CompositeConservative([]) + ExternalPotential(nfw_static)
    pos_static, _, _, _, _ = _runner(
        td2_pos.copy(), td2_vel.copy(), np.array([1.0]),
        integrator, NullSelfGravity(), static_ext_force, NullForce(),
        0.0, td2_t_end.value, td2_dt.value, td2_dt.value,
        return_self_gravity_pot=False, return_self_gravity_acc=False,
    )

    # Trajectories must differ (growing starts weaker)
    final_diff = np.linalg.norm(pos_growing[-1] - pos_static[-1])
    assert final_diff > 1e-6, (
        f"Growing and static potentials produced the same trajectory (diff={final_diff}). "
        "Time is likely not being forwarded to the external force function."
    )

from tambora.interop._galpy.bridge import _galpy_pot_to_pot_fn

def test_time_dependent_potential_energy_matches_galpy():
    '''
    Compare the total energy trajectory E(t) = KE + PE from tambora
    against galpy for a time-dependent potential.

    In a time-dependent potential total energy is NOT conserved — it
    changes as dE/dt = dPhi/dt.  The correct check is that both
    integrators agree on the energy trajectory, confirming that the
    time argument is threaded correctly through the force evaluation.
    '''
    nfw_e = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    smooth_e = DehnenSmoothWrapperPotential(pot=nfw_e, tform=0., tsteady=2. * u.Gyr)
    smooth_e.turn_physical_on()

    # Short enough integration with fine timestep for close agreement
    e_t_end = 0.1 * u.Gyr
    e_dt = 1e-3 * u.Gyr
    e_dt_out = e_dt
    e_ts = np.arange(0, e_t_end.value + e_dt.value, e_dt.value) * u.Gyr

    e_R, e_vR, e_vT, e_z, e_vz, e_phi = 8., 0.1, 220.0, 0., 0.5, 0.
    e_pos = np.array([cyl_to_rect(e_R, e_phi, e_z)])
    e_vel = np.array([(cyl_to_rect_vec(e_vR, e_vT, e_vz, e_phi)
                       * u.km / u.s).to(u.kpc / u.Gyr).value])

    # --- tambora integration ---
    integrator = LeapfrogIntegrator()
    ext_force = _CompositeConservative([]) + ExternalPotential(smooth_e)
    e_pos_out, e_vel_out, e_ts_out, _, _ = _runner(
        e_pos, e_vel, np.array([1.0]),
        integrator, NullSelfGravity(), ext_force, NullForce(),
        0.0, e_t_end.value, e_dt.value, e_dt_out.value, return_self_gravity_pot=False, return_self_gravity_acc=False,
    )
    nsnaps, npart = e_pos_out.shape[:2]
    ez_KE = 0.5 * np.sum(e_vel_out ** 2, axis=-1).squeeze()         # (nsnaps,)
    # galpy's DehnenSmoothWrapperPotential._smooth uses scalar comparison on t,
    # so we must evaluate the potential one snapshot at a time.
    ez_PE = np.array([
        ext_force.potential(e_pos_out[i], np.ones(npart), t=e_ts_out[i])
        for i in range(nsnaps)
    ]).squeeze()
    ez_E = ((ez_KE + ez_PE) * u.kpc ** 2 / u.Gyr ** 2).to(u.km ** 2 / u.s ** 2).value

    # --- galpy reference ---
    e_o = Orbit([e_R * u.kpc, e_vR * u.km / u.s, e_vT * u.km / u.s,
                 e_z * u.kpc, e_vz * u.km / u.s, e_phi * u.rad])
    e_o.integrate(e_ts, smooth_e, method='leapfrog_c', dt=e_dt)
    e_o.turn_physical_on()
    galpy_E = e_o.E(e_ts, pot=smooth_e, quantity=True).to(u.km ** 2 / u.s ** 2).value

    # Energy trajectories should agree closely
    np.testing.assert_allclose(
        ez_E, galpy_E, rtol=1e-6,
        err_msg="Energy trajectory does not match galpy for time-dependent potential.",
    )

    # Sanity: energy should NOT be constant (since potential is time-dependent)
    energy_range = np.ptp(ez_E)
    assert energy_range > 1.0, (
        f"Energy barely changed (range={energy_range:.2e} km^2/s^2) — "
        "time-dependent potential may not be evolving."
    )


# --- bare-function hooks fire end-to-end through Sim.add_hook / run ------------- #

def test_bare_function_hook_fires_during_run_with_step_state():
    '''A plain function (no default_cadence) registered via add_hook is invoked
    each output during the run, receiving a live StepState.

    Exercises the full documented path Sim.add_hook -> run -> _fire -> hook(state)
    for the "any bare callable" case, and confirms the argument is a real
    StepState view (its accessors work and carry the component's shape). The
    StepState object is reused across fires, so per-fire data must be copied out
    inside the hook rather than by stashing the state itself.
    '''
    seen_t, seen_shapes, seen_is_state = [], [], []

    def record(state):
        seen_t.append(state.t)
        seen_shapes.append(state.pos().shape)
        seen_is_state.append(isinstance(state, StepState))

    sim = Sim()
    sim.add_particles('c', np.zeros((4, 3)), np.zeros((4, 3)), np.ones(4))
    sim.add_hook(record)                 # bare callable -> EveryOutput cadence

    sim.run(t_end=1.0, dt=0.5, dt_out=0.5, method=None, progress=False)

    # Fires once at t0 then at every output (dt_out == dt here): t = 0.0, 0.5, 1.0.
    np.testing.assert_allclose(seen_t, [0.0, 0.5, 1.0])
    assert all(seen_is_state)                    # really a StepState, not a raw tuple
    assert seen_shapes == [(4, 3), (4, 3), (4, 3)]   # component's live positions

