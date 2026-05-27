from src.tambora.dynamics.integration import _runner
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from src.tambora.util import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn
from src.tambora.simulation import Sim
from src.tambora.dynamics import ExternalGalpyPotential
from src.tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u
import numpy as np

import pytest
import numpy as np

from src.tambora.dynamics.integration import _runner
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from src.tambora.util import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn
from src.tambora.simulation import Sim
from src.tambora.dynamics import ExternalGalpyPotential, DirectSummationGravity, NullBaseForce, NullSelfGravity
from src.tambora.dynamics.forces.CompositeForce import _CompositeConservative
from src.tambora.dynamics.integration.LeapfrogIntegrator import LeapfrogIntegrator
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u

class TestOrbitConservationLaws:
    @classmethod
    def setup_class(cls):
        t_end = 0.01 * u.Gyr
        dt = 5e-6 * u.Gyr
        integrator = LeapfrogIntegrator()
        cls.t_end = t_end
        cls.ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
        pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
        pot.turn_physical_on()
        R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
        pos = np.array([cyl_to_rect(R, phi, z)])
        vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Gyr).value])
        ext_force = _CompositeConservative([]) + ExternalGalpyPotential(pot)
        pos_out, vel_out, cls.ts_out, _, _ = _runner(pos, vel, np.array([1.]), integrator, NullSelfGravity(), ext_force, NullBaseForce(),
                                            t_end.value, dt.value, dt.value,
                                            return_self_gravity_pot=False, return_self_gravity_acc=False)
        nsnaps, npart = vel_out.shape[:2]
        KE = 0.5 * np.sum(vel_out**2, axis=-1)  # (nsnaps, N)
        PE = ext_force.potential(pos_out.reshape(-1, 3), np.zeros(nsnaps * npart), t=0).reshape(nsnaps, npart)

        cls.E_out = ((KE + PE).squeeze() * u.kpc**2/u.Gyr**2).to(u.km**2/u.s**2).value
        cls.Lz_out = (pos_out[...,0]*vel_out[...,1] - pos_out[...,1]*vel_out[...,0]).squeeze()
        
    def test_energy_conservation(self):
        '''
        Test that total energy is conserved when integrating an orbit in an 
        external potential. Ensures that the integrator is working correctly.
        
        Does not test how the integrator handles self-gravity, which is tested separately.
        '''
        np.testing.assert_allclose(self.E_out, self.E_out[0], rtol=1e-10)
        
    def test_angular_momentum_conservation(self):
        np.testing.assert_allclose(self.Lz_out, self.Lz_out[0], rtol=1e-10)
    
    def test_output_times(self):
        np.testing.assert_allclose(self.ts_out, self.ts.value)

class TestOrbitIntegrationAgainstGalpy:
    @classmethod
    def setup_class(cls):
        t_end = 0.01 * u.Gyr
        dt = 5e-6 * u.Gyr
        integrator = LeapfrogIntegrator()
        cls.t_end = t_end
        cls.ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
        pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
        pot.turn_physical_on()
        R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
        pos = np.array([cyl_to_rect(R, phi, z)])
        vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Gyr).value])
        ext_force = _CompositeConservative([]) + ExternalGalpyPotential(pot)
        cls.pos_out, cls.vel_out, cls.ts_out, _, _ = _runner(pos, vel, np.array([1.]), integrator, NullSelfGravity(), ext_force, NullBaseForce(),
                                            t_end.value, dt.value, dt.value,
                                            return_self_gravity_pot=False, return_self_gravity_acc=False)

        cls.o_galpy = Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.rad])
        cls.o_galpy.integrate(cls.ts, pot, method='leapfrog_c', dt=dt)
        cls.o_galpy.turn_physical_on()

        nsnaps, npart = cls.vel_out.shape[:2]
        KE = 0.5 * np.sum(cls.vel_out**2, axis=-1)
        PE = ext_force.potential(cls.pos_out.reshape(-1, 3), np.zeros(nsnaps * npart), t=0).reshape(nsnaps, npart)
        cls.E_out = ((KE + PE).squeeze() * u.kpc**2/u.Gyr**2).to(u.km**2/u.s**2).value
        cls.Lz_out = (cls.pos_out[...,0]*cls.vel_out[...,1] - cls.pos_out[...,1]*cls.vel_out[...,0]).squeeze()
        
    def test_x_against_galpy(self):
        np.testing.assert_allclose(self.pos_out[...,0][...,0], self.o_galpy.x(self.ts).T, rtol=1e-8), "x position does not match galpy output."
        
    def test_vx_against_galpy(self):
        np.testing.assert_allclose(self.vel_out[...,0][...,0], (self.o_galpy.vx(self.ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "Velocity does not match galpy output."

    def test_y_against_galpy(self):
        np.testing.assert_allclose(self.pos_out[...,1][...,0], self.o_galpy.y(self.ts).T, rtol=1e-6), "y position does not match galpy output."

    def test_vy_against_galpy(self):
        np.testing.assert_allclose(self.vel_out[...,1][...,0], (self.o_galpy.vy(self.ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "y velocity does not match galpy output."

    def test_z_against_galpy(self):
        np.testing.assert_allclose(self.pos_out[...,2][...,0], self.o_galpy.z(self.ts).T, rtol=1e-6), "z position does not match galpy output."

    def test_vz_against_galpy(self):
        np.testing.assert_allclose(self.vel_out[...,2][...,0], (self.o_galpy.vz(self.ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "z velocity does not match galpy output."

    def test_energy_against_galpy(self):
        energy_galpy = self.o_galpy.E(self.ts, quantity=True).to(u.km**2/u.s**2).value
        np.testing.assert_allclose(self.E_out, energy_galpy, rtol=1e-6), "Energy does not match galpy output."

    def test_Lz_against_galpy(self):
        Lz_galpy = self.o_galpy.Lz(self.ts, quantity=True).to(u.kpc*u.kpc/u.Gyr).value
        np.testing.assert_allclose(self.Lz_out, Lz_galpy, rtol=1e-6), "Angular momentum does not match galpy output."
