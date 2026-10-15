"""
Accessors shared by Sim and Component, in terms of ``self._sim``, the Sim holding the data, and
``self._sl``, which of its particles.
"""

import warnings

import numpy as np

from ..dynamics import diagnostics
from ..units import unit_handler


class _ParticleView:
    """Accessors shared by Sim and Component."""

    def _snap(self, array, t):
        """Slice a (nsnaps, N, ...) array to these particles at time t."""
        ti = self._sim._ti(t)
        if ti is ...:
            return array[:, self._sl]
        return array[ti, self._sl]

    @property
    def eps(self):
        """
        Softening length of each particle [kpc], NaN where it isn't set.

        Set it to a number, or one value per particle, to change it. Changing it after a
        run clears that run's outputs, so a new run uses it.
        """
        eps = self._sim._eps
        if eps is None:
            return None
        view = eps[self._sl]
        view.flags.writeable = False
        return view

    @eps.setter
    def eps(self, eps):
        self._sim._set_eps(self._sl, eps)

    # --- Position Accessors -------------------------------------------------------------------

    @unit_handler('length')
    def pos(self, t=...) -> np.ndarray:
        '''
        Particle positions (x, y, z) at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        pos : (len(t), n_particles, 3) array or (n_particles, 3) array
            Positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)

    @unit_handler('length')
    def x(self, t=...) -> np.ndarray:
        '''
        x-positions of the particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        x : (len(t), n_particles) array or (n_particles,) array
            x-positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 0]

    @unit_handler('length')
    def y(self, t=...) -> np.ndarray:
        '''
        y-positions of the particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        y : (len(t), n_particles) array or (n_particles,) array
            y-positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 1]

    @unit_handler('length')
    def z(self, t=...) -> np.ndarray:
        '''
        z-positions of the particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        z : (len(t), n_particles) array or (n_particles,) array
            z-positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 2]

    @unit_handler('length')
    def r(self, t=...):
        '''
        Particle spherical radii at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        r : (len(t), n_particles) array or (n_particles,) array
            Spherical radii at *t*.
            Units: `kpc`
        '''
        pos = self.pos(t, return_internal=True)
        return np.linalg.norm(pos, axis=-1)

    def phi(self, t=...):
        '''
        Particle azimuthal angles at *t*.
        
            *Angle present in both spherical and 
            cylindrical coordinates*

        Units: radians

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        phi : (len(t), n_particles) array or (n_particles,) array
            Azimuthal angles at *t*.
            Units: radians
        '''
        pos = self.pos(t, return_internal=True)
        return np.arctan2(pos[..., 1], pos[..., 0])

    def theta(self, t=...):
        '''
        Particle polar angles at *t*.

            *Angle present in spherical coordinates.*

        Units: radians

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        theta : (len(t), n_particles) array or (n_particles,) array
            Polar angles at *t*.
            Units: radians
        '''
        pos = self.pos(t, return_internal=True)
        r = np.linalg.norm(pos, axis=-1)
        return np.arccos(pos[..., 2] / r)

    @unit_handler('length')
    def cylR(self, t=...):
        '''
        Particle cylindrical radii at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        R : (len(t), n_particles) array or (n_particles,) array
            Cylindrical radii at *t*.
            Units: `kpc`
        '''
        return np.sqrt(self.x(t, return_internal=True)**2 + self.y(t, return_internal=True)**2)

    # --- Velocity Accessors -------------------------------------------------------------------

    @unit_handler('velocity')
    def vel(self, t=...) -> np.ndarray:
        '''
        Particle velocities (vx, vy, vz) at *t*.

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vel : (len(t), n_particles, 3) array or (n_particles, 3) array
            Velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)

    @unit_handler('velocity')
    def vx(self, t=...) -> np.ndarray:
        '''
        x-component of particle velocities at *t*.

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vx : (len(t), n_particles) array or (n_particles,) array
            x-component of velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 0]

    @unit_handler('velocity')
    def vy(self, t=...) -> np.ndarray:
        '''
        y-component of particle velocities at *t*.

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vy : (len(t), n_particles) array or (n_particles,) array
            y-component of velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 1]

    @unit_handler('velocity')
    def vz(self, t=...) -> np.ndarray:
        '''
        z-component of particle velocities at *t*.

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vz : (len(t), n_particles) array or (n_particles,) array
            z-component of velocities at *t*.

            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 2]

    @unit_handler('velocity')
    def vr(self, t=...):
        '''
        Spherical coordinates radial velocities at *t*.

        The component of the velocity vector along the position vector, 
        i.e. :math:`v_r = (x*v_x + y*v_y + z*v_z) / r`.

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vr : (len(t), n_particles) array or (n_particles,) array
            Radial velocities at *t*.
            Units: `km / s`
        '''
        pos = self.pos(t, return_internal=True)
        vel = self.vel(t, return_internal=True)
        vr = np.sum(pos * vel, axis=-1) / self.r(t, return_internal=True)
        return vr

    @unit_handler('angular_velocity')
    def vphi(self, t=...):
        '''
        Azimuthal angular velocity at *t* (for both spherical and cylindrical coordinates).

        The angular velocity about the z-axis,
        i.e. :math:`\\omega_{\\phi} = (x \\cdot v_y - y \\cdot v_x) / R^2`.

        Units: `km/s/kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vphi : (len(t), n_particles) array or (n_particles,) array
            Azimuthal angular velocities at *t*.
            Units: `km/s/kpc`
        '''
        return ((self.x(t, return_internal=True) * self.vy(t, return_internal=True) - 
                self.y(t, return_internal=True) * self.vx(t, return_internal=True)) 
                / self.cylR(t, return_internal=True)**2)

    @unit_handler('velocity')
    def vtheta(self, t=...):
        '''
        Polar velocities at *t* (for spherical coordinates).

        The component of the velocity vector along the polar direction, 
        i.e. :math:`v_{theta} = [z(x*vx + y*vy) - R^2*vz] / (r*R)`.*

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        vtheta : (len(t), n_particles) array or (n_particles,) array
            Polar velocities at *t*.
            Units: `km / s`
        '''
        r = self.r(t, return_internal=True)
        return (
            ((self.z(t, return_internal=True) * 
              (self.x(t, return_internal=True) * self.vx(t, return_internal=True) + self.y(t, return_internal=True) * self.vy(t, return_internal=True)))
             - self.cylR(t, return_internal=True)**2 * self.vz(t, return_internal=True)) 
            / (r * self.cylR(t, return_internal=True))
        )

    @unit_handler('velocity')
    def cylvR(self, t=...):
        '''
        Cylindrical coordinates radial velocities at *t*.

        The component of the velocity vector along the cylindrical radius vector, 
        i.e. :math:`v_{cyl,R} = (x*v_x + y*v_y) / R`.*

        Units: `km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        cylvR : (len(t), n_particles) array or (n_particles,) array
            Cylindrical radial velocities at *t*.
            Units: `km / s`
        '''
        return ((self.x(t, return_internal=True) * self.vx(t, return_internal=True) + 
                 self.y(t, return_internal=True) * self.vy(t, return_internal=True)) / 
                 self.cylR(t, return_internal=True))

    # --- Momentum Accessors -------------------------------------------------------------------

    @unit_handler('momentum')
    def p(self, t=...):
        '''
        Particle momenta (px, py, pz) at *t*.
        
        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        
        Returns
        -------
        momentum : (len(t), n_particles, 3) array or (n_particles, 3) array
            Momenta at *t*.
            Units: `Msun km / s`
        '''
        return self.mass[:, None] * self.vel(t, return_internal=True)

    @unit_handler('momentum')
    def px(self, t=...):
        '''
        x-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        px : (len(t), n_particles) array or (n_particles,) array
            x-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self.mass * self.vx(t, return_internal=True)

    @unit_handler('momentum')
    def py(self, t=...):
        '''
        y-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        py : (len(t), n_particles) array or (n_particles,) array
            y-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self.mass * self.vy(t, return_internal=True)

    @unit_handler('momentum')
    def pz(self, t=...):
        '''
        z-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        pz : (len(t), n_particles) array or (n_particles,) array
            z-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self.mass * self.vz(t, return_internal=True)

    @unit_handler('angular_momentum')
    def L(self, t=..., center_pos=None, center_vel=None):
        '''
        Angular momentum of particles at *t*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        center_pos : array-like, optional
            Point to compute angular momentum about. Default is [0,0,0].
            Units: `kpc`
        center_vel : array-like, optional
            Velocity of the center point. Default is [0,0,0].
            Units: `kpc/Gyr`

        Returns
        -------
        L : (len(t), n_particles, 3) array or (n_particles, 3) array
            Angular momentum of each particle at *t* about *center*.
            Units: `Msun km^2 / s`
        '''
        r = self.pos(t, return_internal=True)
        v = self.vel(t, return_internal=True)
        if center_pos is not None:
            r = r - np.asarray(center_pos)
        if center_vel is not None:
            v = v - np.asarray(center_vel)
        return self.mass[:, None] * np.cross(r, v)

    @unit_handler('angular_momentum')
    def Lx(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]):
        '''
        x-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        center_pos : array-like, optional
            Point to compute angular momentum about. Default is [0,0,0].
            Units: `kpc`
        center_vel : array-like, optional
            Velocity of the center point. Default is [0,0,0].
            Units: `kpc/Gyr`

        Returns
        -------
        Lx : (len(t), n_particles) array or (n_particles,) array
            x-component of angular momentum of each particle at *t* about *center*.
            Units: `Msun km^2 / s`
        '''
        return self.L(t, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 0]

    @unit_handler('angular_momentum')
    def Ly(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]):
        '''
        y-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        center_pos : array-like, optional
            Point to compute angular momentum about. Default is [0,0,0].
            Units: `kpc`
        center_vel : array-like, optional
            Velocity of the center point. Default is [0,0,0].
            Units: `kpc/Gyr`

        Returns
        -------
        Ly : (len(t), n_particles) array or (n_particles,) array
            y-component of angular momentum of each particle at *t* about *center*.
            Units: `Msun km^2 / s`
        '''
        return self.L(t, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 1]

    @unit_handler('angular_momentum')
    def Lz(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]):
        '''
        z-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        center_pos : array-like, optional
            Point to compute angular momentum about. Default is [0,0,0].
            Units: `kpc`
        center_vel : array-like, optional
            Velocity of the center point. Default is [0,0,0].
            Units: `kpc/Gyr`

        Returns
        -------
        Lz : (len(t), n_particles) array or (n_particles,) array
            z-component of angular momentum of each particle at *t* about *center*.
            Units: `Msun km^2 / s`
        '''
        return self.L(t, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 2]

    # --- Center of Mass -----------------------------------------------------------------------

    @unit_handler('length')
    def com_pos(self, t=...):
        '''
        Center of mass (x, y, z) of the particles at *t*: their mass-weighted
        mean position.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        com_pos : (len(t), 3) array or (3,) array
            Center of mass at *t*.
            Units: `kpc`
        '''
        return diagnostics.center_of_mass(self.pos(t=t, return_internal=True), self.mass)

    @unit_handler('velocity')
    def com_vel(self, t=...):
        '''
        Velocity (vx, vy, vz) of the center of mass of the particles at *t*:
        their mass-weighted mean velocity.

        Units: `km/s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        com_vel : (len(t), 3) array or (3,) array
            Velocity of the center of mass at *t*.
            Units: `km/s`
        '''
        return diagnostics.center_of_mass(self.vel(t=t, return_internal=True), self.mass)

    # --- External Potential -------------------------------------------------------------------

    @unit_handler('energy')
    def compute_external_pot(self, t=...):
        '''
        External potential of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        ext_pot : (len(t), n_particles) array or (n_particles,) array
            External potential at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        ti = self._ti(t)
        if isinstance(ti, (int, np.integer)):
            t_phys = self._sim._times[ti]
            ext_pot = self._sim._conserv_ext_force.potential(
                pos=self.pos(t=ti, return_internal=True), mass=self.mass, t=t_phys)
        else:
            warnings.warn("Computing external potential on-the-fly for multiple snapshots may be slow.")
            times = self._sim._times
            ext_pot = np.zeros((len(times), self.mass.shape[0]))
            for i, t_i in enumerate(times):
                ext_pot[i] += self._sim._conserv_ext_force.potential(
                    pos=self.pos(t=i, return_internal=True), mass=self.mass, t=t_i)
        return self.mass * ext_pot

    # --- Kinetic Energy -----------------------------------------------------------------------

    @unit_handler('energy')
    def KE(self, t=...):
        '''
        Kinetic energy of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        KE : (len(t), n_particles) array
            Kinetic energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        return 0.5 * self.mass * np.sum(self.vel(t=t, return_internal=True) ** 2, axis=-1)

    # --- External Acceleration ----------------------------------------------------------------

    @unit_handler('acceleration')
    def external_acc(self, t=...):
        '''
        Total external acceleration on each particle at *t*.

        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        ext_acc : (len(t), n_particles, 3) array or (n_particles, 3) array
            External acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        ti = self._ti(t)
        if ti is ...:
            return np.stack([self.external_acc(i, return_internal=True) for i in range(len(self._sim._times))])
        t_phys = self._sim._times[ti]
        pos, vel = self.pos(ti, return_internal=True), self.vel(ti, return_internal=True)
        return (self._sim._conserv_ext_force.acc(pos=pos, mass=self.mass, t=t_phys)
                + self._sim._base_ext_force.acc(pos=pos, vel=vel, mass=self.mass, t=t_phys))

    @unit_handler('acceleration')
    def external_ax(self, t=...):
        '''
        x-component of external acceleration on each particle at *t*.

        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        external_ax : (len(t), n_particles) array or (n_particles,) array
            x-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        return self.external_acc(t=t, return_internal=True)[..., 0]

    @unit_handler('acceleration')
    def external_ay(self, t=...):
        '''
        y-component of external acceleration on each particle at *t*.

        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        external_ay : (len(t), n_particles) array or (n_particles,) array
            y-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        return self.external_acc(t=t, return_internal=True)[..., 1]

    @unit_handler('acceleration')
    def external_az(self, t=...):
        '''
        z-component of external acceleration on each particle at *t*.

        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        external_az : (len(t), n_particles) array or (n_particles,) array
            z-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        return self.external_acc(t=t, return_internal=True)[..., 2]
