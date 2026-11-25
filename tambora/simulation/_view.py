"""
Accessors shared by Sim and Component, in terms of ``self._sim``, the Sim holding the data, and
``self._sl``, which of its particles.
"""

import warnings

import numpy as np

from ..dynamics import diagnostics, SelfGravityForce
from ..units import unit_handler




class _ParticleView:
    """Accessors shared by Sim and Component."""

    def _snap(self, array, i):
        """Slice a (nsnaps, N, ...) array to these particles at snapshot *i* (None for every one)."""
        if i is None:
            return array[:, self._sl]
        return array[i, self._sl]

    def eps(self):
        """Softening length of each particle [kpc], NaN where it isn't set."""
        eps = self._sim._eps
        if eps is None:
            return None
        view = eps[self._sl]
        view.flags.writeable = False
        return view

    def _softening(self, everyone=False):
        """These particles' softening (the whole Sim's, if *everyone*), as one number if shared."""
        eps = self._sim._eps if everyone else self._sim._eps[self._sl]
        if np.isnan(eps).any():
            raise ValueError("Softening isn't set for every particle here. "
                             "Give it with add_particles(..., eps=...).")
        return float(eps[0]) if np.all(eps == eps[0]) else eps

    def _self_gravity(self, i, solver, everyone, potential=False):
        """These particles' self-gravity acceleration (or specific potential) at snapshot *i*.

        With *everyone*, every particle of the Sim is a source. Without, only these particles are.
        Without *solver*, the values the run saved are used if it saved them and *everyone* is a
        source. Otherwise they are computed for one snapshot, by *solver* or by the Sim's own.
        """
        sim = self._sim
        saved = sim._cached_self_pot if potential else sim._cached_self_acc
        if solver is None and everyone and saved is not None:
            return self._snap(saved, i)
        if solver is None:
            solver = sim._self_gravity_force
            if solver is None:
                raise RuntimeError("The run's self-gravity solver couldn't be loaded. Pass solver=... to compute self-gravity.")
        elif not isinstance(solver, SelfGravityForce):
            raise TypeError(f"solver must be a self-gravity solver, e.g. DirectSum(). Got {type(solver).__name__!r}.")
        if i is None:
            raise TypeError("Cannot compute on-the-fly for all times. Please provide an integer index or a float "
                            "time for t. You will have to manually loop over snapshots.")
        of = sim if everyone else self
        compute = solver.potential if potential else solver.acc
        result = compute(of.pos(index=i, return_internal=True), of.mass, of._softening())
        return result[self._sl] if everyone else result

    # --- Position Accessors -------------------------------------------------------------------

    @unit_handler('length')
    def pos(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        pos : (len(t), n_particles, 3) array or (n_particles, 3) array
            Positions at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._positions, i)

    @unit_handler('length')
    def x(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        x : (len(t), n_particles) array or (n_particles,) array
            x-positions at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._positions, i)[..., 0]

    @unit_handler('length')
    def y(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        y : (len(t), n_particles) array or (n_particles,) array
            y-positions at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._positions, i)[..., 1]

    @unit_handler('length')
    def z(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        z : (len(t), n_particles) array or (n_particles,) array
            z-positions at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._positions, i)[..., 2]

    @unit_handler('length')
    def r(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        r : (len(t), n_particles) array or (n_particles,) array
            Spherical radii at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        pos = self.pos(index=i, return_internal=True)
        return np.linalg.norm(pos, axis=-1)

    def phi(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        phi : (len(t), n_particles) array or (n_particles,) array
            Azimuthal angles at *t*.
            Units: radians
        '''
        i = self._sim._ti(t, index)
        pos = self.pos(index=i, return_internal=True)
        return np.arctan2(pos[..., 1], pos[..., 0])

    def theta(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        theta : (len(t), n_particles) array or (n_particles,) array
            Polar angles at *t*.
            Units: radians
        '''
        i = self._sim._ti(t, index)
        pos = self.pos(index=i, return_internal=True)
        r = np.linalg.norm(pos, axis=-1)
        return np.arccos(pos[..., 2] / r)

    @unit_handler('length')
    def cylR(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        R : (len(t), n_particles) array or (n_particles,) array
            Cylindrical radii at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return np.sqrt(self.x(index=i, return_internal=True)**2 + self.y(index=i, return_internal=True)**2)

    # --- Velocity Accessors -------------------------------------------------------------------

    @unit_handler('velocity')
    def vel(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vel : (len(t), n_particles, 3) array or (n_particles, 3) array
            Velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._velocities, i)

    @unit_handler('velocity')
    def vx(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vx : (len(t), n_particles) array or (n_particles,) array
            x-component of velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._velocities, i)[..., 0]

    @unit_handler('velocity')
    def vy(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vy : (len(t), n_particles) array or (n_particles,) array
            y-component of velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._velocities, i)[..., 1]

    @unit_handler('velocity')
    def vz(self, t=..., *, index=None) -> np.ndarray:
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vz : (len(t), n_particles) array or (n_particles,) array
            z-component of velocities at *t*.

            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        return self._snap(self._sim._velocities, i)[..., 2]

    @unit_handler('velocity')
    def vr(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vr : (len(t), n_particles) array or (n_particles,) array
            Radial velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        pos = self.pos(index=i, return_internal=True)
        vel = self.vel(index=i, return_internal=True)
        vr = np.sum(pos * vel, axis=-1) / self.r(index=i, return_internal=True)
        return vr

    @unit_handler('angular_velocity')
    def vphi(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vphi : (len(t), n_particles) array or (n_particles,) array
            Azimuthal angular velocities at *t*.
            Units: `km/s/kpc`
        '''
        i = self._sim._ti(t, index)
        return ((self.x(index=i, return_internal=True) * self.vy(index=i, return_internal=True) - 
                self.y(index=i, return_internal=True) * self.vx(index=i, return_internal=True)) 
                / self.cylR(index=i, return_internal=True)**2)

    @unit_handler('velocity')
    def vtheta(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        vtheta : (len(t), n_particles) array or (n_particles,) array
            Polar velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        r = self.r(index=i, return_internal=True)
        return (
            ((self.z(index=i, return_internal=True) * 
              (self.x(index=i, return_internal=True) * self.vx(index=i, return_internal=True) + self.y(index=i, return_internal=True) * self.vy(index=i, return_internal=True)))
             - self.cylR(index=i, return_internal=True)**2 * self.vz(index=i, return_internal=True)) 
            / (r * self.cylR(index=i, return_internal=True))
        )

    @unit_handler('velocity')
    def cylvR(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        cylvR : (len(t), n_particles) array or (n_particles,) array
            Cylindrical radial velocities at *t*.
            Units: `km / s`
        '''
        i = self._sim._ti(t, index)
        return ((self.x(index=i, return_internal=True) * self.vx(index=i, return_internal=True) + 
                 self.y(index=i, return_internal=True) * self.vy(index=i, return_internal=True)) / 
                 self.cylR(index=i, return_internal=True))

    # --- Momentum Accessors -------------------------------------------------------------------

    @unit_handler('momentum')
    def p(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.
        
        Returns
        -------
        momentum : (len(t), n_particles, 3) array or (n_particles, 3) array
            Momenta at *t*.
            Units: `Msun km / s`
        '''
        i = self._sim._ti(t, index)
        return self.mass[:, None] * self.vel(index=i, return_internal=True)

    @unit_handler('momentum')
    def px(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        px : (len(t), n_particles) array or (n_particles,) array
            x-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        i = self._sim._ti(t, index)
        return self.mass * self.vx(index=i, return_internal=True)

    @unit_handler('momentum')
    def py(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        py : (len(t), n_particles) array or (n_particles,) array
            y-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        i = self._sim._ti(t, index)
        return self.mass * self.vy(index=i, return_internal=True)

    @unit_handler('momentum')
    def pz(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        pz : (len(t), n_particles) array or (n_particles,) array
            z-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        i = self._sim._ti(t, index)
        return self.mass * self.vz(index=i, return_internal=True)

    @unit_handler('angular_momentum')
    def L(self, t=..., center_pos=None, center_vel=None, *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.
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
        i = self._sim._ti(t, index)
        r = self.pos(index=i, return_internal=True)
        v = self.vel(index=i, return_internal=True)
        if center_pos is not None:
            r = r - np.asarray(center_pos)
        if center_vel is not None:
            v = v - np.asarray(center_vel)
        return self.mass[:, None] * np.cross(r, v)

    @unit_handler('angular_momentum')
    def Lx(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0], *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.
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
        i = self._sim._ti(t, index)
        return self.L(index=i, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 0]

    @unit_handler('angular_momentum')
    def Ly(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0], *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.
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
        i = self._sim._ti(t, index)
        return self.L(index=i, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 1]

    @unit_handler('angular_momentum')
    def Lz(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0], *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.
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
        i = self._sim._ti(t, index)
        return self.L(index=i, center_pos=center_pos, center_vel=center_vel, return_internal=True)[..., 2]

    # --- Center of Mass -----------------------------------------------------------------------

    @unit_handler('length')
    def com_pos(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        com_pos : (len(t), 3) array or (3,) array
            Center of mass at *t*.
            Units: `kpc`
        '''
        i = self._sim._ti(t, index)
        return diagnostics.center_of_mass(self.pos(index=i, return_internal=True), self.mass)

    @unit_handler('velocity')
    def com_vel(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        com_vel : (len(t), 3) array or (3,) array
            Velocity of the center of mass at *t*.
            Units: `km/s`
        '''
        i = self._sim._ti(t, index)
        return diagnostics.center_of_mass(self.vel(index=i, return_internal=True), self.mass)

    # --- External Potential -------------------------------------------------------------------

    @unit_handler('energy')
    def compute_external_pot(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        ext_pot : (len(t), n_particles) array or (n_particles,) array
            External potential at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        i = self._sim._ti(t, index)
        if i is not None:
            t_phys = self._sim._times[i]
            ext_pot = self._sim._conserv_ext_force.potential(
                pos=self.pos(index=i, return_internal=True), mass=self.mass, t=t_phys)
        else:
            warnings.warn("Computing external potential on-the-fly for multiple snapshots may be slow.")
            times = self._sim._times
            ext_pot = np.zeros((len(times), self.mass.shape[0]))
            for k, t_k in enumerate(times):
                ext_pot[k] += self._sim._conserv_ext_force.potential(
                    pos=self.pos(index=k, return_internal=True), mass=self.mass, t=t_k)
        return self.mass * ext_pot

    # --- Kinetic Energy -----------------------------------------------------------------------

    @unit_handler('energy')
    def KE(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        KE : (len(t), n_particles) array
            Kinetic energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        i = self._sim._ti(t, index)
        return 0.5 * self.mass * np.sum(self.vel(index=i, return_internal=True) ** 2, axis=-1)

    # --- External Acceleration ----------------------------------------------------------------

    @unit_handler('acceleration')
    def external_acc(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        ext_acc : (len(t), n_particles, 3) array or (n_particles, 3) array
            External acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        i = self._sim._ti(t, index)
        if i is None:
            return np.stack([self.external_acc(index=k, return_internal=True) for k in range(len(self._sim._times))])
        t_phys = self._sim._times[i]
        pos, vel = self.pos(index=i, return_internal=True), self.vel(index=i, return_internal=True)
        return (self._sim._conserv_ext_force.acc(pos=pos, mass=self.mass, t=t_phys)
                + self._sim._base_ext_force.acc(pos=pos, vel=vel, mass=self.mass, t=t_phys))

    @unit_handler('acceleration')
    def external_ax(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        external_ax : (len(t), n_particles) array or (n_particles,) array
            x-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        i = self._sim._ti(t, index)
        return self.external_acc(index=i, return_internal=True)[..., 0]

    @unit_handler('acceleration')
    def external_ay(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        external_ay : (len(t), n_particles) array or (n_particles,) array
            y-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        i = self._sim._ti(t, index)
        return self.external_acc(index=i, return_internal=True)[..., 1]

    @unit_handler('acceleration')
    def external_az(self, t=..., *, index=None):
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
        index : int, optional
            Position of the snapshot in ``sim.times``, in place of *t*. -1 is the last.

        Returns
        -------
        external_az : (len(t), n_particles) array or (n_particles,) array
            z-component of external acceleration of each particle at *t*.
            Units: `km / s^2`
        '''
        i = self._sim._ti(t, index)
        return self.external_acc(index=i, return_internal=True)[..., 2]
