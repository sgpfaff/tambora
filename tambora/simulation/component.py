import numpy as np
from ..dynamics.forces.self_gravity import self_gravity
from ._decorators import _resolve_use_cached, _resolve_t
from ..util.units import unit_handler
import warnings


class Component:
    """Slice view into one component's snapshot data."""

    def __init__(self, sim, sl, name):
        self._sim = sim
        self._name = name
        self._sl = sl

    @property
    def _has_run(self):
        return self._sim._has_run

    def _ti(self, t, vectorized=True):
        return self._sim._ti(t, vectorized=vectorized)

    def _snap(self, array, t):
        """Slice a (nsnaps, N, ...) array to this component at time t."""
        ti = self._sim._ti(t)
        if ti is ...:
            return array[:, self._sl]
        return array[ti, self._sl]


     # --- Position Accessors -----------------------------------------------------------------
    
    @unit_handler('length')
    def pos(self, t=...):
        '''
        Positions (x, y, z) of particles in the component at *t*.

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
        pos : (len(t), n_particles, 3) array or (n_particles,) array
            x, y, z positions at *t*.
            Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)
    
    @unit_handler('length')
    def x(self, t=...):
        '''
        x-positions of all particles in the component at *t*.
        
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
    def y(self, t=...):
        '''
        y-positions of all particles in the component at *t*.
        
        Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 1]

    @unit_handler('length')
    def z(self, t=...):
        '''
        z-positions of all particles in the component at *t*.
        
        Units: `kpc`
        '''
        return self._snap(self._sim._positions, t)[..., 2]
    
    @unit_handler('length')
    def r(self, t=...):
        '''
        Component spherical radii at *t*.

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
        Component azimuthal angles at *t*.
        
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
        Component polar angles at *t*.

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
        Component cylindrical radii at *t*.

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

    
     # --- Velocity Accessors -----------------------------------------------------------------

    @unit_handler('velocity')
    def vel(self, t=...):
        '''
        Velocities (vx, vy, vz) of particles in the component at *t*.
        
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
            x, y, z velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)

    @unit_handler('velocity')
    def vx(self, t=...):
        '''
        x-component of particle velocities in the component at *t*.
        
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
            x-velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 0]

    @unit_handler('velocity')
    def vy(self, t=...):
        '''
        y-component of particle velocities in the component at *t*.
        
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
            y-velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 1]

    @unit_handler('velocity')
    def vz(self, t=...):
        '''
        z-component of particle velocities in the component at *t*.
        
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
            z-velocities at *t*.
            Units: `km / s`
        '''
        return self._snap(self._sim._velocities, t)[..., 2]
    
    @unit_handler('velocity')
    def vr(self, t=...):
        '''
        Spherical coordinates radial velocities at *t*.

        The component of the velocity vector along the position vector, 
        i.e. :math:`v_r = (x*v_x + y*v_y + z*v_z) / r`.*

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

    # --- Momentum Accessors -----------------------------------------------------------------

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
    

    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #

    @unit_handler('energy')
    def compute_external_pot(self, t=...):
        '''
        External potential of particles in the 
        component at *t*.

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
        ext_pot : (len(t), n_particles) array
            External potential at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        ext_pot = np.zeros(self.mass.shape[0])
        ext_pot = self._sim._conserv_ext_force.potential(self.pos(t=t, return_internal=True), self.mass, t)
        return self.mass * ext_pot
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def self_potential(self, t=..., use_cached=True, include_all_components=True, 
                       method='falcON', **kwargs):
        '''
        Self-gravitational potential energy of the 
        particles in the component at *t*. 
        
        Units: `Msun km^2 / s^2`

        NOTE: This is the self-potential of the particles

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use cached potential energy values if available. Default is True.
            Note that all components are included if cached results are used.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing the self-potential.
            If False, will only include the particles in this component when computing the self-potential.
            Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)

        Returns
        -------
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun km^2 / s^2`
        '''
        if use_cached and self._sim._cached_self_pot is not None:
            if not include_all_components:
                warnings.warn("Using cached self-potential, which includes all particles.")
            return self.mass * self._snap(self._sim._cached_self_pot, t)
        elif use_cached and self._sim._cached_self_pot is None:
            raise ValueError("Cached self-potential is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            if include_all_components:
                _, self_pot = self_gravity(self._sim.pos(t=t, return_internal=True), self._sim._mass, method=method, 
                                           **kwargs)
                self_pot = self_pot[self._sl]
            else:
                _, self_pot = self_gravity(self.pos(t=t, return_internal=True), self.mass, method=method, 
                                           **kwargs)
        return self.mass * self_pot
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def PE(self, t=..., use_cached=True, include_all_components=True, 
           method='falcON', **kwargs):
        '''
        Total potential energy of particles 
        in the component at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use cached self-potential if available. Default is True.
            Note that all components are included if cached results are used.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Gravitational softening length (kpc)
        
        Returns
        -------
        PE : (len(t), n_particles) array
            Total potential energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        
        '''
        return (self.self_potential(t=t, method=method, use_cached=use_cached, 
                                   include_all_components=include_all_components, 
                                   return_internal=True, **kwargs) 
                        + self.compute_external_pot(t=t, return_internal=True))
    
    # --- Kinetic Energy --- #

    @unit_handler('energy')
    def KE(self, t=...):
        '''
        Kinetic energy of particles in the component at *t*.

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

    # --- Total Energy --- #
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def energy(self, t=..., use_cached=True, include_all_components=True, 
               method='falcON', **kwargs):
        """
        Energy of the particles in the component at time t.
        
        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use cached self-potential and self-gravity if available. Default is True.
            Note that all components are included if cached results are used.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps: Gravitational softening length (kpc)
        
        Returns
        -------
        energy : (len(t), n_particles) array
            Total energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        """

        return self.KE(t=t, return_internal=True) + self.PE(t=t, use_cached=use_cached, include_all_components=include_all_components, method=method, 
                                                            return_internal=True, **kwargs)
    

    # --- Acceleration Accessors -----------------------------------------------------------------


    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_gravity(self, t=..., use_cached=True, include_all_components=True, 
                     method=None,  **kwargs):
        '''
        Compute the self-gravity acceleration of each 
        particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity if available. Default is True.
            Note that all components are included if cached results are used.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        
        **kwargs
            Additional keyword arguments to pass to the gravity method.
        
        Returns
        -------
        self_acc : (n_snaps, N, 3) array
            Self-gravity acceleration of each particle at each snapshot.
            [ax, ay, az]
            Units: `km / s^2`
        '''
        if self._sim._self_gravity_on:
            if use_cached and self._sim._cached_self_acc is not None:
                return self._snap(self._sim._cached_self_acc, t)
            elif use_cached and self._sim._cached_self_acc is None:
                raise ValueError("Cached self-gravity is not available. Please set use_cached to False and provide a method for computing self-gravity.")
            else:
                if include_all_components:
                    self_acc, _ = self_gravity(self._sim.pos(t=t, return_internal=True), self._sim._mass, method=method, 
                                               **kwargs)
                    self_acc = self_acc[self._sl]
                else:
                    self_acc, _ = self_gravity(self.pos(t=t, return_internal=True), self.mass, method=method, 
                                               **kwargs)
                return self_acc
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_ax(self, t=..., use_cached=True, include_all_components=True,  
                method=None, **kwargs):
        '''
        x-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity if available. Default is True.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
        
        Returns
        -------
        self_ax : (n_snaps, N) array
            x-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, 
                                 include_all_components=include_all_components, 
                                 return_internal=True, **kwargs)[..., 0]
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_ay(self, t=..., use_cached=True, include_all_components=True,  
                method=None, **kwargs):
        '''
        y-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration
            if available. Default is True.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.
        
        Returns
        -------
        self_ay : (n_snaps, N) array
            y-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, 
                                 include_all_components=include_all_components, 
                                 return_internal=True, **kwargs)[..., 1]
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_az(self, t=..., use_cached=True, include_all_components=True, 
                method=None, **kwargs):
        '''
        z-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.        
        use_cached : bool, optional
            Whether to use cached self-gravity from integration
            if available. Default is True.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON': Use the fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': Use direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps : float
                Gravitational softening length.
                Units: kpc
            - theta : float
                Tree opening angle for pyfalcon.
                Smaller = more accurate but slower.
        
        Returns
        -------
        self_az : (n_snaps, N) array
            z-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, 
                                 include_all_components=include_all_components,
                                 return_internal=True,**kwargs)[..., 2]

    # --- External Acceleration --- #

    @unit_handler('acceleration')
    def external_acc(self, t=-1):
        '''
        Total external acceleration on each particle 
        in the component at *t*.
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        ext_acc : (n_snaps, N, 3) array
            External acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        #ext_acc = np.zeros_like(self.vel(t=t, return_internal=True))
        #for fn in self._sim._ext_acc_fns:
        ti = self._ti(t)
        t_phys = self._sim._times[ti]
        ext_acc = (self._sim._conserv_ext_force.acc(pos=self.pos(ti, return_internal=True), mass=self.mass, t=t_phys) + self._sim._base_ext_force.acc(pos=self.pos(ti, return_internal=True), 
                                  vel=self.vel(ti, return_internal=True), 
                                  mass=self.mass, 
                                  t=t_phys))
        return ext_acc
    
    @unit_handler('acceleration')
    def external_ax(self, t=-1):
        '''
        x-component of external acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_ax : (n_snaps, N) array
            x-component of external acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.external_acc(t=t, return_internal=True)[:, 0]

    @unit_handler('acceleration')
    def external_ay(self, t=-1):
        '''
        y-component of external acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_ay : (n_snaps, N) array
            y-component of external acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.external_acc(t=t, return_internal=True)[:, 1]
    
    @unit_handler('acceleration')
    def external_az(self, t=-1):
        '''
        z-component of external acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        
        Returns
        -------
        external_az : (n_snaps, N) array
            z-component of external acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''        
        return self.external_acc(t=t, return_internal=True)[:, 2]
    
    # --- Properties ---------------------------------------------------------------------- #
    
    @property
    def mass(self):
        return self._sim._mass[self._sl]


