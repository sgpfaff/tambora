"""
Simulation class for ezfalcon.
"""

import numpy as np
from .component import Component
from ..dynamics import (_runner, self_gravity, BaseForce, SelfGravityForce, NullSelfGravity,
                        ConservativeForce, ExternalGalpyPotential, 
                        SELF_GRAVITY_METHODS, INTEGRATORS)

from ..dynamics.forces.CompositeForce import _CompositeConservative, _CompositePlain
from ..util.units import unit_handler, KMS_TO_KPCGYR
import warnings
import inspect
from typing import Optional
from ._decorators import _USE_CACHED_DEFAULT, _resolve_use_cached, _resolve_t


class Sim:
    """
    Self-gravitating N-body simulation.
    """

    def __init__(self):
        self._host = None

        # Particle arrays -- built incrementally by add_particles()
        self._init_pos = None     # (N, 3) kpc
        self._init_vel = None     # (N, 3) kpc/Gyr
        self._mass = None         # (N,) Msun
        self._slices = {}         # component name -> slice

        # Snapshot arrays -- populated by run()
        self._positions = None       # (n_snap, N, 3) kpc
        self._velocities = None      # (n_snap, N, 3) kpc/Gyr
        self._cached_self_acc = None # (n_snap, N, 3) kpc/Gyr^2
        # self._cached_ext_acc = None  # (n_snap, N, 3) kpc/Gyr^2
        self._cached_self_pot = None # (n_snap, N)    kpc^2/Gyr^2
  
        self._times = None           # (n_snap,) Gyr
        self._has_run = False
        self._self_gravity_on = True
        self._self_gravity_force = NullSelfGravity()
        self._conserv_ext_force = _CompositeConservative([]) # ConservativeForce external forces
        self._base_ext_force = _CompositePlain([]) # BaseForce external forces

    def _ti(self, t, vectorized=True):
        """
        Resolve *t* to a snapshot index.

        int -> direct index, float -> nearest time.
        """
        if isinstance(t, (int, np.integer)) or t == ...:
            if t == ...:
                if vectorized:
                    return t
                else:
                    raise TypeError("This method is not vectorized, so t cannot be a list or ellipse. Please provide" \
                    " an integer index or a float time.")
            else:
                if t > len(self._times) - 1 or t < -len(self._times):
                    print(f"Time index {t} is out of bounds for simulation with {len(self._times)} snapshots. Please provide an index within [-{len(self._times)}, {len(self._times)-1}].")
                    raise IndexError(f"Time index {t} is out of bounds for simulation with {len(self._times)} snapshots. Please provide an index within [-{len(self._times)}, {len(self._times)-1}].")
                else:
                    return int(t)
        else:
            if not isinstance(t, (float, np.floating)):
                raise TypeError("t must be an int index, a float time, or ellipsis.")
            else:
                if t < self._times[0] or t > self._times[-1]:
                    raise ValueError(f"t={t} Gyr is out of bounds for simulation time range [{self._times[0]}, {self._times[-1]}] Gyr.")
                else:
                    return int(np.argmin(np.abs(self._times - t)))

    def __getattr__(self, name):
        '''
        Access components as attributes, e.g. sim.sat.pos(t=10).
        '''
        if name.startswith("_"):
            raise AttributeError(name)
        slices = self.__dict__.get("_slices", {})
        if name in slices:
            return Component(self, slices[name], name)
        raise AttributeError(
            f"\'{type(self).__name__}\' has no attribute or component named {name!r}"
        )
    
    # --- Setup ---------------------------------------------------------------------------------                                                   

    def turn_self_gravity_on(self):
        '''
        Turn self-gravity on for the simulation. 
        This is on by default.
        '''
        
        self._self_gravity_on = True

    def turn_self_gravity_off(self):
        '''
        Turn self-gravity off for the simulation.

        Methods the acceleration and potential due
        to self-gravity will be zero.
        '''
        self._self_gravity_force = NullSelfGravity()
        self._self_gravity_on = False

    def add_particles(self, name, pos, vel, mass):
        """
        Add a named particle component.
        Provide (pos, vel, mass) directly [kpc, km/s, Msun].

        Parameters
        ----------
        name : str
            Name of the component, e.g. 'sat' or 'host'.
        pos : (N, 3) array
            Initial positions of particles.
            Units: `kpc`
        vel : (N, 3) array
            Initial velocities of particles.
            Units: `km/s`
        mass : (N,) array
            Masses of particles.
            Units: `Msun`

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the component already exists or if the input arrays have incompatible shapes.
        TypeError
            If the input types are incorrect.
        RuntimeError
            If the simulation has already been run.
        """
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64) * KMS_TO_KPCGYR
        mass = np.asarray(mass, dtype=np.float64)
        if self._has_run:
            raise RuntimeError("Cannot add components after run()")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name in self._slices:
            raise ValueError(f"Component \'{name}\' already exists.")
        if pos.ndim != 2  or pos.shape[1] != 3:
            raise ValueError(f"pos must be shape (N, 3), received {pos.shape}")
        if vel.ndim != 2  or vel.shape[1] != 3:
                raise ValueError(f"vel must be shape (N, 3), received {vel.shape}")
        if mass.ndim != 1:
            raise ValueError(f"mass must be shape (N,), received {mass.shape}")
        if not (pos.shape[0] == vel.shape[0] == mass.shape[0]):
            raise ValueError(f"pos, vel, mass must have same number of particles, received {pos.shape[0]}, {vel.shape[0]}, {mass.shape[0]}.")

        # Build slice and append to flat arrays
        n = pos.shape[0]
        offset = 0 if self._mass is None else self._mass.shape[0]
        self._slices[name] = slice(offset, offset + n)

        if self._mass is None:
            self._init_pos = pos
            self._init_vel = vel
            self._mass = mass
        else:
            self._init_pos = np.concatenate([self._init_pos, pos])
            self._init_vel = np.concatenate([self._init_vel, vel])
            self._mass = np.concatenate([self._mass, mass])

        # Build single-snapshot arrays so accessors work immediately
        N = self._mass.shape[0]
        self._positions = self._init_pos.reshape(1, N, 3)
        self._velocities = self._init_vel.reshape(1, N, 3)
        self._times = np.array([0.0])

    def add_external_force(self, force: BaseForce):
        '''
        Add an external force to the simulation.

        Parameters
        ----------
        force : BaseForce
            An instance of a subclass of BaseForce representing an external force.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If force is an instance of SelfGravity. Only accepts external
            forces.
        '''
        if isinstance(force, SelfGravityForce):
            raise TypeError("The provided force is a self-gravity force, not an external force. Please provide a ConservativeForce or BaseForce subclass.")
        elif isinstance(force, ConservativeForce):
            self._conserv_ext_force = self._conserv_ext_force  + force
        elif isinstance(force, BaseForce):
            self._base_ext_forces = self._base_ext_forces + force
        else:
            raise TypeError(
                f"Expected a ConservativeForce or BaseForce subclass, "
                f"got {type(force).__name__!r}."
            )
        
    def add_external_pot(self, potential):
        '''
        Add an external potential to the simulation.

        Parameters
        ----------
        pot : galpy.potential.Potential
            External potential to add.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the potential is not a galpy Potential object.
        
        Warnings
        --------
        UserWarning
            If the provided galpy potential has physical outputs turned off.
        '''
        try:
            import galpy
        except ImportError:
            raise ImportError(
                "galpy is required for external potentials. "
                "Install it with: pip install ezfalcon[galpy]")
        if isinstance(potential, galpy.potential.Potential):
            force = ExternalGalpyPotential(potential)
            self.add_external_force(force)
        else:
            raise TypeError("External potential must be a galpy Potential object.")
        
    def add_subhalos(self, pos, vel, mass):
        '''
        Add Plummer sphere subhalos as a component to the simulation. 

        Parameters
        ----------
        pos : (N, 3) array
            Initial positions of subhalos.
            Units: `kpc`
        vel : (N, 3) array
            Initial velocities of subhalos.
            Units: `km/s`
        mass : (N,) array
            Masses of subhalos.
            Units: `Msun`

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the input arrays have incompatible shapes.
        TypeError
            If the input types are incorrect.
        RuntimeError
            If the simulation has already been run.
        '''
        raise NotImplementedError("Adding subhalos as Plummer spheres is not yet implemented. Please sample subhalo particles with galpysampler() and add them as a component with add_particles().")

    def tag(self, name, mask):
        '''
        Define a new component based on a mask.
        '''
        raise NotImplementedError("Tagging components by mask is not yet implemented.")
    
    def _resolve_eps(self, eps):
        """
        Convert *eps* to a flat (N,) array.
        
        Accepts:
        - scalar: same softening for all particles
        - dict: ``{component_name: scalar_or_array}`` for every component.
          Each value is either a scalar (broadcast to all particles in that
          component) or an array whose length matches the component's particle
          count.  All components must be present.
        """
        if not isinstance(eps, dict):
            if isinstance(eps, (int, float, np.number)):
                return float(eps)
            else:
                raise TypeError(f"eps must be a scalar or dict, got {type(eps)}")
        else:
            # check for missing keys
            missing_keys = set(self._slices.keys()) - set(eps.keys())
            if missing_keys:
                raise ValueError(f"eps dict is missing components: {missing_keys}. Please specify eps for all components: {set(self._slices.keys())}")
            extra_keys = set(eps.keys()) - set(self._slices.keys())
            if extra_keys:
                raise ValueError(f"eps dict has unknown components: {extra_keys}. Known components are: {set(self._slices.keys())}")    
            N = self._mass.shape[0]
            eps_flat = np.empty(N, dtype=np.float64)
            
            for name, val in eps.items():
                s = self._slices[name]
                n_comp = s.stop - s.start
                if isinstance(val, (int, float, np.number)):
                    eps_flat[s] = float(val)
                elif isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == n_comp:
                    eps_flat[s] = val
                else:
                    raise ValueError(f"eps[{name!r}] must be a scalar or 1D array of length {n_comp}, got {type(val)} with shape {val.shape}")
            return eps_flat
        
    # --- Running the Simulation ------------------------------------------------------------------------------------------

    def run(self, t_end: float, dt: float, dt_out: float, 
            method: Optional[str] = 'auto', integration_method: str = 'leapfrog',
            cache_self_gravity_acc: bool = True, cache_self_gravity_pot: bool = True, 
            **kwargs):
        """
        Run the simulation to *t_end* [Gyr].

        Parameters
        ----------
        t_end : float
            End time of the simulation.
            Units: `Gyr`
        dt : float
            Timestep for integration.
            Units: `Gyr`
        dt_out : float
            Output interval. Must be a multiple of dt.
            Units: `Gyr`
        sg_method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
        integration_method: str, optional
            Method to use for integration. Currently only leapfrog is available.
        cache_self_gravity_acc : bool, optional
            Whether to cache the self-gravity acceleration at each output snapshot. Default is True.
        cache_self_gravity_pot : bool, optional
            Whether to cache the self-gravitational potential at each output snapshot. Default is True.
        **kwargs 
            Additional keyword arguments to pass to the gravity method. 

            eps can be provided as:

            - scalar: same softening for every particle.
            - dict: ``{component_name: scalar_or_array}`` per component.
              Each value is a scalar (applied to all particles in that
              component) or an array matching the component's particle count.
              All components must be present in the dict.

            Other kwargs for 'falcON':
            
            - theta (float, optional): Tree opening angle. Default is 0.6. Smaller = more accurate but slower.
            - kernel (int, optional): Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.
        """
        if 'eps' in kwargs:
            kwargs['eps'] = self._resolve_eps(kwargs['eps'])

        if method == 'auto':
            method = 'falcON' if self._self_gravity_on else None

        if self._self_gravity_on:
            supported = sorted(m for m in SELF_GRAVITY_METHODS if m is not None)
            if method not in SELF_GRAVITY_METHODS:
                raise ValueError(
                    f"Unknown method {method!r} for self-gravity. Supported methods: {supported}")
            solver_cls = SELF_GRAVITY_METHODS[method]
            valid = set(inspect.signature(solver_cls.__init__).parameters) - {'self'}
            invalid = set(kwargs) - valid
            if invalid:
                raise ValueError(
                    f"{invalid} is (are) invalid kwarg(s) for {method!r} self-gravity method. "
                    "Only kwargs for self-gravity methods are allowed."
                )
            self._self_gravity_force = solver_cls(**kwargs)
        else:
            if method is not None:
                raise UserWarning("Self-gravity method is not None but self-gravity is turned off. " \
                "Self-gravity will not be used during integration.")
        integrator = INTEGRATORS[integration_method]()

        (self._positions, self._velocities, 
         self._times, self._cached_self_acc, 
         self._cached_self_pot) = _runner(
                    self._init_pos, self._init_vel, self._mass,
                    integrator,
                    self._self_gravity_force,
                    self._conserv_ext_force,
                    self._base_ext_force,
                    t_end = t_end,
                    dt = dt,
                    dt_out = dt_out,
                    return_self_gravity_pot = cache_self_gravity_pot,
                    return_self_gravity_acc = cache_self_gravity_acc,
                )
        self._has_run = True

    # --- Position Accessors -----------------------------------------------------------------

    @unit_handler('length')
    def pos(self, t=...) -> np.ndarray:
        '''
        Particle positions (x,y,z) at *t*.

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
            Units: {unit}
        '''
        return self._positions[self._ti(t)]

    @unit_handler('length')
    def x(self, t=...) -> np.ndarray:
        '''
        Particle x-positions of all particles at *t*.

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
        return self._positions[self._ti(t), :, 0]
 
    @unit_handler('length')
    def y(self, t=...) -> np.ndarray:
        '''
        Particle y-positions of all particles at *t*.

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
        return self._positions[self._ti(t), :, 1]

    @unit_handler('length')
    def z(self, t=...) -> np.ndarray:
        '''
        Particle z-positions of all particles at *t*.

        Units: `kpc`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.

        Returns
        -------
        z : (len(t), n_particles) array or (n_particles,) array
            z-positions at *t*.
            Units: `kpc`
        '''
        return self._positions[self._ti(t), :, 2]

    @unit_handler('length')
    def r(self, t=...) -> np.ndarray:
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

    def phi(self, t=...) -> np.ndarray:
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
    
    def theta(self, t=...) -> np.ndarray:
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
    def cylR(self, t=...) -> np.ndarray:
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
        return np.sqrt(self.x(t, return_internal=True)**2 + 
                       self.y(t, return_internal=True)**2)
    
    # --- Velocity Accessors -----------------------------------------------------------------
    
    @unit_handler('velocity')
    def vel(self, t=...) -> np.ndarray:
        '''
        Particle velocities (vx,vy,vz) at *t*.

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
        return self._velocities[self._ti(t)]

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
        return self._velocities[self._ti(t), :, 0]

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
        return self._velocities[self._ti(t), :, 1]

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
        return self._velocities[self._ti(t), :, 2]

    @unit_handler('velocity')
    def vr(self, t=...) -> np.ndarray:
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
    def vphi(self, t=...) -> np.ndarray:
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
                self.y(t, return_internal=True) * self.vx(t, return_internal=True)) / 
                self.cylR(t, return_internal=True)**2)
    
    @unit_handler('velocity')
    def vtheta(self, t=...) -> np.ndarray:
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
    def cylvR(self, t=...) -> np.ndarray:
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
                self.y(t, return_internal=True) * self.vy(t, return_internal=True)) 
                / self.cylR(t, return_internal=True))

    # --- Momentum Accessors -----------------------------------------------------------------
    
    @unit_handler('momentum')
    def p(self, t=...) -> np.ndarray:
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
        return self._mass[:, None] * self.vel(t, return_internal=True)
    
    @unit_handler('momentum')
    def px(self, t=...) -> np.ndarray:
        '''
        x-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        px : (len(t), n_particles) array or (n_particles,) array
            x-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self._mass * self.vx(t, return_internal=True)

    @unit_handler('momentum')
    def py(self, t=...) -> np.ndarray:
        '''
        y-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        py : (len(t), n_particles) array or (n_particles,) array
            y-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self._mass * self.vy(t, return_internal=True)
    
    @unit_handler('momentum')
    def pz(self, t=...) -> np.ndarray:
        '''
        z-component of particle momenta at *t*.

        Units: `Msun km / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        pz : (len(t), n_particles) array or (n_particles,) array
            z-component of momenta at *t*.
            Units: `Msun km / s`
        '''
        return self._mass * self.vz(t, return_internal=True)

    @unit_handler('angular_momentum')
    def L(self, t=..., center_pos=None, center_vel=None) -> np.ndarray:
        '''
        Angular momentum of particles at *t*

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
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
    def Lx(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]) -> np.ndarray:
        '''
        x-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
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
        return self.L(t, center_pos=center_pos, center_vel=center_vel, 
                      return_internal=True)[..., 0]
    
    @unit_handler('angular_momentum')
    def Ly(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]) -> np.ndarray:
        '''
        y-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
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
        return self.L(t, center_pos=center_pos, center_vel=center_vel, 
                      return_internal=True)[..., 1]
    
    @unit_handler('angular_momentum')
    def Lz(self, t=..., center_pos=[0,0,0], center_vel=[0,0,0]) -> np.ndarray:
        '''
        z-component of particle angular momentum at *t* about *center*.

        Units: `Msun km^2 / s`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
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
        return self.L(t, center_pos=center_pos, center_vel=center_vel, 
                      return_internal=True)[..., 2]
    
    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #

    @unit_handler('energy')
    def compute_external_pot(self, t=...) -> np.ndarray:
        '''
        External potential of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        ext_pot : (len(t), n_particles) array
            External potential at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        ti = self._ti(t, vectorized=True)
        
        if isinstance(ti, (int, np.integer)):
            t_phys = self._times[ti]
            ext_pot = self._conserv_ext_force.potential(pos=self.pos(t=ti, return_internal=True), 
                                                        mass=self.mass, t=t_phys) if self._conserv_ext_force is not None else 0.0
        else:
            warnings.warn("Computing external potential on-the-fly for multiple snapshots may be slow.")
            times = self._times
            ext_pot = np.zeros((len(times), self._mass.shape[0]))
            for i, t_i in enumerate(times):
                ext_pot[i] += self._conserv_ext_force.potential(pos=self.pos(t=t_i, return_internal=True), 
                                                                mass=self.mass, t=t_i,) if self._conserv_ext_force is not None else 0.0
        return self._mass * ext_pot
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def self_potential(self, t=..., use_cached=True, method=None, **kwargs) -> np.ndarray:
        '''
        Self-gravitational potential of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential if available. Default is True.
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
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun km^2 / s^2`
        '''
        if use_cached and self._cached_self_pot is not None:
            return self._mass * self._cached_self_pot[self._ti(t, vectorized=True)]
        elif use_cached and self._cached_self_pot is None:
            raise ValueError("Cached self-potential is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            _, self_pot = self_gravity(self.pos(t=self._ti(t, vectorized=False), return_internal=True),
                                       self._mass, method=method, **kwargs)
            return self._mass * self_pot
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def PE(self, t=..., use_cached=True, method=None, **kwargs) -> np.ndarray:
        '''
        Total potential energy of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential if available. Default is True.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Gravitational softening length (`kpc`)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps: Gravitational softening length (`kpc`)

        Returns
        -------
        PE : (len(t), n_particles) array
            Total potential energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        return (self.self_potential(t=t, method=method, use_cached=use_cached, 
                                   return_internal=True, **kwargs) + 
                self.compute_external_pot(t=t, return_internal=True)
        )
    # --- Kinetic Energy --- #
    @unit_handler('energy')
    def KE(self, t=...) -> np.ndarray:
        '''
        Kinetic energy of particles at *t*.

        Units: `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.

        Returns
        -------
        KE : (len(t), n_particles) array
            Kinetic energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        '''
        return 0.5 * self._mass * np.sum(self.vel(t=t, return_internal=True) ** 2, axis=-1)

    # --- Total Energy --- #

    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def energy(self, t=..., use_cached=True, method=None, **kwargs):
        """
        Energy of particles at *t*.
        
        Units:  `Msun km^2 / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration if available. Default is True.
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
        return self.KE(t=t, return_internal=True) + self.PE(t=t, method=method, use_cached=use_cached,
                                                            return_internal=True, **kwargs)
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def system_energy(self, t=..., use_cached=True, method=None,  **kwargs):
        r"""
        Total conserved system energy at *t*.
        
        Units: `Msun km^2 / s^2`

        .. math::

            E = \sum_i \frac{1}{2} m_i |v_i|^2 + \frac{1}{2} \sum_i m_i \Phi_{\text{self},i} + \sum_i m_i \Phi_{\text{ext},i}


        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        use_cached : bool, optional
            Whether to use cached self-potential from integration 
            if available. Default is True.
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
        energy : float
            Total energy of the system at time t.
            Units: :math:`Msun km^2 / s^2`
        """
        return (np.sum(self.KE(t=t, return_internal=True), axis=-1) + 
                0.5 * np.sum(self.self_potential(t=t, method=method, 
                                                 use_cached=use_cached, return_internal=True, **kwargs), axis=-1) + 
                np.sum(self.compute_external_pot(t=t, return_internal=True), axis=-1)
                )
    
    @_resolve_use_cached
    def dE(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        Percent change in total energy over the simulation time.

        Parameters
        ----------
        method : str
            Method to use for computing self-gravity. Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps: Gravitational softening length (kpc)
        use_cached : bool, optional
            Whether to use cached self-potential from integration if available. Default is True.

        Returns
        -------
        dE : (n_snaps,) array
            Percent change in total energy at each snapshot.
        '''
        if use_cached:
            Es = self.system_energy(t=t, use_cached=use_cached, method=method, return_internal=True, **kwargs)
            E0 = Es[0]
        else:
            if t is ...:
                Es = np.array([self.system_energy(t=t_i, use_cached=False, method=method, return_internal=True, **kwargs) for t_i in self._times])
                E0 = Es[0]
            else:
                Es = self.system_energy(t=t, use_cached=False, method=method, return_internal=True, **kwargs)
                E0 = self.system_energy(t=0, use_cached=False, method=method, return_internal=True, **kwargs)
        return np.abs((Es - E0) / E0)

    # --- Acceleration Accessors -----------------------------------------------------------------

    # --- Self-Gravity --- #

    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_gravity(self, t=..., use_cached=True, method=None,  **kwargs):
        '''
        The self-gravity acceleration (ax, ay, az) 
        of each particle at *t*.
        
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
        if self._self_gravity_on:
            if use_cached and self._cached_self_acc is not None:
                return self._cached_self_acc[self._ti(t, vectorized=True)]
            elif use_cached and self._cached_self_acc is None:
                raise ValueError("Cached self-gravity is not available. Please set use_cached to False and provide a method for computing self-gravity.")
            else:
                return self_gravity(self.pos(self._ti(t, vectorized=False), return_internal=True), 
                                    self.mass, method=method, **kwargs)[0]
        else:
            return np.zeros_like(self.pos(t=t, return_internal=True))
        
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_ax(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        x-component of self-gravity acceleration on each particle at *t*.
        
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
                                return_internal=True, **kwargs)[..., 0]
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_ay(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        y-component of self-gravity acceleration on each particle at *t*.
        
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
                                 return_internal=True, **kwargs)[..., 1]
    
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('acceleration')
    def self_az(self, t=..., use_cached=True, method=None, **kwargs):
        '''
        z-component of self-gravity acceleration on each particle at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is -1, which returns the value at the last snapshot.
        method : str, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON': Use the fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': Use direct summation.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration
            if available. Default is True.
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
                                 return_internal=True, **kwargs)[..., 2]

    # --- External Acceleration --- #

    @unit_handler('acceleration')
    def external_acc(self, t=-1):
        '''
        Total external acceleration on each particle at *t*.
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
        ti = self._ti(t)
        t_phys = self._times[ti]
        # ext_acc = np.zeros_like(self._velocities[ti])
        ext_acc = (self._conserv_ext_force.acc(pos=self.pos(ti, return_internal=True), mass=self.mass, t=t_phys) + self._base_ext_force.acc(pos=self.pos(ti, return_internal=True), 
                                  vel=self.vel(ti, return_internal=True), 
                                  mass=self.mass, 
                                  t=t_phys))# if self._base_ext_force is not None else 0.0)
        # ((self._conserv_ext_force.acc(pos=self.pos(ti, return_internal=True), mass=self.mass, t=t_phys) 
        #              if self._conserv_ext_force is not None else 0.0) +
        # (self._base_ext_force.acc(pos=self.pos(ti, return_internal=True), 
        #                           vel=self.vel(ti, return_internal=True), 
        #                           mass=self.mass, 
        #                           t=t_phys) if self._base_ext_force is not None else 0.0)
        return ext_acc
    
    @unit_handler('acceleration')
    def external_ax(self, t=-1):
        '''
        x-component of external acceleration on each particle at *t*.
        
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
        y-component of external acceleration on each particle at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int or None, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time.
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
        z-component of external acceleration on each particle at *t*.
        
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
    
    # --- Diagnostics -----------------------------------------------------------------

    @_resolve_use_cached
    def plot_energy_diagnostic(self, method=None, use_cached=True, nsnap=None, 
                        filename=None, **kwargs):
        '''
        Plot global energy conservation as a function of 
        time.
        
        Parameters
        ----------
        method : str
            Method to use for computing self-gravity. Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct': direct summation.
        use_cached : bool, optional
            Whether to use cached self-gravity from integration if available. Default is True.
        nsnap : int, optional
            Number of snapshots to use. If None (default), will use all snapshots.
        filename : str, optional
            If provided, will save the plot to the given filename
            instead of showing it.
        **kwargs
            Additional keyword arguments to pass to the gravity method.
            For 'falcON', these include:
            - eps: Gravitational softening length (kpc)
            - theta: Tree opening angle
        '''
        import matplotlib.pyplot as plt

        skip_every = 1 if nsnap is None else max(1, len(self.times) // nsnap)
        
        plt.figure(figsize=(7,5))
        dE = self.dE(t=..., use_cached=use_cached, method=method, **kwargs)[::skip_every]
        plt.plot(self.times[::skip_every], dE, c='k')
        plt.yscale('log')
        plt.xlabel("Time (Gyr)")
        plt.ylabel("$|\\Delta E / E_0|$")
        plt.title("Energy Conservation")
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_momentum_diagnostic(self, filename=None,
                                 plot_components=True):
        '''
        Plot the distribution of particle momenta at *t*.

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        filename : str, optional
            If provided, will save the plot to the given filename
            instead of showing it.
        plot_components : bool, optional
            Whether to plot the time evolution of the total px, py, pz components. 
            Default is True.

        '''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        if plot_components:
            plt.plot(self.times, (np.sum(self.px(), axis=-1)-np.sum(self.px(t=0), axis=-1))/np.sum(self.px(t=0), axis=-1), alpha=0.5, lw=4, label='$p_x$')
            plt.plot(self.times, (np.sum(self.py(), axis=-1)-np.sum(self.py(t=0), axis=-1))/np.sum(self.py(t=0), axis=-1), alpha=0.5, lw=4, label='$p_y$')
            plt.plot(self.times, (np.sum(self.pz(), axis=-1)-np.sum(self.pz(t=0), axis=-1))/np.sum(self.pz(t=0), axis=-1), alpha=0.5, lw=4, label='$p_z$')
        plt.plot(self.times, np.sum((np.sum(self.p(), axis=-1)-np.sum(self.p(t=0), axis=-1)), axis=-1)/ np.sum(self.p(t=0)), c='k', label='Total')
        plt.xlabel("Time (Gyr)")
        plt.ylabel("$|\\Delta p / p_0|$")
        plt.legend()
        plt.title(f"Momentum Conservation")
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
    
    # --- Properties ---------------------------------------------------------------------
    
    @property
    def mass(self):
        return self._mass

    @property
    def times(self):
        return self._times

    # --- I/O ------------------------------------------------------------------------------

    def save(self):
        raise NotImplementedError("Saving and loading simulations is not yet supported.")
    
    def load(self):
        raise NotImplementedError("Saving and loading simulations is not yet supported.")

    # --- Output to external formats -------------------------------------------------------

    def to_galpy_orbit(self, t):
        raise NotImplementedError('Outputting galpy orbit is not yet supported.')
    