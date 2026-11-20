from ..dynamics import diagnostics
from ._view import _ParticleView
from ..units import unit_handler


class Component(_ParticleView):
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

    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #

    @unit_handler('energy')
    def self_potential(self, t=..., include_all_components=True, solver=None):
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
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing the self-potential.
            If False, will only include the particles in this component when computing the self-potential.
            Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun km^2 / s^2`
        '''
        return self.mass * self._self_gravity(t, solver, include_all_components, potential=True)

    @unit_handler('energy')
    def PE(self, t=..., include_all_components=True, solver=None):
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
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        PE : (len(t), n_particles) array
            Total potential energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        
        '''
        return (self.self_potential(t=t, include_all_components=include_all_components, solver=solver, return_internal=True)
                + self.compute_external_pot(t=t, return_internal=True))

    # --- Total Energy --- #
    @unit_handler('energy')
    def energy(self, t=..., include_all_components=True, solver=None):
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
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        energy : (len(t), n_particles) array
            Total energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        """
        return (self.KE(t=t, return_internal=True)
                + self.PE(t=t, include_all_components=include_all_components, solver=solver, return_internal=True))

    # --- Virial Ratio --- #

    def virial_ratio(self, t=..., include_all_components=True, solver=None):
        r'''
        Virial ratio :math:`2K/|W|` of the particles in the component at *t*: 1 in equilibrium.

        .. math::

            K = \frac{1}{2} \sum_i m_i |v_i - v_{\rm cm}|^2, \qquad
            W = \sum_i m_i (x_i - x_{\rm cm}) \cdot a_i,

        about the component's center of mass, where :math:`a_i` is the acceleration from
        self-gravity and the external forces.

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        virial_ratio : float or (n_snaps,) array
            Virial ratio at each snapshot.
        '''
        acc = (self.external_acc(t, return_internal=True)
               + self.self_gravity(t=t, include_all_components=include_all_components, solver=solver, return_internal=True))
        return diagnostics.virial_ratio(self.pos(t, return_internal=True), self.vel(t, return_internal=True),
                                        self.mass, acc)

    # --- Acceleration Accessors -----------------------------------------------------------------


    @unit_handler('acceleration')
    def self_gravity(self, t=..., include_all_components=True, solver=None):
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
            Default is ... (ellipsis), which returns the value at all times.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        self_acc : (n_snaps, N, 3) array
            Self-gravity acceleration of each particle at each snapshot.
            [ax, ay, az]
            Units: `km / s^2`
        '''
        return self._self_gravity(t, solver, include_all_components)

    @unit_handler('acceleration')
    def self_ax(self, t=..., include_all_components=True, solver=None):
        '''
        x-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        self_ax : (n_snaps, N) array
            x-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, include_all_components=include_all_components, solver=solver, return_internal=True)[..., 0]

    @unit_handler('acceleration')
    def self_ay(self, t=..., include_all_components=True, solver=None):
        '''
        y-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        self_ay : (n_snaps, N) array
            y-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, include_all_components=include_all_components, solver=solver, return_internal=True)[..., 1]

    @unit_handler('acceleration')
    def self_az(self, t=..., include_all_components=True, solver=None):
        '''
        z-component of self-gravity acceleration on each particle in the component at *t*.
        
        Units: `km / s^2`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        solver : SelfGravityForce, optional
            A solver to compute the self-gravity afresh with, one snapshot at a time.
            By default the values the run saved are used where they apply, and the
            Sim's solver otherwise.

        Returns
        -------
        self_az : (n_snaps, N) array
            z-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, include_all_components=include_all_components, solver=solver, return_internal=True)[..., 2]

    # --- Properties ---------------------------------------------------------------------- #

    @property
    def mass(self):
        return self._sim._mass[self._sl]

    @property
    def name(self):
        """This component's name."""
        return self._name

    def __repr__(self):
        m = self.mass
        return f"Component({self._name!r}, {len(m)} particles, {m.sum():.2e} Msun)"


