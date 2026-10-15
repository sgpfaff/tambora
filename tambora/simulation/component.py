from ..dynamics.forces.self_gravity import self_gravity
from ..dynamics import diagnostics
from ._view import _ParticleView
from ._decorators import _resolve_use_cached, _resolve_t
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

    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def self_potential(self, t=..., use_cached=True, include_all_components=True, 
                       method=None, **kwargs):
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
            Whether to use the self-potential cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing the self-potential.
            If False, will only include the particles in this component when computing the self-potential.
            Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Softening length (kpc) to use instead of the particles' own

        Returns
        -------
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun km^2 / s^2`
        '''
        if use_cached and self._sim._cached_self_pot is not None:
            # Reaching the cache implies include_all_components=True (enforced by
            # @_resolve_use_cached), so the cached all-components result is valid.
            return self.mass * self._snap(self._sim._cached_self_pot, t)
        elif use_cached and self._sim._cached_self_pot is None:
            raise ValueError("Cached self-potential is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            if 'eps' not in kwargs:
                kwargs['eps'] = self._softening(everyone=include_all_components)
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
           method=None, **kwargs):
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
            Whether to use the self-potential cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method.

            For 'falcON', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
        
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
    
    # --- Total Energy --- #
    @_resolve_use_cached
    @_resolve_t
    @unit_handler('energy')
    def energy(self, t=..., use_cached=True, include_all_components=True, 
               method=None, **kwargs):
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
            Whether to use the self-potential cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
        
        Returns
        -------
        energy : (len(t), n_particles) array
            Total energy of each particle at each snapshot.
            Units: `Msun km^2 / s^2`
        """

        return self.KE(t=t, return_internal=True) + self.PE(t=t, use_cached=use_cached, include_all_components=include_all_components, method=method, 
                                                            return_internal=True, **kwargs)
    

    # --- Virial Ratio --- #

    @_resolve_use_cached
    @_resolve_t
    def virial_ratio(self, t=..., use_cached=True, include_all_components=True,
                     method=None, **kwargs):
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
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method,
            e.g. eps, a softening length (kpc) to use instead of the particles' own.

        Returns
        -------
        virial_ratio : float or (n_snaps,) array
            Virial ratio at each snapshot.
        '''
        acc = (self.external_acc(t, return_internal=True)
               + self.self_gravity(t=t, use_cached=use_cached, include_all_components=include_all_components,
                                   method=method, return_internal=True, **kwargs))
        return diagnostics.virial_ratio(self.pos(t, return_internal=True), self.vel(t, return_internal=True),
                                    self.mass, acc)

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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
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
        if use_cached and self._sim._cached_self_acc is not None:
            return self._snap(self._sim._cached_self_acc, t)
        elif use_cached and self._sim._cached_self_acc is None:
            raise ValueError("Cached self-gravity is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            if 'eps' not in kwargs:
                kwargs['eps'] = self._softening(everyone=include_all_components)
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run, which includes all
            components. Default is True, unless ``method`` is given or
            ``include_all_components`` is False.
        include_all_components : bool, optional
            Whether to include all components in the simulation when computing 
            the self-gravity acceleration. If False, only the particles in this
            component will be used. Default is True.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
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
        self_az : (n_snaps, N) array
            z-component of self-gravity acceleration of 
            each particle at each snapshot.
            Units: `km / s^2`
        '''
        return self.self_gravity(t=t, method=method, use_cached=use_cached, 
                                 include_all_components=include_all_components,
                                 return_internal=True,**kwargs)[..., 2]

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


