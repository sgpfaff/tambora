"""
Simulation class for tambora.
"""

import numpy as np
from .component import Component
from ._view import _ParticleView
from ..dynamics import (_runner, self_gravity, Force, SelfGravityForce, NullSelfGravity, Conservative, ExternalPotential,
                        SELF_GRAVITY_METHODS, INTEGRATORS)

from ..dynamics.forces.CompositeForce import _CompositeConservative, _CompositePlain
from ..units import unit_handler, KMS_TO_KPCGYR
from ..dynamics import diagnostics
from ..ic import ParticleSet
import inspect
from typing import Optional
from ._decorators import _USE_CACHED_DEFAULT, _resolve_use_cached, _resolve_t


def _check_eps(eps, n):
    """``eps`` as one softening length per particle [kpc], from a number or n values."""
    arr = np.asarray(eps)
    if isinstance(eps, bool) or arr.dtype.kind not in 'iuf':
        raise TypeError(f"eps must be a number or an array of numbers, got {type(eps).__name__}")
    arr = np.array(arr, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    elif arr.shape != (n,):
        raise ValueError(f"eps must be a number or one value per particle ({n}), got shape {arr.shape}")
    if not np.all(np.isfinite(arr) & (arr >= 0)):
        raise ValueError("eps must be finite and non-negative")
    return arr


def _hook_dedup_key(hook):
    """A hook's configuration-identity key, or ``None`` if it opts out of dedup.

    Bare callables (e.g. a plain function) have no ``_dedup_key`` and so always
    opt out; ``Hook`` subclasses return ``None`` by default until they override.
    """
    getter = getattr(hook, "_dedup_key", None)
    return getter() if callable(getter) else None


def _render_table(headers, rows, align=None):
    """Render a Unicode box-drawn table as a list of lines.

    ``rows`` is a sequence of same-length string tuples; ``align`` is an
    optional per-column string of ``'l'``/``'r'`` (default all-left).
    """
    ncol = len(headers)
    align = align or 'l' * ncol
    cells = [[str(c) for c in row] for row in rows]
    widths = [max([len(headers[i])] + [r[i] and len(r[i]) or 0 for r in cells])
              for i in range(ncol)]

    def _row(vals):
        parts = [f"{v:>{widths[i]}}" if align[i] == 'r' else f"{v:<{widths[i]}}"
                 for i, v in enumerate(vals)]
        return "│ " + " │ ".join(parts) + " │"

    rule = lambda l, m, r: l + m.join("─" * (w + 2) for w in widths) + r
    out = [rule("┌", "┬", "┐"), _row(headers), rule("├", "┼", "┤")]
    out += [_row(r) for r in cells]
    out.append(rule("└", "┴", "┘"))
    return out


def _force_label(force):
    """Readable one-line label for an external force in ``Sim.__repr__``."""
    name = type(force).__name__
    backend = getattr(force, "backend", None)
    if backend is not None and hasattr(backend, "describe"):
        return f"{name}({backend.describe()})"
    pot = getattr(force, "_pot", None)
    if pot is None:
        return name
    inner = ("+".join(type(p).__name__ for p in pot)
             if isinstance(pot, (list, tuple)) else type(pot).__name__)
    return f"{name}({inner})"


def _cadence_label(cadence):
    """Readable label for a cadence, including its interval where it has one."""
    name = type(cadence).__name__
    n = getattr(cadence, "n", None)
    return f"{name}({n})" if n is not None else name


class Sim(_ParticleView):
    """
    Self-gravitating N-body simulation.
    """

    _sl = slice(None)

    @property
    def _sim(self):
        return self

    def __init__(self):
        self._host = None

        # Particle arrays -- built incrementally by add_particles()
        self._init_pos = None     # (N, 3) kpc
        self._init_vel = None     # (N, 3) kpc/Gyr
        self._mass = None         # (N,) Msun
        self._eps = None          # (N,) kpc, NaN where not set
        self._slices = {}         # component name -> slice

        # Snapshot arrays -- populated by run()
        self._positions = None       # (n_snap, N, 3) kpc
        self._velocities = None      # (n_snap, N, 3) kpc/Gyr
        self._cached_self_acc = None # (n_snap, N, 3) kpc/Gyr^2
        # self._cached_ext_acc = None  # (n_snap, N, 3) kpc/Gyr^2
        self._cached_self_pot = None # (n_snap, N)    kpc^2/Gyr^2
  
        self._times = None           # (n_snap,) Gyr
        self._has_run = False
        self._self_gravity_force = NullSelfGravity()
        self._conserv_ext_force = _CompositeConservative([]) # conservative external forces
        self._base_ext_force = _CompositePlain([]) # non-conservative external forces
        self._hooks = [] # (Hook, Cadence) pairs run during integration

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

    def add_particles(self, name, pos, vel=None, mass=None, eps=None):
        """
        Add a named particle component.
        Provide a :class:`~tambora.ic.ParticleSet`, or (pos, vel, mass) directly [kpc, km/s, Msun]::

            sim.add_particles('cluster', ps)
            sim.add_particles('cluster', pos, vel, mass)

        Parameters
        ----------
        name : str
            Name of the component, e.g. 'sat' or 'host'.
        pos : (N, 3) array or ParticleSet
            Initial positions of particles, or a ParticleSet holding the positions,
            velocities and masses.
            Units: `kpc`
        vel : (N, 3) array, optional
            Initial velocities of particles. Required unless ``pos`` is a ParticleSet.
            Units: `km/s`
        mass : (N,) array, optional
            Masses of particles. Required unless ``pos`` is a ParticleSet.
            Units: `Msun`
        eps : float or (N,) array, optional
            Softening length of the particles, one for all or one each. Required
            to run with self-gravity; it can also be set later through ``eps``.
            Units: `kpc`

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the component already exists or if the input arrays have incompatible shapes.
        TypeError
            If the input types are incorrect, or if a ParticleSet is given together with
            ``vel`` or ``mass``, or neither is given.
        RuntimeError
            If the simulation has already been run.
        """
        if isinstance(pos, ParticleSet):
            if vel is not None or mass is not None:
                raise TypeError("Pass either a ParticleSet or pos, vel and mass, not both.")
            pos, vel, mass = pos
        elif vel is None or mass is None:
            raise TypeError("vel and mass are required unless pos is a ParticleSet.")
        # Copies: the simulation's initial state can't change through the caller's arrays,
        # and a ParticleSet's read-only arrays don't make it read-only.
        pos = np.array(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64) * KMS_TO_KPCGYR
        mass = np.array(mass, dtype=np.float64)
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
        eps = np.full(pos.shape[0], np.nan) if eps is None else _check_eps(eps, pos.shape[0])

        # Build slice and append to flat arrays
        n = pos.shape[0]
        offset = 0 if self._mass is None else self._mass.shape[0]
        self._slices[name] = slice(offset, offset + n)

        if self._mass is None:
            self._init_pos = pos
            self._init_vel = vel
            self._mass = mass
            self._eps = eps
        else:
            self._init_pos = np.concatenate([self._init_pos, pos])
            self._init_vel = np.concatenate([self._init_vel, vel])
            self._mass = np.concatenate([self._mass, mass])
            self._eps = np.concatenate([self._eps, eps])

        # Build single-snapshot arrays so accessors work immediately
        N = self._mass.shape[0]
        self._positions = self._init_pos.reshape(1, N, 3)
        self._velocities = self._init_vel.reshape(1, N, 3)
        self._times = np.array([0.0])

    def _set_eps(self, sl, eps):
        """Set the softening of the particles in *sl*; a previous run's outputs are cleared."""
        if self._eps is None:
            raise RuntimeError("Add particles before setting their softening.")
        self._eps[sl] = _check_eps(eps, len(self._eps[sl]))
        if self._has_run:
            self._clear_outputs()

    def _clear_outputs(self):
        """Back to the initial conditions, as before ``run()``."""
        N = self._mass.shape[0]
        self._positions = self._init_pos.reshape(1, N, 3)
        self._velocities = self._init_vel.reshape(1, N, 3)
        self._times = np.array([0.0])
        self._cached_self_acc = None
        self._cached_self_pot = None
        self._has_run = False

    def add_external_force(self, force: Force):
        '''
        Add an external force to the simulation.

        Parameters
        ----------
        force : Force
            An instance of an ExternalForce subclass (Conservative or not)
            representing an external force.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If force is an instance of SelfGravityForce. Only accepts external
            forces.
        ValueError
            If this exact force instance is already added, or an equivalently
            configured force is already added (see the force's ``_dedup_key``).
            Adding a duplicate would double-count the field.
        '''
        def _assign_force(_force):
            if isinstance(_force, SelfGravityForce):
                raise TypeError("The provided force is a self-gravity force, "
                                "not an external force. Self-gravity forces are included"
                                "in the run() method.")
            if not isinstance(_force, Force):
                raise TypeError(
                    f"Expected a Force (Conservative or not) subclass, "
                    f"got {type(_force).__name__!r}."
                )
            # Dedup against forces already added. Rescanned per member so an
            # incoming composite carrying its own duplicate is caught too.
            existing = self._conserv_ext_force.members + self._base_ext_force.members
            if any(_force is m for m in existing):
                raise ValueError(
                    f"This {type(_force).__name__} instance is already added.")
            key = _force._dedup_key()
            if key is not None and any(m._dedup_key() == key for m in existing):
                raise ValueError(
                    f"An equivalently-configured {type(_force).__name__} is already "
                    f"added; adding it again would double-count the field.")
            if isinstance(_force, Conservative):
                self._conserv_ext_force = self._conserv_ext_force + _force
            else:
                self._base_ext_force = self._base_ext_force + _force
        if isinstance(force, _CompositeConservative) or isinstance(force, _CompositePlain):
            for _force in force.members:
                _assign_force(_force)
        else:
            _assign_force(force)
            
    def add_external_pot(self, potential):
        '''
        Add an external potential from a supported package.

        Shorthand for ``add_external_force(ExternalPotential(potential))``.

        Parameters
        ----------
        potential : object
            A potential from a supported package, e.g. a galpy ``Potential``,
            a galpy ``CompositePotential``, or a list of galpy potentials.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If no installed backend can use ``potential``.
        ValueError
            If the same potential (or an equivalently configured one) is already added.

        Warnings
        --------
        UserWarning
            If the provided galpy potential has physical outputs turned off.
        '''
        self.add_external_force(ExternalPotential(potential))

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
        raise NotImplementedError("Adding subhalos as Plummer spheres is not yet implemented. Please sample subhalo particles with ic.sample() and add them as a component with add_particles().")

    def tag(self, name, mask):
        '''
        Define a new component based on a mask.
        '''
        raise NotImplementedError("Tagging components by mask is not yet implemented.")
    
        
    # --- Running the Simulation ------------------------------------------------------------------------------------------

    def run(self, t_end: float, dt: float, dt_out: float, t0: float=0.0,
            method: Optional[str] = 'falcON', integration_method: str = 'leapfrog',
            cache_self_gravity_acc: bool = True, cache_self_gravity_pot: bool = True,
            progress: bool = True, monitors='auto',
            **kwargs):
        """
        Run the simulation to *t_end* [Gyr].

        Running again starts from the initial conditions, replaces the previous
        run's outputs, and resets every hook's records (see ``Hook.reset``).

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
        method : str or None, optional
            Method to use for computing self-gravity. Included options are:
            - 'falcON' (default): fast multipole method implemented in falcON.
            - 'direct': direct summation.
            - None: no self-gravity.
        integration_method: str, optional
            Method to use for integration. Currently only leapfrog is available.
        cache_self_gravity_acc : bool, optional
            Whether to cache the self-gravity acceleration at each output snapshot. Default is True.
        cache_self_gravity_pot : bool, optional
            Whether to cache the self-gravitational potential at each output snapshot. Default is True.
        progress : bool, optional
            Whether to show the progress bar. Default is True.
        monitors : str, sequence of str, or False, optional
            Conservation diagnostics to attach for this run, via
            :class:`~tambora.dynamics.hooks.ConservationMonitor`. ``'auto'``
            (default) picks a sensible set (currently ``('energy',)``). Pass a
            name or sequence of names (e.g. ``('energy',)``) to choose
            explicitly, or ``False`` to attach none. Ignored if you already
            added your own ``ConservationMonitor``; read the results from
            ``sim.monitor``.
        **kwargs
            Additional keyword arguments to pass to the gravity method. For 'falcON':

            - theta (float, optional): Tree opening angle. Default is 0.6. Smaller = more accurate but slower.
            - kernel (int, optional): Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.

            The softening comes from the particles: see ``add_particles`` and ``eps``.
        """
        supported = sorted(m for m in SELF_GRAVITY_METHODS if m is not None)
        if method not in SELF_GRAVITY_METHODS:
            raise ValueError(
                f"Unknown method {method!r} for self-gravity. "
                f"Supported methods: {supported} (or None to disable self-gravity).")
        solver_cls = SELF_GRAVITY_METHODS[method]
        valid = set(inspect.signature(solver_cls.__init__).parameters) - {'self'}
        invalid = set(kwargs) - valid
        if invalid:
            raise ValueError(
                f"{invalid} is (are) invalid kwarg(s) for {method!r} self-gravity method. "
                "Only kwargs for self-gravity methods are allowed."
            )
        if method is not None:
            unset = [name for name, sl in self._slices.items() if np.isnan(self._eps[sl]).any()]
            if unset:
                raise ValueError(
                    f"Softening isn't set for {', '.join(repr(n) for n in unset)}: give it with "
                    f"add_particles(..., eps=...) or sim.<component>.eps = ..., or run with method=None.")
        self._self_gravity_force = solver_cls(**kwargs)
        # Outside state (e.g. Agama's global units) can invalidate a wrapped
        # potential after it was added; each backend checks for that here.
        for force in self._conserv_ext_force.members + self._base_ext_force.members:
            check = getattr(force, "check_still_valid", None)
            if check is not None:
                check()
        self._add_default_monitors(monitors)
        from ..dynamics.hooks import Hook
        for hook, _ in self._hooks:
            if isinstance(hook, Hook):
                hook.reset()
        integrator = INTEGRATORS[integration_method]()

        (self._positions, self._velocities, 
         self._times, self._cached_self_acc, 
         self._cached_self_pot) = _runner(
                    self._init_pos, self._init_vel, self._mass,
                    integrator,
                    self._self_gravity_force,
                    self._conserv_ext_force,
                    self._base_ext_force,
                    t0 = t0,
                    t_end = t_end,
                    dt = dt,
                    dt_out = dt_out,
                    return_self_gravity_pot = cache_self_gravity_pot,
                    return_self_gravity_acc = cache_self_gravity_acc,
                    slices = self._slices,
                    hooks = self._hooks,
                    progress = progress,
                    eps = self._eps if np.isnan(self._eps).any() else self._softening(everyone=True),
                )
        self._has_run = True

    # --- Hooks -----------------------------------------------------------------------------------

    def add_hook(self, hook, cadence=None):
        '''
        Register a hook to run during integration.

        A hook is a callable ``hook(state)`` receiving a ``StepState`` view of
        the simulation at the current step. Results are typically accumulated on
        the hook object for inspection after ``run()``.

        Parameters
        ----------
        hook : callable
            The hook to run. Subclass ``tambora.dynamics.hooks.Hook`` for a hook
            with a ``default_cadence``, or pass any bare callable.
        cadence : Cadence, optional
            When the hook fires. If None, uses the hook's ``default_cadence`` if
            it has one, otherwise ``EveryOutput()``.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the simulation has already been run.
        TypeError
            If ``hook`` is not callable, or the resolved ``cadence`` (supplied or
            from the hook's ``default_cadence``) is not a ``Cadence`` instance.
        ValueError
            If this exact hook instance has already been registered, or if an
            equivalently-configured hook is already registered (see the hook's
            ``_dedup_key``).
        '''
        from ..dynamics.hooks import EveryOutput, Cadence
        if self._has_run:
            raise RuntimeError("Cannot add hooks after run()")
        if not callable(hook):
            raise TypeError(f"hook must be callable, got {type(hook).__name__!r}.")
        if any(existing is hook for existing, _ in self._hooks):
            raise ValueError("This hook instance is already registered. Construct a "
                             "separate instance to run it more than once.")
        key = _hook_dedup_key(hook)
        if key is not None and any(_hook_dedup_key(h) == key for h, _ in self._hooks):
            # two hooks with equal non-None keys are duplicates.
            raise ValueError(
                f"An equivalently-configured {type(hook).__name__} is already "
                f"registered. Change its configuration, or reuse the existing hook.")
        if cadence is None:
            cadence = getattr(hook, "default_cadence", None) or EveryOutput()
        if not isinstance(cadence, Cadence):
            raise TypeError(f"cadence must be a Cadence instance, got "
                            f"{type(cadence).__name__!r}.")
        self._hooks.append((hook, cadence))

    def _add_default_monitors(self, monitors):
        '''Attach this run's default ``ConservationMonitor`` (called from ``run``).

        ``monitors`` is ``'auto'`` (default is currently ``('energy',)``),
        ``False``/``None`` to disable, or an explicit quantity name / sequence of
        names. Skipped entirely if a ``ConservationMonitor`` is already
        registered, so a monitor you added yourself always wins over the default.
        '''
        from ..dynamics.hooks import ConservationMonitor
        if monitors is False or monitors is None:
            return
        if any(isinstance(hook, ConservationMonitor) for hook, _ in self._hooks):
            return
        if isinstance(monitors, str):
            track = ('energy',) if monitors == 'auto' else (monitors,)
        else:
            track = tuple(monitors)
        if not track:
            return
        self.add_hook(ConservationMonitor(track=track))

    @property
    def hooks(self):
        '''Registered ``(hook, cadence)`` pairs, in registration order.

        Read-only: returns a shallow copy, so the returned sequence cannot be
        used to add or remove hooks. Use ``add_hook`` for that.
        '''
        return tuple(self._hooks)

    @property
    def monitor(self):
        '''The attached :class:`ConservationMonitor`, or None if there isn't one.

        Convenience handle for the monitor ``run`` adds by default::

            sim.run(t_end=1., dt=0.01, dt_out=0.1)
            sim.monitor.drift['energy'][-1]
        '''
        from ..dynamics.hooks import ConservationMonitor
        for hook, _ in self._hooks:
            if isinstance(hook, ConservationMonitor):
                return hook
        return None

    @property
    def components(self) -> tuple:
        '''Component views in insertion order.

        Read-only tuple. Each :class:`Component` carries its own ``name`` and
        the usual accessors, so you can loop directly::

            for c in sim.components:
                print(c.name, len(c.mass), c.mass.sum())

        Lookup by name is unchanged: ``sim.<name>``.
        '''
        return tuple(Component(self, sl, name) for name, sl in self._slices.items())

    def __repr__(self):
        n_total = 0 if self._mass is None else self._mass.shape[0]
        m_total = 0.0 if self._mass is None else float(self._mass.sum())
        n_comp = len(self._slices)

        # -- metadata header --
        if self._has_run and self._times is not None:
            status = (f"run: {len(self._times)} snapshots, "
                      f"t = {self._times[0]:g} -> {self._times[-1]:g} Gyr")
        else:
            status = "not run"
        lines = [f"Sim — {status}",
                 f"  {n_total} particles in {n_comp} "
                 f"component{'' if n_comp == 1 else 's'}, {m_total:.2e} Msun total",
                 ""]

        def _section(title, table_or_none):
            lines.append(f"  {title}")
            if table_or_none is None:
                lines.append("    (none)")
            else:
                lines.extend("  " + ln for ln in table_or_none)

        # -- components --
        comp_rows = [(name, str(len(self._mass[sl])), f"{self._mass[sl].sum():.2e}")
                     for name, sl in self._slices.items()]
        _section("Components",
                 _render_table(["name", "particles", "mass [Msun]"], comp_rows,
                               align='lrr') if comp_rows else None)
        lines.append("")

        # -- external forces --
        forces = list(self._conserv_ext_force.members) + list(self._base_ext_force.members)
        _section("External forces",
                 _render_table(["force"], [(_force_label(f),) for f in forces])
                 if forces else None)
        lines.append("")

        # -- hooks --
        hook_rows = [(type(h).__name__, _cadence_label(c)) for h, c in self._hooks]
        _section("Hooks",
                 _render_table(["hook", "cadence"], hook_rows) if hook_rows else None)

        return "\n".join(lines)

    # --- Energy Accessors -----------------------------------------------------------------

    # --- Potential Energy --- #

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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.
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
        self_pot : (len(t), n_particles) array
            Self-gravitational potential of each particle at each snapshot.

            Units: `Msun km^2 / s^2`
        '''
        if use_cached and self._cached_self_pot is not None:
            return self._mass * self._cached_self_pot[self._ti(t, vectorized=True)]
        elif use_cached and self._cached_self_pot is None:
            raise ValueError("Cached self-potential is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            if 'eps' not in kwargs:
                kwargs['eps'] = self._softening(everyone=True)
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        **kwargs
            Additional keyword arguments to pass to the gravity method. 

            For 'falcON', these include:
            - eps: Softening length (`kpc`) to use instead of the particles' own
            - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

            For 'direct' and 'direct_C', these include:
            - eps: Softening length (`kpc`) to use instead of the particles' own

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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.
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
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.

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

    # --- Virial Ratio --- #

    @_resolve_use_cached
    @_resolve_t
    def virial_ratio(self, t=..., use_cached=True, method=None, **kwargs):
        r'''
        Virial ratio :math:`2K/|W|` of the particles at *t*: 1 in equilibrium.

        .. math::

            K = \frac{1}{2} \sum_i m_i |v_i - v_{\rm cm}|^2, \qquad
            W = \sum_i m_i (x_i - x_{\rm cm}) \cdot a_i,

        about the particles' center of mass, where :math:`a_i` is the acceleration from
        self-gravity and the external forces.

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run. Default is True, unless
            ``method`` is given.
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
               + self.self_gravity(t=t, use_cached=use_cached, method=method, return_internal=True, **kwargs))
        return diagnostics.virial_ratio(self.pos(t, return_internal=True), self.vel(t, return_internal=True),
                                    self._mass, acc)

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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run. Default is True, unless
            ``method`` is given.
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
        if use_cached and self._cached_self_acc is not None:
            return self._cached_self_acc[self._ti(t, vectorized=True)]
        elif use_cached and self._cached_self_acc is None:
            raise ValueError("Cached self-gravity is not available. Please set use_cached to False and provide a method for computing self-gravity.")
        else:
            if 'eps' not in kwargs:
                kwargs['eps'] = self._softening(everyone=True)
            return self_gravity(self.pos(self._ti(t, vectorized=False), return_internal=True),
                                self.mass, method=method, **kwargs)[0]
        
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run. Default is True, unless
            ``method`` is given.
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
            Default is ... (ellipsis), which returns the value at all times.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run. Default is True, unless
            ``method`` is given.
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
            Default is ... (ellipsis), which returns the value at all times.
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        use_cached : bool, optional
            Whether to use the self-gravity cached during the run. Default is True, unless
            ``method`` is given.
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

    # --- Diagnostics -----------------------------------------------------------------

    @_resolve_use_cached
    def plot_energy_diagnostic(self, method=None, use_cached=True, nsnap=None, 
                        filename=None, **kwargs):   # pragma: no cover
        '''
        Plot global energy conservation as a function of 
        time.
        
        Parameters
        ----------
        method : str, optional
            Method to compute self-gravity with, instead of using the cache.
            Included options are:
            - 'falcON': fast multipole method implemented in falcON.
            - 'direct_C': direct summation in C.
            - 'direct': direct summation.
        use_cached : bool, optional
            Whether to use the self-potential cached during the run. Default is True, unless
            ``method`` is given.
        nsnap : int, optional
            Number of snapshots to use. If None (default), will use all snapshots.
        filename : str, optional
            If provided, will save the plot to the given filename
            instead of showing it.
        **kwargs
            Additional keyword arguments to pass to the gravity method.
            For 'falcON', these include:
            - eps: Softening length (kpc) to use instead of the particles' own
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
                                 plot_components=True):   # pragma: no cover
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
    