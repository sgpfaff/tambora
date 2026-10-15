from dataclasses import dataclass
import numpy as np

@dataclass
class StepContext:
    """Per-run constants shared by every StepState. Built once in _runner."""
    slices: dict # component name -> slice (from Sim._slices)
    self_gravity_force: object
    conserv_ext_force: object
    base_ext_force: object
    progress: object = None # _Progress wrapping the run's progress bar, or None
    eps: object = None      # the particles' softening: a number, or (N,) [kpc]


class _Progress:
    """
    Reporting channel from hooks to the run's progress bar.

    Hooks call ``state.report(**fields)``; fields accumulate here and are pushed
    to the (duck-typed) progress bar's ``set_postfix``. Kept tqdm-agnostic so
    the runner owns the bar and StepState/hooks stay decoupled from it.
    """
    def __init__(self, pbar=None):
        self._pbar = pbar
        self._fields = {}

    def report(self, **fields):
        self._fields.update(fields)
        if self._pbar is not None:
            self._pbar.set_postfix(self._fields, refresh=False)


_FULL = slice(None)

class StepState:
    """
    Live view of the simulation at the current step, passed to hooks.

    Mirrors the API of Sim (pos(), vel(), KE(), energy(), self_gravity(), ...)
    with three main differences:
        * only accesses current time (accessors do not take `t`)
        * returns INTERNAL units for speed/consistency
        * borrows arrays + force objects by reference (no copies); expensive
        potentials are computed lazily and cached for the life of the step.
    """
    def __init__(self, result, ctx, sl=_FULL, cache=None):
        self._result = result       # StepResult: live pos/vel/mass/t + precomputed accs/pot
        self._ctx = ctx             # StepContext: slices + force objects
        self._sl = sl               # whole system, or one compontent
        self._cache = {} if cache is None else cache    # shared across component sub-views

    # -- reuse one instance across steps: refresh data, drop cached results --
    def _update(self, result):
        self._result = result
        self._cache.clear()

    # --- step metadata ---
    @property
    def step(self): return self._result.step

    @property
    def t(self): return self._result.t

    @property
    def mass(self):
        """Live particle masses. A property (time-independent), mirroring ``Sim.mass``."""
        return self._result.mass[self._sl]

    # -- reporting: push display fields to the run's progress bar --
    def report(self, **fields):
        '''
        Report display fields (e.g. ``|dE/E0|``) to the run's progress bar.

        Fields accumulate across hooks and persist between updates. No-op if the
        run has no progress bar (e.g. in tests).
        '''
        if self._ctx.progress is not None:
            self._ctx.progress.report(**fields)

    # -- component access: state.sat.pos() or state.component('sat') --
    def component(self, name):
        return StepState(self._result, self._ctx, self._ctx.slices[name], cache=self._cache)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        slices = self._ctx.slices
        if name in slices:
            return self.component(name)
        raise AttributeError(f"StepState has no attribute or component named {name!r}")
    
    # -- kinematics --
    def pos(self):  
        """Live (x, y, z) positions."""
        return self._result.pos[self._sl]
    def vel(self):
        """Live (vx, vy, vz)."""
        return self._result.vel[self._sl]
    def x(self):
        """x-component of live positions."""
        return self.pos()[..., 0]
    def y(self):  
        """y-component of live positions."""  
        return self.pos()[..., 1]
    def z(self):    
        """z-component of live positions."""
        return self.pos()[..., 2]
    def vx(self):   
        """x-component of live velocities."""
        return self.vel()[..., 0]
    def vy(self):   
        """y-component of live velocities."""
        return self.vel()[..., 1]
    def vz(self):   
        """z-component of live velocities."""
        return self.vel()[..., 2]
    def r(self):
        """Live radius of particles."""
        return np.linalg.norm(self.pos(), axis=-1)
    def KE(self):   
        """Live kinetic energy of particles."""
        return 0.5 * self.mass * np.sum(self.vel()**2, axis=-1)

    # -- self-gravity: acceleration & potential share one computation --
    def _self_gravity(self, include_all_components, solver):
        '''(acc, pot, out_slice) for this view under *solver*, cached for the step.

        Fast path: all components with the run's solver; returns the step's
        precomputed ``(self_acc, self_pot)``. Any other combination recomputes
        once via the solver's one-pass ``acc_and_potential`` and caches the pair,
        so ``self_gravity`` and ``self_potential`` never compute it twice.
        '''
        run_solver = self._ctx.self_gravity_force
        solver = solver if solver is not None else run_solver
        sl = self._sl
        if include_all_components:
            if solver is run_solver:
                return self._result.self_acc, self._result.self_pot, sl
            key = ("sg_all", id(solver))
            if key not in self._cache:
                self._cache[key] = solver.acc_and_potential(
                    self._result.pos, self._result.mass, self._ctx.eps)
            acc, pot = self._cache[key]
            return acc, pot, sl
        key = ("sg_iso", id(solver), (sl.start, sl.stop, sl.step))
        if key not in self._cache:
            eps = self._ctx.eps
            self._cache[key] = solver.acc_and_potential(
                self.pos(), self.mass, eps if np.ndim(eps) == 0 else eps[sl])
        acc, pot = self._cache[key]
        return acc, pot, slice(None)

    # -- accelerations -- reuse what the integrator already computed
    def self_gravity(self, include_all_components, solver=None):
        '''
        Self-gravitational acceleration on this view's particles [internal units].

        Mirrors ``Component.self_gravity``.

        Parameters
        ----------
        include_all_components : bool
            If True, the particles' acceleration within the *whole-system*
            self-gravity field. If False, the *isolated* self-gravity of this
            view's particles alone. There is deliberately no default: on a
            component view the two meanings differ, so the choice must be explicit.
        solver : SelfGravityForce, optional
            Configured self-gravity solver to compute with. If None (default),
            the run's own solver, in which case the value is read from the step's precomputed acceleration. Pass a solver
            instance (e.g. ``DirectSummationGravity(eps=...)``) to use a different
            method/softening than the integration did.

        Notes
        -----
        The fast path: ``include_all_components=True`` with the run's solver;
        reuses ``StepResult.self_acc`` and does no work. Every other combination
        recomputes once per step and caches the result for the step's lifetime.
        '''
        acc, _, sl = self._self_gravity(include_all_components, solver)
        return acc[sl]
    
    def external_acc(self):
        '''Live (x, y, z) acceleration due to external forces.'''
        c, b = self._result.conserv_ext_acc, self._result.base_ext_acc
        if c is None and b is None:
            return np.zeros_like(self.pos())    # no external force; allocate only here
        # Seeded at 0.0 (not at c[sl]) so the accumulation cannot write through
        # the view into the integrator's own array.
        out = 0.0
        if c is not None: out = out + c[self._sl]
        if b is not None: out = out + b[self._sl]
        return out
    
    # -- potentials --
    def self_potential(self, include_all_components, solver=None):
        '''
        Self-gravitational potential energy of this view's particles [internal units].

        Mirrors ``Component.self_potential``.

        Parameters
        ----------
        include_all_components : bool
            If True, the particles' potential within the *whole-system*
            self-gravity field. If False, the *isolated* self-potential of this
            view's particles alone. There is deliberately no default: on a
            component view the two meanings differ, so the choice must be explicit.
        solver : SelfGravityForce, optional
            Configured self-gravity solver to compute with. If None (default),
            the run's own solver, in which case the value is read from the step's precomputed potential. Pass a solver
            instance (e.g. ``DirectSummationGravity(eps=...)``) to use a different
            method/softening than the integration did.

        Notes
        -----
        The fast path: ``include_all_components=True`` with the run's solver;
        reuses ``StepResult.self_pot`` and does no work. Every other combination
        recomputes once per step and caches the result for the step's lifetime.
        '''
        _, phi, sl = self._self_gravity(include_all_components, solver)
        return self.mass * phi[sl]
    
    def external_pot(self):
        '''Live potential from external forces.'''
        # lazy compute on full array and slice after
        phi = self._cache.get("ext_pot")
        if phi is None:
            phi = self._ctx.conserv_ext_force.potential(
                self._result.pos, self._result.mass, self._result.t)
            self._cache["ext_pot"] = phi
        return self.mass * phi[self._sl]
    
    def PE(self, include_all_components, solver=None):
        """Live potential energy (self + external) of this view's particles.

        ``include_all_components`` (required) and ``solver`` are forwarded to
        ``self_potential`` (which see); the external potential is always the full
        external field, independent of those choices.
        """
        return self.self_potential(include_all_components, solver) + self.external_pot()

    def energy(self, include_all_components, solver=None):
        """Live total energy (KE + PE) of this view's particles.

        ``include_all_components`` is required and forwarded to ``PE`` -- on a
        component view its two meanings differ, so the choice must be explicit.
        """
        return self.KE() + self.PE(include_all_components, solver)

    def system_energy(self, solver=None):
        """Live energy of the entire system [internal units].

        A whole-system total, so ``include_all_components`` does not apply; pass
        ``solver`` to evaluate the self-potential with a non-run solver.
        """
        return (np.sum(self.KE())
                + 0.5 * np.sum(self.self_potential(include_all_components=True, solver=solver))
                + np.sum(self.external_pot()))

    # -- boundedness (iterative unbinding), cached per step --
    def bound_mask(self, component=None, *, eps, method='falcON', theta=0.6, max_iter=50):
        '''
        Boolean mask of self-bound particles, via iterative unbinding.

        Cached for the life of the step and keyed by the particle subset and
        parameters, so multiple hooks asking on the same step (at any cadence)
        share a single computation.

        Binding is measured relative to the tested particles alone, in isolation.

        Parameters
        ----------
        component : str, optional
            Component to test. None (default) uses this view's particles (the
            whole system for a top-level state).
        eps : float
            Softening length [kpc]. Required.
        method, theta, max_iter
            Passed through to the iterative unbinding solver.
        '''
        from ..diagnostics import bound_mask as _bound_mask
        c = self if component is None else self.component(component)
        sl = c._sl
        # key by the resolved slice (not the component name) so a whole-system
        # call and a component call can never collide in the shared cache.
        key = ("bound_mask", (sl.start, sl.stop, sl.step), eps, method, theta, max_iter)
        if key not in self._cache:
            self._cache[key] = _bound_mask(c.pos(), c.vel(), c.mass,
                                           eps=eps, method=method, theta=theta,
                                           max_iter=max_iter)
        return self._cache[key]
