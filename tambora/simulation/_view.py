"""
Accessors shared by Sim and Component, in terms of ``self._sim``, the Sim holding the data, and
``self._sl``, which of its particles.
"""

import copy
import warnings

import numpy as np
from tqdm import tqdm

from ..dynamics import diagnostics, NullSelfGravity
from ..units import unit_handler

_PROGRESS_DELAY = 1.0   # s a loop over snapshots runs before showing its progress
_GUEST_MASS = 1e-30     # x the sources' mean mass, for solved particles that aren't sources (falcON: NaN at 0)


class _External:
    """All of a Sim's external forces, as ``sources``: ``sim.external``."""

    def __init__(self, sim):
        self._sim = sim

    def __repr__(self):
        n = len(self._sim._conserv_ext_force.members) + len(self._sim._base_ext_force.members)
        return f"<external forces of a Sim ({n})>"


def _one_if_shared(eps):
    """*eps* as one number if every particle has it: falcON then gives the run's values bit for bit."""
    return float(eps[0]) if np.all(eps == eps[0]) else eps


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

    def _softening(self, everyone=False):
        """These particles' softening (the whole Sim's, if *everyone*), as one number if shared."""
        eps = self._sim._eps if everyone else self._sim._eps[self._sl]
        if np.isnan(eps).any():
            raise ValueError("Softening isn't set for every particle here: set it with sim.eps or "
                             "sim.<component>.eps, or pass eps=... to use one value for all.")
        return _one_if_shared(eps)

    # --- Sources of acceleration and potential -----------------------------------------------

    def _sources(self, sources, solver):
        """*sources* as (a boolean mask of the particles, or None; whether the external forces count)."""
        sim = self._sim
        if sources is None:
            mask, external = np.ones(len(sim._mass), dtype=bool), True
        else:
            items = list(sources) if isinstance(sources, (list, tuple)) else [sources]
            if not items:
                raise ValueError("sources is empty: leave it out for everything, or say what to include.")
            mask, external = np.zeros(len(sim._mass), dtype=bool), False
            for src in items:
                if isinstance(src, _ParticleView) and src._sim is src:
                    raise ValueError("sources=sim is ambiguous: use sim.particles for all the particles, "
                                     "sim.external for the external forces, or leave sources out for both.")
                if not isinstance(src, (_ParticleView, _External)):
                    raise TypeError("sources must be particles (e.g. sim.particles, sim.sat), sim.external, "
                                    f"or a list of these, got {type(src).__name__}")
                if src._sim is not sim:
                    raise ValueError("sources must belong to the same Sim.")
                if isinstance(src, _External):
                    external = True
                else:
                    mask[src._sl] = True
        if not mask.any():
            mask = None
        if mask is None and solver is not None:
            raise ValueError("solver computes the particles' gravity, but sources has no particles.")
        if mask is not None and solver is None and not sim._has_run:
            raise ValueError("No run yet, so no run's solver: pass solver=... for the particles' gravity.")
        return mask, external

    def _softening_for(self, solver, solved):
        """The solver to call and the softening to give it, for a solve over the particles *solved*."""
        own = getattr(solver, 'eps', None)
        if isinstance(solver, NullSelfGravity) or (own is not None and np.ndim(own) == 0):
            return solver, None
        if own is None:
            eps = self._sim._eps[solved]
            if np.isnan(eps).any():
                raise ValueError("Softening isn't set for every particle in this solve: set it with sim.eps "
                                 "or sim.<component>.eps, or use a solver with its own eps.")
            return solver, _one_if_shared(eps)
        if len(own) != len(solved):
            raise ValueError(f"The solver's own eps has {len(own)} values; in a Sim it needs one per "
                             f"particle ({len(solved)}).")
        solver = copy.copy(solver)
        solver.eps = None           # its own values go in with the call, for the solved particles only
        return solver, own[solved]

    def _gravity(self, ti, sources, solver):
        """(acc, specific potential) on these particles at snapshot *ti* from the particles in *sources*
        [internal units], solving the sources and these particles together."""
        sim = self._sim
        solver = sim._self_gravity_force if solver is None else solver
        targets = np.zeros_like(sources)
        targets[self._sl] = True
        solved = sources | targets
        mass = sim._mass[solved]
        guests = ~sources[solved]       # feel the field without adding to it
        if guests.any():
            mass = np.where(guests, _GUEST_MASS * mass[~guests].mean(), mass)
        solver, eps = self._softening_for(solver, solved)
        acc, pot = solver.acc_and_potential(sim._positions[ti][solved], mass, eps)
        keep = targets[solved]
        return acc[keep], pot[keep]

    def _external_acc(self, ti):
        """The external forces' acceleration on these particles at snapshot *ti* [internal units]."""
        sim = self._sim
        t, pos, vel = sim._times[ti], self.pos(ti, return_internal=True), self.vel(ti, return_internal=True)
        return (sim._conserv_ext_force.acc(pos=pos, mass=self.mass, t=t)
                + sim._base_ext_force.acc(pos=pos, vel=vel, mass=self.mass, t=t))

    def _each_snapshot(self, ti, at):
        """``at(i)`` at snapshot *ti*, or stacked over every snapshot, with a progress bar if that's slow."""
        if ti is not ...:
            return at(ti)
        n = len(self._sim._times)
        return np.stack([at(i) for i in tqdm(range(n), desc='snapshots', delay=_PROGRESS_DELAY, leave=False)])

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

    # --- Acceleration -------------------------------------------------------------------------

    @unit_handler('acceleration')
    def acc(self, t=..., sources=None, solver=None):
        '''
        Acceleration of each particle at *t*, from *sources*.

        Units: `km / s / Gyr`

        Parameters
        ----------
        t : float or int, optional
            Time of snapshot to access.
            If float, will return snapshot closest to that time (in Gyr).
            If int, will return snapshot at that index.
            Default is ... (ellipsis), which returns the value at all times.
        sources : optional
            What the acceleration comes from. Default: everything the run integrated, the
            particles and the external forces. Otherwise ``sim.particles`` (the particles'
            gravity), ``sim.external`` (the external forces), a component such as ``sim.sat``
            (its gravity alone), or a list of these.
        solver : SelfGravityForce, optional
            Solver for the particles' gravity, e.g. ``FalcONGravity(theta=0.3)``. Its own
            ``eps``, if it has one, replaces the particles' softening: a number, or one value
            per particle of the Sim. Default: the run's values when every particle is a
            source, otherwise the run's solver. Needed before a run.

        Returns
        -------
        acc : (n_snaps, n_particles, 3) array or (n_particles, 3) array
            Units: `km / s / Gyr`

        Notes
        -----
        Unless the run's values serve, each snapshot takes its own solve; a loop over every
        snapshot shows a progress bar once it has run for a second.
        '''
        sim = self._sim
        mask, external = self._sources(sources, solver)
        ti = self._ti(t)
        cached = mask is not None and mask.all() and solver is None and sim._cached_self_acc is not None
        solve = mask is not None and not cached

        def at(i):
            a = np.zeros((len(self.mass), 3))
            if solve:
                a += self._gravity(i, mask, solver)[0]
            if external:
                a += self._external_acc(i)
            return a

        acc = self._each_snapshot(ti, at) if solve or external else 0.0
        if cached:
            acc = acc + self._snap(sim._cached_self_acc, ti)
        return acc

    @unit_handler('acceleration')
    def ax(self, t=..., sources=None, solver=None):
        '''x-component of :meth:`acc`, which see. Units: `km / s / Gyr`'''
        return self.acc(t, sources, solver, return_internal=True)[..., 0]

    @unit_handler('acceleration')
    def ay(self, t=..., sources=None, solver=None):
        '''y-component of :meth:`acc`, which see. Units: `km / s / Gyr`'''
        return self.acc(t, sources, solver, return_internal=True)[..., 1]

    @unit_handler('acceleration')
    def az(self, t=..., sources=None, solver=None):
        '''z-component of :meth:`acc`, which see. Units: `km / s / Gyr`'''
        return self.acc(t, sources, solver, return_internal=True)[..., 2]
