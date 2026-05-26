Simulation Outputs
==================

Accessing Snapshots
-------------------

All accessor methods accept a ``t`` parameter that controls which snapshot(s)
are returned:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Behaviour
   * - ``int``
     - Snapshot by index (supports negative indexing, e.g. ``t=-1`` for last)
   * - ``float``
     - Closest snapshot to the given time in Gyr
   * - ``...`` (Ellipsis)
     - All snapshots, returned as ``(nsnap, N, ...)``


Saving and Loading a :class:`~tambora.simulation.Sim`
------------------------------------------------------
🚧 *Still working on it...* 🚧

Position Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.pos`
     - Full position vector (N, 3)
     - kpc
   * - :meth:`~tambora.simulation.Sim.x`
     - Cartesian x
     - kpc
   * - :meth:`~tambora.simulation.Sim.y`
     - Cartesian y
     - kpc
   * - :meth:`~tambora.simulation.Sim.z`
     - Cartesian z
     - kpc
   * - :meth:`~tambora.simulation.Sim.r`
     - Spherical radius
     - kpc
   * - :meth:`~tambora.simulation.Sim.phi`
     - Azimuthal angle
     - rad
   * - :meth:`~tambora.simulation.Sim.theta`
     - Polar angle
     - rad
   * - :meth:`~tambora.simulation.Sim.cylR`
     - Cylindrical radius
     - kpc


Velocity Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.vel`
     - Full velocity vector (N, 3)
     - km/s
   * - :meth:`~tambora.simulation.Sim.vx`
     - Cartesian vx
     - km/s
   * - :meth:`~tambora.simulation.Sim.vy`
     - Cartesian vy
     - km/s
   * - :meth:`~tambora.simulation.Sim.vz`
     - Cartesian vz
     - km/s
   * - :meth:`~tambora.simulation.Sim.vr`
     - Spherical radial velocity
     - km/s
   * - :meth:`~tambora.simulation.Sim.vphi`
     - Azimuthal angular velocity
     - km/s/kpc
   * - :meth:`~tambora.simulation.Sim.vtheta`
     - Polar velocity
     - km/s
   * - :meth:`~tambora.simulation.Sim.cylvR`
     - Cylindrical radial velocity
     - km/s


Momentum Accessors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.p`
     - Full momentum vector (N, 3)
     - Msun km/s
   * - :meth:`~tambora.simulation.Sim.px`
     - Momentum x-component
     - Msun km/s
   * - :meth:`~tambora.simulation.Sim.py`
     - Momentum y-component
     - Msun km/s
   * - :meth:`~tambora.simulation.Sim.pz`
     - Momentum z-component
     - Msun km/s


Angular Momentum Accessors
--------------------------

All angular momentum methods also accept ``center_pos`` and ``center_vel``
keyword arguments.

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.L`
     - Full angular momentum vector (N, 3)
     - Msun kpc km/s
   * - :meth:`~tambora.simulation.Sim.Lx`
     - Angular momentum x-component
     - Msun kpc km/s
   * - :meth:`~tambora.simulation.Sim.Ly`
     - Angular momentum y-component
     - Msun kpc km/s
   * - :meth:`~tambora.simulation.Sim.Lz`
     - Angular momentum z-component
     - Msun kpc km/s


Energy Accessors
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.KE`
     - Kinetic energy per particle
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.self_potential`
     - Self-gravitational potential energy per particle
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.compute_external_pot`
     - External potential energy per particle
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.PE`
     - Total potential energy per particle (self + external)
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.energy`
     - Total energy per particle (KE + PE)
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.system_energy`
     - Total system energy (sum over all particles)
     - Msun km²/s²
   * - :meth:`~tambora.simulation.Sim.dE`
     - Fractional energy change \|ΔE/E₀\|
     - dimensionless


Acceleration Accessors
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Method
     - Description
     - Units
   * - :meth:`~tambora.simulation.Sim.self_gravity`
     - Self-gravity acceleration vector (N, 3)
     - km/s²
   * - :meth:`~tambora.simulation.Sim.self_ax`
     - Self-gravity x-component
     - km/s²
   * - :meth:`~tambora.simulation.Sim.self_ay`
     - Self-gravity y-component
     - km/s²
   * - :meth:`~tambora.simulation.Sim.self_az`
     - Self-gravity z-component
     - km/s²
   * - :meth:`~tambora.simulation.Sim.external_acc`
     - External acceleration vector (N, 3)
     - km/s²
   * - :meth:`~tambora.simulation.Sim.external_ax`
     - External acceleration x-component
     - km/s²
   * - :meth:`~tambora.simulation.Sim.external_ay`
     - External acceleration y-component
     - km/s²
   * - :meth:`~tambora.simulation.Sim.external_az`
     - External acceleration z-component
     - km/s²


Making Diagnostic Plots
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - :meth:`~tambora.simulation.Sim.plot_energy_diagnostic`
     - Plot fractional energy change as a function of time
   * - :meth:`~tambora.simulation.Sim.plot_momentum_diagnostic`
     - Plot momentum conservation

.. _component_accessors:

Component Accessors
-------------------

Each named component is accessible as an attribute of the ``Sim`` object
(e.g. ``sim.stars``). The returned :class:`~tambora.simulation.Component`
object provides the same accessor interface as ``Sim``, scoped to that
component's particles.

🚧 *Still working on it...*


Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 15 55 15

   * - Property
     - Description
     - Units
   * - :attr:`~tambora.simulation.Sim.mass`
     - Particle masses
     - Msun
   * - :attr:`~tambora.simulation.Sim.times`
     - Snapshot times
     - Gyr


API
---

.. autoclass:: tambora.simulation.Sim
   :members:
   :undoc-members:
   :exclude-members: add_particles, run, add_external_pot, add_external_acc

.. autoclass:: tambora.simulation.Component
   :members:
   :undoc-members:
