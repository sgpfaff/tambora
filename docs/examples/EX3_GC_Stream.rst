Globular Cluster Stream
=======================

This example demonstrates how tambora works with galpy to simulate the formation of a
globular cluster tidal stream in the Milky Way potential.

The full runnable script for this example is available as a :download:`Python script <EX3_GC_Stream.py>`
or as a :download:`Jupyter notebook <EX3_GC_Stream.ipynb>`.

Setting Up the Simulation
--------------------------

We start by loading the Milky Way potential from galpy and creating a :class:`~tambora.simulation.Sim` instance:

.. code-block:: python

   from galpy.potential import MWPotential2014
   from tambora.simulation import Sim
   import astropy.units as u

   host = MWPotential2014[2]  # NFW component
   host.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)

   stream = Sim()
   stream.add_external_pot(host)

Sampling Initial Conditions
-----------------------------

We place the progenitor 10 kpc from the Galactic center and sample a King sphere
of :math:`10^7\ \mathrm{M}_\odot` using :func:`~tambora.tools.mkKing_galpy`. The tidal
radius is computed with galpy:

.. code-block:: python

   import numpy as np
   from tambora.tools import mkKing_galpy

   center_R = 10 * u.kpc
   center_v = host.vcirc(center_R, quantity=True).to(u.km / u.s) * 0.8

   sat_mass = 1e7 * u.Msun
   rtidal = host.rtide(R=center_R, z=0 * u.kpc, M=sat_mass, quantity=True).to(u.kpc)

   pos, vel, mass = mkKing_galpy(
       m=1e7, n=20000,
       center_pos=[center_R.value, 0, 0],
       center_vel=[0., center_v.value, 0.],
       W0=4, rt=rtidal.value,
   )
   stream.add_particles('stars', pos=pos, vel=vel, mass=mass)

The initial particle distribution is compact and well-mixed in phase space:

.. figure:: _figures/EX3/fig01_cell15.png
   :alt: Initial conditions: position and velocity distributions in 6 projections
   :width: 100%

   Initial conditions of the globular cluster progenitor. Top row: spatial projections
   (*x–y*, *x–z*, *y–z*). Bottom row: velocity projections. The cluster is placed at
   :math:`(x, y, z) = (10, 0, 0)\ \mathrm{kpc}` with a tangential velocity 80% of circular.

Before running, we can sanity-check the balance between external (host) and internal
(self-gravity) accelerations:

.. figure:: _figures/EX3/fig02_cell19.png
   :alt: Acceleration vectors showing external field vs self-gravity
   :width: 60%
   :align: center

   Acceleration vectors at :math:`t=0`. Blue arrows show the external MW field;
   red arrows show the falcON self-gravity. The two are comparable in magnitude,
   confirming the cluster is self-bound.

Running the Simulation
-----------------------

We evolve the system for 1.5 Gyr under falcON self-gravity and the MW external field.
This takes approximately 2 minutes:

.. code-block:: python

   stream.run(t_end=1.5, dt=0.0005, dt_out=0.01, method='falcON', eps=0.01)

Results
--------

After 1.5 Gyr the progenitor has fully disrupted, leaving a clear tidal stream:

.. figure:: _figures/EX3/GC_stream_evolution.gif
   :alt: Tidal stream in the x-y plane at t = 1.5 Gyr
   :width: 60%
   :align: center

   Particle positions over :math:`t = 1.5\ \mathrm{Gyr}` of evolution in the $x-y$ plane.
   The red cross marks the Galactic center.

Performance
-----------

Energy conservation can be checked two ways. First, using the values recorded during
integration:

.. figure:: _figures/EX3/fig04_cell27.png
   :alt: Energy conservation over time from integration values
   :width: 70%
   :align: center

   Fractional energy error :math:`|\Delta E/E_0|` over the orbit. Spikes occur near
   pericenter when tidal stripping is strongest.

Second, by recomputing the energy with direct summation at a handful of snapshots
(more accurate but expensive):

.. figure:: _figures/EX3/fig05_cell29.png
   :alt: Energy conservation from direct summation spot-checks
   :width: 70%
   :align: center

   :math:`|\Delta E/E_0|` at 5 snapshots computed with direct N-body summation.
   The accumulated drift remains below :math:`10^{-6}` over 1.5 Gyr.
