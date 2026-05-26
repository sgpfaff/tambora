Disk Evolution
==============

🚧 *Still working on it...*

This example generates an exponential disk in approximate equilibrium and evolves it
under self-gravity alone, then again with a stabilizing NFW halo, demonstrating the
disk instability and how an external potential suppresses it.

The full runnable script for this example is available as a :download:`Python script <EX2_Disk.py>`
or as a :download:`Jupyter notebook <EX2_Disk.ipynb>`.

Generating Initial Conditions
------------------------------

An exponential disk is sampled with a sech² vertical profile using the built-in
``expdisk`` helper (based on the `gravhopper <https://github.com/jobovy/gravhopper>`_ recipe):

.. code-block:: python

   from tambora.simulation import Sim
   import astropy.units as u

   Ndisk = 10000
   vals = expdisk(
       N=Ndisk,
       sigma0=200 * u.Msun / u.pc**2,
       Rd=2 * u.kpc,
       z0=0.2 * u.kpc,
       sigmaR_Rd=20 * u.km / u.s,
   )
   pos, vel, mass = vals['pos'], vals['vel'], vals['mass']

Case 1: Disk with No Halo
--------------------------

Without a stabilizing potential the disk develops a bar instability:

.. code-block:: python

   disk = Sim()
   disk.add_particles('stars', pos.value, vel.to(u.kpc / u.Myr).value, mass.value)
   disk.run(t_end=500, dt=1.0, dt_out=10, eps=0.5, theta=0.6)

Results
^^^^^^^

🚧 *Run the script to regenerate figures.*

Case 2: Disk with NFW Halo
---------------------------

Adding an NFW halo slows the bar growth:

.. code-block:: python

   from galpy.potential import NFWPotential

   vhalo = 150 * u.km / u.s
   rhalo = 100 * u.kpc

   diskhalo = Sim()
   halo = NFWPotential(vmax=vhalo, rmax=rhalo)
   diskhalo.add_particles('stars', pos.value, vel.to(u.kpc / u.Myr).value, mass.value)
   diskhalo.add_external_pot(halo)
   diskhalo.run(t_end=500, dt=1.0, dt_out=10, eps=0.5, theta=0.6)

Results
^^^^^^^

🚧 *Run the script to regenerate figures.*
