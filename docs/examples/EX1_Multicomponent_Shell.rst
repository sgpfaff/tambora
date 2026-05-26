Stellar Shell Formation from Multicomponent Progenitor
======================================================

🚧 *Still working on it...*

This example simulates the tidal disruption of a two-component progenitor — a dark matter halo
and a stellar body — orbiting inside an NFW host potential. Self-gravity is computed between
all particles across both components, producing realistic shell structures in the stellar
component as the system disrupts.

The full runnable script for this example is available as a :download:`Python script <EX1_Multicomponent_Shell.py>`
or as a :download:`Jupyter notebook <EX1_Multicomponent_Shell.ipynb>`.

Setup
-----

We create the simulation and sample both the dark matter and stellar components from
isotropic Plummer distribution functions using :func:`~tambora.tools.galpydfsampler`:

.. code-block:: python

   from galpy.potential import PlummerPotential, NFWPotential
   from galpy.df import isotropicPlummerdf
   from tambora.simulation import Sim
   from tambora.tools import galpydfsampler
   import astropy.units as u

   shell = Sim()

   # Dark matter component: 10^11 Msun, scale radius 3 kpc
   progpotDM = PlummerPotential(amp=1e11 * u.Msun, b=3 * u.kpc)
   dfDM = isotropicPlummerdf(pot=progpotDM)
   posDM, velDM, massesDM = galpydfsampler(
       dfDM, n=60000, m_total=1e11,
       center_pos=[15, 0, 0], center_vel=[0, 0.066, 0]
   )
   shell.add_particles('dm', posDM, velDM, massesDM)

   # Stellar component: 10^9 Msun, scale radius 2 kpc
   progpotStars = PlummerPotential(amp=1e9 * u.Msun, b=2 * u.kpc)
   dfStars = isotropicPlummerdf(pot=progpotStars)
   posStars, velStars, massesStars = galpydfsampler(
       dfStars, n=30000, m_total=1e9,
       center_pos=[15, 0, 0], center_vel=[0, 0.066, 0]
   )
   shell.add_particles('stars', posStars, velStars, massesStars)

Adding the External Potential
------------------------------

We add an NFW host potential to drive the orbit:

.. code-block:: python

   host_pot = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
   shell.add_external_pot(host_pot)

Running
-------

We run for 400 Myr using falcON self-gravity:

.. code-block:: python

   shell.run(t_end=400, dt=0.01, dt_out=3., eps=0.5, theta=0.5)

Results
-------

🚧 *Run the script to regenerate figures.*
