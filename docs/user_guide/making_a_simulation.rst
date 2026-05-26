Making a Simulation 
===================

The :class:`~tambora.simulation.Sim` class is the core of :ref:`home`. It orchestrates building, running, 
and analyzing n-body simulations and is what you will interact with the most as a result.
Instantiating a simulation is super simple:

.. code-block:: python

    from tambora import Sim
    sim = Sim()


Adding Particles
----------------

To add particles to the simulation, use the :func:`~tambora.simulation.Sim.add_particles` method. This method takes a component name (e.g. 'stars', 'dark_matter') and arrays of positions, velocities, and masses.

.. code-block:: python

    sim.add_particles('stars', pos=pos_array, vel=vel_array, mass=mass_array)


Adding Multiple Components
-------------------------

You can add as many components as you'd like following the same procedure, as long as you give it a unique name. For example, you could add a 'dark_matter' component in addition to 'stars':

.. code-block:: python

    sim.add_particles('dark_matter', pos=dm_pos_array, vel=dm_vel_array, mass=dm_mass_array)

You can easily access the properties of individual components using the component name:

.. code-block:: python

    star_positions = sim.stars.pos()
    dm_velocities = sim.dark_matter.vel()

You can also still access the all particles as usual, for example:

.. code-block:: python

    all_positions = sim.pos()

Methods for accessing the properties of individual components is discussed in more detail in the :ref:`component_accessors` section of the user guide.

API
---

.. autoclass:: tambora.simulation.Sim

.. automethod:: tambora.simulation.Sim.add_particles