User Guide
==========

In-depth explanations of tambora's core concepts. Start with **All About Simulations**
to build and run a simulation, dive into **Forces** for an understanding of how self-gravity and external forces are handled, or
jump to **Tools & Interoperability** for everything around the edges.


All About Simulations
--------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Setting Up a Simulation
      :link: setting_up_a_simulation
      :link-type: doc

      Instantiating a simulation.

   .. grid-item-card:: Generating Initial Conditions
      :link: generating_initial_conditions
      :link-type: doc
      :img-bottom: sampled_plummer_vs_analytic.png

      Generate equilibrium initial conditions for with tambora's built-in IC generation convenience functions.

   .. grid-item-card:: Adding Particles to a Simulation
      :link: adding_particles_to_a_sim
      :link-type: doc
      :img-bottom: multicomponent_IC.png

      Adding a set (or sets) of particles to a simulation.

   .. grid-item-card:: Adding External Forces to a Simulation
      :link: adding_external_forces_to_a_sim
      :link-type: doc

      Adding external forces to a simulation, including conservative forces from potentials and non-conservative forces from user-defined force classes.

   .. grid-item-card:: Running a Simulation
      :link: running_a_simulation
      :link-type: doc

      Configure the integrator and launch a run.
   
   .. grid-item-card:: Hooks 🪝
      :link: hooks
      :link-type: doc

      Observe and record quantities during a run  at a cadence you control, independent of the saved snapshots.

   .. grid-item-card:: Integrators
      :link: integrators
      :link-type: doc

      Overview of how integrators are implemented in tambora and the options available to users for customization.

   .. grid-item-card:: Simulation Accessors 
      :link: simulation_outputs
      :link-type: doc

      Access particle positions, velocities, energies, accelerations, and more from a simulation.
   
   .. grid-item-card:: Evaluating Properties of Multiple Sets of Particles
      :link: evaluating_properties_of_multiple_sets_of_particles
      :link-type: doc
      

      Accessing the properties of multiple sets of particles in a simulation.



Forces
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Overview of Force Classes
      :link: force_types
      :link-type: doc

      Detailing tambora's treatment of forces as being external or interaction forces, conservative or non-conservative.

   .. grid-item-card:: Usage as a Standalone Force Calculator
      :link: standalone_force_usage
      :link-type: doc

      Calculating forces outside of a simulation using force classes directly.

   .. grid-item-card:: Custom Forces

      Implementing custom forces in tambora.

Conservative Forces
^^^^^^^^^^^^^^^^^^^
.. grid:: 1 2 2 2
   
   .. grid-item-card:: Self-Gravity Solvers
      :link: self_gravity_force_and_solvers
      :link-type: doc
      :img-bottom: tree_visualization_v2.gif

      Compute self-gravity forces with falcON, direct summation, or Barnes-Hut. Per-component softening lengths.

   .. grid-item-card:: External Conservative Forces and their Potentials
      :link: external_conservative_forces
      :link-type: doc
      :img-bottom: NFW_acc_vector_field.png

      Compute forces from external potentials such as those in galpy.


Tools & Interoperability
------------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Evaluating Particle Boundedness
      :link: satellite_tools
      :link-type: doc

      Convenient methods to calculate the boundedness of particles to a satellite at a
      cadence independent of the simulation output.

   .. grid-item-card:: Interoperability
      :link: interoperability
      :link-type: doc

      Move data to and from galpy, astropy, and friends.

   .. grid-item-card:: Units
      :link: units
      :link-type: doc

      tambora's internal and user-facing unit conventions.


.. The toctrees below populate the left sidebar, grouped by caption. They
.. replace the old stub pages (simulation.rst / self_gravity.rst) that existed
.. only to hold a nested toctree.

.. toctree::
   :hidden:
   :caption: All About Simulations
   :maxdepth: 2

   setting_up_a_simulation
   generating_initial_conditions
   adding_particles_to_a_sim
   adding_external_forces_to_a_sim
   running_a_simulation
   hooks
   integrators
   simulation_outputs
   evaluating_properties_of_multiple_sets_of_particles

.. toctree::
   :hidden:
   :caption: Forces
   :maxdepth: 2

   force_types
   standalone_force_usage
   custom_forces
   self_gravity_force_and_solvers
   external_conservative_forces

.. toctree::
   :hidden:
   :caption: Tools & Interoperability
   :maxdepth: 1

   satellite_tools
   interoperability
   units
