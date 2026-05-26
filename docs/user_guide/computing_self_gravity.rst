Computing Self-Gravity 
================================

You can calculate the self-gravity acceleration within a :class:`tambora.simulation.Sim` instance with

.. code-block:: python

    sim.self_gravity()  # returns (N, 3) array of accelerations
    sim.self_ax()       # returns (N,) array of x-accelerations
    sim.self_ay()       # returns (N,) array of y-accelerations
    sim.self_az()       # returns (N,) array of z-accelerations


Or you can use the standalone :func:`tambora.dynamics.acceleration.self_gravity` function, which takes arrays of positions, masses, and softening length(s) and returns the self-gravity acceleration:

.. code-block:: python

    from tambora.dynamics.acceleration import self_gravity

    acc, pot = self_gravity(pos, mass, eps=0.1)  # returns (N, 3) array of accelerations

Note that :func:`tambora.dynamics.acceleration.self_gravity` also returns the potential.

API
---

.. autofunction:: tambora.dynamics.acceleration.self_gravity

.. automethod:: tambora.simulation.Sim.self_gravity

.. automethod:: tambora.simulation.Sim.self_ax

.. automethod:: tambora.simulation.Sim.self_ay

.. automethod:: tambora.simulation.Sim.self_az

.. automethod:: tambora.simulation.Sim.self_potential