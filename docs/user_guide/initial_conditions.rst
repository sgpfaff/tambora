Generating Initial Conditions
=============================


:ref:`home` has several specialized tools for generating initial conditions 
that can be passed directly to :func:`~tambora.simulation.Sim.add_particles`. 

In particular, we currently support generating initial conditions for 
Plummer (:func:`~tambora.tools.mkPlummer_galpy`), King (:func:`~tambora.tools.mkKing_galpy`), 
and NFW (:func:`~tambora.tools.mkNFW_galpy`) profiles, using `galpy distribution function sampling <https://docs.galpy.org/en/latest/reference/sphericaldfsample.html>`_ 
in the backend. For example, the following code generates 1000 sample positions, velocities, and masses of a $10^5 \\text{ M}_{\\odot}$ Plummer sphere with a scale radius of 1.0 kpc:

.. code-block:: python

    from tambora.tools import mkPlummer_galpy

    pos, vel, mass = mkPlummer_galpy(m=1e5, b=1.0, n=1000) 

You can also generate a sample from an 
arbitrary spherical potential or spherical distribution function defined in galpy
using an Eddington inversion with :func:`~tambora.tools.galpysampler` and :func:`~tambora.tools.galpydfsampler`,
respectively.

API
---

.. autofunction:: tambora.tools.mkPlummer_galpy

.. autofunction:: tambora.tools.mkKing_galpy

.. autofunction:: tambora.tools.mkNFW_galpy

.. autofunction:: tambora.tools.galpydfsampler

.. autofunction:: tambora.tools.galpysampler