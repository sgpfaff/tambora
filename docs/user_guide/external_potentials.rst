Adding External Potentials
==========================

To add an external potential to the :class:`~tambora.simulation.Sim`, use the :func:`~tambora.simulation.Sim.add_external_pot` method. 
This method takes a (galpy) potential and adds it to the simulation. 

.. code-block:: python

    from galpy.potential import NFWPotential

    extpot = NFWPotential()
    sim.add_external_pot(extpot)

You can add as many external potentials as you'd like, and they will be summed together to compute the total external acceleration on each particle.
You can also use the :func:`~tambora.simulation.Sim.compute_external_pot` method to compute the external potential and the
:func:`~tambora.simulation.Sim.external_acc`, :func:`~tambora.simulation.Sim.external_ax`, :func:`~tambora.simulation.Sim.external_ay`, :func:`~tambora.simulation.Sim.external_az` methods to compute the external acceleration 
from the added potentials at any point in time during the simulation.

galpy
-----

:ref:`home` supports external potentials from `galpy <https://docs.galpy.org>`_.
The following potentials are supported, along with any composition of them:

**Spherical**

- `Burkert potential <https://docs.galpy.org/en/latest/reference/potentialburkert.html>`_ (``BurkertPotential``)
- `Spherical Cored Dehnen potential <https://docs.galpy.org/en/latest/reference/potentialcoredehnen.html>`_ (``DehnenCoreSphericalPotential``)
- `Spherical Dehnen potential <https://docs.galpy.org/en/latest/reference/potentialdehnen.html>`_ (``DehnenSphericalPotential``)
- `Einasto potential <https://docs.galpy.org/en/latest/reference/potentialeinasto.html>`_ (``EinastoPotential``)
- `Hernquist potential <https://docs.galpy.org/en/latest/reference/potentialhernquist.html>`_ (``HernquistPotential``)
- `Homogeneous sphere potential <https://docs.galpy.org/en/latest/reference/potentialhomogsphere.html>`_ (``HomogeneousSpherePotential``)
- `Interpolated spherical potential <https://docs.galpy.org/en/latest/reference/potentialinterpsphere.html>`_ (``interpSphericalPotential``)
- `Isochrone potential <https://docs.galpy.org/en/latest/reference/potentialisochrone.html>`_ (``IsochronePotential``)
- `Jaffe potential <https://docs.galpy.org/en/latest/reference/potentialjaffe.html>`_ (``JaffePotential``)
- `Kepler potential <https://docs.galpy.org/en/latest/reference/potentialkepler.html>`_ (``KeplerPotential``)
- `King potential <https://docs.galpy.org/en/latest/reference/potentialking.html>`_ (``KingPotential``)
- `NFW potential <https://docs.galpy.org/en/latest/reference/potentialnfw.html>`_ (``NFWPotential``)
- `Plummer potential <https://docs.galpy.org/en/latest/reference/potentialplummer.html>`_ (``PlummerPotential``)
- `Power-law density spherical potential <https://docs.galpy.org/en/latest/reference/potentialpowerspher.html>`_ (``PowerSphericalPotential``)
- `Power-law density spherical potential with an exponential cut-off <https://docs.galpy.org/en/latest/reference/potentialpowerspherwcut.html>`_ (``PowerSphericalPotentialwCutoff``)
- `Pseudo-isothermal potential <https://docs.galpy.org/en/latest/reference/potentialpseudoiso.html>`_ (``PseudoIsothermalPotential``)
- `Spherical Shell potential <https://docs.galpy.org/en/latest/reference/potentialsphericalshell.html>`_ (``SphericalShellPotential``)
- `Double power-law density spherical potential <https://docs.galpy.org/en/latest/reference/potentialdoublepowerspher.html>`_ (``TwoPowerSphericalPotential``)

**Axisymmetric**

- `Double exponential disk potential <https://docs.galpy.org/en/latest/reference/potentialdoubleexp.html>`_ (``DoubleExponentialDiskPotential``)
- `Flattened power-law potential <https://docs.galpy.org/en/latest/reference/potentialflattenedpower.html>`_ (``FlattenedPowerPotential``)
- `Interpolated axisymmetric potential <https://docs.galpy.org/en/latest/reference/potentialinterprz.html>`_ (``interpRZPotential``)
- `Kuzmin disk potential <https://docs.galpy.org/en/latest/reference/potentialkuzmindisk.html>`_ (``KuzminDiskPotential``)
- `Kuzmin-Kutuzov Staeckel potential <https://docs.galpy.org/en/latest/reference/potentialkuzminkutuzov.html>`_ (``KuzminKutuzovStaeckelPotential``)
- `Logarithmic halo potential <https://docs.galpy.org/en/latest/reference/potentialloghalo.html>`_ (``LogarithmicHaloPotential``)
- `Miyamoto-Nagai potential <https://docs.galpy.org/en/latest/reference/potentialmiyamoto.html>`_ (``MiyamotoNagaiPotential``)
- `Three Miyamoto-Nagai disk approximation to an exponential disk <https://docs.galpy.org/en/latest/reference/potential3mn.html>`_ (``MN3ExponentialDiskPotential``)
- `Razor-thin exponential disk potential <https://docs.galpy.org/en/latest/reference/potentialrazorexp.html>`_ (``RazorThinExponentialDiskPotential``)
- `Ring potential <https://docs.galpy.org/en/latest/reference/potentialring.html>`_ (``RingPotential``)

**Ellipsoidal triaxial**

- `Perfect Ellipsoid potential <https://docs.galpy.org/en/latest/reference/potentialperfectellipsoid.html>`_ (``PerfectEllipsoidPotential``)
- `Triaxial Gaussian potential <https://docs.galpy.org/en/latest/reference/potentialtriaxialgaussian.html>`_ (``TriaxialGaussianPotential``)
- `Triaxial Hernquist potential <https://docs.galpy.org/en/latest/reference/potentialtriaxialhernquist.html>`_ (``TriaxialHernquistPotential``)
- `Triaxial Jaffe potential <https://docs.galpy.org/en/latest/reference/potentialtriaxialjaffe.html>`_ (``TriaxialJaffePotential``)
- `Triaxial NFW potential <https://docs.galpy.org/en/latest/reference/potentialtriaxialnfw.html>`_ (``TriaxialNFWPotential``)
- `Double power-law density triaxial potential <https://docs.galpy.org/en/latest/reference/potentialdoublepowertriaxial.html>`_ (``TwoPowerTriaxialPotential``)

**Spiral, bar, and miscellaneous**

- `Dehnen bar potential <https://docs.galpy.org/en/latest/reference/potentialdehnenbar.html>`_ (``DehnenBarPotential``)
- `Ferrers potential <https://docs.galpy.org/en/latest/reference/potentialferrers.html>`_ (``FerrersPotential``)
- `Softened-needle bar potential <https://docs.galpy.org/en/latest/reference/potentialsoftenedneedle.html>`_ (``SoftenedNeedleBarPotential``)
- `Spiral arms potential <https://docs.galpy.org/en/latest/reference/potentialspiralarms.html>`_ (``SpiralArmsPotential``)
- `Constant (null) potential <https://docs.galpy.org/en/latest/reference/potentialnull.html>`_ (``NullPotential``)
- ``MWPotential2014`` (composite: NFW + Miyamoto-Nagai + power-law bulge)


**Potential Wrappers**

- `Adiabatic contraction wrapper 🚧 <https://docs.galpy.org/en/latest/reference/potentialadiabaticcontractwrapper.html>`_ (``AdiabaticContractionWrapperPotential``)
- `Any time-dependent amplitude wrapper 🚧 <https://docs.galpy.org/en/latest/reference/potentialtimedependentamplitude.html>`_ (``TimeDependentAmplitudeWrapperPotential``)
- `Corotating rotation wrapper <https://docs.galpy.org/en/latest/reference/potentialcorotwrapper.html>`_ (``CorotatingRotationWrapperPotential``)
- `Cylindrically-separable wrapper 🚧 <https://docs.galpy.org/en/latest/reference/potentialcylindricallyseparablewrapper.html>`_ (``CylindricallySymmetricWrapperPotential``)
- `Dehnen-like smoothing wrapper <https://docs.galpy.org/en/latest/reference/potentialdehnensmoothwrapper.html>`_ (``DehnenSmoothWrapperPotential``)
- `Gaussian-modulated amplitude wrapper <https://docs.galpy.org/en/latest/reference/potentialgaussampwrapper.html>`_ (``GaussianAmplitudeWrapperPotential``)
- `Kuzmin-like wrapper <https://docs.galpy.org/en/latest/reference/potentialkuzminlikewrapper.html>`_ (``KuzminLikeWrapperPotential``)
- `Oblate Staeckel wrapper 🚧 <https://docs.galpy.org/en/latest/reference/potentialoblatestaeckelwrapper.html>`_ (``OblatePerfectEllipsoidWrapperPotential``)
- `Solid-body rotation wrapper <https://docs.galpy.org/en/latest/reference/potentialsolidbodyrotationwrapper.html>`_ (``SolidBodyRotationWrapperPotential``)
- `Rotate-and-tilt wrapper <https://docs.galpy.org/en/latest/reference/potentialrotateandtiltwrapper.html>`_ (``RotateAndTiltWrapperPotential``)

agama 🚧
-----------------------
🚧 *Still working on it...* 🚧

API
---
.. automethod:: tambora.simulation.Sim.add_external_pot

