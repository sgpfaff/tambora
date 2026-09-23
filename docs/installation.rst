Installation
============


🚧🚨 **Warning!** 🚧🚨 This code is still under development. Backwards compatibility is not guaranteed.

Basic Installation
------------------
:ref:`home` is currently published as a pre-release. pip ignores pre-releases by
default, so ``--pre`` is required:

.. code-block:: bash

    pip install --pre tambora

To pin an exact version instead:

.. code-block:: bash

    pip install tambora==0.1.0a1


Optional Dependencies
---------------------


`galpy`_ is required to be installed in the same environment as :ref:`home` to use features including galpy external potentials and distribution function sampling. Please refer to the `galpy installation guide`_ for installing galpy.


.. _galpy: https://docs.galpy.org/en/stable/
.. _galpy installation guide: https://docs.galpy.org/en/stable/installation.html