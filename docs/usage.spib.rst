Running State Predicted Information Bottleneck (SPIB)
======================================================

This document shows how to run the SPIB to get an conformation
embedding after the classical MD.

After the classical MD and AMINO, we have identified the most relevant
features to describe the conformational spaces of the protein.
SPIB uses a variational autoencoder (VAE) to learn a low-dimensional
embedding of the conformational space.
This low-dimensional space, or the latent space, is most useful when it is 2D.

In SPIB, each input point is labeled a state.
The loss function of SPIB is designed to find the encoder and decoder
that best learn the state of the system at :math:`t + dt` given the
input features at time :math:`t`. A good latent space should
be able to predict where the system will be at some time later.
This ``dt`` is the time lag, arguably the most important parameter in the model.

By default, SPIB will use a linear encoder. This ensures the two latent
variables are linear combinations of the input features.
This generally provides better interpretability, and allows for easier
enhanced sampling in the latent space.

The code to run SPIB is simple. Simply import the ``spib`` module and
create a :py:class:`~af2rave.spib.spib.SPIBProcess` with the data.

.. code-block:: python3

    import af2rave.spib as af2spib
    import glob

    colvar_file = glob.glob("colvar/*.dat")
    spib = af2spib.SPIBProcess(colvar_file)

    dt = [300, 1000, 3000, 10000]
    for t in dt:
        result = spib.run(t)
        result.to_file(f"dt_{t}.pickle")

This is a simple script that loads all COLVAR files and
run SPIB on four different ``dt``. The method :py:meth:`~af2rave.spib.spib.SPIBProcess.run`
will return a :py:class:`~af2rave.spib.spib_result.SPIBResult` object.
This object can be pickled for later analysis. The unit of ``dt``
is determined by the COLVAR files. If the COLVAR file was collected
every 1 ps, then ``dt=3000`` means 3 ns.

The documents for the :py:class:`~af2rave.spib.spib_result.SPIBResult` object
showed the common analysis methods one would like to perform.
