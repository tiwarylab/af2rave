Running the MD simulation
==========================

The MD simulation is most commonly run on a computer cluster. This step requires two files from previous steps:

1. The protein structure (a ``.pdb`` file)
2. The CV indices (a ``.npy`` file)

The two input files should be uploaded to the cluster. Our package provides a fast way to run the simulation using the OpenMM package.

.. note:: 
    The user is also welcomed to use their own method or software to perform the simulation. The end product of the simulation should contain:

    1. A trajectory file (a ``.xtc`` file)
    2. A CV file (a ``.dat`` file)




.. tip:: It is recommended to first check if your OpenMM is correctly installed with GPU support. To check, simply run in the GPU cluster:
    ``python -m openmm.testInstallation``. More details can be found at http://docs.openmm.org/latest/userguide/application/01_getting_started.html

The following code snippet shows how to run the simulation:

.. code-block:: python3

    import af2rave.simulation
    import numpy as np

    filename = "sample_box.pdb"
    idx = np.load("cv_indices.npy")

    ubs = af2rave.simulation.UnbiasedSimulation(
        filename, list_of_index=idx,
        xtc_file=f"sample.xtc", xtc_freq=50000,
        cv_file=f"sample.dat", cv_freq=500
    )

    ubs.run(50000000)
    ubs.save_pdb(f"restart.pdb")
    ubs.save_checkpoint(f"restart.chk")

This can be put in a script and submitted to the cluster. The simulation will run for 100 ns and save the trajectory in ``sample.xtc`` and the CVs in ``sample.dat``. The simulation can be restarted from the checkpoint file ``restart.chk``.