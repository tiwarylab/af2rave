Preparing for MD simulation
===========================

This document shows how to prepare the cluster centers we get from ``FeatureSelection`` module to run an MD simulation. A few things to consider are as follows:

1. **Prepare the protein structure**: The structures acquired from `AlphaFold2` does not contain hydrogens. Hydrogens need to be added to the structures. Also, water and salt need to be added to the simulation box.

2. **Monitor the CVs**: The selected a few hundred CVs need to be output throughout the simulation. We need to know their atom indices. Note that these indices will change after hydrogens are added to the protein structure.

Prepare the protein structure
-----------------------------

The :class:`af2rave.simulation.utils.SimulationBox` takes care of solvation and generating a MD-ready structure. A sample code is

.. code-block:: python

    import af2rave.simulation 
    box = af2rave.simulation.SimulationBox("sample.pdb")
    box.create_box(ionicStrength=0.15)
    box.save_pdb("sample_box.pdb")

You can provide other parameters to the ``create_box`` method, such as how thick the water padding should be, the type of water model (default: TIP3P) and salt (default: NaCl).


.. caution:: By default the package uses CHARMM36m force field. If you want to specify a different one, you need to pass a OpenMM ``ForceField`` class to the ``SimulationBox`` constructor. See the documentation for details.

Monitor the CVs
---------------

The pairwise distance CVs are represented by a list of tuples, where each tuple contains two atom indices. The monitored CVs can be read from the ``FeatureSelection`` module. For example:

.. code-block:: python3

    indices = [tuple(fs.atom_indices[n]) for n in names]

However, after adding hydrogens, these indices will change. The ``SimulationBox`` object created can take care of this change:

.. code-block:: python3

    import numpy as np

    new_indices = box.map_atom_index(indices)
    np.save("cv_indices.npy", new_indices)

This saved file can be subsequently transferred to your computer cluster for MD simulation.