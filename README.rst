af2rave package
******************

.. image:: https://readthedocs.org/projects/af2rave/badge/?version=latest
    :target: https://af2rave.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

AlphaFold2-RAVE package generates boltzman-ranked non-native conformations for proteins.

* Free software: MIT license
* Documentation: https://af2rave.readthedocs.io


.. image:: docs/scheme.png
    :target: https://doi.org/10.26434/chemrxiv-2025-q3mwr
    :align: center

Installation
----------------

It is strongly recommended a separate environment for this package, either with ``conda`` or ``venv``. 
After activating the environment, simply run:

.. code-block:: bash

    pip install git+https://github.com/tiwarylab/af2rave.git

If you want the folding module installed, too. You need to install ColabFold separately. One way to do it is with ``conda`` and download its parameters.

.. code-block:: bash

    conda install colabfold
    python -m colabfold.download

Bibliography
----------------

The main article describing the method is:

* Da Teng, Vanessa J. Meraz, Akashnathan Aranganathan, Xinyu Gu, and Pratyush Tiwary, AlphaFold2-RAVE: Protein Ensemble Generation with Physics-Based Sampling, ChemRxiv (2025) https://doi.org/10.26434/chemrxiv-2025-q3mwr

AlphaFold2-RAVE:

1. Bodhi P. Vani, Akashnathan Aranganathan, Dedi Wang, and Pratyush Tiwary, AlphaFold2-RAVE: From Sequence to Boltzmann Ranking, *J. Chem. Theory Comput.* 2023, 19, 14, 4351–4354, https://doi.org/10.1021/acs.jctc.3c00290
2. Bodhi P. Vani, Akashnathan Aranganathan and Pratyush Tiwary, Exploring Kinase Asp-Phe-Gly (DFG) Loop Conformational Stability with AlphaFold2-RAVE, *J. Chem. Inf. Model.* 2024, 64, 7, 2789–2797, https://doi.org/10.1021/acs.jcim.3c01436
3. Xinyu Gu, Akashnathan Aranganathan and Pratyush Tiwary, Empowering AlphaFold2 for protein conformation selective drug discovery with AlphaFold2-RAVE, *eLife*, 2024, https://doi.org/10.7554/eLife.99702.3

SPIB: 

* Dedi Wang and Pratyush Tiwary, State predictive information bottleneck, *J. Chem. Phys.* 154, 134111 (2021), https://doi.org/10.1063/5.0038198

AMINO: 

* Pavan Ravindra, Zachary Smith and Pratyush Tiwary, Automatic mutual information noise omission (AMINO): generating order parameters for molecular systems, *Mol. Syst. Des. Eng.*, 2020,5, 339-348, https://doi.org/10.1039/C9ME00115H



Credits
-------

This package was created with `Cookiecutter`_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage