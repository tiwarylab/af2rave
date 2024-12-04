af2rave
=======

![image](https://img.shields.io/pypi/v/af2rave.svg)
![image](https://img.shields.io/travis/k9cdt/af2rave.svg)
![image](https://readthedocs.org/projects/af2rave/badge/?version=latest)

AlphaFold2-RAVE package generates boltzman-ranked non-native conformations for proteins


* Free software: MIT license
* Documentation: https://af2rave.readthedocs.io

Installation
---------

It is strongly recommended a separate environment for this package. 
The environment.yml will take care of most of the dependencies.
If you choose to install the dependencies this way, ColabFold will be selected as the default AlphaFold2 model.

```bash
git clone git@github.com:tiwarylab/af2rave.git af2rave
cd af2rave
conda env create -n af2rave python=3.11 -f environment.yml
```

Then use pip to install the package.

```bash
pip install .
```

You may need to download the model parameters for ColabFold with

```bash
python -m colabfold.download
```

Bibliography
--------
AlphaFold2-RAVE:

1. Bodhi P. Vani, Akashnathan Aranganathan, Dedi Wang, and Pratyush Tiwary, AlphaFold2-RAVE: From Sequence to Boltzmann Ranking, *J. Chem. Theory Comput.* 2023, 19, 14, 4351–4354, [[doi: 10.1021/acs.jctc.3c00290]](https://doi.org/10.1021/acs.jctc.3c00290)
2. Bodhi P. Vani, Akashnathan Aranganathan and Pratyush Tiwary, Exploring Kinase Asp-Phe-Gly (DFG) Loop Conformational Stability with AlphaFold2-RAVE, *J. Chem. Inf. Model.* 2024, 64, 7, 2789–2797, [[doi: 10.1021/acs.jcim.3c01436]](https://doi.org/10.1021/acs.jcim.3c01436)
3. Xinyu Gu, Akashnathan Aranganathan and Pratyush Tiwary, Empowering AlphaFold2 for protein conformation selective drug discovery with AlphaFold2-RAVE, *eLife*, 2024, [[doi: 10.7554/eLife.99702.3]](https://doi.org/10.7554/eLife.99702.3)

SPIB: 

* Dedi Wang and Pratyush Tiwary, State predictive information bottleneck, *J. Chem. Phys.* 154, 134111 (2021), [[doi: 10.1063/5.0038198]](https://doi.org/10.1063/5.0038198)

AMINO: 

* Pavan Ravindra, Zachary Smith and Pratyush Tiwary, Automatic mutual information noise omission (AMINO): generating order parameters for molecular systems, *Mol. Syst. Des. Eng.*, 2020,5, 339-348 [[doi: 10.1039/C9ME00115H]](https://doi.org/10.1039/C9ME00115H)

Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.
