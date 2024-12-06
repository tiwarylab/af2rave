.. highlight:: shell

============
Installation
============


Stable release
--------------

It is strongly recommended a separate environment for this package. 
The environment.yml will take care of most of the dependencies.
If you choose to install the dependencies this way, 
ColabFold will be selected as the default AlphaFold2 model.

```bash

.. code-block:: console

    $ git clone git@github.com:tiwarylab/af2rave.git af2rave
    $ cd af2rave
    $ conda env create -n af2rave 
    $ conda activate af2rave
    $ conda install python=3.11 
    $ conda install -f environment.yml

Then use `pip` to install it 

.. code-block:: console

    $ pip install .

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
