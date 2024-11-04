'''
The simulation module for AF2RAVE performs molecular dynamics simulations.
This module is mostly a wrapper around OpenMM, and provides utilies that create
simulation boxes, run simulations, and analyze trajectories.
'''

import openmm.app as app
import os

_charmm36FF = f"{__path__[0]}/forcefield/charmm36m_protein.xml"
if not os.path.exists(_charmm36FF):
    Charmm36mFF = app.ForceField(_charmm36FF, "charmm36/water.xml")
else:
    Charmm36mFF = None

from .reporter import CVReporter
from .simulation import UnbiasedSimulation
from .utils import *