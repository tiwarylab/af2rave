'''
The simulation module for AF2RAVE performs molecular dynamics simulations.
This module is mostly a wrapper around OpenMM, and provides utilies that create
simulation boxes, run simulations, and analyze trajectories.
'''

import openmm.app as app

Charmm36mFF = app.ForceField(f"{__path__[0]}/forcefield/charmm36m_protein.xml",
                             "charmm36/water.xml")

from .cv_reporter import CVReporter
from .simulation import UnbiasedSimulation
from .utils import *