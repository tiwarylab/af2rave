'''
The OpenMM simulation package for AF2RAVE.
'''

import openmm.app as app

Charmm36mFF = app.ForceField(f"{__path__[0]}/forcefield/charmm36m_protein.xml",
                             "charmm36/water.xml")

from .cv_reporter import CVReporter
from .simulation import UnbiasedSimulation
from .utils import *