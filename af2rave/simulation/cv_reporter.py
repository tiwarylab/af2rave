'''
This CVReporter generates a PLUMED-style COLVAR file of the features.
'''

import numpy as np
from openmm.unit import angstroms

class CVReporter(object):
    '''
    An OpenMM reporter that writes a PLUMED-style COLVAR file of the features.
    This reporter writes to the `file` every `reportInterval` steps.
    The first column is the number of steps instead of time.
    Distances are in the units of Angstorms.
    '''

    def __init__(self, file: str = "COLVAR.dat",
                 reportInterval=100,
                 list_of_indexes: list[tuple[int, int]] = None,
                 append=False):
        '''
        Initialize the CVReporter object.

        :param file: The name of the file to write the CVs to. Default: COLVAR.dat
        :type file: str
        :param reportInterval: The interval at which to write the CVs. Default: 100
        :type reportInterval: int
        :param list_of_indexes: The list of indexes to calculate the CVs. Default: None
        :type list_of_indexes: list[tuple[int, int]]
        :param append: Append to existing file 
	    :type append: bool
	    '''
        
        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval
        self.list_of_cv = list_of_indexes
        self.n_cv = len(list_of_indexes)
        assert self.n_cv > 0, "No CVs added."

        self.buffer = np.zeros(self.n_cv)
        self.format = "{} " + "{:.4f} " * self.n_cv + "\n"
        self._out.write("#! FIELD time " + 
                        " ".join([f"dist_{i}_{j}" for i, j in self.list_of_cv]) + "\n")

    def __del__(self):
        self._out.flush()
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        step = simulation.currentStep
        coord = state.getPositions(asNumpy=True)
        for i, (a, b) in enumerate(self.list_of_cv):
            self.buffer[i] = np.linalg.norm((coord[a]-coord[b]).value_in_unit(angstroms))
        self._out.write(self.format.format(step, *self.buffer))