'''
This CVReporter generates a PLUMED-style COLVAR file of the features.
'''

import numpy as np
import openmm
from openmm.unit import angstroms


class CVReporter(object):
    '''
    An OpenMM reporter that writes a PLUMED-style COLVAR file of the features.
    This reporter writes to the `file` every `reportInterval` steps.
    The first column is the number of steps instead of time.
    Distances are in the units of Angstorms.
    '''

    def __init__(self, file: str = "COLVAR.dat",
                 reportInterval: int = 100,
                 list_of_indexes: list[tuple[int, int]] = None,
                 append: bool = False):
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
        if not append:
            self._out.write("#! FIELD time " +
                            " ".join([f"dist_{i}_{j}" for i, j in self.list_of_cv]) +
                            "\n")

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
            self.buffer[i] = np.linalg.norm((coord[a] - coord[b]).value_in_unit(angstroms))
        self._out.write(self.format.format(step, *self.buffer))


if openmm.version.short_version >= '8.1.0':
    # The class can have any name but it must subclass MinimizationReporter.
    class MinimizationReporter(openmm.MinimizationReporter):

        def __init__(self, maxIter: int = 500):
            super().__init__()
            self._maxIter = maxIter
            self._round = 1
            print("Minimizing using maxIteration per epoch: ", self._maxIter)

        # you must override the report method and it must have this signature.
        def report(self, iteration, x, grad, args):
            '''
            the report method is called every iteration of the minimization.

            Args:
                iteration (int): The index of the current iteration. This refers
                                to the current call to the L-BFGS optimizer.
                                Each time the minimizer increases the restraint strength,
                                the iteration index is reset to 0.

                x (array-like): The current particle positions in flattened order:
                                the three coordinates of the first particle,
                                then the three coordinates of the second particle, etc.

                grad (array-like): The current gradient of the objective function
                                (potential energy plus restraint energy) with
                                respect to the particle coordinates, in flattened order.

                args (dict): Additional statistics  about the current state of minimization.
                    In particular:
                    "system energy": the current potential energy of the system
                    "restraint energy": the energy of the harmonic restraints
                    "restraint strength": the force constant of the restraints (in kJ/mol/nm^2)
                    "max constraint error": the maximum relative error in the length of any constraint

            Returns:
                bool : Specify if minimization should be stopped.
            '''

            if iteration == self._maxIter - 1:
                print(f"Minimization epoch {self._round}, "
                     f"constraint k = {args['restraint strength']:.2e}. "
                     f"Max constraint error: {args['max constraint error']:.5e}"
                )
                self._round += 1

            return False
