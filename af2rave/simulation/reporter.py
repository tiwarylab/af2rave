'''
This CVReporter generates a PLUMED-style COLVAR file of the features.
'''

import numpy as np
import openmm
from openmm.unit import angstroms


class CVReporter(object):
    '''
    An OpenMM reporter that writes a PLUMED-style COLVAR file of the features.
    This reporter writes to the ``file`` every ``reportInterval`` steps.
    The first column is the number of steps instead of time.
    Distances are in the units of Angstorms.

    This reporter calculates distances without a PBC. This is equivalent to NOPBC
    option in PLUMED. This should do the right thing most of the time.

    :param file: The name of the file to write the CVs to. Default: COLVAR.dat
    :type file: str
    :param reportInterval: The interval at which to write the CVs. Default: 100
    :type reportInterval: int
    :param list_of_indexes: The list of indexes to calculate the CVs. Default: None
    :type list_of_indexes: list[tuple[int, int]]
    :param append: Append to existing file
    :type append: bool
    '''

    def __init__(self, file: str = "COLVAR.dat",
                 reportInterval: int = 100,
                 list_of_indexes: list[tuple[int, int]] = None,
                 append: bool = False):

        self._out = open(file, 'a' if append else 'w')
        self._reportInterval = reportInterval

        # Book keeping
        self.list_of_cv = np.asarray(list_of_indexes)
        self._atom_idx = self.list_of_cv.flatten()
        self._n_cv = len(list_of_indexes)
        self._n_atom = 2 * self._n_cv
        if not self._n_cv > 0:
            raise ValueError("No CVs added.")
        if self.list_of_cv.shape[1] != 2:
            raise ValueError("Each CV must be a pair of indexes.")

        # This is the part used to monitor big coordinate jumps
        self._x_t0 = np.zeros((self._n_atom, 3), dtype=float)
        self._images = np.zeros((self._n_atom, 3), dtype=int)
        self._started = False

        self._buffer = np.zeros(self._n_cv)
        self._format = "{} " + "{:.4f} " * self._n_cv + "\n"
        if not append:
            self._out.write("#! FIELD time " +
                            " ".join([f"dist_{i}_{j}" for i, j in self.list_of_cv]) +
                            "\n")

    def __del__(self):
        self._out.flush()
        self._out.close()

    def describeNextReport(self, simulation):
        '''
        OpenMM reporter method to describe the next report.
        '''
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        if not self._started:
            # I am not sure if this is the most robust way to do this.
            # It is uncertain if the first alignment happens at which stage of step 0.
            # But it works for now.
            self._started = True
            state = simulation.context.getState(getPositions=True)
            coord = state.getPositions(asNumpy=True).value_in_unit(angstroms)
            self._x_t0[:] = np.array([coord[k] for k in self._atom_idx], dtype=float)
            return (1, True, False, False, False, False)
        else:
            return (steps, True, False, False, False, False)
        # for openmm >= 8.2.0
        # return {
        #     'steps': steps, 
        #     'periodic': None, 
        #     'include':['positions']
        # }

    def report(self, simulation, state):
        '''
        OpenMM reporter method to report the current state.
        '''

        step = simulation.currentStep
        coord = state.getPositions(asNumpy=True).value_in_unit(angstroms)
        box = np.diag(state.getPeriodicBoxVectors().value_in_unit(angstroms))

        x_t = np.array([coord[k] for k in self._atom_idx], dtype=float)
        if self._started:
            self._images += np.rint((x_t - self._x_t0) / box).astype(int)  
        else:
            self._started = True

        # Store old value and compute new, imaged coordinates
        self._x_t0 = x_t.copy()
        x_t -= self._images * box

        # Compute the distances and write files
        self._buffer[:] = np.linalg.norm(
            np.array([x_t[a] - x_t[b] for a, b in np.arange(self._n_atom).reshape(-1, 2)]), 
            axis=1
        )
        self._out.write(self._format.format(step, *self._buffer))


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
