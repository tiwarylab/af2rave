'''
The MetadynamicsSimulation class for running metaD simulations.
'''
from . import UnbiasedSimulation

import numpy as np
import af2rave.spib as af2spib

import openmm
import openmm.app as app
from openmm.unit import kilojoules_per_mole

from .reporter import MinimizationReporter

class MetadynamicsSimulation(UnbiasedSimulation):
    '''
    :param pdb_file: Path to OpenMM.app.pdbfile.PDBFile object
    :type pdb_file: str
    :param list_of_index: List of indices to calculate the CVs for biasing 
    :type list_of_index: list[tuple[int, int]]
    :param spib_file: Path to SPIB pickle result
    :type spib_file: str
    :param sigma: Standard deviation of Colvar data. Default: 1.5  
    :type sigma: float
    :param width: Gaussian width. Default: 1.0
    :type width: float
    :param gamma_factor: Multiplicative factor for gaussian sigma. Default: 1/20
    :type gamma_factor: int
    :param grid_bins: Number of bins for bias variable grid. Default: 150
    :type grid_bins: int
    :param grid_min: minimum value latent variable can explore
    :type grid_min: np.ndarray
    :param grid_max: maximum value latent variable can explore
    :type grid_max: np.ndarray
    :param bias_factor: Scaling factor for Gaussian height. Default: 10.0
    :type bias_factor: float
    :param height: Gaussian height. Default: 1.5 kilojoules per mole
    :type height: float
    :param bias_directory: Path to bias directory. Default: "bias/"
    :type bias_directory: str 
    '''
    def __init__(self, pdb_file, spib_file, **kwargs):
        super().__init__(pdb_file, **kwargs)
        
        self._spib_result = af2spib.SPIBResult.from_file(spib_file)
        self._spib_weights = self._spib_result.apparent_weight
        self._spib_bias = self._spib_result.apparent_bias       #unused
        
        self._gauss_sigma = kwargs.get('sigma', 1.5)
        self._gauss_width = kwargs.get('width', 1.0)
        self._gamma = kwargs.get('gamma_factor', 1/20)

        self._grid = kwargs.get('grid_bins', 150)
        self._grid_min = kwargs.get('grid_min')
        self._grid_max = kwargs.get('grid_max')

        if self._grid_min is None or self._grid_max is None:
            raise ValueError("grid_min and grid_max must be specified.")
        if len(self._grid_min) != len(self._grid_max):
            raise ValueError("grid_min and grid_max must have the same dimension.")

        self._dim = len(self._grid_min)
        self._grid_edges = np.column_stack([self._grid_min, self._grid_max])

        width = self._gauss_sigma * self._gauss_width * self._gamma
        self._width = np.array([width] * self._dim)

        # openmm metadynamics params
        self._bias_factor = kwargs.get('bias_factor', 10.0)
        self._gauss_height = kwargs.get('height', 1.5 * kilojoules_per_mole)
        self._biasDir = kwargs.get('bias_directory', "bias/")
#        self._rbiasDir = self._biasDir
        self.metadynamics = self._metad()
        

    def run(self, steps: int = 50000000):
        '''
        Run metadynamics from given pdb file. Default: 50 million steps (100 ns).

        :param steps: Number of steps to run the simulation.
            Default: 50 million steps (100 ns)
        :type steps: int
        '''
        self._add_reporter(self._get_thermo_reporter(steps))
        self.simulation.context.reinitialize()
        self.simulation.context.setPositions(self._pos)
        
        self.simulation.minimizeEnergy(
            maxIterations=500, reporter=MinimizationReporter(500)
        )

        self.save_pdb(self._prefix + "_minimized.pdb")
        self.metadynamics.step(self.simulation, steps)

        return self.simulation

    def _metad(self, **kwargs):
        
        cvs = [openmm.openmm.CustomBondForce("c*r") for _ in range(self._dim)]
        for cv in cvs:
            cv.addPerBondParameter("c")
        
        for i, atompair in enumerate(self._list_of_index):
            for j in range(self._dim):
                cvs[j].addBond(*atompair, [self._spib_weights[j][i]])
        
        # Since we used column_stack, for n-D case _grid_edges.shape will be (n, 2)
        # This holds even if for 1D case.
        CVs = [
            app.metadynamics.BiasVariable(
                cv,
                self._grid_edges[i][0],  # min
                self._grid_edges[i][1],  # max
                self._width[i],
                periodic=False,
                gridWidth=self._grid
            )
            for i, cv in enumerate(cvs)
        ]

        return app.metadynamics.Metadynamics(
            self.simulation.context.getSystem(),
            CVs,
            self._temp,
            biasFactor=self._bias_factor,
            height=self._gauss_height,
            frequency=self._progress_every,
            biasDir=self._biasDir,
            saveFrequency=self._progress_every,
            # rbiasDir=self._rbiasDir
        )
