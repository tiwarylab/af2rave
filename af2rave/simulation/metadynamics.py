'''
The MetadynamicsSimulation class for running metaD simulations.
'''
from . import UnbiasedSimulation

import numpy as np
import af2rave.spib as af2spib

import openmm
import openmm.app as app
from openmm.unit import kilojoules_per_mole

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
    :type grid_min: ndarray of shape (1,) or (2,)
    :param grid_max: maximum value latent variable can explore
    :type grid_max: ndarray of shape (1,) or (2,)
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
        self._spib_weights = self._spib_result._z_mean_encoder["weight"]
        self._spib_bias = self._spib_result._z_mean_encoder["bias"] #unused
        
        self._gauss_sigma = kwargs.get('sigma', 1.5)
        self._gauss_width = kwargs.get('width', 1.0)
        width = np.array([self._gauss_width, self._gauss_width])
        self._gamma = kwargs.get('gamma_factor', 1/20)
        self.width = self._gauss_sigma*width*self._gamma

        self._grid = kwargs.get('grid_bins', 150)
        self._grid_min = kwargs.get('grid_min')
        self._grid_max = kwargs.get('grid_max')
        
        if len(self._grid_min) == 1:
            self._dim = 1
            self._grid_edges = np.array([self._grid_min, self._grid_max])
            print(self._grid_edges)
            
        elif len(self._grid_min) == 2:
            self._dim = 2
            self._grid_edges = np.array([[self._grid_min[0], self._grid_max[0]],
                                         [self._grid_min[1], self._grid_max[1]]])

        # openmm metadynamics params
        self._bias_factor = kwargs.get('bias_factor', 10.0)
        self._gauss_height = kwargs.get('height', 1.5*kilojoules_per_mole)
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
        
        if openmm.version.short_version >= "8.1.0":
            from .reporter import MinimizationReporter
            self.simulation.minimizeEnergy(
                maxIterations=500, reporter=MinimizationReporter(500)
            )
        else:
            self.simulation.minimizeEnergy(maxIterations=500)
        self.save_pdb(self._prefix + "_minimized.pdb")
        self.metadynamics.step(self.simulation, steps)

        return self.simulation

    def _metad(self, **kwargs):
        
        cvs = [openmm.openmm.CustomBondForce("c*r") for i in range(self._dim)]
        [cv.addPerBondParameter("c") for cv in cvs]
        
        for i, atompair in enumerate(self._list_of_index):
            [cvs[j].addBond(atompair[0], atompair[1], [self._spib_weights[j][i]]) for j in range(self._dim)]
        
        if self._dim == 1:
            CVs = [app.metadynamics.BiasVariable(cv,
                                                 self._grid_edges[0], #min
                                                 self._grid_edges[1], #max
                                                 self.width[i],
                                                 periodic=False,
                                                 gridWidth=self._grid) for i, cv in enumerate(cvs)]
        elif self._dim == 2:
            CVs = [app.metadynamics.BiasVariable(cv,
                                                 self._grid_edges[i][0], #min
                                                 self._grid_edges[i][1], #max
                                                 self.width[i],
                                                 periodic=False,
                                                 gridWidth=self._grid) for i, cv in enumerate(cvs)]
        
        return app.metadynamics.Metadynamics(self.simulation.context.getSystem(),
                                              CVs,
                                              self._temp,
                                              biasFactor=self._bias_factor,
                                              height=self._gauss_height,
                                              frequency=self._progress_every,
                                              biasDir=self._biasDir,
                                              saveFrequency=self._progress_every,
#                                              rbiasDir=self._rbiasDir
                                              )
