'''
The simulation module for AF2RAVE performs molecular dynamics simulations.
This module is mostly a wrapper around OpenMM, and provides utilies that create
simulation boxes, run simulations, and analyze trajectories.
'''
import os
import sys
from sys import stdout
import pickle

from mdtraj.reporters import XTCReporter

import openmm
import openmm.app as app
from openmm.unit import angstroms, picoseconds, kelvin

import numpy as np

class UnbiasedSimulation():
    '''
    The goal here is the user will use this module like this:

    > import af2rave.simulation as af2sim
    > sim = af2sim.UnbiasedSimulation(<some arguments>)
    > sim.run(<some other arguments, preferably as few as possible>)

    Then throw this 3-line python script to a cluster.
    '''

    
    def __init__(self, pdb_file,
                forcefield,
                list_of_indexes,
                temp: int = 310,
                pressure: int = 1,
                dt: float = 0.002,
                cutoff: float = 10.0,
                steps: int = 50000000,
                **kwargs,
                ):
        '''
        Simulation parameters
        
        :param pdb_file: Path to OpenMM.app.pdbfile.PDBFile object
        :type pdb_file: str
        :param forcefield: OpenMM.app.ForceField object
        :type forcefield: OpenMM.app.ForceField
        :param list_of_indexes: List of indexes to calculate the CVs
        :type list_of_indexes: list[tuple[int, int]]
        :param temp: Temperature of the system. Default: 310 K
        :type temp: float
        :param pressure: Pressure of the system. Default: 1 Bar
        :type pressure: float
        :param dt: Time step of the simulation. Default: 0.002 ps
        :type dt: float
        :param cutoff: Nonbonded cutoff. Default: 10.0 Angstroms
        :type cutoff: float
        :param steps: Simulation steps. Default:  50 million steps (100 ns)
        :type steps: int
        :param cv_file: File to write CVs to. Default: COLVAR.dat
        :type cv_file: str
        :param reportInterval: The interval at which to write the CVs. Default: 100
        :type reportInterval: int
        :param out_filename: File to write output XTC trajectory file to. Default: traj.xtc
        :type out_filename: str
        :param out_freq: Frequency of systems' state written to a trajectory file. Default: 1000
        :type out_freq: int
        :param append: Appends to existing file. Default: False
        :type append: bool
        
        '''
        
        self.list_of_indexes = list_of_indexes
        
        self.pdb_file = app.pdbfile.PDBFile(pdb_file)
        self.positions = self.pdb_file.positions
        self.topology = self.pdb_file.topology
        
        self.forcefield = forcefield
        self.temp = temp
        self.pressure = pressure
        self.dt = dt
        self.cutoff = cutoff
        self.steps = steps

        # Reporter parameters
        self.cv_file = kwargs.get('cv_file', "COLVAR.dat")
        self.reportInterval = kwargs.get('reportInterval', 100)
        self.out_filename = kwargs.get('out_filename', "traj.xtc")
        self.out_freq = kwargs.get('out_freq', 1000)
        self.append = kwargs.get('append', False)

        # Check platforms and devices, etc.
        n_platforms = openmm.Platform.getNumPlatforms()
        platforms = [openmm.Platform.getPlatform(i) for i in range(n_platforms)]
        platform_names = [platform.getName() for platform in platforms]
        
        # We will use platforms in the following order
        pltfm_selection = ["CUDA", "OpenCL", "CPU"]
        self.platform = None
        try:
            for i, plt in enumerate(pltfm_selection):
                if plt in platform_names:
                    print(f"Using {plt} platform.")
                    self.platform = platforms[i]
                    break
        except:
            print("No suitable platform found. Attempted platforms: CUDA, OpenCL, CPU")
        
    def _get_system_integrator(self) -> app.Simulation:
        '''
        Create the integrator for the system using LangevinMiddleIntegrator. 
        Finds the CUDA platform if available and will fallback to CPU if not.
        Returns the OpenMM simulation object.

        '''
        
        system = self.forcefield.createSystem(self.topology,
                                              nonbondedMethod=app.PME,
                                              nonbondedCutoff=self.cutoff*angstroms,
                                              constraints=app.HBonds)
                                              
        integrator = openmm.LangevinMiddleIntegrator(self.temp*kelvin, 
                                                     1/picoseconds, 
                                                     self.dt*picoseconds)

        simulation = app.Simulation(self.topology, system, integrator, self.platform)
            
        return simulation
    
    def run(self,
            barostat: bool = True,
            save: bool = True,
            restart: bool = False):
        '''
        Run the simulation from given pdb file. Default: 50 million steps (100 ns).
        
        :param barostat:
        :type barostat: bool
        :param save: Saves simulation if True. Default: True
        :type save: bool
        :param restart: Restarts simulation from saved checkpoint if True. Default: False
        :type restart: bool
        '''
        simulation = self._get_system_integrator()
        
        if restart == True and os.path.exists("checkpoint.chk"):
            simulation.loadCheckpoint("checkpoint.chk")
        else:
            raise FileNotFoundError("Checkpoint file does not exist")
        
        if restart == False:
            simulation.context.setPositions(self.positions)
            simulation.minimizeEnergy()
            
        # adding barostat to simulation if True
        if barostat == True:
            simulation.context.getSystem().addForce(MonteCarloBarostat(self.pressure*bar, self.temp*kelvin))
            simulation.context.reinitialize(preserveState=True)
        
        simulation.reporters.append(CVReporter(self.cv_file, self.reportInterval, self.list_of_indexes, self.append))
        simulation.reporters.append(XTCReporter(self.out_filename, self.out_freq, append=self.append,))
        simulation.reporters.append(StateDataReporter(stdout, self.out_freq, step=True,
            potentialEnergy=True, temperature=True, volume=True, density=True,
            progress=True, remainingTime=True, totalSteps=self.steps, elapsedTime=True, speed=True, separator="\t", append=self.append))
        simulation.step(self.steps)

        if save == True:
            simulation.saveCheckpoint("checkpoint.chk")

        return simulation


