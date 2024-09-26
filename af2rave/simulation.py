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

import torch
import numpy as np


def create_simulation_box(filename: str,
                          forcefield,
                          outfile: str = None,
                          **kwargs) -> tuple[list, app.Topology]:
    """
    Generate the simulation box from a raw pdb file.
    Currently only soluble proteins are supported as we can only add water.
    Membrane systems will need to be addressed later.

    This function performs the following tasks:
    1. use pdbfixer to add missing atoms, residues, and terminals
    2. add hydrogen, at the given pH
    3. solvate the system with water

    :param filename: path to the pdb file
    :type filename: str
    :param forcefield: forcefield to be used for adding hydrogens
    :type forcefield: OpenMM.app.ForceField
    :param outfile: Path to the output PDB file. None to suppress file output.
    :type outfile: str or None
    :param pH: float: pH of the system. Default is 7.0
    :type pH: float
    :param padding: padding around the protein. Default is 10. Unit: Angstrom.
    :type padding: float
    :param water_model: water model to be used. Default is 'tip3p'
    :type water_model: str
    :param positiveIon: positive ion used to neutralize the system. Default is 'Na+'
    :type positiveIon: str
    :param negativeIon: negative ion used to neutralize the system. Default is 'Cl-'
    :type negativeIon: str
    :param ionicStrength: ionic strength of the system. Default is 0.0. Unit: molar
    :type ionicStrength: float

    :return: positions, topology.
    :rtype: tuple[list, OpenMM.app.Topology]
    """

    import pdbfixer
    from openmm.unit import angstrom, molar

    # fixer instance
    ifs = open(filename, 'r')
    fixer = pdbfixer.PDBFixer(pdbfile=ifs)

    # finding and adding missing residues including terminals
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)

    # create modeller instance
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # add hydrogens
    pH = kwargs.get('pH', 7.0)
    modeller.addHydrogens(forcefield, pH=pH)

    # add solvent
    padding = kwargs.get('padding', 10 * angstrom)
    water_model = kwargs.get('water_model', 'tip3p')
    positive_ion = kwargs.get('positiveIon', 'Na+')
    negative_ion = kwargs.get('negativeIon', 'Cl-')
    ionic_strength = kwargs.get('ionicStrength', 0.0 * molar)
    modeller.addSolvent(forcefield,
                        padding=padding,
                        model=water_model,
                        neutralize=True,
                        positiveIon=positive_ion,
                        negativeIon=negative_ion,
                        ionicStrength=ionic_strength)

    if outfile is not None:
        with open(outfile, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    return modeller.positions, modeller.topology


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
        self._out.write("#! TIME " + " ".join([f"dist_{i}_{j}" for i, j in self.list_of_cv]) + "\n")

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
        
    def _get_system_integrator(self) -> app.Simulation:
        '''
        Create the integrator for the system using LangevinMiddleIntegrator. 
        Finds the CUDA platform if available and will fallback to CPU if not.
        Returns the OpenMM simulation object.

        '''
        # finds device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        system = self.forcefield.createSystem(self.topology,
                                              nonbondedMethod=app.PME,
                                              nonbondedCutoff=self.cutoff*angstroms,
                                              constraints=app.HBonds)
                                              
        integrator = openmm.LangevinMiddleIntegrator(self.temp*kelvin, 1/picoseconds, self.dt*picoseconds)

        if device.type == "cpu":
            simulation = app.Simulation(self.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName("CUDA")
            simulation = app.Simulation(self.topology, system, integrator, platform)
            
        return simulation
            
