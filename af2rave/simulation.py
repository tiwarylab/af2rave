'''
The simulation module for AF2RAVE performs molecular dynamics simulations.
This module is mostly a wrapper around OpenMM, and provides utilies that create
simulation boxes, run simulations, and analyze trajectories.
'''

import openmm.app as app
from openmm.unit import angstroms, picoseconds, kelvin

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
    :param outfile: path to the output box PDB file. If None then no file is written.
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
    # Note that 
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
    modeller.addHydrogens(forcefield, pH = pH)

    # add solvent
    padding = kwargs.get('padding', 10 * angstrom)
    water_model = kwargs.get('water_model', 'tip3p')
    positive_ion = kwargs.get('positiveIon', 'Na+')
    negative_ion = kwargs.get('negativeIon', 'Cl-')
    ionic_strength = kwargs.get('ionicStrength', 0.0 * molar)
    modeller.addSolvent(forcefield, 
                        padding = padding , 
                        model = water_model, 
                        neutralize = True, 
                        positiveIon = positive_ion, 
                        negativeIon = negative_ion,
                        ionicStrength = ionic_strength)

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

    def __init__(self, file: str = "COLVAR.dat", reportInterval = 100, list_of_indexes: list[tuple[int, int]] = None):
        '''
        Initialize the CVReporter object. 

        :param file: The name of the file to write the CVs to. Default: COLVAR.dat
        :type file: str
        :param reportInterval: The interval at which to write the CVs. Default: 100
        :type reportInterval: int
        :param list_of_indexes: The list of indexes to calculate the CVs. Default: None
        :type list_of_indexes: list[tuple[int, int]]
        '''

        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.list_of_cv = list_of_indexes
        self.n_cv = len(list_of_indexes)
        try:
            assert self.n_cv > 0
        except:
            raise ValueError("No CVs added.")
        
        self.buffer = np.zeros(self.n_cv)
        self.format = "{} " + "{:.4f} " * self.n_cv + "\n"
        self._out.write("#! TIME " + " ".join([f"dist_{i}_{j}" for i, j in self.list_of_cv]) + "\n")

    def __del__(self):
        self._out.flush()
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        step = simulation.currentStep
        coord = state.getPositions(asNumpy=True)
        for i, (a, b) in enumerate(self.list_of_cv):
            self.buffer[i] = np.linalg.norm(coord[a]-coord[b]).value_in_unit(angstroms)
        self._out.write(self.format.format(step, *self.buffer))