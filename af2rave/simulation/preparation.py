import openmm.app as app
from openmm.unit import angstrom, molar

import pdbfixer

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