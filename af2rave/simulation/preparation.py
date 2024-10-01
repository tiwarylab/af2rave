import openmm.app as app
from openmm.unit import angstrom, molar

import pdbfixer
from . import DefaultForcefield


class SimulationBox:

    def __init__(self, filename: str,
                 forcefield: app.ForceField = DefaultForcefield,
                 **kwargs):
        """
        Generate the simulation box from a raw pdb file.
        Currently only soluble proteins are supported as we can only add water.

        This function performs the following tasks:
        1. use pdbfixer to add missing atoms, residues, and terminals
        2. add hydrogen, at the given pH
        3. solvate the system with water

        :param filename: path to the pdb file
        :type filename: str
        :param forcefield: forcefield to be used for adding hydrogens
        :type forcefield: OpenMM.app.ForceField
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
        """

        # store the input parameters
        self._filename = filename
        self._forcefield = forcefield
        self._kwargs = kwargs

        # keep track of the old topology
        self._old_top = app.PDBFile(filename).topology

        # create the simulation box according to kwargs
        self.pos, self.top = self.create_box()

        # maintain a mapping between old and new atom indices
        self._atom_index_map = self._generate_mapping_table()

    def create_box(self) -> tuple[list, app.Topology]:

        # fixer instance
        with open(self._filename, 'r') as ifs:
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
        pH = self._kwargs.get('pH', 7.0)
        modeller.addHydrogens(self._forcefield, pH=pH)

        # add solvent
        padding = self._kwargs.get('padding', 10 * angstrom)
        water_model = self._kwargs.get('water_model', 'tip3p')
        positive_ion = self._kwargs.get('positiveIon', 'Na+')
        negative_ion = self._kwargs.get('negativeIon', 'Cl-')
        ionic_strength = self._kwargs.get('ionicStrength', 0.0 * molar)
        modeller.addSolvent(self._forcefield,
                            padding=padding,
                            model=water_model,
                            neutralize=True,
                            positiveIon=positive_ion,
                            negativeIon=negative_ion,
                            ionicStrength=ionic_strength)

        return modeller.positions, modeller.topology

    def write_pdb(self, filename: str):
        """
        Write the simulation box to a pdb file.

        :param filename: path to the output pdb file
        :type filename: str
        """
        with open(filename, 'w') as f:
            app.PDBFile.writeFile(self.top, self.pos, f, keepIds=True)

    def _generate_mapping_table(self):

        index_lookup = {}
        forward_map = {}

        # Note that xxx.index counts from 0 and xxx.id counts from 1
        for a in self._old_top.atoms():

            # retrieve information for the old map
            index = a.index
            name = a.name
            resid = a.residue.id
            chain = a.residue.chain.id
            index_lookup[(chain, resid, name)] = index

        # now looks up this information in the new topology
        for a in self.top.atoms():
            index = a.index
            name = a.name
            resid = a.residue.id
            chain = a.residue.chain.id
            try:
                old_index = index_lookup[(chain, resid, name)]
                forward_map[old_index] = index
            except KeyError:
                pass

        return forward_map

    type AtomIndexLike = int | set[int] | list[int] | list[set[int]]

    def translate_atom_index(self, index: AtomIndexLike) -> AtomIndexLike:
        '''
        Translate atom index from input PDB to the output PDB file.

        After adding hydrogen, the atom index will be changed. This function
        translates the old atom index to the new atom index.

        :param index: atom index or a list of atom indices
        :type index: AtomIndexLike: int or set[int] or list[int] or list[set[int]]
        '''
        try:
            if isinstance(index, int):
                return self._atom_index_map[index]
            elif isinstance(index, set):
                return {self._atom_index_map[i] for i in index}
            elif isinstance(index, list):
                return [self.translate_atom_index(i) for i in index]
        except KeyError as e:
            raise ValueError("Atom index not found in the topology.") from e
