import openmm.app as app
import openmm.unit as unit
import mdtraj as md
from openmm.unit import angstrom, molar

import pdbfixer

from . import Charmm36mFF

# python < 3.12 backward compatibility
from typing import Union
AtomIndexLike = Union[int, set[int], list[int], list[set[int]]]
TopologyLike = Union[app.Topology, md.Topology, str]


class TopologyMap:

    def __init__(self, old_top: TopologyLike, new_top: TopologyLike):
        '''
        Create a mapping between the old and new topology.

        :param old_top: old topology, either a path or an OpenMM or MDTraj topology object
        :type old_top: app.Topology or md.Topology or str
        :param new_top: new topology, either a path or an OpenMM or MDTraj topology object
        :type new_top: app.Topology or md.Topology or str
        '''

        if isinstance(old_top, str):
            self._old_top = app.PDBFile(old_top).topology
        elif isinstance(old_top, md.Topology):
            self._old_top = old_top.to_openmm()
        elif isinstance(old_top, app.Topology):
            self._old_top = old_top
        else:
            raise ValueError("old_top must be a path to a PDB file or an OpenMM or MDTraj topology object.")

        if isinstance(new_top, str):
            self._new_top = app.PDBFile(new_top).topology
        elif isinstance(new_top, md.Topology):
            self._new_top = new_top.to_openmm()
        elif isinstance(new_top, app.Topology):
            self._new_top = new_top
        else:
            raise ValueError("new_top must be a path to a PDB file or an OpenMM or MDTraj topology object.")

        self._atom_index_map = self._generate_mapping_table()

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
        for a in self._new_top.atoms():
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

    def map_atom_index(self, index: AtomIndexLike) -> AtomIndexLike:
        '''
        Map atom index from input PDB to the output PDB file.

        After adding hydrogen, the atom index will be changed. This function
        translates the old atom index to the new atom index.

        :param index: atom index or a list of atom indices
        :type index: AtomIndexLike: int or set[int] or list[int] or list[set[int]]
        '''
        try:
            if isinstance(index, set):
                return {self._atom_index_map[i] for i in index}
            elif isinstance(index, tuple):
                return tuple(self.map_atom_index(i) for i in index)
            elif isinstance(index, list):
                return [self.map_atom_index(i) for i in index]
            else:
                # mostly likely index is either an int or np.int64
                return self._atom_index_map[int(index)]
        except KeyError as e:
            raise ValueError(f"Atom index {e} in the new topology does "
                             "not exist in the old topology.") from e


class SimulationBox:

    def __init__(self, filename: str,
                 forcefield: app.ForceField = Charmm36mFF):
        '''
        Create a simulation box from a protein PDB box.

        :param filename: path to the pdb file
        :type filename: str
        :param forcefield: forcefield to be used for adding hydrogens
        :type forcefield: OpenMM.app.ForceField
        '''

        self._filename = filename
        self._forcefield = forcefield

    def create_box(self, **kwargs) -> tuple[list, app.Topology]:
        """
        Generate the simulation box from a raw pdb file.
        Currently only soluble proteins are supported as we can only add water.

        This function performs the following tasks:
        1. use pdbfixer to add missing atoms, residues, and terminals
        2. add hydrogen, at the given pH
        3. solvate the system with water, neutralize the system by adding ions

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

        '''
        PDB fixer automatically add disulfide bonds when two cys-S are close.
        First, this may not always be the case. Second, this behavior is not
        verbose in AMBER ffs as they actually have the CYM residue for disulfide-
        bonded CYS. Meanwhile, CHARMM ffs do not have CYM and modeller will
        complain. We will remove the disulfide bond here.
        '''
        disulfide_bonds = []
        for bond in modeller.topology.bonds():
            if bond.atom1.name == 'SG' and bond.atom2.name == 'SG':
                disulfide_bonds.append(bond)
                print(f"Removed sulfide bond between {bond.atom1} and {bond.atom2}")
        modeller.delete(disulfide_bonds)

        # add hydrogens
        self.pH = kwargs.get('pH', 7.0)
        modeller.addHydrogens(self._forcefield, pH=self.pH)

        # add solvent
        self.padding = kwargs.get('padding', 10 * angstrom)
        self.water_model = kwargs.get('water_model', 'tip3p')
        self.positive_ion = kwargs.get('positiveIon', 'Na+')
        self.negative_ion = kwargs.get('negativeIon', 'Cl-')
        self.ionic_strength = kwargs.get('ionicStrength', 0.0 * molar)

        if not unit.is_quantity(self.padding):
            self.padding *= angstrom
        if not unit.is_quantity(self.ionic_strength):
            self.ionic_strength *= molar

        modeller.addSolvent(self._forcefield,
                            padding=self.padding,
                            model=self.water_model,
                            neutralize=True,
                            positiveIon=self.positive_ion,
                            negativeIon=self.negative_ion,
                            ionicStrength=self.ionic_strength)

        self.top = modeller.topology
        self.pos = modeller.positions

        # initialize a mapping object
        self.top_map = TopologyMap(app.PDBFile(self._filename).topology, self.top)
        self.map_atom_index = self.top_map.map_atom_index

        return modeller.positions, modeller.topology

    def save_pdb(self, filename: str):
        """
        Write the simulation box to a pdb file.

        :param filename: path to the output pdb file
        :type filename: str
        """
        with open(filename, 'w') as f:
            app.PDBFile.writeFile(self.top, self.pos, f, keepIds=True)

    def __str__(self):
        return (f"SimulationBox('{self._filename}') at pH {self.pH}. "
                f"Solvated: {self.padding} {self.water_model} water "
                f"with {self.ionic_strength} {self.positive_ion}{self.negative_ion} ions.")
