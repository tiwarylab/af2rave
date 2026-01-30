import openmm.app as app
import openmm.unit as unit
import mdtraj as md
import numpy as np
from openmm.unit import angstrom, molar

import pdbfixer
from numbers import Integral

from . import Charmm36mFF

AtomIndexLike = int | set[int] | tuple[int] | list[int] | list[set[int]] | list[tuple[int]]
TopologyLike = app.Topology | md.Topology | str


class TopologyMap:
    '''
    Create a mapping between the old and new topology.

    :param old_top: old topology, either a path or an OpenMM or MDTraj topology object
    :type old_top: openmm.app.Topology or mdtraj.Topology or str
    :param new_top: new topology, either a path or an OpenMM or MDTraj topology object
    :type new_top: openmm.app.Topology or mdtraj.Topology or str
    :param resid_offset: 
        Offset to be added to the residue id to align two topologies. Default is 0.
        For example, if a residue in the old topology has resid 1 and 
        in the new topology has resid 101, then resid_offset = 100.
    '''

    def __init__(self, old_top: TopologyLike, new_top: TopologyLike, resid_offset = 0):

        self._old_top = self._load_topology(old_top)
        self._new_top = self._load_topology(new_top)
        self._atom_index_map = self._generate_mapping_table(resid_offset)

    def _load_topology(self, top: TopologyLike) -> app.Topology:
        '''
        Load the topology from a file or a topology object.

        :param top: topology, either a path or an OpenMM or MDTraj topology object
        :type top: openmm.app.Topology or mdtraj.Topology or str
        '''

        if isinstance(top, str):
            return app.PDBFile(top).topology
        elif isinstance(top, md.Topology):
            return top.to_openmm()
        elif isinstance(top, app.Topology):
            return top
        else:
            raise ValueError("top must be a path to a PDB file or an OpenMM or MDTraj topology object.")

    def _generate_mapping_table(self, resid_offset: int) -> dict[int, int]:

        index_lookup = {}
        forward_map = {}

        # Note that xxx.index counts from 0 and xxx.id counts from 1
        for a in self._old_top.atoms():

            # retrieve information for the old map
            index = a.index
            name = a.name
            resid = int(a.residue.id)
            chain = a.residue.chain.id
            index_lookup[(chain, resid, name)] = index

        # now looks up this information in the new topology
        for a in self._new_top.atoms():
            index = a.index
            name = a.name
            resid = int(a.residue.id)
            chain = a.residue.chain.id
            try:
                old_index = index_lookup[(chain, resid-resid_offset, name)]
                forward_map[old_index] = index
            except KeyError:
                pass

        return forward_map

    def map_atom_index(self, index: AtomIndexLike) -> AtomIndexLike:
        '''
        Map atom index (0-based) from input PDB to the output PDB file.

        After adding hydrogen, the atom index will be changed. This function
        translates the old atom index to the new atom index.

        :param index: atom index or a list of atom indices
        :type index: AtomIndexLike: int or set[int] or list[int] or list[set[int]]
        '''
        try:
            if isinstance(index, np.ndarray):
                return np.vectorize(self.map_atom_index)(index)
            elif isinstance(index, (set, tuple, list)):
                return type(index)(map(self.map_atom_index, index))
            elif isinstance(index, Integral):
                return self._atom_index_map[int(index)]
            else:
                raise TypeError("Unrecognized type for index.")
        except KeyError as e:
            atom_missing = list(self._old_top.atoms())[int(e.args[0])]
            raise ValueError(f"{atom_missing} in the new topology does "
                             "not exist in the old topology.")


class SimulationBox:
    '''
    Create a simulation box from a protein PDB box.

    :param filename: path to the pdb file
    :type filename: str
    :param forcefield: forcefield to be used for adding hydrogens
    :type forcefield: OpenMM.app.ForceField
    '''

    def __init__(self, filename: str,
                 forcefield: app.ForceField = Charmm36mFF):

        self._filename = filename
        self._forcefield = forcefield

        # fixer instance
        with open(self._filename, 'r') as ifs:
            fixer = pdbfixer.PDBFixer(pdbfile=ifs)

        # finding and adding missing residues including terminals
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)

        '''
        PDB fixer automatically add disulfide bonds when two cys-S are close.
        First, this may not always be the case. Second, this behavior is not
        verbose in AMBER ffs as they actually have the CYM residue for disulfide-
        bonded CYS. Meanwhile, CHARMM ffs do not have CYM and modeller will
        complain. We will remove the disulfide bond here.
        '''
        modeller = app.Modeller(fixer.topology, fixer.positions)
        ds_bonds = []
        for bond in modeller.topology.bonds():
            if bond.atom1.name == 'SG' and bond.atom2.name == 'SG':
                ds_bonds.append(bond)
        modeller.delete(ds_bonds)

        self.top = modeller.topology
        self.pos = modeller.positions

    def add_disulfide_bond(self, resid: list[tuple[int, int]]):
        '''
        By default the simulation box after create_box() does not have any disulfide bonds.
        This method will add back the bonds when necessary.
        This method currently only works with AMBER forcefields.

        :param resid: list of tuples of CYS residues that are bonded
        :type resid: list[tuple[int, int]]
        '''

        # create modeller instance
        modeller = app.Modeller(self.top, self.pos)

        # add disulfide bonds
        for i, j in resid:
            residues = {int(r.index): r for r in modeller.topology.residues()}
            residue_i, residue_j = residues[i], residues[j]
            if residue_i.name != "CYS" or residue_j.name != "CYS":
                raise ValueError(f"Residues {i}({residue_i.name}) {j}({residue_j.name}) both need to be cysteines.")
            atoms_i = {atom.name: atom for atom in residue_i.atoms()}
            atoms_j = {atom.name: atom for atom in residue_j.atoms()}

            if "SG" in atoms_i and "SG" in atoms_j:
                modeller.topology.addBond(atoms_i["SG"], atoms_j["SG"])
#                residue_i.name, residue_j.name = 'CYX', 'CYX'
            else:
                if "SG" not in atoms_i:
                    raise ValueError(f"Residues {i} do not have an SG atom.")
                else:
                    raise ValueError(f"Residues {j} do not have an SG atom.")
            
            to_delete = []
            if "HG" in atoms_i:
                to_delete.append(atoms_i["HG"])
            if "HG" in atoms_j:
                to_delete.append(atoms_j["HG"])
            modeller.delete(to_delete)
        
        self.top = modeller.topology
        self.pos = modeller.positions

    def create_box(self, **kwargs) -> tuple[list, app.Topology]:
        """
        Generate the simulation box from a raw pdb file.
        Currently only soluble proteins are supported as we can only add water.

        This function performs the following tasks:
        1. use pdbfixer to add missing atoms, residues, and terminals
        2. add hydrogen, at the given pH
        3. solvate the system with water, neutralize the system by adding ions

        :param float pH: float: pH of the system. Default is 7.0
        :param float padding: padding around the protein. Default is 10. Unit: Angstrom.
        :param str water_model: water model to be used. Default is 'tip3p'
        :param str positiveIon: positive ion used to neutralize the system. Default is 'Na+'.
        :param str negativeIon: negative ion used to neutralize the system. Default is 'Cl-'
        :param float ionicStrength: ionic strength of the system. Default is 0.0. Unit: molar
        """

        # create modeller instance
        modeller = app.Modeller(self.top, self.pos)

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
                            ionicStrength=self.ionic_strength
        )

        self.top = modeller.topology
        self.pos = modeller.positions

        # initialize a mapping object
        self.top_map = TopologyMap(app.PDBFile(self._filename).topology, self.top)
        self._map_atom_index = self.top_map.map_atom_index

        return modeller.positions, modeller.topology

    @property
    def map_atom_index(self):
        '''
        Map atom index from input PDB to the output PDB file.
        This is a callable property.

        After adding hydrogen, the atom index will be changed. This function
        translates the old atom index to the new atom index.

        Example:

        .. code-block:: python

                box = SimulationBox(filename)
                box.create_box()
                box.map_atom_index(1)

        :param index: atom index or a list of atom indices
        :type index: AtomIndexLike: int or set[int] or list[int] or list[set[int]]
        '''
        return self._map_atom_index

    def save_pdb(self, filename: str):
        """
        Write the simulation box to a pdb file.

        :param str filename: path to the output pdb file
        """
        with open(filename, 'w') as f:
            app.PDBFile.writeFile(self.top, self.pos, f, keepIds=True)

    def __str__(self):
        return (f"SimulationBox('{self._filename}') at pH {self.pH}. "
                f"Solvated: {self.padding} {self.water_model} water "
                f"with {self.ionic_strength} {self.positive_ion}{self.negative_ion} ions.")
