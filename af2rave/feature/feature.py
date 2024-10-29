import numpy as np
import mdtraj as md
from functools import cached_property

from numpy.typing import ArrayLike


class Feature(object):

    def __init__(self, atom_pair: set[int], top: md.Topology, ts: ArrayLike):
        n_atoms = len(atom_pair)
        if n_atoms != 2:
            raise ValueError(f"Accept only two atoms as atom_pairs. Got {n_atoms}")
        else:
            self._ap = tuple(map(int, atom_pair))
        self._top = top
        self._ts = ts
        self._atom_i = self._top.atom(self._ap[0])
        self._atom_j = self._top.atom(self._ap[1])

    @cached_property
    def name(self):

        residue_i = self._atom_i.residue
        residue_j = self._atom_j.residue

        chain_i = residue_i.chain.chain_id
        chain_j = residue_j.chain.chain_id
        resid_i = residue_i.resSeq
        resid_j = residue_j.resSeq
        resname_i = residue_i.name
        resname_j = residue_j.name
        atom_name_i = self._atom_i.name
        atom_name_j = self._atom_j.name

        return f"{resname_i}{resid_i}{chain_i}-{atom_name_i}_{resname_j}{resid_j}{chain_j}-{atom_name_j}"

    @property
    def ap(self):
        return self._ap

    @property
    def ts(self):
        return self._ts

    @cached_property
    def mean(self):
        return np.mean(self._ts)

    @cached_property
    def std(self):
        return np.std(self._ts)

    @cached_property
    def cv(self):
        return self.std / self.mean

    def __len__(self):
        return self._ts.size

    def __eq__(self, value) -> bool:
        return self._ap == value._ap

    def get_plot_script(self):

        script = ""

        chain_i = self._atom_i.residue.chain.chain_id
        chain_j = self._atom_j.residue.chain.chain_id
        resid_i = self._atom_i.residue.resSeq
        resid_j = self._atom_j.residue.resSeq
        atom_name_i = self._atom_i.name
        atom_name_j = self._atom_j.name

        script += ("distance "
                   f"/{chain_i}:{resid_i}@{atom_name_i} "
                   f"/{chain_j}:{resid_j}@{atom_name_j}\n")

        if atom_name_i != "CA":
            script += f"show /{chain_i}:{resid_i} a\n"
        if atom_name_j != "CA":
            script += f"show /{chain_j}:{resid_j} a\n"

        return script
