import numpy as np
import mdtraj as md
from functools import cached_property

from numba import njit
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

    @cached_property
    def name(self):
        i, j = self._ap
        resname_i = str(self._top.atom(i))
        resname_j = str(self._top.atom(j))
        return f"{resname_i}_{resname_j}"
    
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
        return self.std/self.mean

    def __len__(self):
        return self._ts.size

    def __eq__(self, value) -> bool:
        return self._ap == value._ap

    def get_plot_script(self):

        script = ""

        i, j = self._ap
        resid_i = self._top.atom(i).residue.index + 1
        resid_j = self._top.atom(j).residue.index + 1
        atom_name_i = self._top.atom(i).name
        atom_name_j = self._top.atom(j).name

        script += f"distance :{resid_i}@{atom_name_i} :{resid_j}@{atom_name_j}\n"

        if atom_name_i != "CA":
            script += f"show :{resid_i} a\n"
        if atom_name_j != "CA":
            script += f"show :{resid_j} a\n"

        return script


