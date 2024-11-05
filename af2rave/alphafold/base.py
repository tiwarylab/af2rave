'''
This is the base class of the AlphaFold class. It will be inherited
by either LocalColabFold interface or OpenFold interface.
'''

import os

class AlphaFoldBase(object):

    def __init__(self, **kwargs):

        self._msa = kwargs.get('msa', "8:16")
        self._seeds = kwargs.get('seeds', 128)
        self._sequence = kwargs.get("sequence", None)

        if not os.path.exists(self._sequence):
            raise FileNotFoundError(f"MSA file {self._sequence} not found.")
        
    def predict(self, seqs, output_dir, **kwargs):
        raise NotImplementedError("AlphaFoldBase::predict() is a pure virtual function.")
        